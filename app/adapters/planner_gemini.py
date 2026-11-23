"""Live semantic scene planner using Google Gemini Interactions API with structured Pydantic output."""

from typing import List, Optional

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from app.adapters.base import PlannedScene, ScenePlannerAdapter
from app.adapters.gemini_pool import gemini_key_pool
from app.adapters.mock_adapters import (
    MockScenePlannerAdapter,
    align_explicit_scenes_to_duration,
    extract_visual_search_query,
    segment_words_into_scenes,
)
from app.logging_config import get_logger
from app.pipeline.safety import analyze_scene_safety
from app.schemas import SceneHintInput, WordTimestamp

logger = get_logger("planner_gemini")


class LLMSceneVisualPlan(BaseModel):
    scene_index: int
    search_query: str = Field(
        description="Concise 3-7 word ENGLISH stock-video search query matching the scene's contextual meaning (never literal homonyms)."
    )
    alt_queries: List[str] = Field(
        default_factory=list,
        description="2 fallback English stock search queries with slightly broader visual framing.",
    )
    generative_prompt: str = Field(
        description="Photorealistic cinematic visual description for AI generation if stock footage is weak. Never include real people's faces or brand logos."
    )
    disambiguation_notes: Optional[str] = Field(
        default=None,
        description="Explanation if a homonym, metaphor, statistic, or sensitive topic was resolved.",
    )
    prefer_generative: bool = Field(
        default=False,
        description="True only for surreal, abstract, or futuristic visuals where stock footage is typically poor AND no real person/brand is mentioned.",
    )


class LLMPlanBatch(BaseModel):
    scenes: List[LLMSceneVisualPlan]


class GeminiScenePlannerAdapter(ScenePlannerAdapter):
    provider_name = "gemini"

    def __init__(self, api_key: str, model_name: str = "gemini-3.5-flash-lite") -> None:
        self.api_key = api_key
        self.model_name = model_name
        self._fallback = MockScenePlannerAdapter()

    def plan_scenes(
        self,
        transcript: str,
        words: List[WordTimestamp],
        total_duration: float,
        style: str,
        avoid: List[str],
        explicit_scenes: Optional[List[SceneHintInput]] = None,
        min_scene_duration: float = 2.5,
        max_scene_duration: float = 9.0,
    ) -> List[PlannedScene]:
        # Step 1: Deterministic ASR boundary alignment (guarantees 100% coverage + sentence/phrase cuts + no 0.5s flashing!)
        if explicit_scenes:
            segments = align_explicit_scenes_to_duration(explicit_scenes, words, total_duration)
        else:
            raw = segment_words_into_scenes(words, total_duration, min_scene_duration, max_scene_duration)
            segments = [(s, e, txt, None) for s, e, txt in raw]

        if not self.api_key:
            return self._fallback.plan_scenes(
                transcript, words, total_duration, style, avoid, explicit_scenes, min_scene_duration, max_scene_duration
            )

        # Build concise prompt with all scenes + safety/avoid rules
        scene_lines = []
        for idx, (s, e, txt, hint) in enumerate(segments):
            hint_str = f" | visual_hint: {hint}" if hint else ""
            scene_lines.append(f"[{idx}] ({s:.2f}s-{e:.2f}s): {txt}{hint_str}")

        prompt = (
            "You are a senior documentary B-roll video editor. For each narration scene below, "
            "design the best visual search query (in clear ENGLISH even if the narration is Hindi or Tamil) "
            "that captures the CONTEXTUAL MEANING of the line — never literal keyword traps.\n\n"
            "STRICT RULES:\n"
            "1. Disambiguate homonyms using the full script (e.g., 'cloud computing' -> server room/data center, NOT sky; "
            "'python code' -> developer coding, NOT snake; 'bank loan' -> finance/office, NOT river; 'value' -> quality/craft or financial chart, NOT pile of cash).\n"
            "2. For abstract lines ('trust is earned slowly'), pick a concrete visual metaphor (e.g., 'two colleagues shaking hands warm office').\n"
            "3. For real named people or brands: NEVER depict their face or invent logos — use generic editorial/category B-roll.\n"
            "4. For sad/tragic/sensitive news: pick dignified, subdued, calm visuals — never smiling or upbeat scenes.\n"
            f"5. Desired visual style: {style}\n"
            f"6. STRICTLY AVOID: {', '.join(avoid) if avoid else 'none'}\n"
            "7. If a visual_hint is provided, honor it closely.\n"
            "8. For search queries, use 3-5 specific, professional domain nouns (e.g. 'data center server racks technology', 'doctor examining patient hospital'). Never include ambiguous decorative words like 'lights', 'fire', 'tree', 'candle', 'night', 'holiday', or 'celebration' unless explicitly requested by the script.\n\n"
            f"FULL TRANSCRIPT:\n{transcript}\n\n"
            "SCENES:\n" + "\n".join(scene_lines)
        )

        llm_by_idx = {}
        def _do_plan(key: str) -> types.GenerateContentResponse:
            client = genai.Client(api_key=key, http_options={"api_version": "v1beta"})
            return client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=LLMPlanBatch,
                    temperature=0.2,
                ),
            )

        try:
            if self.api_key:
                gemini_key_pool.add_key(self.api_key)
            resp = gemini_key_pool.execute_with_rotation(_do_plan)
            batch = LLMPlanBatch.model_validate_json(resp.text)
            llm_by_idx = {item.scene_index: item for item in batch.scenes}
        except Exception as exc:
            logger.warning("gemini_planner_fallback", error=str(exc))

        planned: List[PlannedScene] = []
        for idx, (start, end, text, hint) in enumerate(segments):
            safety = analyze_scene_safety(text, transcript, avoid)
            llm_item = llm_by_idx.get(idx)

            if llm_item:
                search_query = llm_item.search_query.strip()
                alt_queries = [q.strip() for q in llm_item.alt_queries if q.strip()][:2]
                gen_prompt = llm_item.generative_prompt.strip()
                disambig = llm_item.disambiguation_notes
                prefer_gen = llm_item.prefer_generative and not safety.block_ai_generation
            else:
                search_query = hint or safety.suggested_visual_override or extract_visual_search_query(text)
                alt_queries = [f"{search_query} cinematic wide", f"{search_query} professional documentary"]
                gen_prompt = f"{search_query}, {style}, photorealistic, no text, no logos"
                disambig = "; ".join(safety.homonym_resolutions) if safety.homonym_resolutions else None
                prefer_gen = bool(safety.is_abstract and not safety.block_ai_generation)

            # Hard safety override: never allow brand/person AI generation and keep homonym note visible
            if safety.suggested_visual_override and (safety.named_brands or safety.homonym_resolutions):
                if safety.named_brands:
                    search_query = safety.suggested_visual_override
                if safety.homonym_resolutions and not disambig:
                    disambig = "; ".join(safety.homonym_resolutions)

            planned.append(
                PlannedScene(
                    scene_index=idx,
                    start_time=start,
                    end_time=end,
                    narration_text=text,
                    visual_hint=hint,
                    search_query=search_query,
                    alt_queries=alt_queries,
                    generative_prompt=gen_prompt,
                    disambiguation_notes=disambig,
                    prefer_generative=prefer_gen,
                    prefer_statcard=bool(safety.has_statistics),
                    stat_headline=safety.extracted_stat,
                    safety_flags=safety.flags,
                    negative_keywords=safety.negative_keywords,
                    block_ai_generation=safety.block_ai_generation,
                    warnings=safety.warnings,
                )
            )
        return planned
