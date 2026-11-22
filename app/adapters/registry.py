"""Config-driven adapter factory supporting seamless swapping between mock and live providers."""

from typing import Dict, List, Optional, Set

from app.adapters.asr_groq import GroqWhisperASRAdapter
from app.adapters.base import (
    ASRAdapter,
    FootageSourceAdapter,
    ScenePlannerAdapter,
    TTSAdapter,
    VLMScorerAdapter,
)
from app.adapters.footage_generative import GeminiGenerativeFootageAdapter
from app.adapters.footage_pexels import PexelsFootageAdapter
from app.adapters.footage_pixabay import PixabayFootageAdapter
from app.adapters.footage_statcard import StatCardFootageAdapter
from app.adapters.mock_adapters import (
    MockASRAdapter,
    MockScenePlannerAdapter,
    MockStockFootageAdapter,
    MockTTSAdapter,
    MockVLMScorerAdapter,
)
from app.adapters.planner_gemini import GeminiScenePlannerAdapter
from app.adapters.tts_gemini import GeminiTTSAdapter
from app.adapters.vlm_scorer import GroqVLMScorerAdapter
from app.config import Settings


class AdapterRegistry:
    """Resolves every external AI/media provider from Settings (Tools Rule #3)."""

    def __init__(self, settings: Settings, runtime_disabled_sources: Optional[Set[str]] = None) -> None:
        self.settings = settings
        self.runtime_disabled = {s.lower() for s in (runtime_disabled_sources or set())}

    def get_asr(self, hint_transcript: Optional[str] = None) -> ASRAdapter:
        provider = self.settings.effective_provider(self.settings.asr_provider, "groq")
        if provider == "groq" and self.settings.groq_api_key:
            return GroqWhisperASRAdapter(self.settings.groq_api_key, self.settings.groq_asr_model)
        return MockASRAdapter(hint_transcript=hint_transcript)

    def get_tts(self) -> TTSAdapter:
        provider = self.settings.effective_provider(self.settings.tts_provider, "gemini")
        if provider == "gemini" and self.settings.gemini_api_key:
            return GeminiTTSAdapter(self.settings.gemini_api_key, self.settings.gemini_tts_model)
        return MockTTSAdapter()

    def get_planner(self) -> ScenePlannerAdapter:
        provider = self.settings.effective_provider(self.settings.planner_provider, "gemini")
        if provider == "gemini" and self.settings.gemini_api_key:
            return GeminiScenePlannerAdapter(self.settings.gemini_api_key, self.settings.gemini_planner_model)
        return MockScenePlannerAdapter()

    def get_vlm_scorer(self) -> VLMScorerAdapter:
        provider = self.settings.effective_provider(self.settings.vlm_scorer_provider, "groq")
        if provider == "groq" and self.settings.groq_api_key:
            return GroqVLMScorerAdapter(self.settings.groq_api_key, self.settings.groq_vlm_model)
        return MockVLMScorerAdapter()

    def get_footage_sources(self) -> Dict[str, FootageSourceAdapter]:
        sources: Dict[str, FootageSourceAdapter] = {}
        is_mock = self.settings.adapter_mode == "mock"

        for name in self.settings.enabled_footage_sources:
            if name in self.runtime_disabled:
                continue
            if name == "pexels":
                if not self.settings.pexels_api_key:
                    sources["pexels"] = MockStockFootageAdapter("pexels", is_generative=False)
                else:
                    sources["pexels"] = PexelsFootageAdapter(self.settings.pexels_api_key)
            elif name == "pixabay":
                if not self.settings.pixabay_api_key:
                    sources["pixabay"] = MockStockFootageAdapter("pixabay", is_generative=False)
                else:
                    sources["pixabay"] = PixabayFootageAdapter(self.settings.pixabay_api_key)
            elif name == "generative":
                if not self.settings.gemini_api_key:
                    sources["generative"] = MockStockFootageAdapter("generative", is_generative=True)
                else:
                    sources["generative"] = GeminiGenerativeFootageAdapter(
                        self.settings.gemini_api_key, self.settings.gemini_image_model
                    )
            elif name == "statcard":
                sources["statcard"] = StatCardFootageAdapter()

        # Always guarantee programmatic statcard fallback exists if everything else is disabled
        if "statcard" not in sources and "statcard" not in self.runtime_disabled:
            sources["statcard"] = StatCardFootageAdapter()
        return sources

    def active_source_order(self) -> List[str]:
        return list(self.get_footage_sources().keys())
