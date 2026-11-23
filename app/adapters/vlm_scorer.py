"""Independent Vision-Language Model (VLM) relevance scorer using Groq Llama-4 Vision (Outstanding Level)."""

import base64
import json
from pathlib import Path
import re
import time
from typing import Dict, List, Optional

import cv2
import httpx

from app.adapters.base import VLMScorerAdapter, VLMScoreResult
from app.adapters.mock_adapters import MockVLMScorerAdapter
from app.logging_config import get_logger

logger = get_logger("vlm_scorer")


def extract_keyframe_base64(video_path: Path, position_ratio: float = 0.5) -> Optional[str]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        idx = max(0, int(total * position_ratio))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
        if not ok or frame is None:
            return None
        h, w = frame.shape[:2]
        if w > 640:
            frame = cv2.resize(frame, (640, int(h * 640 / w)), interpolation=cv2.INTER_AREA)
        ok_enc, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok_enc:
            return None
        return base64.b64encode(buf.tobytes()).decode("ascii")
    finally:
        cap.release()


class GroqVLMScorerAdapter(VLMScorerAdapter):
    """Uses Groq Llama-4 Vision (independent from Gemini planner) to rate B-roll relevance 1-5."""

    provider_name = "groq"
    _circuit_open_until: float = 0.0
    _circuit_reason: str = ""
    _cache: Dict[str, VLMScoreResult] = {}

    def __init__(self, api_key: str, model_name: str = "meta-llama/llama-4-scout-17b-16e-instruct") -> None:
        self.api_key = api_key
        self.model_name = model_name
        self._fallback = MockVLMScorerAdapter()

    def score_clip(
        self,
        video_path: Path,
        narration_text: str,
        visual_intent: str,
        avoid_terms: List[str],
        negative_keywords: List[str],
        candidate_metadata: str = "",
    ) -> VLMScoreResult:
        cache_key = f"{video_path.name}_{hashlib.sha1((narration_text + visual_intent).encode()).hexdigest()[:12]}"
        if cache_key in GroqVLMScorerAdapter._cache:
            return GroqVLMScorerAdapter._cache[cache_key]

        if not self.api_key or time.time() < GroqVLMScorerAdapter._circuit_open_until:
            res = self._fallback.score_clip(
                video_path, narration_text, visual_intent, avoid_terms, negative_keywords, candidate_metadata
            )
            GroqVLMScorerAdapter._cache[cache_key] = res
            return res

        frame_b64 = extract_keyframe_base64(video_path, 0.45)
        if not frame_b64:
            return self._fallback.score_clip(
                video_path, narration_text, visual_intent, avoid_terms, negative_keywords, candidate_metadata
            )

        avoid_str = ", ".join(avoid_terms) if avoid_terms else "none"
        prompt = (
            "You are an independent video quality evaluator grading B-roll footage against a voiceover line.\n"
            f"NARRATION LINE: {narration_text}\n"
            f"INTENDED VISUAL: {visual_intent}\n"
            f"MUST AVOID: {avoid_str}\n\n"
            "Inspect the video frame and return STRICT JSON ONLY with keys:\n"
            '{"relevance_score": <float 1.0 to 5.0>, '
            '"rationale": "<1 concise sentence>", '
            '"violates_avoid": <true|false>, '
            '"avoid_reason": <string or null>, '
            '"subject_focus_x": <float 0.0 left to 1.0 right where the main subject sits horizontally>}\n'
            "Scoring guide: 5=exact meaningful match, 4=strong clear fit, 3=acceptable generic fit, "
            "2=weak/literal-keyword mismatch, 1=wrong or violates avoid/watermark."
        )

        body = {
            "model": self.model_name,
            "temperature": 0.1,
            "max_tokens": 220,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"}},
                    ],
                }
            ],
        }

        try:
            with httpx.Client(timeout=2.5) as client:
                resp = client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=body,
                )
            if resp.status_code >= 400:
                logger.warning("groq_vlm_http_error", status=resp.status_code)
                if resp.status_code in (401, 403, 404, 429):
                    GroqVLMScorerAdapter._circuit_open_until = time.time() + 120.0
                    GroqVLMScorerAdapter._circuit_reason = f"HTTP {resp.status_code}"
                return self._fallback.score_clip(
                    video_path, narration_text, visual_intent, avoid_terms, negative_keywords, candidate_metadata
                )
            content = resp.json()["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            score = max(1.0, min(5.0, float(parsed.get("relevance_score", 3.5))))
            violates = bool(parsed.get("violates_avoid", False))
            if violates:
                score = min(score, 1.8)
            fx_raw = parsed.get("subject_focus_x")
            focus_x = max(0.0, min(1.0, float(fx_raw))) if fx_raw is not None else None
            res = VLMScoreResult(
                relevance_score=round(score, 2),
                rationale=str(parsed.get("rationale") or "Scored by independent Groq Llama-4 Vision."),
                violates_avoid=violates,
                avoid_reason=parsed.get("avoid_reason"),
                subject_focus_x=focus_x,
                provider=self.provider_name,
                model=self.model_name,
                estimated_cost_usd=0.0002,
            )
            GroqVLMScorerAdapter._cache[cache_key] = res
            return res
        except Exception as exc:
            logger.warning("groq_vlm_fallback", error=str(exc))
            GroqVLMScorerAdapter._circuit_open_until = time.time() + 30.0
            GroqVLMScorerAdapter._circuit_reason = str(exc)
            res = self._fallback.score_clip(
                video_path, narration_text, visual_intent, avoid_terms, negative_keywords, candidate_metadata
            )
            GroqVLMScorerAdapter._cache[cache_key] = res
            return res
