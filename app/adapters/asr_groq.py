"""Live ASR adapter using Groq Whisper (whisper-large-v3-turbo) with word-level timestamps (EN/HI/TA)."""

from pathlib import Path
from typing import List, Optional

import httpx

from app.adapters.base import ASRAdapter, ASRResult
from app.errors import AppError
from app.schemas import WordTimestamp


class GroqWhisperASRAdapter(ASRAdapter):
    provider_name = "groq"

    def __init__(self, api_key: str, model_name: str = "whisper-large-v3-turbo") -> None:
        self.api_key = api_key
        self.model_name = model_name

    def transcribe(self, audio_path: Path, language_hint: Optional[str] = None) -> ASRResult:
        if not self.api_key:
            raise AppError("PROVIDER_UNAVAILABLE", "GROQ_API_KEY is not configured.", status_code=503, retryable=False)

        url = "https://api.groq.com/openai/v1/audio/transcriptions"
        data = {
            "model": self.model_name,
            "response_format": "verbose_json",
            "timestamp_granularities[]": "word",
        }
        if language_hint:
            data["language"] = language_hint.split("-")[0].lower()

        try:
            with audio_path.open("rb") as f:
                files = {"file": (audio_path.name, f, "application/octet-stream")}
                with httpx.Client(timeout=120.0) as client:
                    resp = client.post(
                        url,
                        headers={"Authorization": f"Bearer {self.api_key}"},
                        data=data,
                        files=files,
                    )
        except Exception as exc:
            raise AppError(
                "PROVIDER_UNAVAILABLE",
                f"Groq Whisper request failed: {exc}",
                status_code=503,
                retryable=True,
            ) from exc

        if resp.status_code >= 400:
            raise AppError(
                "PROVIDER_UNAVAILABLE",
                f"Groq Whisper returned HTTP {resp.status_code}: {resp.text[:200]}",
                status_code=503,
                retryable=resp.status_code in (429, 500, 502, 503, 504),
            )

        payload = resp.json()
        text = str(payload.get("text") or "").strip()
        duration = float(payload.get("duration") or 0.0)
        language = str(payload.get("language") or language_hint or "en")

        raw_words = payload.get("words") or []
        words: List[WordTimestamp] = []
        for item in raw_words:
            w = str(item.get("word") or "").strip()
            if not w:
                continue
            s = float(item.get("start") or 0.0)
            e = float(item.get("end") or (s + 0.1))
            words.append(WordTimestamp(word=w, start=round(s, 3), end=round(max(s + 0.05, e), 3)))

        # Music-only / non-speech detection via segment no_speech_prob and empty transcript
        segments = payload.get("segments") or []
        avg_no_speech = (
            sum(float(seg.get("no_speech_prob") or 0.0) for seg in segments) / len(segments)
            if segments
            else 0.0
        )
        speech_detected = bool(text and len(words) >= 2 and avg_no_speech < 0.85)

        # Fallback word timing if model returned segments but omitted word timestamps
        if text and not words and segments:
            for seg in segments:
                seg_text = str(seg.get("text") or "").strip()
                seg_s = float(seg.get("start") or 0.0)
                seg_e = float(seg.get("end") or (seg_s + 1.0))
                toks = seg_text.split()
                if not toks:
                    continue
                step = (seg_e - seg_s) / len(toks)
                for idx, tok in enumerate(toks):
                    words.append(
                        WordTimestamp(
                            word=tok,
                            start=round(seg_s + idx * step, 3),
                            end=round(seg_s + (idx + 1) * step, 3),
                        )
                    )
            speech_detected = bool(words and avg_no_speech < 0.85)

        # Groq Whisper Turbo pricing: ~$0.04 per audio hour ($0.00067/min)
        est_cost = round((duration / 3600.0) * 0.04, 6)
        return ASRResult(
            text=text,
            language=language[:2].lower() if len(language) >= 2 else "en",
            duration_seconds=round(duration, 3),
            words=words,
            provider=self.provider_name,
            model=self.model_name,
            estimated_cost_usd=est_cost,
            speech_detected=speech_detected,
        )
