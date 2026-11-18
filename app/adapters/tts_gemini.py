"""Live multilingual TTS adapter using Google Gemini 3.8 Flash-Lite TTS (EN-IN, HI-IN, TA-IN)."""

import base64
from pathlib import Path
import subprocess
from typing import Dict, Optional, Tuple

from google import genai

from app.adapters.base import TTSAdapter, TTSResult
from app.adapters.mock_adapters import infer_language_from_voice_or_text
from app.errors import AppError

# Map requested voice presets to Gemini TTS voices + style directives
VOICE_PRESETS: Dict[str, Tuple[str, str]] = {
    "en-in-female": ("Kore", "clear, warm, natural Indian English narration, steady documentary pace"),
    "en-in-male": ("Puck", "confident, clear Indian English narration, calm informative pace"),
    "en-us-female": ("Aoede", "bright, articulate American English narration"),
    "en-us-male": ("Charon", "warm, clear American English narration"),
    "en-gb-female": ("Leda", "clear British English documentary narration"),
    "hi-in-female": ("Kore", "natural, clear Hindi narration with warm Indian accent"),
    "hi-in-male": ("Puck", "clear, articulate Hindi narration with steady pace"),
    "ta-in-female": ("Kore", "natural, clear Tamil narration with warm Indian accent"),
    "ta-in-male": ("Puck", "clear, steady Tamil narration"),
}


class GeminiTTSAdapter(TTSAdapter):
    provider_name = "gemini"

    def __init__(self, api_key: str, model_name: str = "gemini-3.8-flash-lite-tts") -> None:
        self.api_key = api_key
        self.model_name = model_name

    def synthesize(
        self,
        text: str,
        voice: str,
        out_path: Path,
        language_hint: Optional[str] = None,
    ) -> TTSResult:
        if not self.api_key:
            raise AppError("PROVIDER_UNAVAILABLE", "GEMINI_API_KEY is not configured.", status_code=503)

        voice_key = (voice or "en-in-female").strip().lower()
        gemini_voice, style_prompt = VOICE_PRESETS.get(
            voice_key,
            ("Kore", f"clear, natural narration matching voice preset {voice}"),
        )
        lang = language_hint or infer_language_from_voice_or_text(voice, text)

        try:
            client = genai.Client(api_key=self.api_key)
            interaction = client.interactions.create(
                model=self.model_name,
                input=[
                    {
                        "type": "user_input",
                        "content": [
                            {
                                "type": "text",
                                "text": text,
                                "annotations": [
                                    {
                                        "type": "speech_metadata",
                                        "style": style_prompt,
                                    }
                                ],
                            }
                        ],
                    }
                ],
                response_format={
                    "type": "audio",
                    "mime_type": "audio/wav",
                    "sample_rate": 24000,
                },
                generation_config={
                    "speech_config": [{"voice": gemini_voice}],
                },
            )
        except Exception as exc:
            raise AppError(
                "PROVIDER_UNAVAILABLE",
                f"Gemini TTS synthesis failed: {exc}",
                status_code=503,
                retryable=True,
            ) from exc

        output_audio = getattr(interaction, "output_audio", None)
        if not output_audio or not getattr(output_audio, "data", None):
            raise AppError("PROVIDER_UNAVAILABLE", "Gemini TTS returned no audio payload.", status_code=503, retryable=True)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        raw_bytes = base64.b64decode(output_audio.data)
        out_path.write_bytes(raw_bytes)

        proc = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(out_path)],
            capture_output=True,
            text=True,
            timeout=20,
        )
        duration = float((proc.stdout or "0").strip() or "0")
        # Conservative token-based estimate (~$0.004 per minute on Flash-Lite TTS paid tier; $0 on free tier)
        est_cost = round((duration / 60.0) * 0.004, 6)
        return TTSResult(
            audio_path=out_path,
            duration_seconds=round(duration, 3),
            language=lang,
            provider=self.provider_name,
            model=self.model_name,
            estimated_cost_usd=est_cost,
        )
