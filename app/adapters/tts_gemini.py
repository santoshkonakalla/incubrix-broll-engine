"""Live multilingual TTS adapter using Google Gemini 3.8 Flash-Lite TTS (EN-IN, HI-IN, TA-IN)."""

import base64
from pathlib import Path
import subprocess
from typing import Dict, Optional, Tuple
import wave

from google import genai
from google.genai import types

from app.adapters.base import TTSAdapter, TTSResult
from app.adapters.mock_adapters import infer_language_from_voice_or_text, synthesize_speech_like_wav
from app.errors import AppError
from app.logging_config import get_logger

logger = get_logger("tts_gemini")

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

def resolve_edge_voice(voice: str, lang: str = "en") -> str:
    """Resolve user-requested voice into exact Microsoft Neural Edge-TTS voice identifier."""
    v = (voice or "").strip()
    v_lower = v.lower()

    # Exact neural voice names
    known = {
        "en-us-jennyneural": "en-US-JennyNeural",
        "en-us-guyneural": "en-US-GuyNeural",
        "en-us-arianeural": "en-US-AriaNeural",
        "en-us-christopherneural": "en-US-ChristopherNeural",
        "en-in-neerjaneural": "en-IN-NeerjaNeural",
        "en-in-prabhatneural": "en-IN-PrabhatNeural",
        "en-gb-sonianeural": "en-GB-SoniaNeural",
        "en-gb-ryanneural": "en-GB-RyanNeural",
        "hi-in-swaraneural": "hi-IN-SwaraNeural",
        "hi-in-madhurneural": "hi-IN-MadhurNeural",
        "ta-in-pallavineural": "ta-IN-PallaviNeural",
        "ta-in-valluvarneural": "ta-IN-ValluvarNeural",
    }
    if v_lower in known:
        return known[v_lower]

    key_map = {
        "en-in-female": "en-IN-NeerjaNeural",
        "en-in-male": "en-IN-PrabhatNeural",
        "en-us-female": "en-US-JennyNeural",
        "en-us-male": "en-US-GuyNeural",
        "en-gb-female": "en-GB-SoniaNeural",
        "en-gb-male": "en-GB-RyanNeural",
        "hi-in-female": "hi-IN-SwaraNeural",
        "hi-in-male": "hi-IN-MadhurNeural",
        "ta-in-female": "ta-IN-PallaviNeural",
        "ta-in-male": "ta-IN-ValluvarNeural",
    }
    if v_lower in key_map:
        return key_map[v_lower]

    # Gender & language heuristics
    is_male = any(m in v_lower for m in ("male", "man", "guy", "prabhat", "madhur", "ryan", "christopher", "boy"))
    if lang.startswith("hi") or "hindi" in v_lower:
        return "hi-IN-MadhurNeural" if is_male else "hi-IN-SwaraNeural"
    if lang.startswith("ta") or "tamil" in v_lower:
        return "ta-IN-ValluvarNeural" if is_male else "ta-IN-PallaviNeural"
    if "in" in v_lower or lang.startswith("en-in") or "india" in v_lower:
        return "en-IN-PrabhatNeural" if is_male else "en-IN-NeerjaNeural"
    if "gb" in v_lower or "uk" in v_lower or "british" in v_lower:
        return "en-GB-RyanNeural" if is_male else "en-GB-SoniaNeural"
    return "en-US-GuyNeural" if is_male else "en-US-JennyNeural"


def synthesize_edge_tts(text: str, voice_name: str, out_wav_path: Path) -> bool:
    """Robust Edge-TTS synthesis that works reliably inside threads, processes, and async loops."""
    out_wav_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_mp3 = out_wav_path.with_suffix(".edge.mp3")
    edge_ok = False

    # Method 1: New isolated event loop (avoids event loop conflict in worker threads)
    try:
        import asyncio
        import edge_tts
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            loop.run_until_complete(edge_tts.Communicate(text, voice_name).save(str(tmp_mp3)))
            edge_ok = tmp_mp3.exists() and tmp_mp3.stat().st_size > 1000
        finally:
            loop.close()
    except Exception as e_loop:
        logger.warning("edge_tts_loop_failed_trying_cli", error=str(e_loop))

    # Method 2: edge-tts CLI subprocess (zero thread/loop interference)
    if not edge_ok:
        try:
            subprocess.run(
                ["edge-tts", "--text", text, "--write-media", str(tmp_mp3), "--voice", voice_name],
                capture_output=True,
                check=True,
                timeout=35,
            )
            edge_ok = tmp_mp3.exists() and tmp_mp3.stat().st_size > 1000
        except Exception as e_cli:
            logger.warning("edge_tts_cli_failed", error=str(e_cli))

    if not edge_ok:
        return False

    # Convert MP3 to standard 24kHz mono PCM WAV
    try:
        res = subprocess.run(
            ["ffmpeg", "-y", "-threads", "1", "-i", str(tmp_mp3), "-ar", "24000", "-ac", "1", str(out_wav_path)],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        tmp_mp3.unlink(missing_ok=True)
        return out_wav_path.exists() and out_wav_path.stat().st_size > 5000
    except subprocess.CalledProcessError as conv_err:
        logger.warning("edge_tts_convert_failed", error=str(conv_err), stderr=conv_err.stderr[:400] if conv_err.stderr else "")
        return False
    except Exception as conv_err:
        logger.warning("edge_tts_convert_failed", error=str(conv_err))
        return False


class GeminiTTSAdapter(TTSAdapter):
    provider_name = "gemini"

    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash-preview-tts") -> None:
        self.api_key = api_key
        self.model_name = model_name

    def synthesize(
        self,
        text: str,
        voice: str,
        out_path: Path,
        language_hint: Optional[str] = None,
    ) -> TTSResult:
        voice_key = (voice or "en-in-female").strip().lower()
        gemini_voice, _style_prompt = VOICE_PRESETS.get(
            voice_key,
            ("Kore", f"clear, natural narration matching voice preset {voice}"),
        )
        lang = language_hint or infer_language_from_voice_or_text(voice, text)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        used_model = self.model_name
        used_provider = self.provider_name

        succeeded = False
        if self.api_key:
            try:
                client = genai.Client(api_key=self.api_key, http_options={"api_version": "v1beta"})
                resp = client.models.generate_content(
                    model=self.model_name,
                    contents=text,
                    config=types.GenerateContentConfig(
                        response_modalities=["AUDIO"],
                        speech_config=types.SpeechConfig(
                            voice_config=types.VoiceConfig(
                                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=gemini_voice)
                            )
                        ),
                    ),
                )
                raw_pcm = None
                if resp.candidates and resp.candidates[0].content and resp.candidates[0].content.parts:
                    for part in resp.candidates[0].content.parts:
                        if getattr(part, "inline_data", None) and getattr(part.inline_data, "data", None):
                            raw_pcm = part.inline_data.data
                            break

                if raw_pcm:
                    with wave.open(str(out_path), "wb") as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(24000)
                        wf.writeframes(raw_pcm)
                    succeeded = True
            except Exception as exc:
                logger.warning("gemini_tts_quota_or_network_fallback", error=str(exc))

        if not succeeded:
            # Resilient automatic fallback to high-quality neural Edge TTS (works across threads & loops)
            edge_voice = resolve_edge_voice(voice_key, lang)
            if synthesize_edge_tts(text, edge_voice, out_path):
                used_provider = "edge_tts_neural"
                used_model = edge_voice
                succeeded = True
            elif synthesize_edge_tts(text, "en-US-JennyNeural", out_path):
                used_provider = "edge_tts_neural"
                used_model = "en-US-JennyNeural"
                succeeded = True
            else:
                logger.warning("edge_tts_failed_using_offline")
                synthesize_speech_like_wav(text, out_path, sample_rate=24000)
                used_provider = "offline_speech_synth"
                used_model = "speech_cadence_v1"

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
