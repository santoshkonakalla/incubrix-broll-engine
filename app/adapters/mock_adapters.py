"""Zero-API-key deterministic mock adapters for offline unit/integration/E2E testing (Integration #1 & Tool Rule #3)."""

import hashlib
import math
import os
from pathlib import Path
import re
import struct
import subprocess
from typing import List, Optional, Tuple
import wave

import cv2
import numpy as np

from app.adapters.base import (
    ASRAdapter,
    ASRResult,
    FootageCandidate,
    FootageSourceAdapter,
    PlannedScene,
    ScenePlannerAdapter,
    TTSAdapter,
    TTSResult,
    VLMScorerAdapter,
    VLMScoreResult,
)
from app.pipeline.safety import analyze_scene_safety, violates_avoid_list
from app.schemas import SceneHintInput, WordTimestamp


def infer_language_from_voice_or_text(voice: str, text: str = "") -> str:
    v = (voice or "").lower()
    if v.startswith("hi") or any("\u0900" <= ch <= "\u097f" for ch in text):
        return "hi"
    if v.startswith("ta") or any("\u0b80" <= ch <= "\u0bff" for ch in text):
        return "ta"
    return "en"


def synthesize_speech_like_wav(text: str, out_path: Path, sample_rate: int = 24000) -> Tuple[Path, float]:
    """Generate an audible, speech-cadence modulated WAV file offline (0 API keys, passes silence check)."""
    words = [w for w in re.split(r"\s+", text.strip()) if w]
    duration = max(3.0, min(580.0, len(words) * 0.42 + 0.5))
    total_samples = int(duration * sample_rate)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)

        chunk_size = 4096
        for offset in range(0, total_samples, chunk_size):
            n = min(chunk_size, total_samples - offset)
            t = (np.arange(offset, offset + n, dtype=np.float32)) / sample_rate
            # Multi-harmonic speech-like envelope (~180Hz fundamental + 3Hz syllable modulation)
            env = 0.55 + 0.45 * np.sin(2.0 * math.pi * 3.1 * t)
            sig = (
                0.30 * np.sin(2.0 * math.pi * 195.0 * t)
                + 0.18 * np.sin(2.0 * math.pi * 390.0 * t)
                + 0.10 * np.sin(2.0 * math.pi * 585.0 * t)
            ) * env
            pcm = np.clip(sig * 32767.0, -32767, 32767).astype(np.int16)
            wf.writeframes(pcm.tobytes())

    return out_path, round(duration, 3)


def _measure_wav_duration(file_path: Path) -> float:
    try:
        with wave.open(str(file_path), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            return round(frames / float(rate), 3)
    except Exception:
        return 6.0


def generate_mock_video_clip(
    out_path: Path,
    *,
    label: str,
    duration: float = 6.0,
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
    edge_subject_x: Optional[float] = None,
    seed: int = 42,
) -> Path:
    """
    Create a deterministic, smoothly animated H.264 test clip with distinct visuals per query.
    Supports placing a bright moving subject near the left/right edge to verify Hard Case #5 smart crop.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw = out_path.with_suffix(".raw.mp4")

    h_int = int(hashlib.sha256(f"{label}:{seed}".encode()).hexdigest(), 16)
    base_b = 50 + (h_int % 110)
    base_g = 60 + ((h_int >> 8) % 110)
    base_r = 70 + ((h_int >> 16) % 110)

    total_frames = max(int(round(duration * fps)), fps * 2)
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
    freq_x = 1.0 + (h_int % 5) * 0.7
    phase = ((h_int >> 4) % 10) * 0.3
    try:
        xs = np.linspace(0.0, 2.0 * math.pi * freq_x, width, dtype=np.float32) + phase
        for i in range(total_frames):
            t = i / max(1, total_frames - 1)
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            # Smooth continuous wave background (no hard wrap seam, never black, never frozen)
            wave = 0.82 + 0.18 * np.sin(xs + i * 0.09)
            frame[:, :, 0] = np.clip(base_b * wave, 35, 220).astype(np.uint8)
            frame[:, :, 1] = np.clip(base_g * wave, 35, 220).astype(np.uint8)
            frame[:, :, 2] = np.clip(base_r * wave, 35, 220).astype(np.uint8)

            # Distinct spatial background geometry per domain
            is_health = any(k in label.lower() for k in ("health", "healthcare", "doctor", "doctors", "medical", "medicine", "hospital", "clinic", "patient", "disease", "surgery", "diagnostic", "scan", "biomedical"))
            is_tech = any(k in label.lower() for k in ("robot", "ai", "intelligence", "tech", "computer", "cloud", "code", "python", "smart", "algorithm", "software"))
            is_nature = any(k in label.lower() for k in ("planet", "earth", "nature", "tree", "trees", "water", "forest", "green", "environment", "plastic", "waste", "recycle", "solar", "garden", "wildlife", "climate", "ocean", "river"))
            is_finance = any(k in label.lower() for k in ("finance", "money", "budget", "savings", "growth", "invest", "banking", "wealth", "economy", "market", "sales"))
            is_sports = any(k in label.lower() for k in ("cricket", "sport", "sports", "football", "soccer", "basketball", "tennis", "match", "tournament", "stadium", "player", "athlete", "game"))

            if is_health:
                # Clinical medical slate-cyan gradient
                for gy in range(height):
                    frac = gy / float(height)
                    frame[gy, :] = (int(65 + 25 * frac), int(55 + 30 * frac), int(25 + 20 * frac))
                # High-tech animated EKG / heart pulse wave moving across the screen
                pts = []
                for px in range(0, width, 6):
                    norm_x = (px / float(width) - 0.5) * 4.0
                    ekg = (85.0 * math.sin((norm_x - t * 0.4) * 8.0) * math.exp(-((norm_x - t * 0.4) ** 2) * 5.0)) if abs(norm_x - t * 0.4) < 0.8 else 0.0
                    py = int(height * 0.55 + ekg + 5.0 * math.sin(px * 0.04 + t * 4.0))
                    pts.append([px, py])
                cv2.polylines(frame, [np.array(pts, np.int32)], False, (255, 230, 100), 3, cv2.LINE_AA)
            elif is_tech:
                frame[:, :] = (26, 20, 14)  # Deep cyber dark navy
                # High-tech digital grid
                for gx in range(0, width, 160):
                    cv2.line(frame, (gx, 0), (gx, height), (48, 38, 26), 1)
                for gy in range(0, height, 120):
                    cv2.line(frame, (0, gy), (width, gy), (48, 38, 26), 1)
                # Animated neural network nodes
                for n_idx, (nx, ny) in enumerate([(180, 160), (320, 110), (1100, 180), (940, 130), (1060, 540)]):
                    pulse = int(12 * math.sin(t * 6 + n_idx))
                    cv2.circle(frame, (nx, ny), 7, (255, 200, 40), -1)
                    cv2.circle(frame, (nx, ny), 14 + pulse // 3, (120, 90, 20), 1)
            elif is_nature:
                # Lush organic gradient
                for gy in range(height):
                    frac = gy / float(height)
                    frame[gy, :] = (int(32 + 18 * frac), int(68 + 42 * frac), int(26 + 18 * frac))
                # Sun / atmosphere warmth in corner
                cv2.circle(frame, (int(width * 0.15), int(height * 0.18)), 80, (140, 230, 255), -1)
            elif is_finance:
                # Deep corporate royal blue gradient
                for gy in range(height):
                    frac = gy / float(height)
                    frame[gy, :] = (int(60 + 30 * frac), int(30 + 15 * frac), int(16 + 10 * frac))
                # Rising financial growth line
                f_pts = []
                for px in range(0, width, 8):
                    f_frac = px / float(width)
                    f_y = int(height * (0.75 - 0.45 * (f_frac ** 1.3) + 0.04 * math.sin(f_frac * 12 + t * 4)))
                    f_pts.append([px, f_y])
                cv2.polylines(frame, [np.array(f_pts, np.int32)], False, (60, 220, 100), 3, cv2.LINE_AA)
            elif is_sports:
                # Deep athletic stadium field gradient
                for gy in range(height):
                    frac = gy / float(height)
                    frame[gy, :] = (int(26 + 18 * frac), int(85 + 45 * frac), int(32 + 22 * frac))
                # Stadium floodlight glow across top
                cv2.ellipse(frame, (width // 2, int(height * 0.12)), (int(width * 0.45), 55), 0, 0, 180, (140, 240, 220), 2, cv2.LINE_AA)
                cv2.circle(frame, (int(width * 0.2), int(height * 0.12)), 25, (220, 255, 245), -1)
                cv2.circle(frame, (int(width * 0.8), int(height * 0.12)), 25, (220, 255, 245), -1)
            elif edge_subject_x is None:
                # Smooth continuous full-frame 2D gradient (NO SPLIT-SCREEN VERTICAL BLOCKS)
                for gy in range(height):
                    frac_y = 0.75 + 0.45 * (gy / float(height))
                    frame[gy, :, 0] = np.clip(base_b * wave * frac_y, 25, 230).astype(np.uint8)
                    frame[gy, :, 1] = np.clip(base_g * wave * frac_y, 25, 230).astype(np.uint8)
                    frame[gy, :, 2] = np.clip(base_r * wave * frac_y, 25, 230).astype(np.uint8)

            # High-contrast subject (can sit at the edge for Hard Case #5 9:16 smart-crop tests!)
            if edge_subject_x is not None:
                cx = int(width * edge_subject_x)
                cy = int(height * (0.48 + 0.05 * math.sin(t * 2 * math.pi)))
            else:
                cx = int(width * (0.30 + 0.35 * t))
                cy = int(height * (0.46 + 0.08 * math.sin(t * 4 * math.pi)))

            if is_health:
                # Medical Cross Badge with glowing circle
                cv2.circle(frame, (cx, cy), 85, (230, 200, 80), 3, cv2.LINE_AA)
                cv2.circle(frame, (cx, cy), 78, (45, 32, 18), -1)
                arm_w, arm_h = 24, 70
                cv2.rectangle(frame, (cx - arm_w // 2, cy - arm_h // 2), (cx + arm_w // 2, cy + arm_h // 2), (255, 255, 255), -1)
                cv2.rectangle(frame, (cx - arm_h // 2, cy - arm_w // 2), (cx + arm_h // 2, cy + arm_w // 2), (255, 255, 255), -1)
            elif is_tech:
                # Robotic / AI Avatar
                r_rad = int(105 + 6 * math.sin(t * 4 * math.pi))
                cv2.circle(frame, (cx, cy), r_rad, (255, 215, 0), 2, cv2.LINE_AA)
                cv2.circle(frame, (cx, cy), 88, (38, 30, 18), -1)
                # Robot Head
                cv2.rectangle(frame, (cx - 48, cy - 42), (cx + 48, cy + 34), (225, 220, 215), -1)
                cv2.rectangle(frame, (cx - 48, cy - 42), (cx + 48, cy + 34), (95, 85, 75), 3)
                # Glowing cyan eyes
                cv2.circle(frame, (cx - 20, cy - 10), 12, (255, 240, 0), -1)
                cv2.circle(frame, (cx + 20, cy - 10), 12, (255, 240, 0), -1)
                cv2.line(frame, (cx - 18, cy + 18), (cx + 18, cy + 18), (75, 65, 55), 3)
                # Antenna
                cv2.line(frame, (cx, cy - 42), (cx, cy - 68), (195, 185, 175), 3)
                cv2.circle(frame, (cx, cy - 70), 7, (255, 120, 0), -1)
            elif is_nature and any(k in label.lower() for k in ("planet", "earth", "globe")):
                # Animated Earth Globe with continents and blue oceans
                rot_offset = int(25 * math.sin(t * 2 * math.pi))
                cv2.circle(frame, (cx, cy), 125, (215, 135, 30), -1) # Blue ocean
                cv2.ellipse(frame, (cx - 30 + rot_offset, cy - 20), (55, 38), 20, 0, 360, (50, 185, 70), -1)
                cv2.ellipse(frame, (cx + 45 + rot_offset, cy + 30), (45, 32), -15, 0, 360, (50, 185, 70), -1)
                cv2.circle(frame, (cx - 20 + rot_offset, cy + 60), 24, (50, 185, 70), -1)
                cv2.circle(frame, (cx, cy), 130, (255, 225, 130), 3) # Atmosphere glow
            elif is_nature:
                # Tree with lush green canopy and brown trunk
                sway = int(8 * math.sin(t * 4 * math.pi))
                cv2.rectangle(frame, (cx - 20, cy), (cx + 20, cy + 160), (30, 60, 110), -1)
                cv2.circle(frame, (cx + sway, cy - 50), 90, (40, 160, 60), -1)
                cv2.circle(frame, (cx - 55 + sway, cy - 25), 72, (45, 185, 75), -1)
                cv2.circle(frame, (cx + 55 + sway, cy - 25), 72, (45, 185, 75), -1)
                cv2.circle(frame, (cx + sway, cy - 85), 68, (60, 215, 95), -1)
            elif is_finance:
                # Financial bar graph & symbol
                cv2.rectangle(frame, (cx - 60, cy - 20), (cx - 25, cy + 80), (60, 180, 80), -1)
                cv2.rectangle(frame, (cx - 15, cy - 60), (cx + 20, cy + 80), (70, 210, 95), -1)
                cv2.rectangle(frame, (cx + 30, cy - 100), (cx + 65, cy + 80), (80, 240, 110), -1)
            elif is_sports:
                # Animated cricket / sports ball with dynamic seam and motion glow
                ball_r = 52
                cv2.circle(frame, (cx, cy), ball_r + 14, (110, 240, 180), 2, cv2.LINE_AA)
                cv2.circle(frame, (cx, cy), ball_r, (40, 45, 195), -1)  # Deep red cricket ball
                # Cricket ball white seam curve
                cv2.ellipse(frame, (cx, cy), (ball_r - 2, int(ball_r * 0.55)), int(t * 180), 0, 360, (245, 245, 250), 3, cv2.LINE_AA)
            else:
                # Clean modern cinematic orb / lens flare (no box / split rectangle)
                glow_r = int(75 + 10 * math.sin(t * 3 * math.pi))
                cv2.circle(frame, (cx, cy), glow_r, (min(255, base_b + 80), min(255, base_g + 80), min(255, base_r + 80)), -1, cv2.LINE_AA)
                cv2.circle(frame, (cx, cy), int(glow_r * 0.65), (255, 255, 255), -1, cv2.LINE_AA)

            # Elegant scene title pill badge
            if is_health and any(k in label.lower() for k in ("ai", "tech", "artificial")):
                title_text = "AI Healthcare & Medical Technology"
            elif is_health and any(k in label.lower() for k in ("doctor", "patient", "care")):
                title_text = "Doctors & Advanced Patient Care"
            elif is_health and any(k in label.lower() for k in ("scan", "detect", "disease", "diagnostic")):
                title_text = "Diagnostic Imaging & Technology"
            elif is_health:
                title_text = "Modern Healthcare & Medicine"
            elif is_nature and any(k in label.lower() for k in ("planet", "earth")):
                title_text = "Protecting Our Planet Earth"
            elif is_nature and any(k in label.lower() for k in ("tree", "forest", "woodland", "plant")):
                title_text = "Planting Trees & Preserving Nature"
            elif is_nature and any(k in label.lower() for k in ("water", "ocean", "river")):
                title_text = "Saving Water & Clean Rivers"
            elif is_tech and "robot" in label.lower():
                title_text = "Robotics & Everyday AI"
            elif is_tech:
                title_text = "Artificial Intelligence & Technology"
            elif is_finance:
                title_text = "Financial Growth & Analytics"
            elif is_sports and "cricket" in label.lower():
                title_text = "Cricket Match & Stadium Sports"
            elif is_sports:
                title_text = "Athletics & Competitive Sports"
            else:
                clean_lbl = re.sub(r"^[a-zA-Z0-9_\-]+:\s*", "", label).strip().title()
                title_text = clean_lbl[:45]
            pill_w = min(width - 40, max(280, len(title_text) * 14 + 40))
            px0 = max(20, (width - pill_w) // 2)
            cv2.rectangle(frame, (px0, height - 68), (px0 + pill_w, height - 20), (12, 14, 20), -1)
            cv2.rectangle(frame, (px0, height - 68), (px0 + pill_w, height - 20), (80, 85, 110), 1)
            cv2.putText(
                frame,
                title_text,
                (px0 + 20, height - 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            writer.write(frame)
    finally:
        writer.release()

    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(raw),
            "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", "-r", str(fps),
            str(out_path),
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    raw.unlink(missing_ok=True)
    return out_path


class MockTTSAdapter(TTSAdapter):
    provider_name = "mock"
    model_name = "mock-tts-v1"

    def synthesize(
        self,
        text: str,
        voice: str,
        out_path: Path,
        language_hint: Optional[str] = None,
    ) -> TTSResult:
        lang = language_hint or infer_language_from_voice_or_text(voice, text)
        if "PYTEST_CURRENT_TEST" in os.environ:
            wav_path, dur = synthesize_speech_like_wav(text, out_path)
            return TTSResult(
                audio_path=wav_path,
                duration_seconds=dur,
                language=lang,
                provider=self.provider_name,
                model=self.model_name,
                estimated_cost_usd=0.0,
            )
        # Priority 1: High-fidelity Neural Edge-TTS (0 API keys required, speaks real natural human words)
        try:
            from app.adapters.tts_gemini import resolve_edge_voice, synthesize_edge_tts
            edge_voice = resolve_edge_voice(voice, lang)
            if synthesize_edge_tts(text, edge_voice, out_path) or synthesize_edge_tts(text, "en-US-JennyNeural", out_path):
                dur = _measure_wav_duration(out_path)
                if dur >= 0.5:
                    return TTSResult(
                        audio_path=out_path,
                        duration_seconds=dur,
                        language=lang,
                        provider="edge_tts_neural",
                        model=edge_voice,
                        estimated_cost_usd=0.0,
                    )
        except Exception:
            pass

        # Priority 2: Resilient offline speech cadence fallback (for air-gapped unit tests)
        wav_path, dur = synthesize_speech_like_wav(text, out_path)
        return TTSResult(
            audio_path=wav_path,
            duration_seconds=dur,
            language=lang,
            provider=self.provider_name,
            model=self.model_name,
            estimated_cost_usd=0.0,
        )


class MockASRAdapter(ASRAdapter):
    provider_name = "mock"
    model_name = "mock-asr-v1"

    def __init__(self, hint_transcript: Optional[str] = None) -> None:
        self.hint_transcript = hint_transcript

    def transcribe(self, audio_path: Path, language_hint: Optional[str] = None) -> ASRResult:
        # Measure real duration via ffprobe
        proc = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(audio_path)],
            capture_output=True,
            text=True,
            timeout=20,
        )
        duration = float((proc.stdout or "6.0").strip() or "6.0")

        # Detect intentionally music-only mock test files by filename convention
        if "music_only" in audio_path.name.lower():
            return ASRResult(
                text="",
                language=language_hint or "en",
                duration_seconds=duration,
                words=[],
                provider=self.provider_name,
                model=self.model_name,
                speech_detected=False,
            )

        text = (self.hint_transcript or "").strip()
        if not text:
            text = (
                "Building long term financial habits starts with simple daily choices. "
                "Modern cloud software helps teams track budgets and measure growth clearly. "
                "When people save consistently, their confidence grows every single month."
            )

        tokens = [t for t in re.split(r"\s+", text) if t]
        if not tokens:
            return ASRResult(
                text="",
                language=language_hint or "en",
                duration_seconds=duration,
                words=[],
                provider=self.provider_name,
                model=self.model_name,
                speech_detected=False,
            )

        # Distribute word timestamps proportionally by character weight across exact audio duration
        weights = [max(2, len(t)) for t in tokens]
        total_w = float(sum(weights))
        words: List[WordTimestamp] = []
        cursor = 0.0
        for tok, w in zip(tokens, weights):
            span = duration * (w / total_w)
            start = round(cursor, 3)
            end = round(min(duration, cursor + span), 3)
            words.append(WordTimestamp(word=tok, start=start, end=max(start + 0.05, end)))
            cursor += span
        if words:
            words[-1].end = round(duration, 3)

        lang = language_hint or infer_language_from_voice_or_text("", text)
        return ASRResult(
            text=text,
            language=lang,
            duration_seconds=round(duration, 3),
            words=words,
            provider=self.provider_name,
            model=self.model_name,
            estimated_cost_usd=0.0,
            speech_detected=True,
        )


def segment_words_into_scenes(
    words: List[WordTimestamp],
    total_duration: float,
    min_scene_duration: float = 2.5,
    max_scene_duration: float = 9.0,
) -> List[Tuple[float, float, str]]:
    """
    Segment word-level timestamps strictly on sentence/phrase boundaries, guaranteeing
    100% coverage of [0, total_duration] and sensible pacing (no flashing 0.5s clips).
    """
    if not words:
        return [(0.0, round(total_duration, 3), "")]

    raw_sentences: List[Tuple[float, float, str]] = []
    buf: List[WordTimestamp] = []

    for i, w in enumerate(words):
        buf.append(w)
        dur = buf[-1].end - buf[0].start
        is_sentence_end = w.word.strip().endswith((".", "!", "?", "।", "॥"))
        is_phrase_break = w.word.strip().endswith((",", ";", ":")) and dur >= min_scene_duration
        is_long_pause = (i + 1 < len(words)) and (words[i + 1].start - w.end >= 0.55) and dur >= min_scene_duration

        if is_sentence_end or is_phrase_break or is_long_pause or dur >= max_scene_duration:
            raw_sentences.append((buf[0].start, buf[-1].end, " ".join(x.word for x in buf)))
            buf = []

    if buf:
        raw_sentences.append((buf[0].start, buf[-1].end, " ".join(x.word for x in buf)))

    # Merge overly short fragments (< min_scene_duration) into neighbors so no flashing clips exist
    merged: List[Tuple[float, float, str]] = []
    for seg in raw_sentences:
        s, e, txt = seg
        if merged and (e - s) < min_scene_duration:
            prev_s, _prev_e, prev_txt = merged[-1]
            merged[-1] = (prev_s, e, f"{prev_txt} {txt}".strip())
        elif merged and (merged[-1][1] - merged[-1][0]) < min_scene_duration:
            prev_s, _prev_e, prev_txt = merged[-1]
            merged[-1] = (prev_s, e, f"{prev_txt} {txt}".strip())
        else:
            merged.append((s, e, txt))

    # Snap boundaries continuously so EVERY second of narration from 0.0 to total_duration is covered
    continuous: List[Tuple[float, float, str]] = []
    for idx, (s, e, txt) in enumerate(merged):
        start = 0.0 if idx == 0 else continuous[-1][1]
        if idx == len(merged) - 1:
            end = round(total_duration, 3)
        else:
            next_s = merged[idx + 1][0]
            end = round((e + next_s) / 2.0, 3) if next_s > e else round(e, 3)
        if end <= start:
            end = round(start + 0.5, 3)
        continuous.append((round(start, 3), round(end, 3), txt))

    if continuous:
        s_last, _e_last, t_last = continuous[-1]
        continuous[-1] = (s_last, round(total_duration, 3), t_last)
    return continuous


def align_explicit_scenes_to_duration(
    explicit_scenes: List[SceneHintInput],
    words: List[WordTimestamp],
    total_duration: float,
) -> List[Tuple[float, float, str, Optional[str]]]:
    """Align an IncuBrix scene list to exact narration duration using word count proportions."""
    weights = [max(1, len(re.findall(r"\S+", s.text))) for s in explicit_scenes]
    total_w = float(sum(weights))
    aligned: List[Tuple[float, float, str, Optional[str]]] = []
    cursor = 0.0
    for idx, (scene, w) in enumerate(zip(explicit_scenes, weights)):
        start = round(cursor, 3)
        span = total_duration * (w / total_w)
        end = round(total_duration, 3) if idx == len(explicit_scenes) - 1 else round(cursor + span, 3)
        aligned.append((start, max(start + 0.5, end), scene.text.strip(), scene.visual_hint))
        cursor += span
    return aligned


def extract_visual_search_query(text: str) -> str:
    """
    Extract high-precision visual concepts, mapping narrative abstractions into concrete video search queries.
    Prevents abstract trap words ('protecting', 'responsibility', 'everyone') from matching irrelevant videos.
    """
    t = text.lower()

    # 1. Healthcare / Medical / Diagnostic domain
    has_medical = any(k in t for k in ("health", "healthcare", "doctor", "doctors", "patient", "patients", "hospital", "clinic", "medical", "medicine", "disease", "diseases", "surgery", "surgeon", "nurse", "scan", "diagnosis", "diagnostic", "treatment", "care", "biomedical"))
    has_ai_tech = any(k in t for k in ("artificial intelligence", "ai", "technology", "tech", "advanced", "smart", "digital", "algorithm", "robot", "robotics", "computer", "software", "machine learning"))

    if has_medical and has_ai_tech:
        if any(k in t for k in ("scan", "detect", "disease", "diseases", "diagnos")):
            return "medical diagnostic technology scan"
        if any(k in t for k in ("doctor", "physician", "surgeon", "nurse")):
            return "doctors medical technology healthcare"
        return "artificial intelligence healthcare medical"

    if has_medical:
        if any(k in t for k in ("doctor", "doctors", "physician", "surgeon", "nurse")):
            return "doctors hospital patient care"
        if any(k in t for k in ("patient", "patients", "care")):
            return "healthcare patient medical care"
        if any(k in t for k in ("scan", "x-ray", "mri", "detect", "disease", "diagnosis")):
            return "medical diagnostic scan laboratory"
        return "healthcare medical doctor"

    # 2. Technology / Robotics / AI domain
    if any(k in t for k in ("robot", "robotics", "bot", "android", "cyborg", "humanoid")):
        return "robot humanoid robotics"
    if any(k in t for k in ("artificial intelligence", "ai", "machine learning", "neural", "deep learning", "algorithm")):
        return "artificial intelligence technology"
    if any(k in t for k in ("computer", "software", "code", "coding", "developer", "programming", "cloud", "data", "cyber", "server")):
        return "software developer coding computer"

    # 3. Nature / Planet / Environmental domain
    if any(k in t for k in ("planet", "earth", "globe", "world")):
        return "planet earth"
    if any(k in t for k in ("tree", "trees", "plant", "planting", "forest", "woodland", "canopy", "leaves")):
        return "nature forest trees"
    if any(k in t for k in ("water", "ocean", "river", "sea", "lake", "saving water")):
        return "clean water ocean river"
    if any(k in t for k in ("plastic", "waste", "recycle", "recycling", "pollution", "clean")):
        return "recycling environment nature"
    if any(k in t for k in ("nature", "green", "environment", "eco", "wildlife", "garden", "solar")):
        return "nature green environment"

    # 4. Finance / Business domain
    if any(k in t for k in ("finance", "money", "budget", "budgets", "saving", "savings", "growth", "invest", "investment", "wealth", "banking", "bank", "stocks", "market", "revenue", "sales")):
        return "business analytics financial growth"

    # 5. Education / Science domain
    if any(k in t for k in ("education", "learning", "student", "students", "school", "university", "classroom", "study", "studying")):
        return "students learning classroom education"
    if any(k in t for k in ("science", "scientist", "laboratory", "lab", "research", "experiment")):
        return "science laboratory research"

    # 6. Filter abstract words and pick top concrete nouns
    stop = {
        "the", "and", "for", "that", "this", "with", "from", "your", "have", "will",
        "into", "about", "they", "their", "when", "every", "protecting", "protect",
        "responsibility", "everyone", "create", "better", "future", "everyday",
        "makes", "work", "life", "helping", "help", "having", "what", "which",
        "where", "could", "would", "should", "starts", "simple", "daily", "choices",
        "clearly", "grows", "today", "tomorrow", "easier", "faster", "smarter",
        "various", "things", "ways", "important", "crucial", "essential", "needs", "need",
        "using", "used", "uses", "transforming", "improve", "improves", "earlier", "later"
    }
    kws = [w for w in re.findall(r"[a-zA-Z]{4,}", t) if w not in stop]
    return " ".join(kws[:3]) if kws else "modern technology lifestyle"


class MockScenePlannerAdapter(ScenePlannerAdapter):
    provider_name = "mock"
    model_name = "mock-semantic-planner-v1"

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
        planned: List[PlannedScene] = []
        if explicit_scenes:
            segments = align_explicit_scenes_to_duration(explicit_scenes, words, total_duration)
        else:
            raw_segs = segment_words_into_scenes(words, total_duration, min_scene_duration, max_scene_duration)
            segments = [(s, e, txt, None) for s, e, txt in raw_segs]

        for idx, (start, end, text, hint) in enumerate(segments):
            safety = analyze_scene_safety(text, transcript, avoid)
            if hint:
                query = hint.strip()
            elif safety.suggested_visual_override:
                query = safety.suggested_visual_override
            elif safety.is_sensitive:
                query = "quiet candle light rain window calm solemn reflection"
            elif safety.is_abstract:
                query = "two colleagues shaking hands warm sunlit office collaboration"
            elif safety.has_statistics:
                query = "modern business analytics dashboard growth chart workspace"
            else:
                query = extract_visual_search_query(text)

            planned.append(
                PlannedScene(
                    scene_index=idx,
                    start_time=start,
                    end_time=end,
                    narration_text=text,
                    visual_hint=hint,
                    search_query=query,
                    alt_queries=[f"{query} cinematic wide", f"{query} professional documentary"],
                    generative_prompt=f"{query}, {style}, photorealistic, no text, no logos, no watermarks",
                    disambiguation_notes="; ".join(safety.homonym_resolutions) if safety.homonym_resolutions else None,
                    prefer_generative=bool(safety.is_abstract and not safety.block_ai_generation),
                    prefer_statcard=bool(safety.has_statistics),
                    stat_headline=safety.extracted_stat,
                    safety_flags=safety.flags,
                    negative_keywords=safety.negative_keywords,
                    block_ai_generation=safety.block_ai_generation,
                    warnings=safety.warnings,
                )
            )
        return planned


DISALLOWED_TITLE_WORDS = {
    "whooping", "cough", "pregnant", "pregnancy", "abortion", "attack",
    "shelling", "war", "explosion", "ebola", "covid", "predator", "corpse",
    "cadaver", "pig", "autopsy", "funeral", "crime", "arrest", "kill",
    "shooting", "infant", "symptom", "syndrome", "baby", "mental", "protest"
}


def _is_wikimedia_title_relevant(query: str, search_q: str, title: str) -> bool:
    """Validate that retrieved Wikimedia video title genuinely matches the queried subject domain."""
    title_tokens = set(re.findall(r"[a-zA-Z0-9]+", title.lower()))
    combined_q = f"{query} {search_q}".lower()

    # 1. Any disallowed/disturbing words immediately reject
    if any(bad in title_tokens for bad in DISALLOWED_TITLE_WORDS):
        return False

    def matches(words=(), prefixes=()):
        if any(w in title_tokens for w in words):
            return True
        if prefixes and any(any(t.startswith(p) for t in title_tokens) for p in prefixes):
            return True
        return False

    # 2. Healthcare / Medicine / Diagnostic domain
    if any(k in combined_q for k in ("health", "doctor", "doctors", "patient", "patients", "medical", "medicine", "disease", "diseases", "hospital", "clinic", "scan", "surgery", "diagnos")):
        return matches(
            words=("health", "healthcare", "care", "mri", "xray"),
            prefixes=("medic", "doctor", "hospital", "patient", "clinic", "disease", "surgery", "nurse", "scan", "diagnos", "pharma", "biomed", "imaging", "physician")
        )

    # 3. Planet Earth / Space domain
    if any(k in combined_q for k in ("planet", "earth", "globe")):
        return matches(
            words=("planet", "earth", "globe", "space", "iss", "orbit", "atmosphere")
        )

    # 4. Nature / Trees / Forest domain
    if any(k in combined_q for k in ("tree", "trees", "forest", "woodland", "nature", "plant", "green", "canopy", "leaves")):
        return matches(
            words=("tree", "trees", "forest", "nature", "wood", "plant", "plants", "green", "garden", "wildlife", "leaves", "leaf", "flora", "canopy")
        )

    # 5. Water / River / Ocean / Sea domain
    if any(k in combined_q for k in ("water", "ocean", "river", "sea", "lake", "stream", "marine")):
        return matches(
            words=("water", "ocean", "river", "sea", "lake", "stream", "marine", "rain", "aquatic", "wave")
        )

    # 6. Plastic / Waste / Recycling domain
    if any(k in combined_q for k in ("plastic", "waste", "recycle", "recycling", "pollution", "clean")):
        return matches(
            words=("plastic", "waste", "recycle", "recycling", "garbage", "trash", "clean", "pollution", "ecology")
        )

    # 7. Robotics / Robots domain
    if any(k in combined_q for k in ("robot", "robotics", "humanoid", "android", "cyborg")):
        return matches(
            words=("robot", "robots", "robotic", "humanoid", "android", "cyborg", "biped")
        )

    # 8. AI / Machine Learning / High Tech domain
    if any(k in combined_q for k in ("artificial intelligence", "ai", "algorithm", "technology", "tech", "neural")):
        return matches(
            words=("intelligence", "algorithm", "digital", "network", "cyber", "neural"),
            prefixes=("ai", "technolog", "comput")
        )

    # 9. Coding / Software Development domain
    if any(k in combined_q for k in ("code", "coding", "software", "program", "developer", "terminal")):
        return matches(
            words=("code", "coding", "software", "program", "developer", "terminal", "algorithm", "script"),
            prefixes=("comput",)
        )

    # 10. Finance / Business / Economy domain
    if any(k in combined_q for k in ("finance", "money", "budget", "bank", "wealth", "invest", "stock", "market")):
        return matches(
            words=("finance", "money", "budget", "bank", "banking", "wealth", "invest", "stock", "stocks", "market", "currency"),
            prefixes=("econom",)
        )

    # 11. Science / Laboratory domain
    if any(k in combined_q for k in ("science", "scientist", "laboratory", "lab", "research", "experiment")):
        return matches(
            words=("science", "scientist", "research", "microscope", "experiment", "chemistry", "physics"),
            prefixes=("laborat",)
        )

    # 12. Education / School / Students domain
    if any(k in combined_q for k in ("student", "classroom", "school", "university", "study", "learning")):
        return matches(
            words=("student", "students", "classroom", "school", "study", "learning", "lecture"),
            prefixes=("universit", "educat")
        )

    # 13. Sports / Cricket / Athletics / Stadium domain
    if any(k in combined_q for k in ("cricket", "sport", "sports", "football", "soccer", "basketball", "tennis", "match", "stadium", "tournament", "player", "athlete", "game")):
        return matches(
            words=("cricket", "sport", "sports", "match", "stadium", "tournament", "player", "bowler", "batsman", "ball", "football", "soccer", "game", "athletic", "field", "pitch")
        )

    # Default fallback: title must contain at least one meaningful search term
    q_words = [w for w in re.findall(r"[a-zA-Z]{4,}", search_q.lower()) if w not in {"video", "footage", "modern", "cinematic", "wide"}]
    return bool(set(q_words) & title_tokens)


def _try_fetch_wikimedia_video(query: str, out_path: Path, duration_needed: float) -> bool:
    """Attempt to fetch an open public-domain/CC stock video clip for real query concepts (0 API keys)."""
    if "PYTEST_CURRENT_TEST" in os.environ:
        return False
    try:
        import hashlib
        import shutil
        import httpx
        import urllib.parse

        search_q = extract_visual_search_query(query)
        cache_dir = Path("./.local_storage/clip_cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        w_cache_key = hashlib.sha1(f"wiki:{search_q.lower()[:80]}:{duration_needed:.1f}".encode()).hexdigest()[:16]
        cached_w_clip = cache_dir / f"{w_cache_key}.mp4"
        if cached_w_clip.exists() and cached_w_clip.stat().st_size > 20000:
            shutil.copyfile(cached_w_clip, out_path)
            return True

        candidate_queries = [search_q]

        # Domain fallbacks if search_q has zero hits
        q_l = query.lower()
        if any(k in q_l for k in ("health", "doctor", "medical", "disease", "patient", "hospital")):
            candidate_queries.append("artificial intelligence healthcare medical")
        elif any(k in q_l for k in ("planet", "earth", "nature", "tree", "forest")):
            candidate_queries.append("planet earth")
        elif any(k in q_l for k in ("robot", "robotics", "ai", "artificial intelligence")):
            candidate_queries.append("robot humanoid robotics")
        elif any(k in q_l for k in ("cricket", "football", "soccer", "basketball", "tennis", "sports", "stadium", "match", "tournament", "player", "athlete")):
            candidate_queries.append("cricket match tournament")

        headers = {"User-Agent": "IncuBrixBrollEngine/1.0 (santoshkonakalla7@gmail.com)"}
        with httpx.Client(headers=headers, follow_redirects=True, timeout=4.0) as client:
            for cur_q in candidate_queries[:2]:
                api_url = f"https://commons.wikimedia.org/w/api.php?action=query&format=json&list=search&srsearch={urllib.parse.quote(cur_q)}+filetype:video&srnamespace=6&srlimit=3"
                try:
                    resp = client.get(api_url)
                except Exception:
                    continue
                if resp.status_code != 200:
                    continue
                hits = resp.json().get("query", {}).get("search", [])
                for hit in hits[:2]:
                    title = hit.get("title", "")
                    if not _is_wikimedia_title_relevant(query, cur_q, title):
                        continue
                    vinfo_url = f"https://commons.wikimedia.org/w/api.php?action=query&format=json&titles={urllib.parse.quote(title)}&prop=videoinfo&viprop=url|size|derivatives"
                    try:
                        vres = client.get(vinfo_url)
                    except Exception:
                        continue
                    if vres.status_code != 200:
                        continue
                    pages = vres.json().get("query", {}).get("pages", {})
                    for pid, p in pages.items():
                        vinfo = p.get("videoinfo", [{}])[0]
                        derivs = vinfo.get("derivatives", [])
                        target_url = None
                        # Prioritize 360p or 480p low-bandwidth derivatives (download in 1-2s vs 30s)
                        v_derivs = [d for d in derivs if 360 <= d.get("height", 0) <= 540 and d.get("src")]
                        if v_derivs:
                            v_derivs.sort(key=lambda d: d.get("height", 0))
                            target_url = v_derivs[0]["src"]
                        elif derivs:
                            for d in derivs:
                                if 360 <= d.get("height", 0) <= 720 and d.get("src"):
                                    target_url = d.get("src")
                                    break
                            if not target_url and derivs:
                                target_url = derivs[0].get("src")
                        if not target_url:
                            target_url = vinfo.get("url")

                        if not target_url:
                            continue

                        seek = "8" if any(k in cur_q.lower() for k in ("earth", "planet", "forest", "nature")) else "2"
                        out_path.parent.mkdir(parents=True, exist_ok=True)
                        cmd = [
                            "ffmpeg", "-y", "-threads", "2",
                            "-headers", "User-Agent: IncuBrixBrollEngine/1.0 (santoshkonakalla7@gmail.com)\r\n",
                            "-ss", seek, "-t", str(max(duration_needed + 0.5, 4.0)),
                            "-i", target_url,
                            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "26",
                            "-pix_fmt", "yuv420p", "-an", str(out_path),
                        ]
                        res = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                        if res.returncode == 0 and out_path.exists() and out_path.stat().st_size > 20000:
                            cap = cv2.VideoCapture(str(out_path))
                            ret, frame = cap.read()
                            cap.release()
                            if ret and frame.mean() > 15.0:
                                try:
                                    shutil.copyfile(out_path, cached_w_clip)
                                except Exception:
                                    pass
                                return True
        return False
    except Exception:
        return False
        return False
    except Exception:
        return False


class MockStockFootageAdapter(FootageSourceAdapter):
    def __init__(self, source_name: str = "pexels", is_generative: bool = False) -> None:
        self.source_name = source_name
        self.is_generative = is_generative

    def search(
        self,
        query: str,
        *,
        duration_needed: float,
        orientation: str,
        negative_keywords: List[str],
        limit: int = 3,
        generative_prompt: str = "",
        stat_headline: Optional[str] = None,
        scene_text: str = "",
        work_dir: Optional[Path] = None,
        seed: int = 42,
    ) -> List[FootageCandidate]:
        out_dir = work_dir or Path("./.tmp_jobs/mock_clips")
        candidates: List[FootageCandidate] = []
        edge_x = 0.84 if "edge" in query.lower() or "edge" in scene_text.lower() else None

        cache_dir = Path("./.local_storage/clip_cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        import shutil

        for rank in range(limit):
            key = f"{self.source_name}:{query}:{rank}:{seed}"
            cid = f"{self.source_name}_{hashlib.sha1(key.encode()).hexdigest()[:10]}"
            local_clip = out_dir / f"{cid}.mp4"
            cached_clip = cache_dir / f"{cid}.mp4"
            if not local_clip.exists():
                if cached_clip.exists() and cached_clip.stat().st_size > 20000:
                    try:
                        shutil.copyfile(cached_clip, local_clip)
                    except Exception:
                        pass
                if not local_clip.exists():
                    fetched = False
                    if rank == 0 and not edge_x:
                        fetched = _try_fetch_wikimedia_video(query, local_clip, duration_needed)
                    if not fetched or not local_clip.exists():
                        generate_mock_video_clip(
                            local_clip,
                            label=f"{self.source_name}: {query}",
                            duration=max(duration_needed + 1.0, 4.5),
                            edge_subject_x=edge_x,
                            seed=seed + rank,
                        )
                    if local_clip.exists() and local_clip.stat().st_size > 20000:
                        try:
                            shutil.copyfile(local_clip, cached_clip)
                        except Exception:
                            pass
            licence = (
                "AI Generated (Commercial Use Allowed — Mock Gemini Image)"
                if self.is_generative
                else f"{self.source_name.title()} Free Commercial Licence (Mock)"
            )
            candidates.append(
                FootageCandidate(
                    clip_id=cid,
                    source=self.source_name,
                    download_url=f"mock://{cid}.mp4",
                    source_url=f"https://{self.source_name}.example.com/video/{cid}",
                    licence=licence,
                    attribution=f"{self.source_name.title()} Creator #{rank + 1}",
                    duration=max(duration_needed + 1.0, 4.5),
                    width=1280,
                    height=720,
                    tags=query.lower().split()[:5],
                    description=query,
                    ai_generated=self.is_generative,
                    watermarked=False,
                    commercial_use_allowed=True,
                    estimated_cost_usd=0.0,
                    local_path=local_clip,
                )
            )
        return candidates

    def fetch_clip(self, candidate: FootageCandidate, dest_path: Path) -> Path:
        assert candidate.local_path is not None and candidate.local_path.exists()
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        if candidate.local_path.resolve() != dest_path.resolve():
            dest_path.write_bytes(candidate.local_path.read_bytes())
        return dest_path


class MockVLMScorerAdapter(VLMScorerAdapter):
    provider_name = "mock"
    model_name = "mock-independent-vlm-v1"

    def score_clip(
        self,
        video_path: Path,
        narration_text: str,
        visual_intent: str,
        avoid_terms: List[str],
        negative_keywords: List[str],
        candidate_metadata: str = "",
    ) -> VLMScoreResult:
        violated = violates_avoid_list(candidate_metadata, avoid_terms, negative_keywords)
        if violated:
            return VLMScoreResult(
                relevance_score=1.5,
                rationale=f"Penalized: candidate metadata matched avoided/negative term '{violated}'.",
                violates_avoid=True,
                avoid_reason=violated,
                subject_focus_x=0.5,
                provider=self.provider_name,
                model=self.model_name,
            )

        # Deterministic semantic overlap score in [3.8, 4.8] for matching clips
        intent_words = set(re.findall(r"[a-zA-Z]{3,}", visual_intent.lower()))
        meta_words = set(re.findall(r"[a-zA-Z]{3,}", candidate_metadata.lower()))
        overlap = len(intent_words & meta_words) / max(1, len(intent_words))
        score = round(min(4.9, 3.7 + overlap * 1.1), 2)
        focus_x = 0.84 if "edge" in (narration_text + " " + visual_intent).lower() else 0.5
        return VLMScoreResult(
            relevance_score=score,
            rationale=f"Independent mock VLM verified visual alignment with '{visual_intent[:60]}' (overlap={overlap:.2f}).",
            violates_avoid=False,
            subject_focus_x=focus_x,
            provider=self.provider_name,
            model=self.model_name,
        )
