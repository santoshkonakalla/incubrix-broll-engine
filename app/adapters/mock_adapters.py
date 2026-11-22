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
    layout_variant = h_int % 4
    try:
        xs = np.linspace(0.0, 2.0 * math.pi, width, dtype=np.float32)
        for i in range(total_frames):
            t = i / max(1, total_frames - 1)
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            # Smooth continuous wave background (no hard wrap seam, never black, never frozen)
            wave = 0.85 + 0.15 * np.sin(xs + i * 0.09)
            frame[:, :, 0] = np.clip(base_b * wave, 35, 220).astype(np.uint8)
            frame[:, :, 1] = np.clip(base_g * wave, 35, 220).astype(np.uint8)
            frame[:, :, 2] = np.clip(base_r * wave, 35, 220).astype(np.uint8)

            # Distinct spatial background geometry per seed (for realistic perceptual dHash variation)
            is_tech = any(k in label.lower() for k in ("robot", "ai", "intelligence", "tech", "computer", "cloud", "code", "python", "smart"))
            if is_tech:
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
            elif edge_subject_x is None:
                if layout_variant == 0:
                    cv2.rectangle(frame, (0, int(height * 0.62)), (width, height), (base_b // 2, base_g // 2, base_r // 2), -1)
                elif layout_variant == 1:
                    cv2.rectangle(frame, (0, 0), (int(width * 0.42), height), (min(230, base_b + 40), base_g, base_r), -1)
                elif layout_variant == 2:
                    pts = np.array([[0, height], [width // 2, int(height * 0.25)], [width, height]], np.int32)
                    cv2.fillPoly(frame, [pts], (base_b, min(230, base_g + 45), base_r // 2))
                else:
                    cv2.rectangle(frame, (int(width * 0.55), 0), (width, height), (base_b // 2, base_g, min(230, base_r + 45)), -1)

            # High-contrast subject (can sit at the edge for Hard Case #5 9:16 smart-crop tests!)
            if edge_subject_x is not None:
                cx = int(width * edge_subject_x)
                cy = int(height * (0.48 + 0.05 * math.sin(t * 2 * math.pi)))
            else:
                cx = int(width * (0.30 + 0.35 * t))
                cy = int(height * (0.46 + 0.08 * math.sin(t * 4 * math.pi)))

            if is_tech:
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
            else:
                cv2.circle(frame, (cx, cy), 95, (255, 245, 220), -1)
                cv2.circle(frame, (cx, cy), 60, (40, 210, 120), -1)
                cv2.rectangle(frame, (cx - 42, cy - 28), (cx + 42, cy + 28), (20, 40, 90), 5)

            # Elegant scene title pill badge
            title_text = label[:45]
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
            import asyncio
            import edge_tts
            v_lower = (voice or "").lower()
            edge_voice = "en-US-JennyNeural"
            if lang.startswith("hi"):
                edge_voice = "hi-IN-SwaraNeural"
            elif lang.startswith("ta"):
                edge_voice = "ta-IN-PallaviNeural"
            elif "in" in v_lower or lang.startswith("en-in"):
                edge_voice = "en-IN-NeerjaNeural"
            elif "gb" in v_lower:
                edge_voice = "en-GB-SoniaNeural"

            out_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_mp3 = out_path.with_suffix(".edge.mp3")
            asyncio.run(edge_tts.Communicate(text, edge_voice).save(str(tmp_mp3)))
            proc = subprocess.run(
                ["ffmpeg", "-y", "-threads", "1", "-i", str(tmp_mp3), "-ar", "24000", "-ac", "1", str(out_path)],
                capture_output=True,
                check=True,
                timeout=45,
            )
            if tmp_mp3.exists():
                tmp_mp3.unlink()
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
                # Extract top meaningful content words
                stop = {"the", "and", "for", "that", "this", "with", "from", "your", "have", "will", "into", "about", "they", "their", "when", "every"}
                kws = [w for w in re.findall(r"[a-zA-Z]{4,}", text.lower()) if w not in stop]
                query = " ".join(kws[:4]) if kws else "modern lifestyle workspace sunlit"

            planned.append(
                PlannedScene(
                    scene_index=idx,
                    start_time=start,
                    end_time=end,
                    narration_text=text,
                    visual_hint=hint,
                    search_query=query,
                    alt_queries=[f"{query} cinematic wide", "calm everyday workspace modern"],
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


def _try_fetch_wikimedia_video(query: str, out_path: Path, duration_needed: float) -> bool:
    """Attempt to fetch an open public-domain/CC stock video clip for real query concepts (0 API keys)."""
    if "PYTEST_CURRENT_TEST" in os.environ:
        return False
    try:
        import httpx
        import urllib.parse
        # Extract the most descriptive keyword
        stop = {"the", "and", "for", "with", "this", "that", "from", "your", "into", "help", "helping", "makes", "life", "everyday"}
        terms = [w for w in re.findall(r"[a-zA-Z]{4,}", query.lower()) if w not in stop]
        if not terms:
            return False
        search_q = terms[0]

        api_url = f"https://commons.wikimedia.org/w/api.php?action=query&format=json&list=search&srsearch={urllib.parse.quote(search_q)}+filetype:video&srnamespace=6&srlimit=3"
        headers = {"User-Agent": "IncuBrixBrollEngine/1.0 (santoshkonakalla7@gmail.com)"}
        with httpx.Client(headers=headers, follow_redirects=True, timeout=6.0) as client:
            resp = client.get(api_url)
            if resp.status_code != 200:
                return False
            hits = resp.json().get("query", {}).get("search", [])
            for hit in hits:
                title = hit.get("title", "")
                iurl = f"https://commons.wikimedia.org/w/api.php?action=query&format=json&titles={urllib.parse.quote(title)}&prop=imageinfo&iiprop=url"
                iresp = client.get(iurl)
                if iresp.status_code != 200:
                    continue
                pages = iresp.json().get("query", {}).get("pages", {})
                for pid, p in pages.items():
                    info = p.get("imageinfo", [{}])[0]
                    vurl = info.get("url", "")
                    if vurl and (vurl.endswith(".webm") or vurl.endswith(".mp4")):
                        tmp_src = out_path.with_suffix(".src" + Path(vurl).suffix)
                        r = client.get(vurl)
                        if len(r.content) > 10000:
                            tmp_src.write_bytes(r.content)
                            out_path.parent.mkdir(parents=True, exist_ok=True)
                            subprocess.run(
                                [
                                    "ffmpeg", "-y", "-threads", "1",
                                    "-i", str(tmp_src),
                                    "-c:v", "libx264", "-preset", "ultrafast", "-crf", "26",
                                    "-pix_fmt", "yuv420p", "-t", str(max(duration_needed + 0.5, 4.0)),
                                    "-an", str(out_path)
                                ],
                                capture_output=True,
                                check=True,
                                timeout=45,
                            )
                            tmp_src.unlink(missing_ok=True)
                            return True
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

        for rank in range(limit):
            key = f"{self.source_name}:{query}:{rank}:{seed}"
            cid = f"{self.source_name}_{hashlib.sha1(key.encode()).hexdigest()[:10]}"
            local_clip = out_dir / f"{cid}.mp4"
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
