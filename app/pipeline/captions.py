"""Multilingual SRT caption generation (English, Hindi, Tamil) and FFmpeg burn-in helpers."""

from pathlib import Path
from typing import List, Tuple

from app.schemas import WordTimestamp


def _format_srt_ts(seconds: float) -> str:
    ms_total = max(0, int(round(seconds * 1000)))
    hours = ms_total // 3_600_000
    rem = ms_total % 3_600_000
    minutes = rem // 60_000
    rem %= 60_000
    secs = rem // 1000
    ms = rem % 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def build_caption_cues(
    words: List[WordTimestamp],
    max_words_per_cue: int = 7,
    max_chars_per_cue: int = 44,
    max_cue_duration: float = 4.0,
) -> List[Tuple[float, float, str]]:
    """Group word-level timestamps into readable, phrase-aligned caption cues."""
    if not words:
        return []

    cues: List[Tuple[float, float, str]] = []
    buf: List[WordTimestamp] = []

    def flush() -> None:
        nonlocal buf
        if not buf:
            return
        start = buf[0].start
        end = max(buf[-1].end, start + 0.35)
        text = " ".join(w.word.strip() for w in buf if w.word.strip())
        if text:
            cues.append((start, end, text))
        buf = []

    for w in words:
        token = w.word.strip()
        if not token:
            continue
        if buf:
            projected_text = " ".join([x.word for x in buf] + [token])
            projected_dur = w.end - buf[0].start
            gap = w.start - buf[-1].end
            if (
                len(buf) >= max_words_per_cue
                or len(projected_text) > max_chars_per_cue
                or projected_dur > max_cue_duration
                or gap > 0.65
            ):
                flush()
        buf.append(w)
        if token.endswith((".", "!", "?", "।", "॥")):
            flush()

    flush()

    # Ensure strictly monotonic non-overlapping cues
    cleaned: List[Tuple[float, float, str]] = []
    for idx, (s, e, text) in enumerate(cues):
        if cleaned and s < cleaned[-1][1]:
            s = cleaned[-1][1]
        if e <= s:
            e = s + 0.25
        cleaned.append((round(s, 3), round(e, 3), text))
    return cleaned


def write_srt(cues: List[Tuple[float, float, str]], out_path: Path) -> Path:
    lines: List[str] = []
    for i, (start, end, text) in enumerate(cues, start=1):
        lines.append(str(i))
        lines.append(f"{_format_srt_ts(start)} --> {_format_srt_ts(end)}")
        lines.append(text)
        lines.append("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def ffmpeg_escape_subtitles_path(srt_path: Path) -> str:
    """Escape path safely for FFmpeg subtitles filter on both Linux and Windows."""
    resolved = str(srt_path.resolve()).replace("\\", "/")
    resolved = resolved.replace(":", "\\:").replace("'", "\\'")
    return resolved


def select_caption_font(language: str) -> str:
    lang = (language or "en").lower()
    if lang.startswith("hi"):
        return "Noto Sans Devanagari"
    if lang.startswith("ta"):
        return "Noto Sans Tamil"
    return "Noto Sans"
