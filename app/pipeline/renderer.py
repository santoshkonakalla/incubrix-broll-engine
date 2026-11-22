"""FFmpeg video normalizer, color/luma matcher, per-scene segment cache, and fast swap-a-clip assembler."""

from functools import lru_cache
import json
from pathlib import Path
import subprocess
import textwrap
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from app.errors import AppError
from app.logging_config import get_logger
from app.pipeline.captions import ffmpeg_escape_subtitles_path, select_caption_font, write_srt
from app.pipeline.framing import compute_crop_expression

logger = get_logger("renderer")


@lru_cache(maxsize=1)
def ffmpeg_has_subtitles_filter() -> bool:
    try:
        proc = subprocess.run(["ffmpeg", "-hide_banner", "-filters"], capture_output=True, text=True, timeout=15)
        return " subtitles " in proc.stdout
    except Exception:
        return False


def _load_caption_font(language: str, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    lang = (language or "en").lower()
    candidates = []
    if lang.startswith("hi"):
        candidates.extend([
            "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Bold.ttf",
            "C:/Windows/Fonts/NirmalaB.ttf",
            "C:/Windows/Fonts/Nirmala.ttf",
            "C:/Windows/Fonts/mangalb.ttf",
        ])
    elif lang.startswith("ta"):
        candidates.extend([
            "/usr/share/fonts/truetype/noto/NotoSansTamil-Bold.ttf",
            "C:/Windows/Fonts/NirmalaB.ttf",
            "C:/Windows/Fonts/Nirmala.ttf",
            "C:/Windows/Fonts/lathab.ttf",
        ])
    candidates.extend([
        "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "C:/Windows/Fonts/segoeuib.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
    ])
    for fp in candidates:
        if Path(fp).exists():
            try:
                return ImageFont.truetype(fp, size=size)
            except Exception:
                pass
    return ImageFont.load_default()


def _render_caption_png(text: str, width: int, height: int, language: str, out_png: Path) -> Path:
    """Render a transparent RGBA caption pill overlay PNG (fallback when FFmpeg lacks libass)."""
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    font_size = max(28, int(min(width, height) * 0.042))
    font = _load_caption_font(language, font_size)

    max_chars = 38 if width >= height else 26
    wrapped = "\n".join(textwrap.wrap(text.strip(), width=max_chars)[:3])
    bbox = draw.multiline_textbbox((0, 0), wrapped, font=font, align="center", spacing=8)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    pad_x, pad_y = 26, 16
    box_w = tw + pad_x * 2
    box_h = th + pad_y * 2
    x0 = max(16, (width - box_w) // 2)
    y0 = max(16, height - box_h - int(height * 0.08))
    draw.rounded_rectangle([x0, y0, x0 + box_w, y0 + box_h], radius=14, fill=(10, 14, 24, 195))
    draw.multiline_text(
        (x0 + pad_x, y0 + pad_y),
        wrapped,
        font=font,
        fill=(255, 255, 255, 255),
        align="center",
        spacing=8,
    )
    img.save(out_png, format="PNG")
    return out_png

RESOLUTION_PROFILES: Dict[str, Dict[str, Tuple[int, int]]] = {
    "720p": {
        "16:9": (1280, 720),
        "9:16": (720, 1280),
        "1:1": (720, 720),
    },
    "1080p": {
        "16:9": (1920, 1080),
        "9:16": (1080, 1920),
        "1:1": (1080, 1080),
    },
}


def get_aspect_dimensions(resolution: Optional[str] = None) -> Dict[str, Tuple[int, int]]:
    try:
        from app.config import get_settings
        res = resolution or get_settings().render_resolution
        return RESOLUTION_PROFILES.get(res, RESOLUTION_PROFILES["720p"])
    except Exception:
        return RESOLUTION_PROFILES["720p"]


ASPECT_DIMENSIONS: Dict[str, Tuple[int, int]] = RESOLUTION_PROFILES["720p"]

RATIO_FILE_TAGS: Dict[str, str] = {
    "16:9": "16x9",
    "9:16": "9x16",
    "1:1": "1x1",
}


_probe_cache: Dict[str, Dict[str, float | int | str]] = {}
_color_eq_cache: Dict[str, str] = {}


def probe_video_stream(video_path: Path) -> Dict[str, float | int | str]:
    k = str(video_path.resolve())
    if k in _probe_cache:
        return _probe_cache[k]

    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,codec_name,pix_fmt,r_frame_rate,duration:format=duration",
        "-of", "json",
        str(video_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if proc.returncode != 0:
        raise AppError(
            "RENDER_FAILED",
            f"ffprobe failed on clip {video_path.name}: {proc.stderr[:250]}",
            status_code=500,
            retryable=True,
        )
    data = json.loads(proc.stdout or "{}")
    streams = data.get("streams") or []
    if not streams:
        raise AppError("RENDER_FAILED", f"Clip {video_path.name} has no video stream.", status_code=500)
    st = streams[0]
    dur = float(st.get("duration") or (data.get("format") or {}).get("duration") or 0.0)
    res = {
        "width": int(st.get("width") or 1920),
        "height": int(st.get("height") or 1080),
        "codec_name": str(st.get("codec_name") or ""),
        "pix_fmt": str(st.get("pix_fmt") or ""),
        "r_frame_rate": str(st.get("r_frame_rate") or "30/1"),
        "duration": dur,
    }
    _probe_cache[k] = res
    return res


def compute_color_eq_filter(video_path: Path, target_luma: float = 0.46, target_sat: float = 0.38) -> str:
    """
    Measure clip average brightness & saturation. If the footage is already balanced,
    returns an empty string to preserve pristine original camera colors without artificial distortion.
    """
    k = f"{video_path.resolve()}_{target_luma}_{target_sat}"
    if k in _color_eq_cache:
        return _color_eq_cache[k]

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return ""
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        idxs = [0, max(0, total // 2), max(0, total - 2)]
        lumas, sats = [], []
        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            small = cv2.resize(frame, (160, 90))
            hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV).astype(np.float32) / 255.0
            lumas.append(float(np.mean(hsv[:, :, 2])))
            sats.append(float(np.mean(hsv[:, :, 1])))

        if not lumas:
            _color_eq_cache[k] = ""
            return ""

        mean_luma = float(np.mean(lumas))
        mean_sat = float(np.mean(sats))

        # Do not artificially apply colour correction if source footage is already balanced
        if abs(target_luma - mean_luma) <= 0.08 and (0.22 <= mean_sat <= 0.52):
            _color_eq_cache[k] = ""
            return ""

        # Gently nudge brightness & saturation toward cohesive target without clipping
        brightness_delta = max(-0.10, min(0.10, (target_luma - mean_luma) * 0.40))
        sat_mult = max(0.88, min(1.15, target_sat / max(0.15, mean_sat) if mean_sat > 0.05 else 1.0))
        sat_mult = 0.7 + 0.3 * sat_mult  # heavily damped adjustment
        eq_val = f"eq=brightness={brightness_delta:.3f}:contrast=1.01:saturation={sat_mult:.3f}"
        _color_eq_cache[k] = eq_val
        return eq_val
    finally:
        cap.release()


def render_scene_segment(
    src_video: Path,
    out_segment: Path,
    *,
    duration: float,
    aspect_ratio: str,
    focus_x: float = 0.5,
    fps: int = 30,
    color_normalize: bool = True,
    scene_cues: Optional[List[Tuple[float, float, str]]] = None,
    language: str = "en",
    work_dir: Optional[Path] = None,
) -> Path:
    """
    Normalize a single source clip to exact target resolution, 30fps, yuv420p, H.264,
    subject-aware crop, color normalization, exact duration, and optional scene-local burned captions.
    Produces a stream-copy-compatible segment cached for fast swap-a-clip.
    """
    target_w, target_h = get_aspect_dimensions()[aspect_ratio]
    meta = probe_video_stream(src_video)
    src_w = int(meta["width"])
    src_h = int(meta["height"])
    src_dur = float(meta["duration"])

    crop_w, crop_h, crop_x, crop_y = compute_crop_expression(
        src_w=src_w,
        src_h=src_h,
        target_w=target_w,
        target_h=target_h,
        focus_x=focus_x,
    )

    filters = [
        f"crop={crop_w}:{crop_h}:{crop_x}:{crop_y}",
        f"scale={target_w}:{target_h}:flags=fast_bilinear",
        f"fps={fps}",
        "setsar=1",
    ]
    if color_normalize:
        eq_filt = compute_color_eq_filter(src_video)
        if eq_filt:
            filters.append(eq_filt)

    # Optional scene-local burned-in captions (enables 1-second -c:v copy swap even with burn_in=True!)
    scene_srt_path: Optional[Path] = None
    run_cwd: Optional[Path] = None
    overlay_inputs: List[str] = []
    overlay_cues: List[Tuple[float, float, Path]] = []

    if scene_cues:
        srt_dir = (work_dir or out_segment.parent).resolve()
        srt_dir.mkdir(parents=True, exist_ok=True)
        if ffmpeg_has_subtitles_filter():
            scene_srt_path = srt_dir / f"{out_segment.stem}.srt"
            write_srt(scene_cues, scene_srt_path)
            run_cwd = srt_dir
            font_name = select_caption_font(language)
            font_size = 18 if aspect_ratio == "16:9" else 15
            margin_v = 36 if aspect_ratio == "16:9" else 56
            style = (
                f"FontName={font_name},FontSize={font_size},PrimaryColour=&H00FFFFFF,"
                f"OutlineColour=&H00000000,BackColour=&H80000000,BorderStyle=3,Outline=1.5,"
                f"Shadow=0,Alignment=2,MarginV={margin_v}"
            )
            filters.append(f"subtitles=filename={scene_srt_path.name}:force_style='{style}'")
        else:
            for c_idx, (c_start, c_end, c_text) in enumerate(scene_cues):
                png_path = srt_dir / f"{out_segment.stem}_cap_{c_idx}.png"
                _render_caption_png(c_text, target_w, target_h, language, png_path)
                overlay_inputs.extend(["-i", str(png_path)])
                overlay_cues.append((c_start, c_end, png_path))

    out_segment.parent.mkdir(parents=True, exist_ok=True)
    input_args: List[str] = []
    if src_dur > 0 and src_dur < duration:
        # Loop seamlessly if clip is shorter than narration segment (never freeze-frame!)
        input_args = ["-stream_loop", "-1"]

    if overlay_cues:
        base_chain = ",".join(filters)
        fc_parts = [f"[0:v]{base_chain}[v0]"]
        prev_label = "v0"
        for c_idx, (c_start, c_end, _png) in enumerate(overlay_cues, start=1):
            next_label = f"v{c_idx}"
            fc_parts.append(
                f"[{prev_label}][{c_idx}:v]overlay=0:0:format=yuv420:enable='between(t,{c_start:.3f},{c_end:.3f})'[{next_label}]"
            )
            prev_label = next_label
        fc_parts.append(f"[{prev_label}]format=yuv420p[vout]")
        filter_args = ["-filter_complex", ";".join(fc_parts), "-map", "[vout]"]
    else:
        filters.append("format=yuv420p")
        filter_args = ["-vf", ",".join(filters)]

    cmd = [
        "ffmpeg",
        "-y",
        "-threads", "1",
        *input_args,
        "-i", str(src_video.resolve()),
        *overlay_inputs,
        "-t", f"{duration:.3f}",
        "-an",
        *filter_args,
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-tune", "fastdecode",
        "-crf", "26",
        "-maxrate", "2000k",
        "-bufsize", "4000k",
        "-profile:v", "main",
        "-pix_fmt", "yuv420p",
        "-colorspace", "bt709",
        "-color_primaries", "bt709",
        "-color_trc", "bt709",
        "-color_range", "tv",
        "-r", str(fps),
        "-video_track_timescale", "90000",
        "-g", str(fps * 2),
        "-movflags", "+faststart",
        str(out_segment.resolve()),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180, cwd=str(run_cwd) if run_cwd else None)
    if proc.returncode != 0:
        raise AppError(
            "RENDER_FAILED",
            f"FFmpeg scene render failed: {proc.stderr[-350:]}",
            status_code=500,
            retryable=True,
        )
    return out_segment


def assemble_video_from_segments(
    segment_paths: List[Path],
    narration_audio: Path,
    output_video: Path,
    work_dir: Path,
) -> Path:
    """
    Assemble normalized cached scene segments via fast FFmpeg concat demuxer (-c:v copy)
    and mux the narration audio track cleanly with zero drift.
    """
    if not segment_paths:
        raise AppError("RENDER_FAILED", "No video segments provided for assembly.", status_code=500)

    concat_list = work_dir / f"concat_{output_video.stem}.txt"
    lines = []
    for p in segment_paths:
        escaped = str(p.resolve()).replace("\\", "/").replace("'", "'\\''")
        lines.append(f"file '{escaped}'")
    concat_list.write_text("\n".join(lines) + "\n", encoding="utf-8")

    output_video.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-threads", "1",
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_list),
        "-i", str(narration_audio),
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-ar", "48000",
        "-ac", "2",
        "-shortest",
        "-movflags", "+faststart",
        str(output_video),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if proc.returncode != 0:
        raise AppError(
            "RENDER_FAILED",
            f"FFmpeg concat assembly failed: {proc.stderr[-350:]}",
            status_code=500,
            retryable=True,
        )
    return output_video


def slice_cues_for_scene(
    all_cues: List[Tuple[float, float, str]],
    scene_start: float,
    scene_end: float,
) -> List[Tuple[float, float, str]]:
    """Extract and rebase caption cues overlapping [scene_start, scene_end) to scene-local 0-based time."""
    local_cues: List[Tuple[float, float, str]] = []
    scene_dur = max(0.1, scene_end - scene_start)
    for start, end, text in all_cues:
        if end <= scene_start + 0.02 or start >= scene_end - 0.02:
            continue
        rel_s = max(0.0, start - scene_start)
        rel_e = min(scene_dur, end - scene_start)
        if rel_e - rel_s >= 0.15:
            local_cues.append((round(rel_s, 3), round(rel_e, 3), text))
    return local_cues
