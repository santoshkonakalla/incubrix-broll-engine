"""Automated video QA checks (Must-Have #4): resolution, fps, pix_fmt, black/freeze frames, audio drift."""

import json
from pathlib import Path
import re
import subprocess
from typing import Any, Dict

from app.pipeline.renderer import ASPECT_DIMENSIONS, get_aspect_dimensions


def run_video_qa(
    video_path: Path,
    aspect_ratio: str,
    expected_duration: float,
    expected_fps: int = 30,
) -> Dict[str, Any]:
    target_w, target_h = get_aspect_dimensions()[aspect_ratio]

    probe_cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "stream=codec_type,codec_name,width,height,pix_fmt,r_frame_rate,duration:format=duration",
        "-of", "json",
        str(video_path),
    ]
    proc = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=30)
    info = json.loads(proc.stdout or "{}") if proc.returncode == 0 else {}
    streams = info.get("streams") or []
    v_stream = next((s for s in streams if s.get("codec_type") == "video"), {})
    a_stream = next((s for s in streams if s.get("codec_type") == "audio"), {})

    fps_str = str(v_stream.get("r_frame_rate") or "0/1")
    num, _, den = fps_str.partition("/")
    actual_fps = round(float(num) / max(1.0, float(den or 1)), 2)

    v_dur = float(v_stream.get("duration") or (info.get("format") or {}).get("duration") or 0.0)
    a_dur = float(a_stream.get("duration") or v_dur)
    audio_drift = round(abs(v_dur - a_dur), 3)
    narration_diff = round(abs(v_dur - expected_duration), 3)

    # Black-frame & frozen-frame detection (quick single-pass FFmpeg filter)
    detect_cmd = [
        "ffmpeg",
        "-threads", "0",
        "-i", str(video_path),
        "-vf", "blackdetect=d=0.4:pix_th=0.10,freezedetect=n=-60dB:d=1.5",
        "-f", "null",
        "-",
    ]
    det = subprocess.run(detect_cmd, capture_output=True, text=True, timeout=90)
    black_events = re.findall(r"black_duration:([0-9.]+)", det.stderr)
    freeze_events = re.findall(r"freeze_duration:([0-9.]+)", det.stderr)

    passed = (
        v_stream.get("codec_name") == "h264"
        and a_stream.get("codec_name") == "aac"
        and int(v_stream.get("width") or 0) == target_w
        and int(v_stream.get("height") or 0) == target_h
        and v_stream.get("pix_fmt") == "yuv420p"
        and abs(actual_fps - expected_fps) <= 0.5
        and audio_drift <= 0.25
        and narration_diff <= 0.5
        and len(black_events) == 0
        and len(freeze_events) == 0
    )

    return {
        "passed": passed,
        "video_codec": v_stream.get("codec_name"),
        "audio_codec": a_stream.get("codec_name"),
        "resolution": f"{v_stream.get('width')}x{v_stream.get('height')}",
        "expected_resolution": f"{target_w}x{target_h}",
        "pix_fmt": v_stream.get("pix_fmt"),
        "fps": actual_fps,
        "video_duration": round(v_dur, 3),
        "audio_duration": round(a_dur, 3),
        "audio_drift_seconds": audio_drift,
        "narration_duration_diff_seconds": narration_diff,
        "black_frame_events": len(black_events),
        "freeze_frame_events": len(freeze_events),
    }
