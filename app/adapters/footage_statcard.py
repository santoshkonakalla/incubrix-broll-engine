"""Programmatic animated motion-graphic card generator (stats, numbers, & total-outage fallback)."""

import hashlib
from pathlib import Path
import subprocess
import textwrap
from typing import List, Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from app.adapters.base import FootageCandidate, FootageSourceAdapter
from app.errors import AppError

PALETTES = [
    ((18, 28, 48), (32, 82, 149), (74, 222, 128)),    # Deep navy + blue + emerald accent
    ((24, 24, 38), (79, 70, 229), (56, 189, 248)),    # Slate + indigo + sky
    ((22, 34, 38), (15, 118, 110), (251, 191, 36)),   # Dark teal + amber
    ((34, 24, 38), (126, 34, 206), (244, 114, 182)),  # Plum + violet
]


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size=size)
            except Exception:
                pass
    return ImageFont.load_default()


def render_animated_card_video(
    out_path: Path,
    *,
    headline: str,
    subtitle: str,
    duration: float,
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
    seed: int = 42,
    is_stat_chart: bool = True,
) -> Path:
    """
    Generate a clean, continuously animated 30fps H.264 motion-graphic B-roll clip.
    Continuously animates subtle background waves + growing data bars so there are
    zero frozen frames and zero black frames.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw_avi = out_path.with_suffix(".raw.mp4")

    palette_idx = int(hashlib.sha256(f"{headline}:{seed}".encode()).hexdigest(), 16) % len(PALETTES)
    bg_dark, bg_mid, accent = PALETTES[palette_idx]

    total_frames = max(int(round(duration * fps)), fps * 2)
    font_big = _load_font(84 if is_stat_chart else 56)
    font_sub = _load_font(34)

    wrapped_sub = "\n".join(textwrap.wrap(subtitle.strip(), width=42)[:3])
    short_head = headline.strip()[:36]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(raw_avi), fourcc, float(fps), (width, height))
    if not writer.isOpened():
        raise AppError("RENDER_FAILED", "OpenCV VideoWriter failed for statcard.", status_code=500)

    try:
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
        xn = xx / width
        yn = yy / height

        for frame_idx in range(total_frames):
            t = frame_idx / max(1, total_frames - 1)
            phase = frame_idx * 0.08

            # Smooth animated gradient background (guarantees continuous subtle motion -> 0 freeze frames)
            wave = 0.5 + 0.5 * np.sin(xn * 4.2 + yn * 2.1 + phase)
            r = np.clip(bg_dark[0] * (1.0 - 0.35 * wave) + bg_mid[0] * (0.35 * wave), 0, 255).astype(np.uint8)
            g = np.clip(bg_dark[1] * (1.0 - 0.35 * wave) + bg_mid[1] * (0.35 * wave), 0, 255).astype(np.uint8)
            b = np.clip(bg_dark[2] * (1.0 - 0.35 * wave) + bg_mid[2] * (0.35 * wave), 0, 255).astype(np.uint8)
            rgb = np.stack([r, g, b], axis=2)

            img = Image.fromarray(rgb, mode="RGB")
            draw = ImageDraw.Draw(img)

            # Animated subtle geometric grid / rising chart bars
            ease = 1.0 - (1.0 - min(1.0, t * 1.4)) ** 3
            bar_count = 6
            base_y = int(height * 0.84)
            bar_w = int(width * 0.055)
            gap = int(width * 0.025)
            start_x = int(width * 0.52)

            if is_stat_chart:
                bar_heights = [0.16, 0.24, 0.31, 0.42, 0.52, 0.64]
                for i in range(bar_count):
                    bh = int(height * bar_heights[i] * (0.25 + 0.75 * ease) * (1.0 + 0.03 * np.sin(phase + i)))
                    x0 = start_x + i * (bar_w + gap)
                    y0 = base_y - bh
                    color = accent if i == bar_count - 1 else (
                        min(255, bg_mid[0] + 55),
                        min(255, bg_mid[1] + 55),
                        min(255, bg_mid[2] + 55),
                    )
                    draw.rounded_rectangle([x0, y0, x0 + bar_w, base_y], radius=8, fill=color)
            else:
                # Moving abstract accent rings for non-stat fallback cards
                cx = int(width * (0.72 + 0.04 * np.sin(phase * 0.7)))
                cy = int(height * (0.50 + 0.04 * np.cos(phase * 0.6)))
                for rad in (90, 150, 210):
                    draw.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], outline=accent, width=4)

            # Accent bar + text on left half
            left_x = int(width * 0.08)
            top_y = int(height * 0.26)
            bar_len = int(140 * (0.4 + 0.6 * ease))
            draw.rounded_rectangle([left_x, top_y - 26, left_x + bar_len, top_y - 16], radius=4, fill=accent)
            draw.text((left_x, top_y), short_head, font=font_big, fill=(245, 248, 255))
            draw.multiline_text(
                (left_x, top_y + 118),
                wrapped_sub,
                font=font_sub,
                fill=(205, 218, 236),
                spacing=10,
            )

            bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            writer.write(bgr)
    finally:
        writer.release()

    # Transcode to clean H.264 yuv420p
    cmd = [
        "ffmpeg", "-y",
        "-threads", "2",
        "-i", str(raw_avi),
        "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", "-r", str(fps),
        str(out_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    raw_avi.unlink(missing_ok=True)
    if proc.returncode != 0:
        raise AppError("RENDER_FAILED", f"StatCard FFmpeg transcode failed: {proc.stderr[-250:]}", status_code=500)
    return out_path


class StatCardFootageAdapter(FootageSourceAdapter):
    source_name = "statcard"
    is_generative = True

    def search(
        self,
        query: str,
        *,
        duration_needed: float,
        orientation: str,
        negative_keywords: List[str],
        limit: int = 1,
        generative_prompt: str = "",
        stat_headline: Optional[str] = None,
        scene_text: str = "",
        work_dir: Optional[Path] = None,
        seed: int = 42,
    ) -> List[FootageCandidate]:
        headline = stat_headline or (query.title()[:28] if query else "Key Insight")
        digest = hashlib.sha1(f"{headline}:{scene_text}:{seed}".encode("utf-8")).hexdigest()[:12]
        clip_id = f"statcard_{digest}"
        w, h = (1280, 720) if orientation != "portrait" else (720, 1280)
        out_dir = work_dir or Path("./.tmp_jobs/statcards")
        out_path = out_dir / f"{clip_id}.mp4"
        render_animated_card_video(
            out_path,
            headline=headline,
            subtitle=scene_text or query,
            duration=max(duration_needed + 0.5, 3.0),
            width=w,
            height=h,
            seed=seed,
            is_stat_chart=bool(stat_headline),
        )
        return [
            FootageCandidate(
                clip_id=clip_id,
                source=self.source_name,
                download_url=f"local://{out_path}",
                source_url="https://incubrix.local/programmatic-motion-card",
                licence="MIT / Engine-Generated Original Graphic (Commercial Use Allowed)",
                attribution="Generated programmatically by IncuBrix B-roll Engine",
                duration=max(duration_needed + 0.5, 3.0),
                width=w,
                height=h,
                tags=["motion-graphic", "data-card", "commercial-safe"],
                description=f"Animated graphic card: {headline}",
                ai_generated=True,
                watermarked=False,
                commercial_use_allowed=True,
                estimated_cost_usd=0.0,
                local_path=out_path,
            )
        ]

    def fetch_clip(self, candidate: FootageCandidate, dest_path: Path) -> Path:
        if candidate.local_path and candidate.local_path.exists():
            if candidate.local_path.resolve() != dest_path.resolve():
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                dest_path.write_bytes(candidate.local_path.read_bytes())
            return dest_path
        raise AppError("RENDER_FAILED", "StatCard local clip missing.", status_code=500)
