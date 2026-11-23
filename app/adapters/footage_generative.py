"""Generative AI visual adapter: Gemini Image Generation -> smooth 30fps Ken Burns motion video."""

import base64
import hashlib
from pathlib import Path
import subprocess
from typing import List, Optional

from google import genai

from app.adapters.base import FootageCandidate, FootageSourceAdapter
from app.adapters.gemini_pool import gemini_key_pool
from app.errors import AppError


def image_to_ken_burns_video(
    image_path: Path,
    out_video: Path,
    *,
    duration: float,
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
) -> Path:
    """Animate a generated still image with smooth continuous zoom + gentle pan (0 freeze frames)."""
    out_video.parent.mkdir(parents=True, exist_ok=True)
    frames = max(int(round(duration * fps)), fps * 2)
    # Slow cinematic zoom from 1.00 -> 1.14 centered smoothly
    intermediate_w = int(width * 1.15)
    intermediate_h = int(height * 1.15)
    vf = (
        f"scale={intermediate_w}:{intermediate_h}:force_original_aspect_ratio=increase,"
        f"crop={intermediate_w}:{intermediate_h},"
        f"zoompan=z='min(zoom+0.0008,1.15)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={frames}:s={width}x{height}:fps={fps},"
        f"format=yuv420p"
    )
    cmd = [
        "ffmpeg", "-y",
        "-threads", "1",
        "-loop", "1",
        "-i", str(image_path),
        "-t", f"{duration:.3f}",
        "-vf", vf,
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "26", "-pix_fmt", "yuv420p", "-r", str(fps),
        str(out_video),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:
        raise AppError("RENDER_FAILED", f"Ken Burns animation failed: {proc.stderr[-250:]}", status_code=500)
    return out_video


class GeminiGenerativeFootageAdapter(FootageSourceAdapter):
    source_name = "generative"
    is_generative = True

    def __init__(self, api_key: str, model_name: str = "gemini-3.1-flash-lite-image") -> None:
        self.api_key = api_key
        self.model_name = model_name

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
        if not self.api_key:
            raise AppError("PROVIDER_UNAVAILABLE", "GEMINI_API_KEY is not configured.", status_code=503, retryable=True)

        prompt_text = generative_prompt or query
        neg_clause = (
            f" Strictly avoid: {', '.join(negative_keywords[:8])}."
            if negative_keywords
            else ""
        )
        full_prompt = (
            f"Cinematic B-roll still frame, high photographic quality, natural lighting: {prompt_text}. "
            f"No text, no watermarks, no brand logos, no recognizable real public figures.{neg_clause}"
        )

        aspect = "16:9" if orientation != "portrait" else "9:16"
        w, h = (1280, 720) if aspect == "16:9" else (720, 1280)

        try:
            import concurrent.futures
            pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            def _call_gemini():
                if self.api_key:
                    gemini_key_pool.add_key(self.api_key)

                def _do_img(key: str):
                    client = genai.Client(api_key=key, http_options={"api_version": "v1beta", "timeout": 6000})
                    return client.interactions.create(
                        model=self.model_name,
                        input=full_prompt,
                        response_format={"type": "image", "aspect_ratio": aspect},
                    )

                return gemini_key_pool.execute_with_rotation(_do_img)

            try:
                future = pool.submit(_call_gemini)
                interaction = future.result(timeout=6.0)
            finally:
                pool.shutdown(wait=False, cancel_futures=True)
        except Exception as exc:
            raise AppError(
                "PROVIDER_UNAVAILABLE",
                f"Gemini image generation unavailable or quota exceeded ({exc}); failing over to stock/statcard.",
                status_code=503,
                retryable=False,
            ) from exc

        out_img = getattr(interaction, "output_image", None)
        if not out_img or not getattr(out_img, "data", None):
            raise AppError("PROVIDER_UNAVAILABLE", "Gemini image generation returned no image.", status_code=503)

        digest = hashlib.sha1(f"{full_prompt}:{seed}".encode("utf-8")).hexdigest()[:12]
        clip_id = f"genai_{digest}"
        out_dir = work_dir or Path("./.tmp_jobs/generative")
        out_dir.mkdir(parents=True, exist_ok=True)

        img_path = out_dir / f"{clip_id}.png"
        img_path.write_bytes(base64.b64decode(out_img.data))

        vid_path = out_dir / f"{clip_id}.mp4"
        image_to_ken_burns_video(
            img_path,
            vid_path,
            duration=max(duration_needed + 0.5, 4.0),
            width=w,
            height=h,
        )

        return [
            FootageCandidate(
                clip_id=clip_id,
                source=self.source_name,
                download_url=f"local://{vid_path}",
                source_url=f"https://ai.google.dev/gemini-api/docs/image-generation#{self.model_name}",
                licence=f"Google Gemini API Terms (Commercial Use Permitted — {self.model_name})",
                attribution=f"AI-generated visual via Google {self.model_name} + FFmpeg Ken Burns motion",
                duration=max(duration_needed + 0.5, 4.0),
                width=w,
                height=h,
                tags=["ai-generated", "image-to-video"],
                description=prompt_text,
                ai_generated=True,
                watermarked=False,
                commercial_use_allowed=True,
                estimated_cost_usd=0.02,
                local_path=vid_path,
            )
        ]

    def fetch_clip(self, candidate: FootageCandidate, dest_path: Path) -> Path:
        if candidate.local_path and candidate.local_path.exists():
            if candidate.local_path.resolve() != dest_path.resolve():
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                dest_path.write_bytes(candidate.local_path.read_bytes())
            return dest_path
        raise AppError("RENDER_FAILED", "Generative clip file missing.", status_code=500)
