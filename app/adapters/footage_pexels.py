"""Pexels Video API adapter (commercial use allowed under Pexels Licence, no watermarks)."""

from pathlib import Path
from typing import List, Optional

import httpx

from app.adapters.base import FootageCandidate, FootageSourceAdapter
from app.errors import AppError
from app.pipeline.safety import violates_avoid_list


class PexelsFootageAdapter(FootageSourceAdapter):
    source_name = "pexels"
    is_generative = False

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    def search(
        self,
        query: str,
        *,
        duration_needed: float,
        orientation: str,
        negative_keywords: List[str],
        limit: int = 4,
        generative_prompt: str = "",
        stat_headline: Optional[str] = None,
        scene_text: str = "",
        work_dir: Optional[Path] = None,
        seed: int = 42,
    ) -> List[FootageCandidate]:
        if not self.api_key:
            raise AppError("PROVIDER_UNAVAILABLE", "PEXELS_API_KEY is not configured.", status_code=503, retryable=True)

        params = {
            "query": query,
            "per_page": max(8, limit * 2),
            "orientation": orientation if orientation in ("landscape", "portrait", "square") else "landscape",
        }
        try:
            with httpx.Client(timeout=6.0) as client:
                resp = client.get(
                    "https://api.pexels.com/videos/search",
                    headers={"Authorization": self.api_key},
                    params=params,
                )
        except Exception as exc:
            raise AppError("PROVIDER_UNAVAILABLE", f"Pexels search error: {exc}", status_code=503, retryable=False) from exc

        if resp.status_code >= 400:
            raise AppError(
                "PROVIDER_UNAVAILABLE",
                f"Pexels returned HTTP {resp.status_code}",
                status_code=503,
                retryable=False if resp.status_code in (400, 401, 403) else True,
            )

        videos = resp.json().get("videos") or []
        candidates: List[FootageCandidate] = []
        for v in videos:
            vid_id = f"pexels_{v.get('id')}"
            v_url = str(v.get("url") or f"https://www.pexels.com/video/{v.get('id')}/")
            user_name = str((v.get("user") or {}).get("name") or "Pexels Contributor")
            dur = float(v.get("duration") or 5.0)
            slug_desc = v_url.rstrip("/").split("/")[-1].replace("-", " ")

            if violates_avoid_list(slug_desc, [], negative_keywords):
                continue

            # Pick cleanest HD mp4 file (prefer ~1080p or 720p, avoid huge 4K downloads for speed)
            files = [f for f in (v.get("video_files") or []) if f.get("file_type") == "video/mp4" and f.get("link")]
            if not files:
                continue
            files.sort(key=lambda f: abs(int(f.get("width") or 1280) - 1280))
            chosen = files[0]

            candidates.append(
                FootageCandidate(
                    clip_id=vid_id,
                    source=self.source_name,
                    download_url=str(chosen["link"]),
                    source_url=v_url,
                    licence="Pexels License (Free for Commercial Use, No Watermark)",
                    attribution=f"Video by {user_name} on Pexels",
                    duration=dur,
                    width=int(chosen.get("width") or 1280),
                    height=int(chosen.get("height") or 720),
                    tags=slug_desc.split()[:8],
                    description=slug_desc,
                    ai_generated=False,
                    watermarked=False,
                    commercial_use_allowed=True,
                    estimated_cost_usd=0.0,
                )
            )
            if len(candidates) >= limit:
                break
        return candidates

    def fetch_clip(self, candidate: FootageCandidate, dest_path: Path) -> Path:
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with httpx.Client(timeout=45.0, follow_redirects=True) as client:
                with client.stream("GET", candidate.download_url) as resp:
                    if resp.status_code >= 400:
                        raise AppError("PROVIDER_UNAVAILABLE", f"Pexels download HTTP {resp.status_code}", status_code=503, retryable=True)
                    with dest_path.open("wb") as f:
                        for chunk in resp.iter_bytes(65536):
                            f.write(chunk)
        except AppError:
            raise
        except Exception as exc:
            raise AppError("PROVIDER_UNAVAILABLE", f"Pexels clip download failed: {exc}", status_code=503, retryable=True) from exc
        return dest_path
