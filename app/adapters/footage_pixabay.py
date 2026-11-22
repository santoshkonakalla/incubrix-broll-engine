"""Pixabay Video API adapter (commercial use allowed under Pixabay Content License, no watermarks)."""

from pathlib import Path
from typing import List, Optional

import httpx

from app.adapters.base import FootageCandidate, FootageSourceAdapter
from app.errors import AppError
from app.pipeline.safety import violates_avoid_list


class PixabayFootageAdapter(FootageSourceAdapter):
    source_name = "pixabay"
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
            raise AppError("PROVIDER_UNAVAILABLE", "PIXABAY_API_KEY is not configured.", status_code=503, retryable=True)

        params = {
            "key": self.api_key,
            "q": query[:100],
            "safesearch": "true",
            "per_page": max(8, limit * 2),
        }
        try:
            with httpx.Client(timeout=20.0) as client:
                resp = client.get("https://pixabay.com/api/videos/", params=params)
        except Exception as exc:
            raise AppError("PROVIDER_UNAVAILABLE", f"Pixabay search error: {exc}", status_code=503, retryable=True) from exc

        if resp.status_code >= 400:
            raise AppError("PROVIDER_UNAVAILABLE", f"Pixabay returned HTTP {resp.status_code}", status_code=503, retryable=False)

        hits = resp.json().get("hits") or []
        candidates: List[FootageCandidate] = []
        for h in hits:
            tags_str = str(h.get("tags") or "")
            if violates_avoid_list(tags_str, [], negative_keywords):
                continue

            videos_map = h.get("videos") or {}
            chosen = videos_map.get("medium") or videos_map.get("large") or videos_map.get("small")
            if not chosen or not chosen.get("url"):
                continue

            vid_id = f"pixabay_{h.get('id')}"
            user_name = str(h.get("user") or "Pixabay Contributor")
            page_url = str(h.get("pageURL") or f"https://pixabay.com/videos/id-{h.get('id')}/")

            candidates.append(
                FootageCandidate(
                    clip_id=vid_id,
                    source=self.source_name,
                    download_url=str(chosen["url"]),
                    source_url=page_url,
                    licence="Pixabay Content License (Free for Commercial Use, No Watermark)",
                    attribution=f"Video by {user_name} on Pixabay",
                    duration=float(h.get("duration") or 6.0),
                    width=int(chosen.get("width") or 1280),
                    height=int(chosen.get("height") or 720),
                    tags=[t.strip() for t in tags_str.split(",") if t.strip()],
                    description=tags_str,
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
                        raise AppError("PROVIDER_UNAVAILABLE", f"Pixabay download HTTP {resp.status_code}", status_code=503, retryable=True)
                    with dest_path.open("wb") as f:
                        for chunk in resp.iter_bytes(65536):
                            f.write(chunk)
        except AppError:
            raise
        except Exception as exc:
            raise AppError("PROVIDER_UNAVAILABLE", f"Pixabay clip download failed: {exc}", status_code=503, retryable=True) from exc
        return dest_path
