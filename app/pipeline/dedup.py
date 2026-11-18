"""Clip deduplication by canonical clip_id, source URL, and perceptual dHash (Must-Have #5)."""

from pathlib import Path
from typing import Set

import cv2
import numpy as np


def compute_video_dhash(video_path: Path, hash_size: int = 8) -> str:
    """Compute a 64-bit difference hash from the middle frame of a video clip."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return ""
    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        target_frame = max(0, frame_count // 2)
        if target_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
        if not ok or frame is None:
            return ""

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
        diff = resized[:, 1:] > resized[:, :-1]
        bits = "".join("1" if b else "0" for b in diff.flatten())
        return f"{int(bits, 2):016x}"
    finally:
        cap.release()


def hamming_distance(hex_a: str, hex_b: str) -> int:
    if not hex_a or not hex_b or len(hex_a) != len(hex_b):
        return 64
    return (int(hex_a, 16) ^ int(hex_b, 16)).bit_count()


class ClipDeduplicator:
    """Guarantees repeat_clip_rate == 0 within a single job."""

    def __init__(self, max_hamming_distance: int = 6) -> None:
        self.used_ids: Set[str] = set()
        self.used_urls: Set[str] = set()
        self.used_hashes: Set[str] = set()
        self.max_hamming_distance = max_hamming_distance

    def is_duplicate_metadata(self, clip_id: str, source_url: str = "") -> bool:
        if clip_id in self.used_ids:
            return True
        if source_url and source_url in self.used_urls:
            return True
        return False

    def is_duplicate_visual(self, video_path: Path) -> bool:
        dhash = compute_video_dhash(video_path)
        if not dhash:
            return False
        for existing in self.used_hashes:
            if hamming_distance(dhash, existing) <= self.max_hamming_distance:
                return True
        return False

    def register(self, clip_id: str, source_url: str = "", video_path: Path | None = None) -> None:
        self.used_ids.add(clip_id)
        if source_url:
            self.used_urls.add(source_url)
        if video_path and video_path.exists():
            dhash = compute_video_dhash(video_path)
            if dhash:
                self.used_hashes.add(dhash)
