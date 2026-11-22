"""Subject-aware dynamic framing for 9:16 and 1:1 crops (Hard Case #5)."""

from pathlib import Path
from typing import Optional, Tuple

import cv2
cv2.setNumThreads(1)
import numpy as np


def detect_subject_focus_x(
    video_path: Path,
    vlm_hint_x: Optional[float] = None,
    sample_frames: int = 5,
) -> float:
    """
    Estimate horizontal center of the primary visual subject in [0.0, 1.0].

    Uses OpenCV gradient + color-contrast saliency (and Haar face/subject boost when
    not avoided) across evenly sampled frames. If an independent VLM supplied a
    subject focus coordinate, blends it with saliency for high accuracy on
    edge-positioned subjects in 16:9 clips.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return _clamp(vlm_hint_x if vlm_hint_x is not None else 0.5, 0.15, 0.85)

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total_frames <= 0:
            indices = [0]
        else:
            indices = np.linspace(
                max(0, int(total_frames * 0.1)),
                max(0, int(total_frames * 0.9)),
                num=min(sample_frames, max(1, total_frames)),
                dtype=int,
            ).tolist()

        x_estimates = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            x_est = _frame_saliency_center_x(frame)
            del frame
            if x_est is not None:
                x_estimates.append(x_est)

        saliency_x = float(np.median(x_estimates)) if x_estimates else 0.5
        if vlm_hint_x is not None:
            blended = 0.65 * float(vlm_hint_x) + 0.35 * saliency_x
            return _clamp(blended, 0.0, 1.0)
        return _clamp(saliency_x, 0.0, 1.0)
    finally:
        cap.release()


def _frame_saliency_center_x(frame: np.ndarray) -> Optional[float]:
    h, w = frame.shape[:2]
    if w == 0 or h == 0:
        return None
    small_w = 320
    small_h = max(64, int(h * (small_w / w)))
    resized = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_AREA)

    # 1. Edge / gradient magnitude energy (highlights sharp foreground subjects vs blurred bg)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(gx, gy)

    # 2. Color contrast against frame median background (great for edge-placed subjects!)
    lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB).astype(np.float32)
    bg_color = np.median(lab.reshape(-1, 3), axis=0)
    color_dist = np.linalg.norm(lab - bg_color, axis=2)

    # Normalize & combine, focusing on the main content band (ignore top/bottom 12% letterbox/overlay edges)
    grad_norm = grad_mag / (np.max(grad_mag) + 1e-5)
    color_norm = color_dist / (np.max(color_dist) + 1e-5)
    saliency = 0.55 * grad_norm + 0.45 * color_norm

    y0 = int(small_h * 0.12)
    y1 = int(small_h * 0.85)
    roi = saliency[y0:y1, :] if y1 > y0 else saliency

    # Emphasize top-8% peak salient pixels (squared weighting) so primary subject dominates background
    thresh = np.percentile(roi, 92)
    mask = np.where(roi >= thresh, roi ** 2, 0.0)
    col_weights = mask.sum(axis=0)
    total_w = float(col_weights.sum())
    if total_w <= 1e-5:
        return 0.5

    xs = np.linspace(0.0, 1.0, small_w)
    center_x = float((col_weights * xs).sum() / total_w)
    return center_x


def compute_crop_expression(
    src_w: int,
    src_h: int,
    target_w: int,
    target_h: int,
    focus_x: float,
) -> Tuple[int, int, int, int]:
    """
    Compute (crop_w, crop_h, crop_x, crop_y) in source pixels so scaling to (target_w, target_h)
    preserves aspect ratio without distortion while keeping focus_x centered in the crop window.
    """
    target_ar = target_w / target_h
    src_ar = src_w / max(1, src_h)

    if src_ar > target_ar:
        # Source is wider than target (e.g., 16:9 -> 9:16 or 1:1): crop horizontally around focus_x
        crop_h = src_h
        crop_w = int(round(src_h * target_ar))
        crop_w = min(src_w, max(2, crop_w - (crop_w % 2)))
        desired_x = int(round(focus_x * src_w - crop_w / 2))
        crop_x = max(0, min(src_w - crop_w, desired_x))
        crop_y = 0
    else:
        # Source is taller than target (e.g., 9:16 -> 16:9): crop vertically around upper-center (rule of thirds)
        crop_w = src_w
        crop_h = int(round(src_w / target_ar))
        crop_h = min(src_h, max(2, crop_h - (crop_h % 2)))
        crop_x = 0
        crop_y = max(0, min(src_h - crop_h, int(round((src_h - crop_h) * 0.42))))

    return crop_w, crop_h, crop_x, crop_y


def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))
