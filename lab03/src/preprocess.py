"""Preprocessing po segmentacji: CLAHE, wycięcie obiektu, normalizacja rozmiaru."""

from __future__ import annotations

import cv2
import numpy as np

TARGET_SIZE = 256


def enhance_colors(img_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """CLAHE na kanale L w Lab — tylko w obszarze obiektu (tło bez zmian)."""
    out = img_bgr.copy()
    lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge([l_eq, a, b]), cv2.COLOR_LAB2BGR)
    out[mask] = enhanced[mask]
    return out


def apply_mask(img_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.zeros_like(img_bgr)
    out[mask] = img_bgr[mask]
    return out


def bbox_with_margin(mask: np.ndarray, margin_frac: float = 0.05) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        h, w = mask.shape
        return 0, 0, h, w
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    h, w = mask.shape
    my = int((y1 - y0 + 1) * margin_frac) + 2
    mx = int((x1 - x0 + 1) * margin_frac) + 2
    return (
        max(0, y0 - my),
        min(h, y1 + my + 1),
        max(0, x0 - mx),
        min(w, x1 + mx + 1),
    )


def crop_and_resize(img_bgr: np.ndarray, mask: np.ndarray, size: int = TARGET_SIZE) -> tuple[np.ndarray, np.ndarray]:
    y0, y1, x0, x1 = bbox_with_margin(mask)
    crop_img = img_bgr[y0:y1, x0:x1]
    crop_mask = mask[y0:y1, x0:x1]
    if crop_img.size == 0:
        return img_bgr, mask
    interp = cv2.INTER_AREA if max(crop_img.shape[:2]) > size else cv2.INTER_LINEAR
    resized_img = cv2.resize(crop_img, (size, size), interpolation=interp)
    resized_mask = cv2.resize(crop_mask.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST) > 0
    return resized_img, resized_mask


def preprocess_image(img_bgr: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    enhanced = enhance_colors(img_bgr, mask)
    isolated = apply_mask(enhanced, mask)
    return crop_and_resize(isolated, mask)
