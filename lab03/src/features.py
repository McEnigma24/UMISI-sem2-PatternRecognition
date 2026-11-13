"""Cechy koloru, tekstury (GLCM) i kształtu z wyciętego owocu."""

from __future__ import annotations

import cv2
import numpy as np
from skimage import measure
from skimage.feature import graycomatrix, graycoprops

GLCM_PROPS = [
    "contrast",
    "dissimilarity",
    "homogeneity",
    "energy",
    "correlation",
    "ASM",
]
GLCM_DISTANCES = (1,)
GLCM_ANGLES = (0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)
GLCM_LEVELS = 32

COLOR_FEATURES = [
    "hue_mean",
    "hue_std",
    "sat_mean",
    "sat_std",
    "val_mean",
    "val_std",
    "lab_a_mean",
    "lab_b_mean",
    "red_ratio",
    "green_ratio",
    "yellow_ratio",
]

SHAPE_FEATURES = [
    "fg_area_ratio",
    "solidity",
    "eccentricity",
    "extent",
    "aspect_ratio",
]

GLCM_FEATURE_NAMES = [*GLCM_PROPS, "entropy"]
MINIMAL_FEATURES = [
    "hue_mean",
    "sat_mean",
    "red_ratio",
    "green_ratio",
    "energy",
    "entropy",
]
ALL_FEATURES = COLOR_FEATURES + GLCM_FEATURE_NAMES + SHAPE_FEATURES


def _masked_pixels(img_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    pts = img_bgr[mask]
    if pts.size == 0:
        return img_bgr.reshape(-1, 3)
    return pts


def color_features(img_bgr: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    h = hsv[:, :, 0][mask].astype(np.float64)
    s = hsv[:, :, 1][mask].astype(np.float64)
    v = hsv[:, :, 2][mask].astype(np.float64)
    a = lab[:, :, 1][mask].astype(np.float64)
    b = lab[:, :, 2][mask].astype(np.float64)

    if h.size == 0:
        h = s = v = a = b = np.array([0.0])

    red = ((h < 15) | (h > 165)) & (s > 35)
    green = (h >= 25) & (h <= 95) & (s > 30)
    yellow = (h >= 15) & (h < 35) & (s > 35)
    n = max(1, h.size)

    return {
        "hue_mean": float(h.mean()),
        "hue_std": float(h.std()),
        "sat_mean": float(s.mean()),
        "sat_std": float(s.std()),
        "val_mean": float(v.mean()),
        "val_std": float(v.std()),
        "lab_a_mean": float(a.mean()),
        "lab_b_mean": float(b.mean()),
        "red_ratio": float(red.sum() / n),
        "green_ratio": float(green.sum() / n),
        "yellow_ratio": float(yellow.sum() / n),
    }


def shape_features(mask: np.ndarray) -> dict[str, float]:
    labeled = measure.label(mask)
    if labeled.max() == 0:
        return {k: 0.0 for k in SHAPE_FEATURES}
    props = max(measure.regionprops(labeled), key=lambda r: r.area)
    min_row, min_col, max_row, max_col = props.bbox
    bw = max_col - min_col
    bh = max_row - min_row
    aspect = min(bw, bh) / max(bw, bh) if max(bw, bh) > 0 else 0.0
    return {
        "fg_area_ratio": float(mask.sum() / mask.size),
        "solidity": float(props.solidity),
        "eccentricity": float(props.eccentricity),
        "extent": float(props.extent),
        "aspect_ratio": float(aspect),
    }


def quantize_levels(gray: np.ndarray, levels: int = GLCM_LEVELS) -> np.ndarray:
    return ((gray.astype(np.float64) / 255.0) * (levels - 1)).astype(np.uint8)


def glcm_entropy(glcm: np.ndarray) -> float:
    p = glcm.astype(np.float64)
    p = p / p.sum()
    p = p.ravel()
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def glcm_features(gray: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """GLCM liczone na szarości w ROI maski (tło = mediana jasności pierwszego planu)."""
    if not mask.any():
        return {k: 0.0 for k in GLCM_FEATURE_NAMES}

    g = gray.copy()
    fill = float(g[mask].mean()) if mask.any() else 128.0
    g[~mask] = fill

    q = quantize_levels(g)
    glcm = graycomatrix(
        q,
        distances=GLCM_DISTANCES,
        angles=GLCM_ANGLES,
        levels=GLCM_LEVELS,
        symmetric=True,
        normed=True,
    )
    out: dict[str, float] = {}
    for prop in GLCM_PROPS:
        out[prop] = float(np.mean(graycoprops(glcm, prop)))
    out["entropy"] = glcm_entropy(glcm)
    return out


def extract_features(img_bgr: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return (
        color_features(img_bgr, mask)
        | glcm_features(gray, mask)
        | shape_features(mask)
    )
