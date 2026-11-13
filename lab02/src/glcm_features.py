"""GLCM texture features (scikit-image graycomatrix / graycoprops + entropy)."""

from pathlib import Path

import numpy as np
from PIL import Image
from skimage.feature import graycomatrix, graycoprops

# Haralick properties available in skimage.feature.graycoprops
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

MINIMAL_FEATURES = ["energy", "entropy"]
ALL_FEATURES = [*GLCM_PROPS, "entropy"]


def load_grayscale(path: Path) -> np.ndarray:
    """Load image as uint8 grayscale (KTH-TIPS color PNGs -> luminance)."""
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8)


def quantize_levels(gray: np.ndarray, levels: int = GLCM_LEVELS) -> np.ndarray:
    return ((gray.astype(np.float64) / 255.0) * (levels - 1)).astype(np.uint8)


def glcm_entropy(glcm: np.ndarray) -> float:
    p = glcm.astype(np.float64)
    p = p / p.sum()
    p = p.ravel()
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def _mean_over_angles_distances(values: np.ndarray) -> float:
    return float(np.mean(values))


def glcm_features(
    gray: np.ndarray,
    *,
    levels: int = GLCM_LEVELS,
    distances: tuple[int, ...] = GLCM_DISTANCES,
    angles: tuple[float, ...] = GLCM_ANGLES,
) -> dict[str, float]:
    q = quantize_levels(gray, levels)
    glcm = graycomatrix(
        q,
        distances=distances,
        angles=angles,
        levels=levels,
        symmetric=True,
        normed=True,
    )
    out: dict[str, float] = {}
    for prop in GLCM_PROPS:
        vals = graycoprops(glcm, prop)
        out[prop] = _mean_over_angles_distances(vals)
    out["entropy"] = glcm_entropy(glcm)
    return out
