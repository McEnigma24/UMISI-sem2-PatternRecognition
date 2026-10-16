from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage
from skimage import measure

SIMPLE_FEATURES = [
    "area",
    "perimeter",
    "equivalent_diameter",
    "feret_diameter_max",
    "bbox_width",
    "bbox_height",
    "bbox_area",
    "convex_area",
]

COMPLEX_FEATURES = [
    "compactness",
    "solidity",
    "extent",
    "rectangularity",
    "aspect_ratio",
    "eccentricity",
    "asymmetry_major",
    "asymmetry_minor",
    "radius_max",
    "radius_min",
    "radius_mean",
    "radius_std",
    "radius_min_max_ratio",
    "radius_mean_max_ratio",
    "radius_cv",
]

ALL_FEATURES = SIMPLE_FEATURES + COMPLEX_FEATURES


def load_mask(path: Path) -> np.ndarray:
    # cv2.imread cannot decode part of the MPEG-7 GIFs, Pillow handles all of them
    return np.array(Image.open(path).convert("L")) > 127


def region(mask: np.ndarray):
    # all foreground pixels form one object: smaller components are real parts
    # of the shape (e.g. a pocket watch chain), not noise
    return measure.regionprops(mask.astype(np.uint8))[0]


def perimeter(mask: np.ndarray) -> float:
    """Length of all boundaries (outer and holes) after simplifying the pixel staircase,
    error < 1% regardless of rotation; skimage.measure.perimeter is off by ~8% at 30°."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    return float(sum(cv2.arcLength(cv2.approxPolyDP(c, 1.0, True), True) for c in contours))


def simple_features(mask: np.ndarray) -> dict[str, float]:
    props = region(mask)
    min_row, min_col, max_row, max_col = props.bbox
    return {
        "area": float(props.area),
        "perimeter": perimeter(mask),
        "equivalent_diameter": float(props.equivalent_diameter_area),
        "feret_diameter_max": float(props.feret_diameter_max),
        "bbox_width": float(max_col - min_col),
        "bbox_height": float(max_row - min_row),
        "bbox_area": float(props.area_bbox),
        "convex_area": float(props.area_convex),
    }


def outer_contour(mask: np.ndarray) -> np.ndarray:
    """Outer boundary points of all parts as (row, col), holes are ignored."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return np.vstack(contours)[:, 0, ::-1].astype(float)


def min_area_rect(props) -> tuple:
    """Smallest rotated rectangle around the object, in OpenCV (x, y) convention.
    Fitted to pixel centers, so sides are extended by 1 px to cover whole pixels."""
    center, (w, h), angle = cv2.minAreaRect(props.coords[:, ::-1].astype(np.float32))
    return center, (w + 1, h + 1), angle


def principal_axes(props) -> tuple[np.ndarray, np.ndarray]:
    """Unit vectors (row, col) of the major and minor axis of inertia."""
    eigvals, eigvecs = np.linalg.eigh(np.cov(props.coords.T))
    return eigvecs[:, 1], eigvecs[:, 0]


def aligned_mask(mask: np.ndarray, props) -> np.ndarray:
    """Mask rotated so the centroid lies in the canvas center and the major axis is horizontal."""
    major, minor = principal_axes(props)
    radius = np.linalg.norm(props.coords - props.centroid, axis=1).max()
    size = 2 * int(np.ceil(radius)) + 3
    center = (size - 1) / 2
    # output (row, col) -> input: rows walk along the minor axis, cols along the major axis
    matrix = np.column_stack([minor, major])
    offset = np.array(props.centroid) - matrix @ [center, center]
    return ndimage.affine_transform(mask.astype(np.uint8), matrix, offset, (size, size), order=0) > 0


def asymmetry(aligned: np.ndarray, flip) -> float:
    """Share of the object that does not overlap its mirror image: 0 = perfectly symmetric."""
    return 1.0 - (aligned & flip(aligned)).sum() / aligned.sum()


def complex_features(mask: np.ndarray) -> dict[str, float]:
    props = region(mask)

    (_, _), (rect_w, rect_h), _ = min_area_rect(props)
    aligned = aligned_mask(mask, props)
    radii = np.linalg.norm(outer_contour(mask) - props.centroid, axis=1)

    return {
        "compactness": perimeter(mask) ** 2 / (4 * np.pi * props.area),
        "solidity": float(props.solidity),
        "extent": float(props.extent),
        "rectangularity": props.area / (rect_w * rect_h),
        "aspect_ratio": min(rect_w, rect_h) / max(rect_w, rect_h),
        "eccentricity": float(props.eccentricity),
        "asymmetry_major": asymmetry(aligned, np.flipud),
        "asymmetry_minor": asymmetry(aligned, np.fliplr),
        "radius_max": float(radii.max()),
        "radius_min": float(radii.min()),
        "radius_mean": float(radii.mean()),
        "radius_std": float(radii.std()),
        "radius_min_max_ratio": radii.min() / radii.max(),
        "radius_mean_max_ratio": radii.mean() / radii.max(),
        "radius_cv": radii.std() / radii.mean(),
    }


def all_features(mask: np.ndarray) -> dict[str, float]:
    return simple_features(mask) | complex_features(mask)


def label_from_filename(path: Path) -> str:
    return path.stem.rsplit("-", 1)[0]
