from pathlib import Path

import numpy as np
from PIL import Image
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


def load_mask(path: Path) -> np.ndarray:
    # cv2.imread cannot decode part of the MPEG-7 GIFs, Pillow handles all of them
    return np.array(Image.open(path).convert("L")) > 127


def simple_features(mask: np.ndarray) -> dict[str, float]:
    # all foreground pixels form one object: smaller components are real parts
    # of the shape (e.g. a pocket watch chain), not noise
    props = measure.regionprops(mask.astype(np.uint8))[0]
    min_row, min_col, max_row, max_col = props.bbox
    return {
        "area": float(props.area),
        "perimeter": float(props.perimeter),
        "equivalent_diameter": float(props.equivalent_diameter_area),
        "feret_diameter_max": float(props.feret_diameter_max),
        "bbox_width": float(max_col - min_col),
        "bbox_height": float(max_row - min_row),
        "bbox_area": float(props.area_bbox),
        "convex_area": float(props.area_convex),
    }


def label_from_filename(path: Path) -> str:
    return path.stem.rsplit("-", 1)[0]
