import cv2
import numpy as np

from features import region

TARGET_SIZE = 256
MARGIN = 2


def center_mask(mask: np.ndarray) -> np.ndarray:
    """Square crop with the centroid in the middle pixel, just big enough to hold the whole object."""
    props = region(mask)
    cy, cx = np.round(props.centroid).astype(int)
    radius = int(np.ceil(np.linalg.norm(props.coords - (cy, cx), axis=1).max())) + MARGIN
    padded = np.pad(mask, radius)
    return padded[cy: cy + 2 * radius + 1, cx: cx + 2 * radius + 1]


def normalize_size(centered: np.ndarray, size: int = TARGET_SIZE) -> np.ndarray:
    """Rescale a centered crop to size x size, so every object gets the same longest radius."""
    shrinking = centered.shape[0] > size
    resized = cv2.resize(centered.astype(np.float32), (size, size),
                         interpolation=cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR)
    return resized > 0.5


VARIANTS = {
    "oryginal": lambda mask: mask,
    "wycentrowane": center_mask,
    "wycentrowane_przeskalowane": lambda mask: normalize_size(center_mask(mask)),
}
