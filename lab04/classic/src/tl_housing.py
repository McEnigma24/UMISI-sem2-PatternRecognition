"""M2: detekcja ciemnych obudów sygnalizatorów (black-hat)."""

from __future__ import annotations

import cv2
import numpy as np

from evaluate import nms
from tl_candidates import TlParams, color_activity_mask


def housing_activity(img: np.ndarray, p: TlParams) -> bool:
    roi_y = int(img.shape[0] * p.roi_y_frac)
    gray = cv2.cvtColor(img[:roi_y], cv2.COLOR_BGR2GRAY)
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, se)
    return int(np.count_nonzero(bh > 30)) > 120


def detect_housing(img: np.ndarray, p: TlParams | None = None, *, night: bool | None = None) -> list[dict]:
    from tl_candidates import default_params

    p = p or default_params()
    h_img, w_img = img.shape[:2]
    roi_y = int(h_img * p.roi_y_frac)
    gray = cv2.cvtColor(img[:roi_y], cv2.COLOR_BGR2GRAY)
    if night is None:
        night = float(gray.mean()) < 55

    if night:
        return []

    masks = []
    for k in (11, 17, 23):
        se = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, se)
        if night:
            _, m = cv2.threshold(bh, 25, 255, cv2.THRESH_BINARY)
        else:
            _, m = cv2.threshold(bh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        masks.append(m)

    mask = masks[0]
    for m in masks[1:]:
        mask = cv2.bitwise_or(mask, m)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[dict] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 140 or area > 5500:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 6 or h < 10:
            continue
        ar = h / w if w else 0
        vert = 1.3 <= ar <= 4.2
        horiz = 0.25 <= ar <= 0.75
        if not (vert or horiz):
            continue
        hull = cv2.convexHull(cnt)
        solid = area / max(cv2.contourArea(hull), 1)
        if solid < 0.45:
            continue
        pad = 6
        x1i = max(0, x - pad)
        y1i = max(0, y - pad)
        x2i = min(w_img, x + w + pad)
        y2i = min(roi_y, y + h + pad)
        patch = gray[y1i:y2i, x1i:x2i]
        ring = gray[max(0, y - h) : min(roi_y, y + 2 * h), max(0, x - w) : min(w_img, x + 2 * w)]
        if patch.size == 0 or ring.size == 0:
            continue
        if float(ring.mean()) - float(patch.mean()) < 18:
            continue
        score = min(1.0, solid * 0.5 + area / 2000.0)
        boxes.append(
            {
                "x1": float(x1i),
                "y1": float(y1i),
                "x2": float(x2i),
                "y2": float(y2i),
                "score": score,
                "color": "unknown",
                "source": "m2",
            }
        )
    return nms(boxes, p.nms_iou)
