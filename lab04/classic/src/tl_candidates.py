"""M1: detekcja plam świateł — HSV + top-hat, rozszerzenie do obudowy."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from evaluate import nms


@dataclass
class TlParams:
    roi_y_frac: float = 0.62
    s_min: int = 100
    v_min: int = 145
    morph_k: int = 3
    min_area: int = 12
    max_area: int = 2500
    min_circularity_small: float = 0.25
    min_circularity_large: float = 0.45
    tophat_k: int = 15
    tophat_v_thr: int = 200
    nms_iou: float = 0.4
    min_score: float = 0.35
    max_boxes: int = 4
    gate_color_pixels: int = 60
    expand: dict[str, dict[str, float]] = field(default_factory=dict)


DEFAULT_EXPAND = {
    "red": {"w_mult": 2.8, "h_mult": 4.5, "cy_shift": -0.35},
    "yellow": {"w_mult": 2.6, "h_mult": 4.2, "cy_shift": 0.0},
    "green": {"w_mult": 2.6, "h_mult": 4.2, "cy_shift": 0.35},
    "unknown": {"w_mult": 2.5, "h_mult": 4.0, "cy_shift": 0.0},
}


def default_params() -> TlParams:
    p = TlParams()
    p.expand = {k: dict(v) for k, v in DEFAULT_EXPAND.items()}
    return p


def _color_masks(hsv: np.ndarray, p: TlParams) -> dict[str, np.ndarray]:
    h, s, v = cv2.split(hsv)
    sat = s >= p.s_min
    val = v >= p.v_min
    masks = {
        "red": (
            (((h <= 10) | (h >= 165)) & sat & val)
            | (sat & (v >= 180) & ((h <= 12) | (h >= 160)))
        ).astype(np.uint8)
        * 255,
        "yellow": (((h >= 15) & (h <= 38)) & sat & val).astype(np.uint8) * 255,
        "green": (((h >= 42) & (h <= 95)) & sat & val).astype(np.uint8) * 255,
    }
    return masks


def _tophat_bright(gray: np.ndarray, p: TlParams) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p.tophat_k, p.tophat_k))
    th = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k)
    _, mask = cv2.threshold(th, max(15, p.tophat_v_thr // 4), 255, cv2.THRESH_BINARY)
    return mask


def _filter_contour(cnt: np.ndarray, p: TlParams) -> tuple[float, float, float] | None:
    area = cv2.contourArea(cnt)
    if area < p.min_area or area > p.max_area:
        return None
    (cx, cy), radius = cv2.minEnclosingCircle(cnt)
    if radius < 1.5:
        return None
    perim = cv2.arcLength(cnt, True)
    if perim <= 0:
        return None
    circ = 4 * np.pi * area / (perim * perim)
    thr = p.min_circularity_small if radius < 6 else p.min_circularity_large
    if circ < thr:
        return None
    x, y, w, h = cv2.boundingRect(cnt)
    if w <= 0 or h <= 0:
        return None
    fill = area / (w * h)
    if fill < 0.2:
        return None
    return float(cx), float(cy), float(radius)


def expand_blob(cx: float, cy: float, radius: float, color: str, p: TlParams, w_img: int, h_img: int) -> dict:
    ex = p.expand.get(color) or p.expand.get("unknown") or DEFAULT_EXPAND["unknown"]
    bw = float(np.clip(ex["w_mult"] * radius, 10, 45))
    bh = float(np.clip(ex["h_mult"] * radius, 16, 80))
    cy_shift = ex["cy_shift"] * bh
    x1 = cx - bw / 2
    y1 = cy + cy_shift - bh / 2
    x2 = x1 + bw
    y2 = y1 + bh
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w_img, x2)
    y2 = min(h_img, y2)
    score = min(1.0, radius / 12.0)
    return {
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "score": score,
        "color": color,
        "source": "m1",
    }


def color_activity_mask(img: np.ndarray, p: TlParams) -> np.ndarray:
    roi_y = int(img.shape[0] * p.roi_y_frac)
    roi = img[:roi_y]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    masks = _color_masks(hsv, p)
    combined = masks["red"]
    for m in masks.values():
        combined = cv2.bitwise_or(combined, m)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    bright = _tophat_bright(gray, p)
    combined = cv2.bitwise_or(combined, bright)
    return combined


def image_has_tl_activity(img: np.ndarray, p: TlParams) -> bool:
    mask = color_activity_mask(img, p)
    return int(np.count_nonzero(mask)) >= p.gate_color_pixels


def detect_color_blobs(img: np.ndarray, p: TlParams | None = None) -> list[dict]:
    p = p or default_params()
    h_img, w_img = img.shape[:2]
    roi_y = int(h_img * p.roi_y_frac)
    roi = img[:roi_y]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p.morph_k, p.morph_k))
    masks = _color_masks(hsv, p)
    bright = _tophat_bright(gray, p)

    boxes: list[dict] = []
    for color, mask in masks.items():
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        if color in ("red", "green"):
            mask = cv2.bitwise_or(mask, cv2.bitwise_and(bright, mask))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            fit = _filter_contour(cnt, p)
            if fit is None:
                continue
            cx, cy, radius = fit
            boxes.append(expand_blob(cx, cy, radius, color, p, w_img, h_img))

    # jasne punkty bez saturacji — szukaj koloru w pierścieniu
    contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        fit = _filter_contour(cnt, p)
        if fit is None:
            continue
        cx, cy, radius = fit
        xi, yi = int(cx), int(cy)
        if not (0 <= xi < w_img and 0 <= yi < roi_y):
            continue
        h0, s0, v0 = hsv[yi, xi]
        if s0 < 40 and v0 < p.tophat_v_thr:
            continue
        color = "unknown"
        if h0 <= 10 or h0 >= 165:
            color = "red"
        elif 15 <= h0 <= 38:
            color = "yellow"
        elif 42 <= h0 <= 95:
            color = "green"
        elif v0 >= 220:
            # pierścień wokół centrum
            r_in = max(2, int(radius))
            r_out = min(roi_y, int(cy + radius * 2))
            patch = hsv[max(0, yi - r_out) : yi + r_out, max(0, xi - r_out) : xi + r_out]
            if patch.size:
                hh = patch[:, :, 0].reshape(-1)
                ss = patch[:, :, 1].reshape(-1)
                vv = patch[:, :, 2].reshape(-1)
                m = (ss > 60) & (vv > 100)
                if m.any():
                    med_h = np.median(hh[m])
                    if med_h <= 10 or med_h >= 165:
                        color = "red"
                    elif 15 <= med_h <= 38:
                        color = "yellow"
                    elif 42 <= med_h <= 95:
                        color = "green"
        boxes.append(expand_blob(cx, cy, radius, color, p, w_img, h_img))

    boxes = [b for b in boxes if float(b.get("score", 0)) >= p.min_score]
    boxes = nms(boxes, p.nms_iou)
    boxes = sorted(boxes, key=lambda b: -float(b["score"]))[: p.max_boxes]
    return boxes


def learn_expansion_from_gt(objects_df, max_samples: int = 500) -> dict[str, dict[str, float]]:
    """Mediany współczynników z GT (proxy: środek ramki, min_side jako średnica diody)."""
    import pandas as pd

    tls = objects_df[objects_df["class"].str.startswith("traffic_light")].copy()
    tls = tls[tls["split"] == "train"]
    color_map = {
        "traffic_light_red": "red",
        "traffic_light_yellow": "yellow",
        "traffic_light_green": "green",
        "traffic_light_unknown": "unknown",
    }
    buckets: dict[str, list[tuple[float, float, float]]] = {k: [] for k in DEFAULT_EXPAND}

    for _, row in tls.iterrows():
        if len(buckets["red"]) + len(buckets["green"]) + len(buckets["yellow"]) > max_samples:
            break
        col = color_map.get(row["class"], "unknown")
        w = row["x2"] - row["x1"]
        h = row["y2"] - row["y1"]
        if w <= 0 or h <= 0:
            continue
        cx = (row["x1"] + row["x2"]) / 2
        cy = (row["y1"] + row["y2"]) / 2
        diam = max(3.0, min(w, h) * 0.45)
        if col == "unknown":
            continue
        cy_shift = (cy - ((row["y1"] + row["y2"]) / 2)) / h
        # dla kolorowych: szacuj pozycję diody w ramce
        if col == "red":
            blob_cy = row["y1"] + h * 0.25
            cy_shift = (blob_cy - cy) / h
        elif col == "green":
            blob_cy = row["y1"] + h * 0.75
            cy_shift = (blob_cy - cy) / h
        else:
            blob_cy = row["y1"] + h * 0.5
            cy_shift = (blob_cy - cy) / h
        buckets[col].append((w / diam, h / diam, cy_shift))

    out: dict[str, dict[str, float]] = {}
    for col, vals in buckets.items():
        if not vals:
            out[col] = dict(DEFAULT_EXPAND[col])
            continue
        arr = np.array(vals)
        out[col] = {
            "w_mult": float(np.median(arr[:, 0])),
            "h_mult": float(np.median(arr[:, 1])),
            "cy_shift": float(np.median(arr[:, 2])),
        }
    out["unknown"] = dict(DEFAULT_EXPAND["unknown"])
    return out
