"""Moduł koloru sygnalizatora: red / yellow / green / unknown."""

from __future__ import annotations

import cv2
import numpy as np


def classify_color_roi(bgr: np.ndarray, *, hinted: str | None = None) -> tuple[str, float]:
  # Returns (color, confidence)
    if bgr.size == 0:
        return "unknown", 0.0
    h, w = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hh, ss, vv = cv2.split(hsv)
    vertical = h >= w
    scores = {"red": 0.0, "yellow": 0.0, "green": 0.0}

    def zone_slice(k: int) -> np.ndarray:
        if vertical:
            z = h // 3
            return hsv[k * z : (k + 1) * z if k < 2 else h, :, :]
        z = w // 3
        return hsv[:, k * z : (k + 1) * z if k < 2 else w, :]

    zones = [zone_slice(i) for i in range(3)]
    color_ranges = {
        "red": lambda h_: (h_ <= 12) | (h_ >= 165),
        "yellow": lambda h_: (h_ >= 15) & (h_ <= 40),
        "green": lambda h_: (h_ >= 42) & (h_ <= 95),
    }
    zone_weights = {
        "red": [1.2, 0.6, 0.3],
        "yellow": [0.5, 1.2, 0.5],
        "green": [0.3, 0.6, 1.2],
    }
    for col, fn in color_ranges.items():
        for zi, zpatch in enumerate(zones):
            zh, zs, zv = cv2.split(zpatch)
            m = fn(zh) & (zs >= 50) & (zv >= 80)
            bright = (zv >= 160) & (zs >= 30)
            cnt = int(m.sum()) + 0.5 * int(bright.sum())
            scores[col] += cnt * zone_weights[col][zi]

    best = max(scores, key=scores.get)
    total = sum(scores.values())
    conf = scores[best] / total if total > 0 else 0.0
    if conf < 0.18 or total < 15:
        if hinted in scores and scores[hinted] > total * 0.1:
            return hinted, float(scores[hinted] / max(total, 1))
        return "unknown", conf
    return best, float(conf)


def assign_colors(img: np.ndarray, boxes: list[dict]) -> list[dict]:
    out = []
    h_img, w_img = img.shape[:2]
    for b in boxes:
        x1, y1, x2, y2 = map(int, (b["x1"], b["y1"], b["x2"], b["y2"]))
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_img, x2), min(h_img, y2)
        crop = img[y1:y2, x1:x2]
        hint = b.get("color")
        if hint == "unknown":
            hint = None
        color, conf = classify_color_roi(crop, hinted=hint)
        nb = dict(b)
        if b.get("source") == "m1" and hint and hint != "unknown" and conf < 0.25:
            color = hint
        nb["color"] = color
        nb["color_conf"] = conf
        out.append(nb)
    return out
