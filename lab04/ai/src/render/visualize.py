"""Wizualizacje: mask_black + overlay z klasami."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from classes import CLASS_COLORS_BGR, CLASS_NAMES


def render_mask_black(rgb: np.ndarray, semantic: np.ndarray) -> np.ndarray:
    out = np.zeros_like(rgb)
    fg = semantic > 0
    out[fg] = rgb[fg]
    return out


def render_overlay(
    bgr: np.ndarray,
    semantic: np.ndarray,
    instances: list | None = None,
    alpha: float = 0.45,
) -> np.ndarray:
    vis = bgr.copy()
    overlay = vis.copy()
    for cls_id in range(1, 6):
        m = semantic == cls_id
        if not m.any():
            continue
        color = CLASS_COLORS_BGR.get(cls_id, (255, 255, 255))
        overlay[m] = color
    vis = cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0)
    if instances:
        for inst in instances:
            x1, y1, x2, y2 = map(int, (inst.x1, inst.y1, inst.x2, inst.y2))
            color = CLASS_COLORS_BGR.get(inst.cls_id, (255, 255, 255))
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            name = CLASS_NAMES[inst.cls_id] if inst.cls_id < len(CLASS_NAMES) else str(inst.cls_id)
            cv2.putText(
                vis,
                f"{name} {inst.score:.2f}",
                (x1, max(12, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )
    return vis


def save_prediction_pair(
    bgr: np.ndarray,
    semantic: np.ndarray,
    instances: list,
    out_stem: Path,
) -> tuple[Path, Path]:
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    black = render_mask_black(rgb, semantic)
    over = render_overlay(bgr, semantic, instances)
    p1 = out_stem.with_name(out_stem.name + "_mask_black.png")
    p2 = out_stem.with_name(out_stem.name + "_overlay.png")
    cv2.imwrite(str(p1), cv2.cvtColor(black, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(p2), over)
    return p1, p2