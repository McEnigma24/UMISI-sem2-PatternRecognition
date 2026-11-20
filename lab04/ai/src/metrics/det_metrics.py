"""P/R/F1 instancji (mask IoU >= prog) — spójne z porównaniem classic."""

from __future__ import annotations

import cv2
import numpy as np

from models.base import SegInstance


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        h, w = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1])
        a, b = a[:h, :w], b[:h, :w]
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 0.0


def instances_from_mask(mask: np.ndarray, *, min_area: int = 16) -> list[SegInstance]:
    """GT: składowe spójne per klasa (bez tła)."""
    out: list[SegInstance] = []
    for cls_id in range(1, 6):
        binary = (mask == cls_id).astype(np.uint8)
        if not binary.any():
            continue
        n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        for lab in range(1, n):
            area = int(stats[lab, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            x, y, w, h = stats[lab, :4]
            inst_mask = labels == lab
            out.append(
                SegInstance(
                    cls_id=cls_id,
                    score=1.0,
                    mask=inst_mask,
                    x1=float(x),
                    y1=float(y),
                    x2=float(x + w),
                    y2=float(y + h),
                )
            )
    return out


def detection_prf1(
    pred: list[SegInstance],
    gt: list[SegInstance],
    *,
    iou_thresh: float = 0.5,
) -> dict[str, float]:
    """Hungarian-free greedy matching per class."""
    tp = fp = fn = 0
    for cls_id in range(1, 6):
        preds = [p for p in pred if p.cls_id == cls_id]
        gts = [g for g in gt if g.cls_id == cls_id]
        matched_gt: set[int] = set()
        for p in sorted(preds, key=lambda x: -x.score):
            best_j, best_iou = -1, 0.0
            for j, g in enumerate(gts):
                if j in matched_gt:
                    continue
                iou = _mask_iou(p.mask, g.mask)
                if iou >= iou_thresh and iou > best_iou:
                    best_iou, best_j = iou, j
            if best_j >= 0:
                tp += 1
                matched_gt.add(best_j)
            else:
                fp += 1
        fn += len(gts) - len(matched_gt)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "tp": float(tp), "fp": float(fp), "fn": float(fn)}