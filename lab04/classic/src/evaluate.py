"""Ewaluacja detekcji traffic_light (IoU>=0.5) i koloru na TP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from dataset import box_iou

TL_COLOR_CLASSES = (
    "traffic_light_red",
    "traffic_light_yellow",
    "traffic_light_green",
    "traffic_light_unknown",
)

COLOR_TO_TL = {
    "red": "traffic_light_red",
    "yellow": "traffic_light_yellow",
    "green": "traffic_light_green",
    "unknown": "traffic_light_unknown",
}


def gt_to_light_class(cls: str) -> str:
    if cls.startswith("traffic_light_"):
        return cls
    return ""


def pred_color_class(color: str) -> str:
    return COLOR_TO_TL.get(color, "traffic_light_unknown")


def box_dict(row: dict | pd.Series) -> dict:
    return {"x1": float(row["x1"]), "y1": float(row["y1"]), "x2": float(row["x2"]), "y2": float(row["y2"])}


def min_side(box: dict) -> float:
    return min(box["x2"] - box["x1"], box["y2"] - box["y1"])


def size_bucket(box: dict) -> str:
    s = min_side(box)
    if s < 10:
        return "<10"
    if s <= 20:
        return "10-20"
    return ">20"


def match_boxes(
    preds: list[dict],
    gts: list[dict],
    iou_thr: float = 0.5,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Zachłanne dopasowanie pred -> gt malejąco po score predykcji."""
    preds_sorted = sorted(enumerate(preds), key=lambda x: -float(x[1].get("score", 0)))
    gt_used = [False] * len(gts)
    tp_pairs: list[tuple[int, int]] = []
    fp: list[int] = []

    for pi, pred in preds_sorted:
        best_j = -1
        best_iou = 0.0
        pb = box_dict(pred)
        for j, gt in enumerate(gts):
            if gt_used[j]:
                continue
            iou = box_iou(pb, box_dict(gt))
            if iou >= iou_thr and iou > best_iou:
                best_iou = iou
                best_j = j
        if best_j >= 0:
            tp_pairs.append((pi, best_j))
            gt_used[best_j] = True
        else:
            fp.append(pi)

    fn = [j for j, u in enumerate(gt_used) if not u]
    return tp_pairs, fp, fn


@dataclass
class EvalResult:
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int
    color_acc: float
    color_correct: int
    color_total: int
    color_matrix: pd.DataFrame
    by_size: pd.DataFrame
    by_timeofday: pd.DataFrame


def evaluate_image_preds(
    objects_df: pd.DataFrame,
    preds_by_image: dict[str, list[dict]],
    *,
    iou_thr: float = 0.5,
    images_df: pd.DataFrame | None = None,
) -> EvalResult:
    tls = objects_df[objects_df["class"].str.startswith("traffic_light")].copy()
    tp = fp = fn = 0
    color_ok = color_n = 0
    cm: dict[tuple[str, str], int] = {}
    size_stats: dict[str, dict[str, int]] = {}
    tod_stats: dict[str, dict[str, int]] = {}

    tod_map: dict[str, str] = {}
    if images_df is not None:
        tod_map = dict(zip(images_df["image_id"], images_df["timeofday"], strict=False))

    image_ids = set(preds_by_image.keys()) | set(tls["image_id"].unique())
    gts_by_image = {i: g.to_dict("records") for i, g in tls.groupby("image_id")}

    for image_id in image_ids:
        gts = gts_by_image.get(image_id, [])
        preds = preds_by_image.get(image_id, [])
        pairs, fp_idx, fn_idx = match_boxes(preds, gts, iou_thr=iou_thr)
        tp += len(pairs)
        fp += len(fp_idx)
        fn += len(fn_idx)

        for pi, gi in pairs:
            gt_cls = gts[gi]["class"]
            pred_col = preds[pi].get("color", "unknown")
            pred_cls = pred_color_class(pred_col)
            if pred_cls == gt_cls:
                color_ok += 1
            color_n += 1
            cm[(gt_cls, pred_cls)] = cm.get((gt_cls, pred_cls), 0) + 1

        for gi in [p[1] for p in pairs]:
            b = size_bucket(box_dict(gts[gi]))
            size_stats.setdefault(b, {"tp": 0, "fn": 0})
            size_stats[b]["tp"] += 1
        for gi in fn_idx:
            b = size_bucket(box_dict(gts[gi]))
            size_stats.setdefault(b, {"tp": 0, "fn": 0})
            size_stats[b]["fn"] += 1

        td = tod_map.get(image_id, "unknown")
        tod_stats.setdefault(td, {"tp": 0, "fn": 0})
        tod_stats[td]["tp"] += len(pairs)
        tod_stats[td]["fn"] += len(fn_idx)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    color_acc = color_ok / color_n if color_n else 0.0

    labels = list(TL_COLOR_CLASSES)
    mat = np.zeros((len(labels), len(labels)), dtype=int)
    idx = {c: i for i, c in enumerate(labels)}
    for (gt_c, pr_c), n in cm.items():
        if gt_c in idx and pr_c in idx:
            mat[idx[gt_c], idx[pr_c]] += n
    color_matrix = pd.DataFrame(mat, index=labels, columns=labels)

    size_rows = []
    for b, st in size_stats.items():
        t, f = st["tp"], st["fn"]
        size_rows.append({"size": b, "tp": t, "fn": f, "recall": t / (t + f) if t + f else 0})
    by_size = pd.DataFrame(size_rows)

    tod_rows = []
    for td, st in tod_stats.items():
        t, f = st["tp"], st["fn"]
        tod_rows.append({"timeofday": td, "tp": t, "fn": f, "recall": t / (t + f) if t + f else 0})
    by_timeofday = pd.DataFrame(tod_rows)

    return EvalResult(
        precision=precision,
        recall=recall,
        f1=f1,
        tp=tp,
        fp=fp,
        fn=fn,
        color_acc=color_acc,
        color_correct=color_ok,
        color_total=color_n,
        color_matrix=color_matrix,
        by_size=by_size,
        by_timeofday=by_timeofday,
    )


def aggregate_preds(
    preds_by_image: dict[str, list[dict]],
) -> list[dict]:
    out: list[dict] = []
    for image_id, preds in preds_by_image.items():
        for p in preds:
            out.append({**p, "image_id": image_id})
    return out


def nms(boxes: list[dict], iou_thr: float = 0.4) -> list[dict]:
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: -float(b.get("score", 0)))
    keep: list[dict] = []
    for b in boxes:
        if all(box_iou(box_dict(b), box_dict(k)) < iou_thr for k in keep):
            keep.append(b)
    return keep
