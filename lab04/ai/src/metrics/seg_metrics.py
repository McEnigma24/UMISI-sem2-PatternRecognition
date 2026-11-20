"""mIoU i macierz pomyłek pikseli."""

from __future__ import annotations

import numpy as np

from classes import NUM_CLASSES


def confusion_matrix(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    k = (gt >= 0) & (gt < NUM_CLASSES)
    return np.bincount(
        NUM_CLASSES * gt[k].astype(int) + pred[k].astype(int),
        minlength=NUM_CLASSES ** 2,
    ).reshape(NUM_CLASSES, NUM_CLASSES)


def miou_from_confusion(cm: np.ndarray) -> tuple[float, float, list[float]]:
    ious = []
    for c in range(NUM_CLASSES):
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp
        fp = cm[:, c].sum() - tp
        denom = tp + fn + fp
        ious.append(float(tp / denom) if denom > 0 else float("nan"))
    valid = [x for x in ious if x == x]
    miou_all = float(np.mean(valid)) if valid else 0.0
    fg = [ious[c] for c in range(1, NUM_CLASSES) if ious[c] == ious[c]]
    miou_fg = float(np.mean(fg)) if fg else 0.0
    return miou_all, miou_fg, ious