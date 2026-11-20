from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from transformers import SegformerConfig, SegformerForSemanticSegmentation

from classes import CLASS_NAMES, NUM_CLASSES
from models.base import SegArchitecture, SegInstance, SegPrediction, imagenet_normalize

HF_ID = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024"


def label_maps() -> dict:
    id2label = {i: n for i, n in enumerate(CLASS_NAMES)}
    return {"id2label": id2label, "label2id": {n: i for i, n in id2label.items()}}


class SegFormerAdapter(SegArchitecture):
    name = "segformer_b2"

    def build(self, *, source: str, num_classes: int = NUM_CLASSES, **hp) -> nn.Module:
        if source == "scratch":
            cfg = SegformerConfig.from_pretrained(HF_ID, num_labels=num_classes, **label_maps())
            return SegformerForSemanticSegmentation(cfg)
        return SegformerForSemanticSegmentation.from_pretrained(
            HF_ID,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
            **label_maps(),
        )

    def forward_logits(self, model: nn.Module, images: torch.Tensor) -> torch.Tensor:
        logits = model(pixel_values=imagenet_normalize(images)).logits
        return torch.nn.functional.interpolate(
            logits, size=images.shape[-2:], mode="bilinear", align_corners=False
        )

    @torch.no_grad()
    def predict(self, model: nn.Module, images: torch.Tensor) -> list[SegPrediction]:
        model.eval()
        preds = self.forward_logits(model, images).argmax(dim=1).cpu().numpy()
        return [
            SegPrediction(semantic=p.astype(np.uint8), instances=_instances_from_sem(p.astype(np.uint8)))
            for p in preds
        ]


def _instances_from_sem(sem: np.ndarray) -> list[SegInstance]:
    import cv2

    inst: list[SegInstance] = []
    for cls_id in range(1, 6):
        binm = (sem == cls_id).astype(np.uint8)
        if not binm.any():
            continue
        n, labels, stats, _ = cv2.connectedComponentsWithStats(binm, connectivity=8)
        for lab in range(1, n):
            x, y, w, h, area = stats[lab]
            if area < 6:
                continue
            inst.append(
                SegInstance(
                    cls_id=cls_id,
                    score=1.0,
                    mask=labels == lab,
                    x1=float(x),
                    y1=float(y),
                    x2=float(x + w),
                    y2=float(y + h),
                )
            )
    return inst
