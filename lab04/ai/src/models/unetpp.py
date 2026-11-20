from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

from classes import NUM_CLASSES
from models.base import SegArchitecture, SegPrediction, imagenet_normalize
from models.segformer import _instances_from_sem


class UnetPPAdapter(SegArchitecture):
    name = "unetpp_convnext_t"

    def build(self, *, source: str, num_classes: int = NUM_CLASSES, **hp) -> nn.Module:
        # SMP dla enkoderów "tu-" (timm): True = pretrenowane wagi timm (ImageNet), None = losowe
        return smp.UnetPlusPlus(
            encoder_name="tu-convnext_tiny",
            encoder_weights=True if source == "downloaded" else None,
            in_channels=3,
            classes=num_classes,
        )

    def forward_logits(self, model: nn.Module, images: torch.Tensor) -> torch.Tensor:
        return model(imagenet_normalize(images))

    @torch.no_grad()
    def predict(self, model: nn.Module, images: torch.Tensor) -> list[SegPrediction]:
        model.eval()
        sem = self.forward_logits(model, images).argmax(dim=1).cpu().numpy()
        return [
            SegPrediction(semantic=s.astype(np.uint8), instances=_instances_from_sem(s))
            for s in sem
        ]
