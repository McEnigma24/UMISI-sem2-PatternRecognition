from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from transformers import Mask2FormerConfig, Mask2FormerForUniversalSegmentation

from classes import NUM_CLASSES
from models.base import SegArchitecture, SegPrediction, imagenet_normalize
from models.segformer import _instances_from_sem, label_maps

HF_ID = "facebook/mask2former-swin-tiny-cityscapes-semantic"


class Mask2FormerAdapter(SegArchitecture):
    name = "mask2former_swin_t"
    # końcowy layernorm Swin nie jest używany przez pixel decoder
    ddp_find_unused = True

    def build(self, *, source: str, num_classes: int = NUM_CLASSES, **hp) -> nn.Module:
        self.aux_weight = float(hp["aux_dice_focal_weight"])
        cfg = Mask2FormerConfig.from_pretrained(HF_ID, num_labels=num_classes, **label_maps())
        if source == "scratch":
            cfg.use_pretrained_backbone = False
            return Mask2FormerForUniversalSegmentation(cfg)
        return Mask2FormerForUniversalSegmentation.from_pretrained(
            HF_ID,
            config=cfg,
            ignore_mismatched_sizes=True,
        )

    @staticmethod
    def _semantic_probs(out, hw: tuple[int, int]) -> torch.Tensor:
        cls = out.class_queries_logits.float().softmax(-1)[..., :-1]
        masks = out.masks_queries_logits.float().sigmoid()
        sem = torch.einsum("bqc,bqhw->bchw", cls, masks)
        return torch.nn.functional.interpolate(sem, size=hw, mode="bilinear", align_corners=False)

    @staticmethod
    def _targets(mask: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Maska semantyczna -> (mask_labels, class_labels) per obraz: jedna binarna maska na klasę."""
        mask_labels, class_labels = [], []
        for m in mask:
            present = torch.unique(m)
            present = present[present < NUM_CLASSES]
            mask_labels.append(torch.stack([(m == c).float() for c in present]))
            class_labels.append(present.long())
        return mask_labels, class_labels

    def forward_logits(self, model: nn.Module, images: torch.Tensor) -> torch.Tensor:
        out = model(pixel_values=imagenet_normalize(images))
        return torch.log(self._semantic_probs(out, images.shape[-2:]).clamp_min(1e-6))

    def training_step_loss(self, model, batch, criterion):
        images, mask = batch["image"], batch["mask"]
        mask_labels, class_labels = self._targets(mask)
        out = model(
            pixel_values=imagenet_normalize(images),
            mask_labels=mask_labels,
            class_labels=class_labels,
        )
        logits = torch.log(self._semantic_probs(out, images.shape[-2:]).clamp_min(1e-6))
        loss = out.loss + self.aux_weight * criterion(logits, mask)
        return {"loss": loss, "logits": logits}

    @torch.no_grad()
    def predict(self, model: nn.Module, images: torch.Tensor) -> list[SegPrediction]:
        model.eval()
        sem = self.forward_logits(model, images).argmax(dim=1).cpu().numpy()
        return [
            SegPrediction(semantic=s.astype(np.uint8), instances=_instances_from_sem(s.astype(np.uint8)))
            for s in sem
        ]
