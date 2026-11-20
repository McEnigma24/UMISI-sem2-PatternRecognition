"""Dice + Focal (multiclass)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceFocalLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        *,
        dice_weight: float,
        focal_weight: float,
        focal_gamma: float,
        class_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.focal_gamma = focal_gamma
        self.register_buffer("class_weights", class_weights if class_weights is not None else torch.ones(num_classes))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits B,C,H,W  targets B,H,W
        probs = F.softmax(logits, dim=1)
        tgt_oh = F.one_hot(targets.clamp(0, self.num_classes - 1), self.num_classes)
        tgt_oh = tgt_oh.permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        inter = (probs * tgt_oh).sum(dims)
        union = probs.sum(dims) + tgt_oh.sum(dims)
        dice = 1.0 - (2 * inter + 1e-6) / (union + 1e-6)
        dice = (dice * self.class_weights).mean()

        ce = F.cross_entropy(logits, targets, weight=self.class_weights, reduction="none")
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.focal_gamma * ce).mean()
        return self.dice_weight * dice + self.focal_weight * focal