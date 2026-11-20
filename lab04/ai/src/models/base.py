"""Wspólny interfejs architektury segmentacji."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
# prefiksy parametrów enkodera (backbone) — dostają lr * backbone_lr_mult
_BACKBONE_PREFIXES = ("segformer.encoder.", "model.pixel_level_module.encoder.", "encoder.")


def imagenet_normalize(images: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(_IMAGENET_MEAN, device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
    std = torch.tensor(_IMAGENET_STD, device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
    return (images - mean) / std


@dataclass
class SegInstance:
    cls_id: int
    score: float
    mask: np.ndarray
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass
class SegPrediction:
    semantic: np.ndarray
    instances: list[SegInstance] = field(default_factory=list)


class SegArchitecture(ABC):
    name: str
    owns_training_loop: bool = False
    ddp_find_unused: bool = False

    @abstractmethod
    def build(self, *, source: str, num_classes: int, **hp) -> nn.Module: ...

    @abstractmethod
    def forward_logits(self, model: nn.Module, images: torch.Tensor) -> torch.Tensor:
        """Logity B,C,H,W w rozdzielczości wejścia."""

    def load_trained(self, run_dir: Path, *, source: str, **hp) -> nn.Module:
        model = self.build(source=source, **hp)
        for name in (f"{self.name}_best_miou.pt", f"{self.name}_final.pt"):
            p = run_dir / name
            if p.is_file():
                model.load_state_dict(torch.load(p, map_location="cpu", weights_only=True), strict=True)
                return model
        raise FileNotFoundError(f"Brak wag {self.name} w {run_dir}")

    def training_step_loss(self, model: nn.Module, batch: dict, criterion: nn.Module) -> dict:
        logits = self.forward_logits(model, batch["image"])
        return {"loss": criterion(logits, batch["mask"]), "logits": logits}

    @abstractmethod
    def predict(self, model: nn.Module, images: torch.Tensor) -> list[SegPrediction]: ...

    def param_groups(
        self,
        model: nn.Module,
        *,
        lr: float,
        backbone_lr_mult: float,
        weight_decay: float,
    ) -> list[dict]:
        groups: dict[tuple[bool, bool], list[nn.Parameter]] = {}
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            is_backbone = name.startswith(_BACKBONE_PREFIXES)
            no_decay = name.endswith(".bias") or "norm" in name.lower() or p.ndim == 1
            groups.setdefault((is_backbone, no_decay), []).append(p)
        return [
            {
                "params": params,
                "lr": lr * (backbone_lr_mult if is_backbone else 1.0),
                "weight_decay": 0.0 if no_decay else weight_decay,
            }
            for (is_backbone, no_decay), params in groups.items()
        ]
