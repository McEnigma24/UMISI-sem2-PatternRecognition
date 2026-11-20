"""LightningModule segmentacji semantycznej."""

from __future__ import annotations

import lightning.pytorch as pl
import torch
import torch.nn as nn
from torchmetrics.classification import MulticlassJaccardIndex

from classes import CLASS_NAMES, NUM_CLASSES
from lit.gpu_augment import gpu_train_augment
from lit.optim import warmup_cosine_scheduler
from models.base import SegArchitecture


class LitSegModule(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        adapter: SegArchitecture,
        criterion: nn.Module,
        *,
        learning_rate: float,
        backbone_lr_mult: float,
        weight_decay: float,
        warmup_steps: int,
        total_steps: int,
    ) -> None:
        super().__init__()
        self.model = model
        self.adapter = adapter
        self.criterion = criterion
        self.learning_rate = learning_rate
        self.backbone_lr_mult = backbone_lr_mult
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.train_iou = MulticlassJaccardIndex(NUM_CLASSES, average="none")
        self.val_iou = MulticlassJaccardIndex(NUM_CLASSES, average="none")
        self.save_hyperparameters(ignore=["model", "criterion", "adapter"])

    def forward(self, x):
        return self.adapter.forward_logits(self.model, x)

    def on_after_batch_transfer(self, batch, dataloader_idx):
        batch["image"] = batch["image"].float().div_(255.0)
        batch["mask"] = batch["mask"].long()
        if self.training:
            batch["image"], batch["mask"] = gpu_train_augment(batch["image"], batch["mask"])
        return batch

    def _step(self, batch, stage: str):
        out = self.adapter.training_step_loss(self.model, batch, self.criterion)
        loss = out["loss"]
        preds = out["logits"].argmax(dim=1)
        bs = batch["image"].shape[0]
        metric = self.train_iou if stage == "train" else self.val_iou
        metric.update(preds, batch["mask"])
        # train/loss bez sync_dist: pasek postępu (tylko rank 0) liczy go w on_train_epoch_end,
        # a all-reduce wywołany tylko na jednym ranku zawiesza DDP
        self.log(f"{stage}/loss", loss, on_step=stage == "train", on_epoch=True,
                 sync_dist=stage == "val", batch_size=bs)
        return loss

    def _log_iou(self, stage: str, metric: MulticlassJaccardIndex) -> None:
        # compute() synchronizuje macierz pomyłek na wszystkich rankach; wynik identyczny — bez sync_dist
        per_cls = metric.compute()
        self.log(f"{stage}/mIoU", per_cls.mean(), prog_bar=stage == "val")
        self.log(f"{stage}/mIoU_fg", per_cls[1:].mean(), prog_bar=stage == "val")
        if stage == "val":
            for i, name in enumerate(CLASS_NAMES):
                self.log(f"val_iou/{name}", per_cls[i])
        metric.reset()

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def on_train_epoch_end(self):
        self._log_iou("train", self.train_iou)

    def on_validation_epoch_end(self):
        self._log_iou("val", self.val_iou)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.adapter.param_groups(
                self.model,
                lr=self.learning_rate,
                backbone_lr_mult=self.backbone_lr_mult,
                weight_decay=self.weight_decay,
            ),
            lr=self.learning_rate,
        )
        sch = warmup_cosine_scheduler(opt, warmup_steps=self.warmup_steps, total_steps=self.total_steps)
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sch, "interval": "step", "frequency": 1},
        }
