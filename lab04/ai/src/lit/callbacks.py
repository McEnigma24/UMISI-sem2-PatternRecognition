"""metrics.csv, best checkpoint po val/mIoU_fg, model_summary.txt, run_meta (tylko rank 0)."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import lightning.pytorch as pl
import torch

from classes import CLASS_NAMES


class SegRunLogCallback(pl.Callback):
    def __init__(self, save_dir: Path, architecture: str) -> None:
        self.save_dir = save_dir
        self.architecture = architecture
        self.metrics_path = save_dir / "metrics.csv"
        self.best_miou_fg = float("-inf")
        self._t0 = 0.0

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._t0 = time.monotonic()
        if trainer.is_global_zero:
            n_params = sum(p.numel() for p in pl_module.model.parameters())
            (self.save_dir / "model_summary.txt").write_text(
                f"architecture: {self.architecture}\nparams: {n_params:,}\n\n{pl_module.model}\n"
            )

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.sanity_checking:
            return
        # odczyt callback_metrics synchronizuje metryki między rankami (all-reduce) — MUSZĄ go zrobić wszystkie ranki
        m = trainer.callback_metrics
        if not trainer.is_global_zero:
            return
        row = {
            "epoch": trainer.current_epoch,
            "step": trainer.global_step,
            "elapsed_s": round(time.monotonic() - self._t0, 1),
            "train_loss": _f(m.get("train/loss_epoch", m.get("train/loss_step"))),
            "val_loss": _f(m.get("val/loss")),
            "val_mIoU": _f(m.get("val/mIoU")),
            "val_mIoU_fg": _f(m.get("val/mIoU_fg")),
            **{f"iou_{n}": _f(m.get(f"val_iou/{n}")) for n in CLASS_NAMES},
        }
        write_header = not self.metrics_path.is_file()
        with self.metrics_path.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if write_header:
                w.writeheader()
            w.writerow(row)

        miou_fg = row["val_mIoU_fg"]
        if miou_fg is not None and miou_fg > self.best_miou_fg:
            self.best_miou_fg = miou_fg
            torch.save(pl_module.model.state_dict(), self.save_dir / f"{self.architecture}_best_miou.pt")

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not trainer.is_global_zero:
            return
        torch.save(pl_module.model.state_dict(), self.save_dir / f"{self.architecture}_final.pt")
        meta_path = self.save_dir / "run_meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
        meta.update(
            {
                "train_total_seconds": round(time.monotonic() - self._t0, 1),
                "epochs_done": trainer.current_epoch + 1,
                "global_step": trainer.global_step,
                "best_val_mIoU_fg": self.best_miou_fg if self.best_miou_fg > float("-inf") else None,
            }
        )
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")


def _f(v) -> float | None:
    if v is None:
        return None
    return float(v.item()) if hasattr(v, "item") else float(v)
