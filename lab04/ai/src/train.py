"""Trening od zera -> saved_models/pretrained/from_scratch.

Wspólna pętla treningu (Lightning DDP lub YOLO); `finetune.py` woła `launch_job` z innym phase/source.

Subprocess: DDP ranki uruchamiają ten plik ponownie z job.json (transformers 5 nie przeżywa pickle).
Zawieszony rank: `kill -USR1 <pid>` → stack na stderr.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
from torch.utils.data import DataLoader

from _utils_.log_print import log_print
from _utils_.run_paths import resolve_save_dir, write_run_meta
from classes import NUM_CLASSES
from data.dataset import BddSegDataset
from lit.callbacks import SegRunLogCallback
from lit.losses import DiceFocalLoss
from lit.optim import scaled_learning_rate
from lit.seg_module import LitSegModule
from models.registry import get_architecture
from models.yolo11_seg import run_yolo_training

PHASE_SCRATCH = "pretrained/from_scratch"
SOURCE_SCRATCH = "scratch"


def _is_rank_zero() -> bool:
    return int(os.environ.get("LOCAL_RANK", "0")) == 0


def launch_job(*, phase: str, source: str, tensorboard_dir: str, preset: dict) -> Path:
    """Trening w osobnym procesie; zwraca katalog runu albo rzuca przy błędzie."""
    save_dir = resolve_save_dir(preset["architecture"], phase=phase, run_tag=preset["run_tag"])
    job = {
        "phase": phase,
        "source": source,
        "tensorboard_dir": tensorboard_dir,
        "save_dir": str(save_dir),
        **preset,
    }
    job_path = save_dir / "job.json"
    job_path.write_text(json.dumps(job, indent=2) + "\n")
    rc = subprocess.call([sys.executable, str(Path(__file__).resolve()), str(job_path)])
    if rc != 0:
        raise RuntimeError(f"Trening {preset['architecture']} zakończony kodem {rc} ({save_dir})")
    return save_dir


def main(tensorboard_dir: str, **preset) -> Path:
    return launch_job(
        phase=PHASE_SCRATCH,
        source=SOURCE_SCRATCH,
        tensorboard_dir=tensorboard_dir,
        preset=preset,
    )


def run_training(
    *,
    architecture: str,
    phase: str,
    source: str,
    run_tag: str,
    tensorboard_dir: str,
    save_dir: str,
    max_train_images: int,
    max_val_images: int,
    epochs: int,
    time_budget_min: float,
    batch_size: int,
    baseline_batch_size: int,
    baseline_learning_rate: float,
    lr_scaling: str,
    learning_rate_max: float,
    lightning_devices: int,
    num_workers: int,
    precision: str,
    dice_weight: float,
    focal_weight: float,
    focal_gamma: float,
    weight_decay: float,
    warmup_ratio: float,
    val_checks_per_epoch: int,
    log_every_n_steps: int,
    backbone_lr_mult: float,
    checkpoint_every: int,
    model_hparams: dict,
) -> Path:
    hparams = {k: v for k, v in locals().items() if k not in ("tensorboard_dir", "save_dir")}
    adapter = get_architecture(architecture)
    save_dir = Path(save_dir)
    os.environ["SEG_ARCHITECTURE"] = architecture
    os.environ["SEG_PHASE"] = phase

    use_gpu = torch.cuda.is_available()
    devices = min(lightning_devices, torch.cuda.device_count()) if use_gpu else 1
    lr = scaled_learning_rate(
        baseline_batch_size=baseline_batch_size,
        baseline_learning_rate=baseline_learning_rate,
        global_batch=batch_size * devices,
        lr_scaling=lr_scaling,
        learning_rate_max=learning_rate_max,
    )
    if _is_rank_zero():
        write_run_meta(
            save_dir,
            {
                **hparams,
                "devices": devices,
                "global_batch": batch_size * devices,
                "learning_rate": lr,
                "started_at": datetime.now().isoformat(),
            },
        )
        log_print(f"Start {architecture} ({phase}, {source}) devices={devices} lr={lr:.2e} -> {save_dir}")

    model = adapter.build(source=source, num_classes=NUM_CLASSES, **model_hparams)

    if adapter.owns_training_loop:
        run_yolo_training(
            model,
            save_dir=save_dir,
            max_train_images=max_train_images,
            max_val_images=max_val_images,
            epochs=epochs,
            batch=batch_size,
            imgsz=int(model_hparams["imgsz"]),
            devices=list(range(devices)) if use_gpu else ["cpu"],
            learning_rate=lr,
            weight_decay=weight_decay,
            warmup_ratio=warmup_ratio,
            num_workers=num_workers,
            precision=precision,
        )
        return save_dir

    train_ds = BddSegDataset("train", augment=True, max_images=max_train_images)
    val_ds = BddSegDataset("val", augment=False, max_images=max_val_images)
    loader_kw = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **loader_kw)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kw)

    steps_per_epoch = max(1, len(train_ds) // (batch_size * devices))
    total_steps = steps_per_epoch * epochs
    criterion = DiceFocalLoss(
        NUM_CLASSES,
        dice_weight=dice_weight,
        focal_weight=focal_weight,
        focal_gamma=focal_gamma,
    )
    lit = LitSegModule(
        model,
        adapter,
        criterion,
        learning_rate=lr,
        backbone_lr_mult=backbone_lr_mult,
        weight_decay=weight_decay,
        warmup_steps=int(total_steps * warmup_ratio),
        total_steps=total_steps,
    )

    ckpt_cb = ModelCheckpoint(
        dirpath=save_dir / "checkpoints",
        filename="epoch_{epoch:04d}",
        auto_insert_metric_name=False,
        every_n_epochs=checkpoint_every,
        save_top_k=-1,
        save_weights_only=True,
        save_on_train_epoch_end=True,
    )
    strategy = DDPStrategy(find_unused_parameters=adapter.ddp_find_unused) if devices > 1 else "auto"
    trainer = pl.Trainer(
        max_epochs=epochs,
        max_time=timedelta(minutes=time_budget_min),
        accelerator="gpu" if use_gpu else "cpu",
        devices=devices,
        strategy=strategy,
        precision=precision,
        logger=TensorBoardLogger(tensorboard_dir, name=f"{phase}/{architecture}", version=save_dir.name),
        log_every_n_steps=log_every_n_steps,
        val_check_interval=1.0 / val_checks_per_epoch,
        gradient_clip_val=1.0,
        callbacks=[ckpt_cb, SegRunLogCallback(save_dir, architecture), LearningRateMonitor("step")],
    )
    trainer.fit(lit, train_loader, val_loader)
    if trainer.is_global_zero and not (save_dir / f"{architecture}_final.pt").is_file():
        raise RuntimeError(f"Trening {architecture} nie zapisał wag final w {save_dir}")
    return save_dir


if __name__ == "__main__":
    import faulthandler
    import signal

    faulthandler.register(signal.SIGUSR1, all_threads=True)
    run_training(**json.loads(Path(sys.argv[1]).read_text()))