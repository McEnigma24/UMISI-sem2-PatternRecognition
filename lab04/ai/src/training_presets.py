"""Trening od zera (losowe wagi) -> saved_models/pretrained/from_scratch/<arch>/<run>.

Smoke vs FULL: w SCRATCH_ALL aktywne wpisy SMOKE albo FULL (drugie zakomentowane) —
uruchomienie zawsze przez ./normal_process_start.sh (main.py).
"""

from __future__ import annotations

from finetune_presets import _SEG_SMOKE

_SCRATCH_BASE = dict(
    max_train_images=0,
    max_val_images=1000,
    eval_val_images=0,
    eval_test_images=0,
    render_images=24,
    epochs=30,
    time_budget_min=600.0,
    batch_size=6,
    baseline_batch_size=12,
    baseline_learning_rate=6e-4,
    lr_scaling="sqrt",
    learning_rate_max=1.5e-3,
    lightning_devices=2,
    num_workers=5,
    precision="bf16-mixed",
    dice_weight=0.5,
    focal_weight=0.5,
    focal_gamma=2.0,
    weight_decay=0.01,
    warmup_ratio=0.06,
    val_checks_per_epoch=2,
    log_every_n_steps=25,
    backbone_lr_mult=1.0,  # od zera: enkoder i głowica z tym samym LR
    checkpoint_every=1,
    model_hparams={},
)

SCRATCH_SEGFORMER_B2 = {**_SCRATCH_BASE, "architecture": "segformer_b2", "run_tag": "scratch-b2", "batch_size": 4}
SCRATCH_MASK2FORMER = {
    **_SCRATCH_BASE,
    "architecture": "mask2former_swin_t",
    "run_tag": "scratch-m2f",
    "batch_size": 3,
    "model_hparams": {"aux_dice_focal_weight": 0.5},
}
SCRATCH_UNETPP = {**_SCRATCH_BASE, "architecture": "unetpp_convnext_t", "run_tag": "scratch-unetpp", "batch_size": 6}
SCRATCH_YOLO11M = {
    **_SCRATCH_BASE,
    "architecture": "yolo11m_seg",
    "run_tag": "scratch-yolo11m",
    "batch_size": 8,
    "baseline_batch_size": 16,
    "baseline_learning_rate": 1e-3,
    "learning_rate_max": 2e-3,
    "model_hparams": {"yolo_weights": "yolo11m-seg.pt", "imgsz": 1024},
}

_YOLO_SMOKE = {**_SEG_SMOKE, "model_hparams": {"yolo_weights": "yolo11m-seg.pt", "imgsz": 640}}

SCRATCH_ALL: list[dict] = [
    # --- SMOKE ---
    {**SCRATCH_SEGFORMER_B2, **_SEG_SMOKE},
    {**SCRATCH_MASK2FORMER, **_SEG_SMOKE},
    {**SCRATCH_UNETPP, **_SEG_SMOKE},
    {**SCRATCH_YOLO11M, **_YOLO_SMOKE},
    # --- FULL ---
    # {**SCRATCH_SEGFORMER_B2},
    # {**SCRATCH_MASK2FORMER},
    # {**SCRATCH_UNETPP},
    # {**SCRATCH_YOLO11M},
]
