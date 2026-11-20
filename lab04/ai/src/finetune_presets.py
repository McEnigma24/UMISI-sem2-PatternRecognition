"""Fine-tuning (wagi pobrane) -> saved_models/finetuned/<arch>/<run>.

Smoke vs FULL: w FINETUNE_ALL aktywne wpisy SMOKE albo FULL (drugie zakomentowane) —
uruchomienie zawsze przez ./normal_process_start.sh (main.py).
"""

from __future__ import annotations

_SEG_BASE = dict(

    max_train_images=0,  # 0 = cały split (70k)
    max_val_images=1000,  # walidacja w trakcie treningu
    eval_val_images=0,  # końcowa ewaluacja main (0 = cały val 10k)
    eval_test_images=0,  # końcowa ewaluacja main (0 = cały test 20k)
    render_images=24,  # ile predykcji zapisać jako PNG na split

    epochs=25,
    time_budget_min=75.0,
    batch_size=6,

    baseline_batch_size=12,
    baseline_learning_rate=3e-4,
    lr_scaling="sqrt",
    learning_rate_max=8e-4,


    lightning_devices=2,
    num_workers=2,
    precision="bf16-mixed",
    dice_weight=0.5,
    focal_weight=0.5,
    focal_gamma=2.0,
    weight_decay=0.01,
    warmup_ratio=0.05,
    val_checks_per_epoch=2,
    log_every_n_steps=25,
    backbone_lr_mult=0.2,
    checkpoint_every=1,
    model_hparams={},
)

# SMOKE / dry run: ten sam pipeline (2 GPU, DDP), okrojony zbiór — całość wszystkich modeli <= 15 min
_SEG_SMOKE = dict(
    max_train_images=16,
    max_val_images=8,
    eval_val_images=8,
    eval_test_images=8,
    render_images=4,
    epochs=1,
    time_budget_min=4.0,
    batch_size=2,
    num_workers=2,
    val_checks_per_epoch=1,
    log_every_n_steps=1,
)

FT_SEGFORMER_B2 = {
    **_SEG_BASE,
    "architecture": "segformer_b2",
    "run_tag": "ft-b2",
    "batch_size": 4
}

FT_MASK2FORMER = {
    **_SEG_BASE,
    "architecture": "mask2former_swin_t",
    "run_tag": "ft-m2f",
    "batch_size": 3,
    "model_hparams": {"aux_dice_focal_weight": 0.5},
}

FT_UNETPP = {
    **_SEG_BASE,
    "architecture": "unetpp_convnext_t",
    "run_tag": "ft-unetpp",
    "batch_size": 6
}

FT_YOLO11M = {
    **_SEG_BASE,
    "architecture": "yolo11m_seg",
    "run_tag": "ft-yolo11m",
    "batch_size": 8,
    "baseline_batch_size": 16,
    "baseline_learning_rate": 1e-3,
    "learning_rate_max": 2e-3,
    "model_hparams": {"yolo_weights": "yolo11m-seg.pt", "imgsz": 1024},
}
# Alternatywy (mniejsze): SegFormer-B1 "nvidia/segformer-b1-finetuned-cityscapes-1024-1024", YOLO11s-seg "yolo11s-seg.pt"

_YOLO_SMOKE = {**_SEG_SMOKE, "model_hparams": {"yolo_weights": "yolo11m-seg.pt", "imgsz": 640}}

FINETUNE_ALL: list[dict] = [

    # --- SMOKE --- #
    # {**FT_SEGFORMER_B2, **_SEG_SMOKE},
    # {**FT_MASK2FORMER, **_SEG_SMOKE},
    # {**FT_UNETPP, **_SEG_SMOKE},
    # {**FT_YOLO11M, **_YOLO_SMOKE},

    # --- FULL --- #
    {**FT_SEGFORMER_B2},
    {**FT_MASK2FORMER},
    {**FT_UNETPP},
    {**FT_YOLO11M},
]
