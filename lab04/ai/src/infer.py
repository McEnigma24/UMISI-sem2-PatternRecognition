"""Ewaluacja (mIoU, IoU per klasa, P/R/F1, img/s), render predykcji i porównanie modeli."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from classes import CLASS_NAMES, NUM_CLASSES
from data.bdd_labels import image_path, label_path
from data.dataset import MASK_ROOT
from metrics.det_metrics import detection_prf1, instances_from_mask
from metrics.seg_metrics import confusion_matrix, miou_from_confusion
from models.base import SegArchitecture
from models.registry import get_architecture
from render.visualize import render_overlay, save_prediction_pair


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _pad_to_32(t: torch.Tensor) -> torch.Tensor:
    _, _, h, w = t.shape
    nh, nw = ((h + 31) // 32) * 32, ((w + 31) // 32) * 32
    if (nh, nw) == (h, w):
        return t
    return torch.nn.functional.pad(t, (0, nw - w, 0, nh - h))


def load_run_model(run_dir: Path) -> tuple[SegArchitecture, object]:
    meta = json.loads((run_dir / "run_meta.json").read_text())
    adapter = get_architecture(meta["architecture"])
    model = adapter.load_trained(run_dir, source=meta["source"], num_classes=NUM_CLASSES, **meta["model_hparams"])
    if not adapter.owns_training_loop:
        model = model.to(_device()).eval()
    return adapter, model


def predict_image(adapter: SegArchitecture, model, bgr: np.ndarray):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    device = _device()
    t = torch.from_numpy(rgb).to(device).permute(2, 0, 1).float().unsqueeze(0).div_(255.0)
    if adapter.owns_training_loop:
        return adapter.predict(model, t.cpu())[0]
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        pred = adapter.predict(model, _pad_to_32(t))[0]
    pred.semantic = pred.semantic[:h, :w]
    for inst in pred.instances:
        inst.mask = inst.mask[:h, :w]
    return pred


def split_stems(split: str, max_images: int) -> list[str]:
    stems = sorted(p.stem for p in (MASK_ROOT / split).glob("*.png"))
    return stems[:max_images] if max_images > 0 else stems


@torch.no_grad()
def evaluate_run(run_dir: Path, *, split: str, max_images: int, render_images: int, out_dir: Path) -> dict:
    adapter, model = load_run_model(run_dir)
    stems = split_stems(split, max_images)
    render_dir = out_dir / split
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    det = {"tp": 0.0, "fp": 0.0, "fn": 0.0}
    infer_s = 0.0
    for i, stem in enumerate(stems):
        bgr = cv2.imread(str(image_path(split, stem)))
        gt = cv2.imread(str(MASK_ROOT / split / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
        t0 = time.perf_counter()
        pred = predict_image(adapter, model, bgr)
        infer_s += time.perf_counter() - t0
        cm += confusion_matrix(pred.semantic, gt)
        prf = detection_prf1(pred.instances, instances_from_mask(gt))
        for k in det:
            det[k] += prf[k]
        if i < render_images:
            save_prediction_pair(bgr, pred.semantic, pred.instances, render_dir / stem)

    miou, miou_fg, per_cls = miou_from_confusion(cm)
    tp, fp, fn = det["tp"], det["fp"], det["fn"]
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    report = {
        "run_dir": str(run_dir),
        "architecture": adapter.name,
        "split": split,
        "n_images": len(stems),
        "mIoU": miou,
        "mIoU_fg": miou_fg,
        "per_class_iou": {CLASS_NAMES[i]: per_cls[i] for i in range(NUM_CLASSES)},
        "det_precision": prec,
        "det_recall": rec,
        "det_f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        "img_per_s": len(stems) / infer_s if infer_s else 0.0,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"metrics_{split}.json").write_text(json.dumps(report, indent=2) + "\n")
    np.save(out_dir / f"confusion_{split}.npy", cm)
    del model
    torch.cuda.empty_cache()
    return report


def _pick_examples(split: str, stems: list[str], n: int) -> list[str]:
    """Po jednym przykładzie: dzień, noc, deszcz, małe światła, wiele znaków — potem reszta."""
    picked: dict[str, str] = {}
    for stem in stems:
        attrs = json.loads(label_path(split, stem).read_text()).get("attributes", {})
        gt = cv2.imread(str(MASK_ROOT / split / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
        lamps = int(np.isin(gt, (3, 4, 5)).sum())
        n_signs = cv2.connectedComponents((gt == 1).astype(np.uint8))[0] - 1
        checks = {
            "day": attrs.get("timeofday") == "daytime",
            "night": attrs.get("timeofday") == "night",
            "rain": attrs.get("weather") == "rainy",
            "small_lights": 0 < lamps < 150,
            "many_signs": n_signs >= 4,
        }
        for key, ok in checks.items():
            if ok and key not in picked and stem not in picked.values():
                picked[key] = stem
    out = list(picked.values())
    out += [s for s in stems if s not in out]
    return out[:n]


def _label(img: np.ndarray, text: str) -> np.ndarray:
    img = img.copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(img, text, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return img


@torch.no_grad()
def compare_models(
    entries: list[tuple[str, Path]],
    reports: list[dict],
    *,
    split: str,
    n_examples: int,
    out_dir: Path,
) -> Path:
    """entries: (etykieta, run_dir). Siatka oryginał | GT | model_1 | ...; CSV + wykres IoU per klasa."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    stems = _pick_examples(split, split_stems(split, 2000), n_examples)
    images = {s: cv2.imread(str(image_path(split, s))) for s in stems}
    columns: list[tuple[str, dict[str, np.ndarray]]] = []
    gt_col = {}
    for s in stems:
        gt = cv2.imread(str(MASK_ROOT / split / f"{s}.png"), cv2.IMREAD_GRAYSCALE)
        gt_col[s] = render_overlay(images[s], gt, instances_from_mask(gt))
    columns.append(("GT", gt_col))
    for label, run_dir in entries:
        adapter, model = load_run_model(run_dir)
        col = {}
        for s in stems:
            pred = predict_image(adapter, model, images[s])
            col[s] = render_overlay(images[s], pred.semantic, pred.instances)
        columns.append((label, col))
        del model
        torch.cuda.empty_cache()

    rows = []
    for s in stems:
        tiles = [_label(images[s], s)] + [_label(col[s], name) for name, col in columns]
        tiles = [cv2.resize(t, (640, 360)) for t in tiles]
        rows.append(np.hstack(tiles))
    if rows:
        cv2.imwrite(str(out_dir / f"compare_grid_{split}.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])

    fields = ["label", "architecture", "split", "n_images", "mIoU", "mIoU_fg", "det_precision",
              "det_recall", "det_f1", "img_per_s", *[f"iou_{n}" for n in CLASS_NAMES]]
    with (out_dir / f"compare_{split}.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for (label, _), r in zip(entries, reports):
            w.writerow({"label": label, **{k: r[k] for k in fields[1:10]},
                        **{f"iou_{n}": r["per_class_iou"][n] for n in CLASS_NAMES}})

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(NUM_CLASSES - 1)
    width = 0.8 / max(len(reports), 1)
    for i, ((label, _), r) in enumerate(zip(entries, reports)):
        vals = [np.nan_to_num(r["per_class_iou"][CLASS_NAMES[c]]) for c in range(1, NUM_CLASSES)]
        ax.bar(x + i * width, vals, width=width, label=f"{label} (mIoU_fg={r['mIoU_fg']:.3f})")
    ax.set_xticks(x + width * (len(reports) - 1) / 2)
    ax.set_xticklabels(CLASS_NAMES[1:], rotation=20, ha="right")
    ax.set_ylabel("IoU")
    ax.set_title(f"IoU per klasa — {split}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f"compare_iou_{split}.png", dpi=120)
    plt.close(fig)
    return out_dir
