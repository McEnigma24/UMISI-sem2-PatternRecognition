"""YOLO11-seg — własna pętla ultralytics, zapis do tej samej struktury saved_models."""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

from _utils_.run_paths import project_root
from classes import NUM_CLASSES
from data.bdd_labels import image_path
from models.base import SegArchitecture, SegInstance, SegPrediction

YOLO_ROOT = project_root() / "data_derived" / "yolo"
DOWNLOADED = project_root() / "saved_models" / "pretrained" / "downloaded" / "ultralytics"
YOLO_NAMES = ("znak", "obudowa", "red", "yellow", "green")
# YOLO id -> klasa semantyczna; obudowa malowana pierwsza, diody/znaki na wierzchu
YOLO_TO_SEM = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5}


def _link_images(split: str, stems: list[str]) -> list[str]:
    img_dir = YOLO_ROOT / split / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for st in stems:
        src = image_path(split, st)
        dst = img_dir / f"{st}.jpg"
        if dst.is_symlink() and os.readlink(dst) != str(src):
            dst.unlink()
        if not dst.exists():
            dst.symlink_to(src)
        paths.append(str(dst))
    return paths


def build_yolo_dataset(out_dir: Path, *, max_train_images: int, max_val_images: int) -> Path:
    """data.yaml + listy obrazów (tylko stemy z maskami, te same co BddSegDataset)."""
    from data.dataset import MASK_ROOT

    out_dir.mkdir(parents=True, exist_ok=True)
    for split, limit in (("train", max_train_images), ("val", max_val_images)):
        stems = sorted(p.stem for p in (MASK_ROOT / split).glob("*.png"))
        if limit > 0:
            stems = stems[:limit]
        missing = [s for s in stems if not (YOLO_ROOT / split / "labels" / f"{s}.txt").is_file()]
        if missing:
            raise FileNotFoundError(f"Brak etykiet YOLO dla {len(missing)} obrazów {split} — uruchom prepare")
        (out_dir / f"{split}.txt").write_text("\n".join(_link_images(split, stems)) + "\n")
    names = "".join(f"  {i}: {n}\n" for i, n in enumerate(YOLO_NAMES))
    data_yaml = out_dir / "data.yaml"
    data_yaml.write_text(f"path: {out_dir}\ntrain: train.txt\nval: val.txt\nnames:\n{names}")
    return data_yaml


class Yolo11SegAdapter(SegArchitecture):
    name = "yolo11m_seg"
    owns_training_loop = True

    def build(self, *, source: str, num_classes: int = NUM_CLASSES, **hp) -> nn.Module:
        from ultralytics import YOLO

        weights: str = hp["yolo_weights"]
        self.imgsz = int(hp["imgsz"])
        if source == "scratch":
            return YOLO(weights.removesuffix(".pt") + ".yaml")
        DOWNLOADED.mkdir(parents=True, exist_ok=True)
        return YOLO(str(DOWNLOADED / weights))

    def load_trained(self, run_dir: Path, **hp):
        from ultralytics import YOLO

        self.imgsz = int(hp["imgsz"])
        weights = run_dir / "ultralytics" / "weights"
        for name in ("best.pt", "last.pt"):
            if (weights / name).is_file():
                return YOLO(str(weights / name))
        raise FileNotFoundError(f"Brak wag YOLO w {weights}")

    def forward_logits(self, model, images: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("YOLO uses ultralytics loop")

    @torch.no_grad()
    def predict(self, model, images: torch.Tensor) -> list[SegPrediction]:
        outs: list[SegPrediction] = []
        for i in range(images.shape[0]):
            rgb = (images[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            res = model.predict(
                cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                imgsz=self.imgsz,
                retina_masks=True,
                verbose=False,
            )[0]
            sem = np.zeros(rgb.shape[:2], dtype=np.uint8)
            inst: list[SegInstance] = []
            if res.masks is not None:
                dets = []
                for j in range(len(res.boxes)):
                    cid = YOLO_TO_SEM.get(int(res.boxes.cls[j].item()), 0)
                    if cid:
                        dets.append((cid, j))
                for cid, j in sorted(dets, key=lambda d: d[0] != 2):
                    m = res.masks.data[j].cpu().numpy().astype(bool)
                    sem[m] = cid
                    x1, y1, x2, y2 = res.boxes.xyxy[j].cpu().numpy().tolist()
                    inst.append(
                        SegInstance(
                            cls_id=cid,
                            score=float(res.boxes.conf[j].item()),
                            mask=m,
                            x1=x1,
                            y1=y1,
                            x2=x2,
                            y2=y2,
                        )
                    )
            outs.append(SegPrediction(semantic=sem, instances=inst))
        return outs


def run_yolo_training(
    model,
    *,
    save_dir: Path,
    max_train_images: int,
    max_val_images: int,
    epochs: int,
    batch: int,
    imgsz: int,
    devices: list[int],
    learning_rate: float,
    weight_decay: float,
    warmup_ratio: float,
    num_workers: int,
    precision: str,
) -> None:
    data_yaml = build_yolo_dataset(
        save_dir / "yolo_data",
        max_train_images=max_train_images,
        max_val_images=max_val_images,
    )
    # check_amp ultralytics pobiera yolo11n.pt do cwd — trzymamy to w katalogu pobranych wag
    prev_cwd = Path.cwd()
    DOWNLOADED.mkdir(parents=True, exist_ok=True)
    os.chdir(DOWNLOADED)
    try:
        _train(model, data_yaml, save_dir, epochs, batch, imgsz, devices, learning_rate,
               weight_decay, warmup_ratio, num_workers, precision)
    finally:
        os.chdir(prev_cwd)


def _train(model, data_yaml, save_dir, epochs, batch, imgsz, devices, learning_rate,
           weight_decay, warmup_ratio, num_workers, precision) -> None:
    # ultralytics `time=` wydłuża trening do budżetu zamiast go ucinać — limitem są epoki
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch * len(devices),
        device=devices,
        workers=num_workers,
        optimizer="AdamW",
        lr0=learning_rate,
        weight_decay=weight_decay,
        warmup_epochs=max(warmup_ratio * epochs, 0.0),
        amp=precision != "32",
        project=str(save_dir),
        name="ultralytics",
        exist_ok=True,
        save_period=1,
        plots=True,
        verbose=False,
    )
