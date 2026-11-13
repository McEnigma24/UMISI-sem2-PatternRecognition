import csv
from pathlib import Path

import cv2
import numpy as np
from joblib import Parallel, delayed
from PIL import Image

from features import ALL_FEATURES, extract_features
from preprocess import preprocess_image
from segment import foreground_ratio, segment_foreground

LAB_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = LAB_DIR / "dataset"
PREPROCESSED_DIR = LAB_DIR / "dataset_przetworzone"
MASKS_DIR = LAB_DIR / "masks"
FEATURES_DIR = LAB_DIR / "features"
SPLITS = ["1_trening", "2_walidacja", "3_test"]


def process_image(path: Path) -> dict:
    img = cv2.imread(str(path))
    if img is None:
        raise OSError(f"Nie można wczytać: {path}")

    mask_full, _ = segment_foreground(img)
    proc_img, proc_mask = preprocess_image(img, mask_full)

    rel = path.relative_to(DATASET_DIR)
    out_img = PREPROCESSED_DIR / rel
    out_mask = MASKS_DIR / rel.with_suffix(".png")
    out_img.parent.mkdir(parents=True, exist_ok=True)
    out_mask.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(cv2.cvtColor(proc_img, cv2.COLOR_BGR2RGB)).save(out_img)
    Image.fromarray(proc_mask.astype(np.uint8) * 255).save(out_mask)

    feats = extract_features(proc_img, proc_mask)
    feats["seg_fg_ratio"] = foreground_ratio(proc_mask)
    return {"file": path.name, "label": path.parent.name} | feats


def save_csv(rows: list[dict], path: Path, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if not DATASET_DIR.is_dir():
        raise SystemExit("Brak dataset/ — uruchom ./split_dataset.sh")

    extra = ["seg_fg_ratio"]
    fieldnames = ["file", "label", *ALL_FEATURES, *extra]

    for split in SPLITS:
        paths = sorted(DATASET_DIR.joinpath(split).rglob("*.jpeg"))
        if not paths:
            paths = sorted(DATASET_DIR.joinpath(split).rglob("*.jpg"))
        if not paths:
            raise SystemExit(f"Brak obrazów w {DATASET_DIR / split}")

        rows = Parallel(n_jobs=-1)(delayed(process_image)(p) for p in paths)
        out = FEATURES_DIR / f"features_{split}.csv"
        save_csv(rows, out, fieldnames)
        ratios = [r["seg_fg_ratio"] for r in rows]
        print(
            f"{split}: {len(rows)} obrazów -> {out.relative_to(LAB_DIR)}  "
            f"(śr. pole FG={sum(ratios)/len(ratios):.2f})"
        )


if __name__ == "__main__":
    main()
