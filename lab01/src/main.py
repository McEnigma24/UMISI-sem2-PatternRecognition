import csv
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from PIL import Image

from features import ALL_FEATURES, all_features, label_from_filename, load_mask
from preprocess import VARIANTS

LAB_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = LAB_DIR / "dataset"
PREPROCESSED_DIR = LAB_DIR / "dataset_przetworzone"
FEATURES_DIR = LAB_DIR / "features"
SPLITS = ["1_trening", "2_walidacja", "3_test"]


def process_image(path: Path, variant: str) -> dict:
    mask = VARIANTS[variant](load_mask(path))
    if variant != "oryginal":
        out = PREPROCESSED_DIR / variant / path.relative_to(DATASET_DIR).with_suffix(".png")
        out.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(mask.astype(np.uint8) * 255).save(out)
    return {"file": path.name, "label": label_from_filename(path)} | all_features(mask)


def save_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "label", *ALL_FEATURES])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    for variant in VARIANTS:
        for split in SPLITS:
            paths = sorted((DATASET_DIR / split).rglob("*.gif"))
            rows = Parallel(n_jobs=-1)(delayed(process_image)(p, variant) for p in paths)
            out = FEATURES_DIR / variant / f"features_{split}.csv"
            save_csv(rows, out)
            print(f"{variant:28s} {split}: {len(rows)} obrazów -> {out.relative_to(LAB_DIR)}")


if __name__ == "__main__":
    main()
