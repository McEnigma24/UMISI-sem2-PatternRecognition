import csv
from pathlib import Path

from joblib import Parallel, delayed

from glcm_features import ALL_FEATURES, glcm_features, load_grayscale

LAB_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = LAB_DIR / "dataset"
FEATURES_DIR = LAB_DIR / "features"
SPLITS = ["1_trening", "2_walidacja", "3_test"]


def process_image(path: Path) -> dict:
    gray = load_grayscale(path)
    label = path.parent.name
    return {"file": path.name, "label": label} | glcm_features(gray)


def save_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "label", *ALL_FEATURES])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    for split in SPLITS:
        paths = sorted(DATASET_DIR.joinpath(split).rglob("*.png"))
        if not paths:
            raise SystemExit(f"Brak obrazów w {DATASET_DIR / split} — uruchom prepare_dataset.py i split_dataset.sh")
        rows = Parallel(n_jobs=-1)(delayed(process_image)(p) for p in paths)
        out = FEATURES_DIR / f"features_{split}.csv"
        save_csv(rows, out)
        print(f"{split}: {len(rows)} obrazów -> {out.relative_to(LAB_DIR)}")


if __name__ == "__main__":
    main()
