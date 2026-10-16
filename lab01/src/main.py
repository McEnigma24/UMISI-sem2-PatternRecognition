import csv
from pathlib import Path

from features import SIMPLE_FEATURES, label_from_filename, load_mask, simple_features

LAB_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = LAB_DIR / "dataset"
FEATURES_DIR = LAB_DIR / "features"
SPLITS = ["1_trening", "2_walidacja", "3_test"]


def compute_split(split: str) -> list[dict]:
    rows = []
    for path in sorted((DATASET_DIR / split).rglob("*.gif")):
        row = {"file": path.name, "label": label_from_filename(path)}
        row.update(simple_features(load_mask(path)))
        rows.append(row)
    return rows


def save_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "label", *SIMPLE_FEATURES])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    for split in SPLITS:
        rows = compute_split(split)
        out = FEATURES_DIR / f"simple_{split}.csv"
        save_csv(rows, out)
        print(f"{split}: {len(rows)} obrazów -> {out.relative_to(LAB_DIR)}")


if __name__ == "__main__":
    main()
