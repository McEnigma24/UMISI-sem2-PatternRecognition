"""Download KTH-TIPS-2 (Kaggle) and copy five texture classes into original/."""

import shutil
from pathlib import Path

import kagglehub

LAB_DIR = Path(__file__).resolve().parent.parent
ORIGINAL_DIR = LAB_DIR / "original"
KAGGLE_DATASET = "ag3ntsp1d3rx/kth-tips-2"
KTH_SUBDIR = "KTH_TIPS"

# Same five classes as in the Roboflow workflow example in the assignment brief.
CLASSES = (
    "styrofoam",
    "cotton",
    "aluminium_foil",
    "brown_bread",
    "corduroy",
)


def kth_tips_root(cache_path: Path) -> Path:
    root = cache_path / KTH_SUBDIR
    if not root.is_dir():
        raise FileNotFoundError(f"Expected {root} inside Kaggle archive")
    return root


def main() -> None:
    cache = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    src_root = kth_tips_root(cache)

    if ORIGINAL_DIR.exists():
        shutil.rmtree(ORIGINAL_DIR)
    ORIGINAL_DIR.mkdir(parents=True)

    for cls in CLASSES:
        src = src_root / cls
        if not src.is_dir():
            raise FileNotFoundError(f"Missing class folder: {src}")
        dst = ORIGINAL_DIR / cls
        shutil.copytree(src, dst)
        n = len(list(dst.glob("*.png")))
        print(f"  {cls}: {n} obrazów")

    print(f"\nZapisano klasy w {ORIGINAL_DIR.relative_to(LAB_DIR)}/")


if __name__ == "__main__":
    main()
