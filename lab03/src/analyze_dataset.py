"""Statystyki zbioru apples/tomatoes i jakości segmentacji."""

import csv
from collections import Counter
from pathlib import Path

LAB_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = LAB_DIR / "data" / "apples_tomatoes"
DATASET_DIR = LAB_DIR / "dataset"
FEATURES_DIR = LAB_DIR / "features"
RESULTS_DIR = LAB_DIR / "results"


def source_prefix(filename: str) -> str:
    parts = filename.split("_")
    return parts[1] if len(parts) > 1 and parts[0] == "img" else "unknown"


def count_tree(root: Path) -> dict[str, Counter]:
    out: dict[str, Counter] = {}
    for split in ("train", "test"):
        split_dir = root / split
        if not split_dir.is_dir():
            continue
        for cls_dir in sorted(split_dir.iterdir()):
            if cls_dir.is_dir():
                out.setdefault(split, Counter())[cls_dir.name] = len(
                    list(cls_dir.glob("*.jpeg")) + list(cls_dir.glob("*.jpg"))
                )
    return out


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    print("=== Oryginalny zbiór (data/apples_tomatoes) ===")
    for split, counts in count_tree(DATA_DIR).items():
        print(f"  {split}: {dict(counts)}  (razem {sum(counts.values())})")

    if DATASET_DIR.is_dir():
        print("\n=== Podział roboczy (dataset/) ===")
        for sub in ("1_trening", "2_walidacja", "3_test"):
            n = len(list((DATASET_DIR / sub).rglob("*.jpeg")))
            print(f"  {sub}: {n} obrazów")

    prefixes = Counter()
    for split in ("train", "test"):
        for path in (DATA_DIR / split).rglob("*.jpeg"):
            prefixes[source_prefix(path.name)] += 1
    print("\n=== Źródła obrazów (prefiks img_p*) ===")
    for pref, n in sorted(prefixes.items()):
        print(f"  {pref}: {n}")

    seg_path = FEATURES_DIR / "features_1_trening.csv"
    if seg_path.is_file():
        with seg_path.open() as f:
            rows = list(csv.DictReader(f))
        ratios = [float(r["seg_fg_ratio"]) for r in rows]
        suspicious = sum(1 for x in ratios if x < 0.05 or x > 0.82)
        print("\n=== Segmentacja (trening) ===")
        print(f"  średnie pole FG: {sum(ratios)/len(ratios):.3f}")
        print(f"  podejrzane maski (FG<5% lub >82%): {suspicious}/{len(ratios)}")

        by_src: dict[str, list[float]] = {}
        for r in rows:
            by_src.setdefault(source_prefix(r["file"]), []).append(float(r["seg_fg_ratio"]))
        print("  średnie FG wg źródła:")
        for pref in sorted(by_src):
            vals = by_src[pref]
            print(f"    {pref}: {sum(vals)/len(vals):.3f}  (n={len(vals)})")

    out_csv = RESULTS_DIR / "analiza_zbioru.txt"
    lines = [
        "Lab03 — jabłka vs pomidory",
        "Mały, bardzo zróżnicowany zbiór (zdjęcia, ilustracje, stock).",
        "Prefiksy img_p1/p2/p3 ≈ różne źródła — segmentacja hybrydowa (progowanie + GrabCut).",
    ]
    out_csv.write_text("\n".join(lines) + "\n")
    print(f"\n-> {out_csv.relative_to(LAB_DIR)}")


if __name__ == "__main__":
    main()
