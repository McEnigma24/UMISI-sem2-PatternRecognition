"""Renders a PNG next to every CSV in features/ and results/."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LAB_DIR = Path(__file__).resolve().parent.parent
CSV_DIRS = [LAB_DIR / "features", LAB_DIR / "results"]
METRICS = ["accuracy", "precision", "recall", "f1"]


def read_csv(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def save(fig, path: Path, dpi: int = 110) -> Path:
    out = path.with_suffix(".png")
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return out


def annotated_heatmap(ax, values, rows, cols, cmap, fmt="{:.2f}", vmin=None, vmax=None, fontsize=9):
    im = ax.imshow(values, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(cols)), cols, rotation=45, ha="right")
    ax.set_yticks(range(len(rows)), rows)
    lo, hi = im.get_clim()
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            dark = (values[i, j] - lo) / (hi - lo or 1) > 0.6
            ax.text(j, i, fmt.format(values[i, j]), ha="center", va="center", fontsize=fontsize,
                    color="white" if dark else "black")
    return im


def plot_features(path: Path, rows: list[dict]) -> Path:
    names = [k for k in rows[0] if k not in ("file", "label")]
    labels = np.array([r["label"] for r in rows])
    X = np.array([[float(r[k]) for k in names] for r in rows])
    classes = sorted(set(labels))
    means = np.array([X[labels == c].mean(axis=0) for c in classes])
    z = (means - means.mean(axis=0)) / (means.std(axis=0) + 1e-12)

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(z, cmap="RdBu_r", vmin=-3, vmax=3, aspect="auto")
    ax.set_xticks(range(len(names)), names, rotation=60, ha="right", fontsize=9)
    ax.set_yticks(range(len(classes)), classes, fontsize=9)
    fig.colorbar(im, fraction=0.03, pad=0.02)
    split = path.stem.removeprefix("features_")
    ax.set_title(f"Odciski klas GLCM — {split} ({len(rows)} obrazów)")
    fig.tight_layout()
    return save(fig, path, dpi=90)


def plot_summary(path: Path, rows: list[dict]) -> Path:
    fig, (ax_table, ax_bar) = plt.subplots(2, 1, figsize=(14, 10), height_ratios=(1.2, 2))
    ax_table.axis("off")
    cells = [
        [
            r["cechy"],
            r["metoda"].replace("_", " "),
            f"{float(r['val_accuracy']):.3f}",
            *(f"{float(r[f'test_{m}']):.3f}" for m in METRICS),
        ]
        for r in rows
    ]
    table = ax_table.table(
        cells,
        colLabels=["cechy", "metoda", "walidacja", *(f"test {m}" for m in METRICS)],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.4)

    labels = [f"{r['cechy']}\n{r['metoda']}" for r in rows]
    x = np.arange(len(rows))
    width = 0.18
    for i, key in enumerate(["val_accuracy", *(f"test_{m}" for m in METRICS)]):
        bars = ax_bar.bar(x + (i - 2) * width, [float(r[key]) for r in rows], width, label=key)
        ax_bar.bar_label(bars, fmt="%.2f", fontsize=7)
    ax_bar.set_xticks(x, labels, fontsize=7)
    ax_bar.set_ylim(0, 1.05)
    ax_bar.legend(fontsize=8)
    ax_bar.grid(axis="y", alpha=0.3)
    fig.suptitle("Podsumowanie klasyfikacji tekstur (GLCM)")
    fig.tight_layout()
    return save(fig, path)


def plot_per_class(path: Path, rows: list[dict]) -> Path:
    methods = list(dict.fromkeys(r["metoda"] for r in rows))
    classes = sorted(set(r["klasa"] for r in rows))
    lookup = {(r["metoda"], r["klasa"]): r for r in rows}
    keys = ["precision", "recall", "f1"]
    values = np.array([[float(lookup[(m, c)][k]) for m in methods for k in keys] for c in classes])
    cols = [f"{m.replace('_', ' ')}\n{k}" for m in methods for k in keys]

    fig, ax = plt.subplots(figsize=(10, 5))
    annotated_heatmap(ax, values, classes, cols, "RdYlGn", "{:.2f}", 0, 1, fontsize=9)
    ax.set_title(path.stem)
    fig.tight_layout()
    return save(fig, path)


def plot_generic(path: Path, rows: list[dict]) -> Path:
    fig, ax = plt.subplots(figsize=(14, 0.4 * len(rows) + 1))
    ax.axis("off")
    table = ax.table([list(r.values()) for r in rows], colLabels=list(rows[0]), loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    ax.set_title(path.name)
    fig.tight_layout()
    return save(fig, path)


def plot_csv(path: Path) -> Path:
    rows = read_csv(path)
    if path.name.startswith("features_"):
        return plot_features(path, rows)
    if path.name == "podsumowanie.csv":
        return plot_summary(path, rows)
    if path.name.startswith("wyniki_per_klasa"):
        return plot_per_class(path, rows)
    return plot_generic(path, rows)


def main() -> None:
    for path in sorted(p for d in CSV_DIRS for p in d.rglob("*.csv")):
        print(f"{path.relative_to(LAB_DIR)} -> {plot_csv(path).relative_to(LAB_DIR)}")


if __name__ == "__main__":
    main()
