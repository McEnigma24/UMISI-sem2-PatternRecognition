"""Renders a PNG next to every CSV in features/ and results/ (same name, .png extension)."""

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


def annotated_heatmap(ax, values: np.ndarray, rows: list[str], cols: list[str], cmap: str, fmt: str = "{:.2f}",
                      vmin=None, vmax=None, fontsize: int = 9):
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
    """Class 'fingerprints': mean of every feature per class, standardized per feature."""
    names = [k for k in rows[0] if k not in ("file", "label")]
    labels = np.array([r["label"] for r in rows])
    X = np.array([[float(r[k]) for k in names] for r in rows])
    classes = sorted(set(labels))
    means = np.array([X[labels == c].mean(axis=0) for c in classes])
    z = (means - means.mean(axis=0)) / (means.std(axis=0) + 1e-12)

    fig, ax = plt.subplots(figsize=(14, 22))
    im = ax.imshow(z, cmap="RdBu_r", vmin=-3, vmax=3, aspect="auto")
    ax.set_xticks(range(len(names)), names, rotation=60, ha="right", fontsize=9)
    ax.set_yticks(range(len(classes)), classes, fontsize=8)
    ax.axvline(7.5, color="black", lw=1.5)
    fig.colorbar(im, fraction=0.03, pad=0.02, label="średnia klasy, odchylenia std. od średniej wszystkich klas")
    variant, split = path.parent.name, path.stem.removeprefix("features_")
    ax.set_title(f"Odciski klas: {variant} / {split} ({len(rows)} obrazów)\n"
                 f"czerwony = klasa ma cechę wyższą niż typowo, niebieski = niższą; "
                 f"linia oddziela cechy proste od złożonych", fontsize=11)
    fig.tight_layout()
    return save(fig, path, dpi=80)


def plot_summary(path: Path, rows: list[dict]) -> Path:
    methods = [r["metoda"] for r in rows]
    fig, (ax_table, ax_bar) = plt.subplots(2, 1, figsize=(12, 9), height_ratios=(1, 2.2))

    ax_table.axis("off")
    cells = [[r["metoda"].replace("_", " "), r["parametry"], f"{float(r['val_accuracy']):.3f}",
              *(f"{float(r[f'test_{m}']):.3f}" for m in METRICS)] for r in rows]
    table = ax_table.table(cells, colLabels=["metoda", "parametry", "walidacja", *(f"test {m}" for m in METRICS)],
                           loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.auto_set_column_width(range(7))
    table.scale(1, 1.6)

    keys = ["val_accuracy", *(f"test_{m}" for m in METRICS)]
    x = np.arange(len(methods))
    width = 0.16
    for i, key in enumerate(keys):
        bars = ax_bar.bar(x + (i - 2) * width, [float(r[key]) for r in rows], width,
                          label=key.replace("_", " "), hatch="//" if key.startswith("val") else None,
                          alpha=0.6 if key.startswith("val") else 1)
        ax_bar.bar_label(bars, fmt="%.2f", fontsize=7)
    ax_bar.set_xticks(x, [m.replace("_", " ") for m in methods])
    ax_bar.set_ylim(0, 1.05)
    ax_bar.legend(loc="lower right", fontsize=9)
    ax_bar.grid(axis="y", alpha=0.3)
    fig.suptitle("Podsumowanie klasyfikacji (cechy wybrane, bez preprocessingu)", fontsize=13)
    fig.tight_layout()
    return save(fig, path)


def plot_preprocessing(path: Path, rows: list[dict]) -> Path:
    variants = list(dict.fromkeys(r["preprocessing"] for r in rows))
    combos = list(dict.fromkeys((r["cechy"], r["metoda"]) for r in rows))
    lookup = {(r["preprocessing"], r["cechy"], r["metoda"]): r for r in rows}
    row_labels = [f"{s} / {m.replace('_', ' ')}" for s, m in combos]
    col_labels = [v.replace("_", " ") for v in variants]

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    for ax, key, title in ((axes[0], "val_accuracy", "walidacja"), (axes[1], "test_accuracy", "test")):
        values = np.array([[float(lookup[(v, s, m)][key]) for v in variants] for s, m in combos])
        annotated_heatmap(ax, values, row_labels, col_labels, "viridis", "{:.3f}", vmin=0.6, vmax=1)
        ax.set_title(f"accuracy — {title}")
    test = np.array([[float(lookup[(v, s, m)]["test_accuracy"]) for v in variants] for s, m in combos])
    delta = (test - test[:, :1]) * 100
    im = annotated_heatmap(axes[2], delta, row_labels, col_labels, "RdBu", "{:+.1f}", vmin=-6, vmax=6)
    axes[2].set_title("zmiana accuracy na teście względem oryginału [p.p.]")
    fig.colorbar(im, ax=axes[2], fraction=0.05)
    for ax in axes:
        for y in range(3, len(combos), 3):
            ax.axhline(y - 0.5, color="white", lw=3)
    fig.suptitle("Porównanie preprocessingu: zestaw cech / metoda × wariant", fontsize=14)
    fig.tight_layout()
    return save(fig, path)


def plot_per_class(path: Path, rows: list[dict]) -> Path:
    methods = list(dict.fromkeys(r["metoda"] for r in rows))
    classes = sorted(set(r["klasa"] for r in rows))
    lookup = {(r["metoda"], r["klasa"]): r for r in rows}
    keys = ["precision", "recall", "f1"]
    values = np.array([[float(lookup[(m, c)][k]) for m in methods for k in keys] for c in classes])
    order = np.argsort(values[:, 2::3].mean(axis=1))
    cols = [f"{m.replace('_', ' ')}\n{k}" for m in methods for k in keys]

    fig, ax = plt.subplots(figsize=(11, 24))
    annotated_heatmap(ax, values[order], [classes[i] for i in order], cols, "RdYlGn", "{:.2f}", 0, 1, fontsize=7)
    ax.tick_params(axis="x", labelsize=8)
    for x in range(3, len(cols), 3):
        ax.axvline(x - 0.5, color="black", lw=2)
    ax.set_title("Wyniki per klasa na teście (posortowane wg średniego F1, najsłabsze na górze)", fontsize=11)
    fig.tight_layout()
    return save(fig, path, dpi=90)


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
    handlers = {
        "podsumowanie.csv": plot_summary,
        "porownanie_preprocessingu.csv": plot_preprocessing,
        "wyniki_per_klasa.csv": plot_per_class,
    }
    return handlers.get(path.name, plot_generic)(path, rows)


def main() -> None:
    for path in sorted(p for d in CSV_DIRS for p in d.rglob("*.csv")):
        print(f"{path.relative_to(LAB_DIR)} -> {plot_csv(path).relative_to(LAB_DIR)}")


if __name__ == "__main__":
    main()
