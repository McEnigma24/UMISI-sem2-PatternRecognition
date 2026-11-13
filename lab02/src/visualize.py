import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from glcm_features import ALL_FEATURES, MINIMAL_FEATURES

LAB_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = LAB_DIR / "dataset"
FEATURES_DIR = LAB_DIR / "features"
PLOTS_DIR = LAB_DIR / "plots"
SPLIT = "1_trening"


def load_features(split: str) -> tuple[list[str], list[str], np.ndarray]:
    with (FEATURES_DIR / f"features_{split}.csv").open() as f:
        rows = list(csv.DictReader(f))
    files = [r["file"] for r in rows]
    labels = [r["label"] for r in rows]
    X = np.array([[float(r[k]) for k in ALL_FEATURES] for r in rows])
    return files, labels, X


def save(fig, name: str, dpi: int = 110) -> Path:
    out = PLOTS_DIR / name
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return out


def plot_texture_samples() -> Path:
    classes = sorted(p.name for p in (DATASET_DIR / SPLIT).iterdir() if p.is_dir())
    fig, axes = plt.subplots(len(classes), 5, figsize=(12, 2.2 * len(classes)))
    if len(classes) == 1:
        axes = np.array([axes])
    for i, cls in enumerate(classes):
        paths = sorted((DATASET_DIR / SPLIT / cls).glob("*.png"))[:5]
        for j, path in enumerate(paths):
            axes[i, j].imshow(Image.open(path).convert("L"), cmap="gray", vmin=0, vmax=255)
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            if j == 0:
                axes[i, j].set_ylabel(cls, fontsize=9)
    fig.suptitle("Przykładowe tekstury (grayscale) — zbiór treningowy", fontsize=13)
    fig.tight_layout()
    return save(fig, "1_przykladowe_tekstury.png")


def plot_feature_boxplots(files: list[str], labels: list[str], X: np.ndarray) -> Path:
    show = ["contrast", "energy", "homogeneity", "entropy", "correlation"]
    idx = [ALL_FEATURES.index(n) for n in show]
    classes = sorted(set(labels))
    fig, axes = plt.subplots(1, len(show), figsize=(14, 4), sharey=False)
    for ax, feat_i, name in zip(axes, idx, show):
        data = [X[np.array(labels) == c, feat_i] for c in classes]
        ax.boxplot(data, tick_labels=classes)
        ax.set_title(name)
        ax.tick_params(axis="x", rotation=45)
    fig.suptitle("Rozkłady cech GLCM w zbiorze treningowym")
    fig.tight_layout()
    return save(fig, "2_rozklady_cech_glcm.png")


def plot_correlation(X: np.ndarray) -> Path:
    corr = np.corrcoef(X.T)
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(ALL_FEATURES)), ALL_FEATURES, rotation=90, fontsize=8)
    ax.set_yticks(range(len(ALL_FEATURES)), ALL_FEATURES, fontsize=8)
    ax.set_title("Korelacja między cechami GLCM (trening)")
    fig.colorbar(im, fraction=0.03)
    fig.tight_layout()
    return save(fig, "3_korelacja_cech.png")


def plot_minimal_vs_energy_entropy(files: list[str], labels: list[str], X: np.ndarray) -> Path:
    ei, ti = ALL_FEATURES.index("energy"), ALL_FEATURES.index("entropy")
    classes = sorted(set(labels))
    colors = plt.cm.tab10(np.linspace(0, 1, len(classes)))
    fig, ax = plt.subplots(figsize=(8, 6))
    y = np.array(labels)
    for c, color in zip(classes, colors):
        m = y == c
        ax.scatter(X[m, ei], X[m, ti], s=18, alpha=0.6, label=c, color=color)
    ax.set_xlabel("energy")
    ax.set_ylabel("entropy")
    ax.set_title("Energia vs entropia (trening) — cechy z p. 7 PDF")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return save(fig, "4_energy_vs_entropy.png")


def main() -> None:
    PLOTS_DIR.mkdir(exist_ok=True)
    files, labels, X = load_features(SPLIT)
    print(f"Wizualizacje ({len(files)} obrazów treningowych, cechy minimalne: {MINIMAL_FEATURES})")
    for fn in (
        plot_texture_samples,
        lambda: plot_feature_boxplots(files, labels, X),
        lambda: plot_correlation(X),
        lambda: plot_minimal_vs_energy_entropy(files, labels, X),
    ):
        path = fn()
        print(f"  -> {path.relative_to(LAB_DIR)}")


if __name__ == "__main__":
    main()
