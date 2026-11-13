import csv
import random
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from features import ALL_FEATURES, MINIMAL_FEATURES
from segment import segment_foreground, superpixel_round_debug

LAB_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = LAB_DIR / "dataset"
PREPROCESSED_DIR = LAB_DIR / "dataset_przetworzone"
MASKS_DIR = LAB_DIR / "masks"
FEATURES_DIR = LAB_DIR / "features"
PLOTS_DIR = LAB_DIR / "plots"
SPLIT = "1_trening"
SEED = 42


def save(fig, name: str, dpi: int = 110) -> Path:
    out = PLOTS_DIR / name
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return out


def plot_superpixel_examples(n_per_class: int = 3) -> Path:
    rng = random.Random(SEED)
    classes = sorted(p.name for p in (DATASET_DIR / SPLIT).iterdir() if p.is_dir())
    fig, axes = plt.subplots(len(classes), n_per_class * 3, figsize=(3 * n_per_class, 2.5 * len(classes)))
    if len(classes) == 1:
        axes = np.array([axes])

    for i, cls in enumerate(classes):
        paths = sorted((DATASET_DIR / SPLIT / cls).glob("*.jpeg"))
        sample = rng.sample(paths, min(n_per_class, len(paths)))
        for j, path in enumerate(sample):
            img = cv2.imread(str(path))
            _, sp_mask, sp_overlay = superpixel_round_debug(img)
            col = j * 3
            panels = [
                (cv2.cvtColor(img, cv2.COLOR_BGR2RGB), "oryginał"),
                (sp_overlay, "SLIC"),
                (sp_overlay, "okrągły SP"),
            ]
            # podświetl wybrany superpixelowy region na trzecim panelu
            highlight = sp_overlay.copy()
            ys, xs = np.where(sp_mask)
            if len(xs):
                y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
                cv2.rectangle(highlight, (x0, y0), (x1, y1), (255, 0, 0), 2)
            panels[2] = (highlight, "wybrany region")
            for ax, (data, title) in zip(axes[i, col : col + 3], panels):
                ax.imshow(data)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(title if i == 0 else "", fontsize=8)
            if j == 0:
                axes[i, 0].set_ylabel(cls, fontsize=9)

    fig.suptitle("Superpixele SLIC → wybór okrągłego segmentu → dalszy GrabCut", fontsize=12)
    fig.tight_layout()
    return save(fig, "1b_superpixel_okraglosc.png")


def plot_segmentation_examples(n_per_class: int = 4) -> Path:
    rng = random.Random(SEED)
    classes = sorted(p.name for p in (DATASET_DIR / SPLIT).iterdir() if p.is_dir())
    fig, axes = plt.subplots(len(classes), n_per_class * 3, figsize=(3 * n_per_class, 2.5 * len(classes)))
    if len(classes) == 1:
        axes = np.array([axes])

    for i, cls in enumerate(classes):
        paths = sorted((DATASET_DIR / SPLIT / cls).glob("*.jpeg"))
        sample = rng.sample(paths, min(n_per_class, len(paths)))
        for j, path in enumerate(sample):
            img = cv2.imread(str(path))
            mask, _ = segment_foreground(img)
            overlay = img.copy()
            overlay[~mask] = (overlay[~mask] * 0.25).astype(np.uint8)

            proc_path = PREPROCESSED_DIR / path.relative_to(DATASET_DIR)
            proc = np.array(Image.open(proc_path).convert("RGB")) if proc_path.is_file() else overlay

            col = j * 3
            for ax, data, title in zip(
                axes[i, col : col + 3],
                [
                    cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                    cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
                    proc,
                ],
                ["oryginał", "maska", "po preprocessingu"],
            ):
                ax.imshow(data)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(title if i == 0 else "", fontsize=8)
            if j == 0:
                axes[i, 0].set_ylabel(cls, fontsize=9)

    fig.suptitle("Segmentacja i preprocessing — przykłady ze zbioru treningowego", fontsize=12)
    fig.tight_layout()
    return save(fig, "1_segmentacja_przyklady.png")


def load_features(split: str) -> tuple[list[str], list[str], np.ndarray]:
    with (FEATURES_DIR / f"features_{split}.csv").open() as f:
        rows = list(csv.DictReader(f))
    files = [r["file"] for r in rows]
    labels = [r["label"] for r in rows]
    X = np.array([[float(r[k]) for k in ALL_FEATURES] for r in rows])
    return files, labels, X


def plot_color_features(labels: list[str], X: np.ndarray) -> Path:
    show = ["hue_mean", "sat_mean", "red_ratio", "green_ratio", "yellow_ratio"]
    idx = [ALL_FEATURES.index(n) for n in show]
    classes = sorted(set(labels))
    fig, axes = plt.subplots(1, len(show), figsize=(14, 4))
    for ax, feat_i, name in zip(axes, idx, show):
        data = [X[np.array(labels) == c, feat_i] for c in classes]
        ax.boxplot(data, tick_labels=classes)
        ax.set_title(name)
        ax.tick_params(axis="x", rotation=45)
    fig.suptitle("Rozkłady cech koloru (trening)")
    fig.tight_layout()
    return save(fig, "2_rozklady_cech_koloru.png")


def plot_glcm_features(labels: list[str], X: np.ndarray) -> Path:
    glcm_names = [n for n in ALL_FEATURES if n in ("contrast", "energy", "homogeneity", "entropy", "correlation")]
    idx = [ALL_FEATURES.index(n) for n in glcm_names]
    classes = sorted(set(labels))
    fig, axes = plt.subplots(1, len(glcm_names), figsize=(14, 4))
    for ax, feat_i, name in zip(axes, idx, glcm_names):
        data = [X[np.array(labels) == c, feat_i] for c in classes]
        ax.boxplot(data, tick_labels=classes)
        ax.set_title(name)
        ax.tick_params(axis="x", rotation=45)
    fig.suptitle("Rozkłady cech GLCM (trening)")
    fig.tight_layout()
    return save(fig, "3_rozklady_cech_glcm.png")


def plot_correlation(X: np.ndarray) -> Path:
    corr = np.corrcoef(X.T)
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(ALL_FEATURES)), ALL_FEATURES, rotation=90, fontsize=7)
    ax.set_yticks(range(len(ALL_FEATURES)), ALL_FEATURES, fontsize=7)
    ax.set_title("Korelacja między cechami (trening)")
    fig.colorbar(im, fraction=0.03)
    fig.tight_layout()
    return save(fig, "4_korelacja_cech.png")


def plot_minimal_scatter(labels: list[str], X_full: np.ndarray) -> Path:
    idx = [ALL_FEATURES.index(n) for n in MINIMAL_FEATURES[:2]]
    classes = sorted(set(labels))
    colors = {"apples": "#c0392b", "tomatoes": "#27ae60"}
    fig, ax = plt.subplots(figsize=(6, 5))
    labels_arr = np.array(labels)
    for cls in classes:
        m = labels_arr == cls
        ax.scatter(X_full[m, idx[0]], X_full[m, idx[1]], alpha=0.6, label=cls, c=colors.get(cls, "gray"))
    ax.set_xlabel(MINIMAL_FEATURES[0])
    ax.set_ylabel(MINIMAL_FEATURES[1])
    ax.set_title("Projekcja 2D (cechy minimalne)")
    ax.legend()
    fig.tight_layout()
    return save(fig, "5_scatter_minimalne.png")


def main() -> None:
    PLOTS_DIR.mkdir(exist_ok=True)
    if not FEATURES_DIR.is_dir():
        raise SystemExit("Brak features/ — uruchom main.py")

    plot_superpixel_examples()
    plot_segmentation_examples()
    files, labels, X = load_features(SPLIT)
    plot_color_features(labels, X)
    plot_glcm_features(labels, X)
    plot_correlation(X)
    plot_minimal_scatter(labels, X)
    print(f"Wykresy zapisane w {PLOTS_DIR.relative_to(LAB_DIR)}/")


if __name__ == "__main__":
    main()
