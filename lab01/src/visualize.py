import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist, squareform
from skimage import measure
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_classif
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import (
    ALL_FEATURES,
    COMPLEX_FEATURES,
    SIMPLE_FEATURES,
    aligned_mask,
    complex_features,
    load_mask,
    min_area_rect,
    outer_contour,
    principal_axes,
    region,
)

LAB_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = LAB_DIR / "dataset"
FEATURES_DIR = LAB_DIR / "features"
PLOTS_DIR = LAB_DIR / "plots"
SPLIT = "1_trening"
VAL_SPLIT = "2_walidacja"

OVERLAY_CLASSES = [
    "apple", "Bone", "fish", "key", "spoon", "HCircle",
    "pocket", "horseshoe", "dog", "bat", "guitar", "hammer",
]


def load_features(split: str) -> tuple[list[str], list[str], np.ndarray]:
    with (FEATURES_DIR / f"features_{split}.csv").open() as f:
        rows = list(csv.DictReader(f))
    files = [r["file"] for r in rows]
    labels = [r["label"] for r in rows]
    X = np.array([[float(r[k]) for k in ALL_FEATURES] for r in rows])
    return files, labels, X


def columns(names: list[str]) -> list[int]:
    return [ALL_FEATURES.index(n) for n in names]


def overlay_examples(files: list[str], labels: list[str]):
    for ax_idx, cls in enumerate(OVERLAY_CLASSES):
        file = files[labels.index(cls)]
        yield ax_idx, file, load_mask(DATASET_DIR / SPLIT / cls / file)


def save(fig, name: str, dpi: int = 110) -> Path:
    out = PLOTS_DIR / name
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return out


def plot_simple_overlays(files: list[str], labels: list[str]) -> Path:
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.ravel()
    for i, file, mask in overlay_examples(files, labels):
        ax = axes[i]
        props = region(mask)

        ax.imshow(mask, cmap="gray_r", alpha=0.35)

        for contour in measure.find_contours(mask, 0.5):
            ax.plot(contour[:, 1], contour[:, 0], color="tab:blue", lw=1)

        min_row, min_col, max_row, max_col = props.bbox
        ax.add_patch(Rectangle((min_col - 0.5, min_row - 0.5), max_col - min_col, max_row - min_row,
                               fill=False, ec="tab:red", lw=1.5, ls="--"))

        hull_pts = props.coords[ConvexHull(props.coords).vertices]
        closed = np.vstack([hull_pts, hull_pts[:1]])
        ax.plot(closed[:, 1], closed[:, 0], color="tab:green", lw=1.2)

        dist = squareform(pdist(hull_pts))
        a, b = np.unravel_index(dist.argmax(), dist.shape)
        ax.plot(hull_pts[[a, b], 1], hull_pts[[a, b], 0], color="tab:purple", lw=2)

        cy, cx = props.centroid
        ax.add_patch(Circle((cx, cy), props.equivalent_diameter_area / 2, fill=False, ec="tab:orange", lw=1.5))

        ax.set_title(f"{file}\npole={props.area:.0f}  Feret={props.feret_diameter_max:.0f}", fontsize=9)
        ax.axis("off")

    handles = [
        plt.Line2D([], [], color="tab:blue", label="kontur (obwód)"),
        plt.Line2D([], [], color="tab:red", ls="--", label="prostokąt otaczający"),
        plt.Line2D([], [], color="tab:green", label="otoczka wypukła"),
        plt.Line2D([], [], color="tab:purple", lw=2, label="średnica Fereta"),
        plt.Line2D([], [], color="tab:orange", label="koło o tym samym polu"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=11)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return save(fig, "1_cechy_proste_na_obrazach.png")


def plot_complex_overlays(files: list[str], labels: list[str]) -> Path:
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.ravel()
    for i, file, mask in overlay_examples(files, labels):
        ax = axes[i]
        props = region(mask)
        f = complex_features(mask)
        cy, cx = props.centroid

        ax.imshow(mask, cmap="gray_r", alpha=0.35)

        box = cv2.boxPoints(min_area_rect(props)) - 0.5
        box = np.vstack([box, box[:1]])
        ax.plot(box[:, 0], box[:, 1], color="tab:red", lw=1.5, ls="--")

        major, minor = principal_axes(props)
        half = f["radius_max"]
        for axis, style in ((major, "-"), (minor, ":")):
            ax.plot([cx - axis[1] * half, cx + axis[1] * half], [cy - axis[0] * half, cy + axis[0] * half],
                    color="black", lw=1, ls=style)

        contour = outer_contour(mask)
        radii = np.linalg.norm(contour - props.centroid, axis=1)
        for idx, color in ((radii.argmax(), "tab:purple"), (radii.argmin(), "tab:cyan")):
            ax.plot([cx, contour[idx, 1]], [cy, contour[idx, 0]], color=color, lw=2.5)
        ax.add_patch(Circle((cx, cy), f["radius_mean"], fill=False, ec="tab:orange", lw=1.5))
        ax.plot(cx, cy, "k+", ms=10)

        ax.set_title(f"{file}\ncompact={f['compactness']:.2f}  solid={f['solidity']:.2f}  "
                     f"rect={f['rectangularity']:.2f}\nasym maj/min={f['asymmetry_major']:.2f}/"
                     f"{f['asymmetry_minor']:.2f}  r min/max={f['radius_min_max_ratio']:.2f}", fontsize=8.5)
        ax.axis("off")

    handles = [
        plt.Line2D([], [], color="tab:red", ls="--", label="min. obrócony prostokąt"),
        plt.Line2D([], [], color="black", label="oś główna"),
        plt.Line2D([], [], color="black", ls=":", label="oś poprzeczna"),
        plt.Line2D([], [], color="tab:purple", lw=2.5, label="najdłuższy promień"),
        plt.Line2D([], [], color="tab:cyan", lw=2.5, label="najkrótszy promień"),
        plt.Line2D([], [], color="tab:orange", label="średni promień"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=10)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return save(fig, "2_cechy_zlozone_na_obrazach.png")


def plot_symmetry(files: list[str], labels: list[str]) -> Path:
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.ravel()
    for i, file, mask in overlay_examples(files, labels):
        aligned = aligned_mask(mask, region(mask))
        mirrored = np.flipud(aligned)
        rgb = np.ones((*aligned.shape, 3))
        rgb[aligned & mirrored] = (0.55, 0.55, 0.55)
        rgb[aligned & ~mirrored] = (0.85, 0.15, 0.15)
        rgb[~aligned & mirrored] = (0.15, 0.4, 0.85)
        axes[i].imshow(rgb)
        axes[i].axhline((aligned.shape[0] - 1) / 2, color="black", lw=1)
        axes[i].set_title(f"{file}\nasymmetry_major = {1 - (aligned & mirrored).sum() / aligned.sum():.2f}",
                          fontsize=9)
        axes[i].axis("off")

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=(0.55, 0.55, 0.55), label="pokrywa się z odbiciem"),
        plt.Rectangle((0, 0), 1, 1, color=(0.85, 0.15, 0.15), label="tylko obiekt"),
        plt.Rectangle((0, 0), 1, 1, color=(0.15, 0.4, 0.85), label="tylko odbicie"),
        plt.Line2D([], [], color="black", label="oś główna (oś odbicia)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=11)
    fig.suptitle("Asymetria: obiekt obrócony osią główną poziomo i odbity względem niej", fontsize=14)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    return save(fig, "3_asymetria.png")


def plot_boxplots(labels: list[str], X: np.ndarray, f_scores: np.ndarray, names: list[str], title: str,
                  filename: str) -> Path:
    labels_arr = np.array(labels)
    classes = sorted(set(labels))
    idx = columns(names)
    ncols = 2
    nrows = int(np.ceil(len(idx) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(26, 7 * nrows))
    axes = axes.ravel()
    for ax, k in zip(axes, sorted(idx, key=lambda k: -f_scores[k])):
        groups = [X[labels_arr == c, k] for c in classes]
        order = np.argsort([np.median(g) for g in groups])
        ax.boxplot([groups[o] for o in order], tick_labels=[classes[o] for o in order], showfliers=False)
        ax.tick_params(axis="x", rotation=90, labelsize=7)
        ax.set_title(f"{ALL_FEATURES[k]}   (F = {f_scores[k]:.1f})", fontsize=13)
        ax.grid(axis="y", alpha=0.3)
    for ax in axes[len(idx):]:
        ax.axis("off")
    fig.suptitle(f"{title} — rozkład w klasach (zbiór treningowy, klasy posortowane wg mediany)", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    return save(fig, filename, dpi=80)


def plot_correlation(X: np.ndarray) -> Path:
    corr = np.corrcoef(X, rowvar=False)
    n = len(ALL_FEATURES)
    fig, ax = plt.subplots(figsize=(16, 14))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(n), ALL_FEATURES, rotation=60, ha="right")
    ax.set_yticks(range(n), ALL_FEATURES)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=7)
    border = len(SIMPLE_FEATURES) - 0.5
    ax.axhline(border, color="black", lw=2)
    ax.axvline(border, color="black", lw=2)
    fig.colorbar(im, fraction=0.04)
    ax.set_title("Korelacja między cechami (linia oddziela proste od złożonych)")
    fig.tight_layout()
    return save(fig, "6_korelacja_cech.png")


def plot_pca(labels: list[str], X: np.ndarray, names: list[str], title: str, filename: str) -> Path:
    pca = PCA(n_components=2)
    Z = pca.fit_transform(StandardScaler().fit_transform(X[:, columns(names)]))
    labels_arr = np.array(labels)
    classes = sorted(set(labels))
    cmap = plt.get_cmap("tab20")

    fig, ax = plt.subplots(figsize=(16, 12))
    for i, c in enumerate(classes):
        pts = Z[labels_arr == c]
        color = cmap(i % 20)
        ax.scatter(pts[:, 0], pts[:, 1], s=14, color=color, alpha=0.6)
        cx, cy = np.median(pts, axis=0)
        ax.text(cx, cy, c, fontsize=8, weight="bold", ha="center", va="center", color=color,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=color, alpha=0.8))
    var = pca.explained_variance_ratio_ * 100
    ax.set_xlabel(f"PC1 ({var[0]:.0f}% wariancji)")
    ax.set_ylabel(f"PC2 ({var[1]:.0f}% wariancji)")
    ax.set_title(f"PCA: {title} (standaryzowane) — etykieta w medianie klasy")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return save(fig, filename, dpi=100)


def quick_knn(labels: list[str], X: np.ndarray) -> None:
    _, val_labels, X_val = load_features(VAL_SPLIT)
    print("\nSzybki test kNN (k=1) na walidacji:")
    for title, names in (("proste", SIMPLE_FEATURES), ("złożone", COMPLEX_FEATURES), ("wszystkie", ALL_FEATURES)):
        idx = columns(names)
        model = make_pipeline(StandardScaler(), KNeighborsClassifier(1)).fit(X[:, idx], labels)
        print(f"  {title:10s} {model.score(X_val[:, idx], val_labels):.3f}")


def main() -> None:
    PLOTS_DIR.mkdir(exist_ok=True)
    files, labels, X = load_features(SPLIT)
    f_scores, _ = f_classif(X, labels)

    print("Siła rozróżniania klas (F ANOVA, więcej = lepiej):")
    for k in np.argsort(-f_scores):
        group = "prosta" if ALL_FEATURES[k] in SIMPLE_FEATURES else "złożona"
        print(f"  {ALL_FEATURES[k]:22s} {f_scores[k]:7.1f}  ({group})")

    outputs = [
        plot_simple_overlays(files, labels),
        plot_complex_overlays(files, labels),
        plot_symmetry(files, labels),
        plot_boxplots(labels, X, f_scores, SIMPLE_FEATURES, "Cechy proste", "4_rozklady_proste.png"),
        plot_boxplots(labels, X, f_scores, COMPLEX_FEATURES, "Cechy złożone", "5_rozklady_zlozone.png"),
        plot_correlation(X),
        plot_pca(labels, X, SIMPLE_FEATURES, "cechy proste", "7_pca_proste.png"),
        plot_pca(labels, X, COMPLEX_FEATURES, "cechy złożone", "8_pca_zlozone.png"),
    ]
    for out in outputs:
        print(f"-> {out.relative_to(LAB_DIR)}")

    quick_knn(labels, X)


if __name__ == "__main__":
    main()
