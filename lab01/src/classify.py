import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from features import ALL_FEATURES, SIMPLE_FEATURES
from preprocess import VARIANTS

LAB_DIR = Path(__file__).resolve().parent.parent
FEATURES_DIR = LAB_DIR / "features"
PLOTS_DIR = LAB_DIR / "plots"
RESULTS_DIR = LAB_DIR / "results"

# one feature from each strongly correlated group (see plots/6_korelacja_cech.png)
SELECTED_FEATURES = [
    "eccentricity",
    "aspect_ratio",
    "rectangularity",
    "solidity",
    "radius_cv",
    "radius_mean_max_ratio",
    "radius_min_max_ratio",
    "asymmetry_major",
    "asymmetry_minor",
    "compactness",
    "equivalent_diameter",
]
# heavy right tail (up to ~35 for shapes with many holes), log brings it to a usable range
LOG_FEATURES = ["compactness"]

FEATURE_SETS = {
    "proste": SIMPLE_FEATURES,
    "wybrane": SELECTED_FEATURES,
    "wszystkie": ALL_FEATURES,
}

METHODS = {
    "regresja_logistyczna": (
        LogisticRegression(max_iter=20000),
        {"C": [0.1, 1, 10, 100, 1000, 10000]},
    ),
    "kNN": (
        KNeighborsClassifier(),
        {"n_neighbors": [1, 3, 5, 7, 9, 15], "weights": ["uniform", "distance"]},
    ),
    "drzewo_decyzyjne": (
        DecisionTreeClassifier(random_state=0),
        {"max_depth": [5, 10, 15, 20, None], "criterion": ["gini", "entropy"]},
    ),
}


def load_split(split: str, names: list[str], variant: str = "oryginal") -> tuple[np.ndarray, np.ndarray]:
    with (FEATURES_DIR / variant / f"features_{split}.csv").open() as f:
        rows = list(csv.DictReader(f))
    X = np.array([[float(r[k]) for k in names] for r in rows])
    y = np.array([r["label"] for r in rows])
    return X, y


def load_all(names: list[str], variant: str = "oryginal"):
    return [load_split(s, names, variant) for s in ("1_trening", "2_walidacja", "3_test")]


def log_columns(X: np.ndarray, idx: list[int]) -> np.ndarray:
    X = X.copy()
    X[:, idx] = np.log(X[:, idx])
    return X


def build(model, names: list[str], scaler=None):
    idx = [names.index(n) for n in LOG_FEATURES if n in names]
    log = FunctionTransformer(log_columns, kw_args={"idx": idx})
    if scaler is False:
        return make_pipeline(log, clone(model))
    return make_pipeline(log, scaler or MinMaxScaler(), clone(model))


def param_grid(grid: dict) -> list[dict]:
    combos = [{}]
    for key, values in grid.items():
        combos = [c | {key: v} for c in combos for v in values]
    return combos


def tune(model, grid: dict, names: list[str], train, val) -> tuple[dict, float, list]:
    history = []
    for params in param_grid(grid):
        pipe = build(clone(model).set_params(**params), names).fit(*train)
        history.append((params, accuracy_score(val[1], pipe.predict(val[0]))))
    best_params, best_acc = max(history, key=lambda h: h[1])
    return best_params, best_acc, history


def metrics(y_true, y_pred) -> dict[str, float]:
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {"accuracy": accuracy_score(y_true, y_pred), "precision": p, "recall": r, "f1": f}


def evaluate(method: str, names: list[str], variant: str) -> dict:
    model, grid = METHODS[method]
    train, val, test = load_all(names, variant)
    best_params, val_acc, history = tune(model, grid, names, train, val)
    pipe = build(clone(model).set_params(**best_params), names).fit(*train)
    y_pred = pipe.predict(test[0])
    return {"params": best_params, "val_accuracy": val_acc, "history": history,
            "y_test": test[1], "y_pred": y_pred, **metrics(test[1], y_pred)}


def save(fig, name: str, dpi: int = 110) -> Path:
    out = PLOTS_DIR / name
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return out


def plot_confusion(y_true, y_pred, title: str, filename: str) -> Path:
    classes = sorted(set(y_true))
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    fig, ax = plt.subplots(figsize=(18, 16))
    im = ax.imshow(cm, cmap="Blues", vmin=0)
    ax.set_xticks(range(len(classes)), classes, rotation=90, fontsize=7)
    ax.set_yticks(range(len(classes)), classes, fontsize=7)
    for i, j in zip(*np.nonzero(cm)):
        ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=6,
                color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("przewidziana klasa")
    ax.set_ylabel("prawdziwa klasa")
    ax.set_title(title, fontsize=14)
    fig.colorbar(im, fraction=0.03)
    fig.tight_layout()
    return save(fig, filename, dpi=90)


def plot_comparison(summary: list[dict]) -> Path:
    names = [s["metoda"] for s in summary]
    keys = ["accuracy", "precision", "recall", "f1"]
    x = np.arange(len(names))
    width = 0.2
    fig, ax = plt.subplots(figsize=(11, 6))
    for i, key in enumerate(keys):
        values = [s[f"test_{key}"] for s in summary]
        bars = ax.bar(x + (i - 1.5) * width, values, width, label=key)
        ax.bar_label(bars, fmt="%.2f", fontsize=8)
    ax.set_xticks(x, [n.replace("_", " ") for n in names])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("wartość (makro-średnia po klasach)")
    ax.set_title("Porównanie metod na zbiorze testowym (cechy wybrane, bez preprocessingu)")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return save(fig, "9_porownanie_metod.png")


def plot_preprocessing_comparison(rows: list[dict]) -> Path:
    variants = list(VARIANTS)
    fig, axes = plt.subplots(1, len(FEATURE_SETS), figsize=(18, 6), sharey=True)
    width = 0.27
    x = np.arange(len(METHODS))
    for ax, set_name in zip(axes, FEATURE_SETS):
        for i, variant in enumerate(variants):
            values = [next(r["test_accuracy"] for r in rows if r["cechy"] == set_name and r["metoda"] == m
                           and r["preprocessing"] == variant) for m in METHODS]
            bars = ax.bar(x + (i - 1) * width, values, width, label=variant.replace("_", " "))
            ax.bar_label(bars, fmt="%.2f", fontsize=8)
        ax.set_xticks(x, [m.replace("_", "\n") for m in METHODS])
        ax.set_title(f"cechy: {set_name} ({len(FEATURE_SETS[set_name])})")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("accuracy na teście")
    axes[0].legend(loc="lower left")
    fig.suptitle("Wpływ preprocessingu na skuteczność", fontsize=14)
    fig.tight_layout()
    return save(fig, "14_wplyw_preprocessingu.png")


def most_confused(y_true, y_pred, n: int = 8) -> list[tuple[str, str, int]]:
    pairs: dict[tuple[str, str], int] = {}
    for t, p in zip(y_true, y_pred):
        if t != p:
            pairs[(t, p)] = pairs.get((t, p), 0) + 1
    return sorted(((t, p, c) for (t, p), c in pairs.items()), key=lambda x: -x[2])[:n]


def write_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def normalization_check() -> None:
    train, val, _ = load_all(SELECTED_FEATURES)
    print("\nWpływ normalizacji cech (kNN, k=1, walidacja):")
    knn = KNeighborsClassifier(1)
    for name, scaler in (("bez normalizacji", False), ("MinMax [0, 1]", MinMaxScaler()),
                         ("standaryzacja", StandardScaler())):
        pipe = build(knn, SELECTED_FEATURES, scaler).fit(*train)
        print(f"  {name:18s} {pipe.score(*val):.3f}")


def classification_report_original() -> None:
    """Points 5-7: selected features, original images, all three methods in detail."""
    summary = []
    per_class = []
    for i, method in enumerate(METHODS):
        r = evaluate(method, SELECTED_FEATURES, "oryginal")
        report_dict = classification_report(r["y_test"], r["y_pred"], zero_division=0, output_dict=True)
        per_class += [{"metoda": method, "klasa": cls, "precision": v["precision"], "recall": v["recall"],
                       "f1": v["f1-score"], "support": v["support"]}
                      for cls, v in report_dict.items() if cls in set(r["y_test"])]
        print(f"\n=== {method} ===")
        print("  walidacja: " + ", ".join(f"{p} -> {a:.3f}" for p, a in r["history"]))
        print(f"  najlepsze: {r['params']}  (walidacja {r['val_accuracy']:.3f})")
        print("  test:      " + "  ".join(f"{k}={r[k]:.3f}" for k in ("accuracy", "precision", "recall", "f1")))
        print("  najczęstsze pomyłki (prawdziwa -> przewidziana):")
        for t, p, c in most_confused(r["y_test"], r["y_pred"], 5):
            print(f"    {t} -> {p}: {c}")

        report = classification_report(r["y_test"], r["y_pred"], zero_division=0, digits=3)
        (RESULTS_DIR / f"raport_{method}.txt").write_text(
            f"{method}, parametry: {r['params']}\ncechy: {', '.join(SELECTED_FEATURES)}\n\n{report}"
        )
        plot_confusion(r["y_test"], r["y_pred"], f"{method.replace('_', ' ')} {r['params']} — test, "
                       f"accuracy = {r['accuracy']:.3f}", f"{10 + i}_macierz_pomylek_{method}.png")
        summary.append({"metoda": method, "parametry": str(r["params"]), "val_accuracy": r["val_accuracy"],
                        **{f"test_{k}": r[k] for k in ("accuracy", "precision", "recall", "f1")}})

    write_csv(summary, RESULTS_DIR / "podsumowanie.csv")
    write_csv(per_class, RESULTS_DIR / "wyniki_per_klasa.csv")
    print(f"\n{'metoda':22s} {'walid.':>7s} {'acc':>7s} {'prec':>7s} {'recall':>7s} {'f1':>7s}")
    for s in summary:
        print(f"{s['metoda']:22s} {s['val_accuracy']:7.3f} {s['test_accuracy']:7.3f} {s['test_precision']:7.3f} "
              f"{s['test_recall']:7.3f} {s['test_f1']:7.3f}")
    print(f"-> {plot_comparison(summary).relative_to(LAB_DIR)}")


def preprocessing_comparison() -> None:
    """Points 8-9: does centering and size normalization help?"""
    jobs = [(v, s, m) for v in VARIANTS for s in FEATURE_SETS for m in METHODS]
    results = Parallel(n_jobs=-1)(delayed(evaluate)(m, FEATURE_SETS[s], v) for v, s, m in jobs)
    rows = [{"preprocessing": v, "cechy": s, "metoda": m, "parametry": str(r["params"]),
             "val_accuracy": r["val_accuracy"], "test_accuracy": r["accuracy"], "test_f1": r["f1"]}
            for (v, s, m), r in zip(jobs, results)]
    write_csv(rows, RESULTS_DIR / "porownanie_preprocessingu.csv")

    print("\n=== Wpływ preprocessingu (accuracy na teście, w nawiasie walidacja) ===")
    header = "".join(f"{v[:26]:>30s}" for v in VARIANTS)
    for set_name in FEATURE_SETS:
        print(f"\ncechy: {set_name}\n{'':22s}{header}")
        for method in METHODS:
            cells = []
            for variant in VARIANTS:
                r = next(r for r in rows if r["preprocessing"] == variant and r["cechy"] == set_name
                         and r["metoda"] == method)
                cells.append(f"{r['test_accuracy']:.3f} ({r['val_accuracy']:.3f})")
            print(f"{method:22s}" + "".join(f"{c:>30s}" for c in cells))
    print(f"\n-> {plot_preprocessing_comparison(rows).relative_to(LAB_DIR)}")


def main() -> None:
    PLOTS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    assert set(SELECTED_FEATURES) <= set(ALL_FEATURES)
    print(f"Cechy wybrane ({len(SELECTED_FEATURES)}): {', '.join(SELECTED_FEATURES)}")

    normalization_check()
    classification_report_original()
    preprocessing_comparison()
    print(f"-> {RESULTS_DIR.relative_to(LAB_DIR)}/ (raporty per klasa, podsumowanie.csv, porownanie_preprocessingu.csv)")


if __name__ == "__main__":
    main()
