import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
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
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from glcm_features import ALL_FEATURES, MINIMAL_FEATURES

LAB_DIR = Path(__file__).resolve().parent.parent
FEATURES_DIR = LAB_DIR / "features"
PLOTS_DIR = LAB_DIR / "plots"
RESULTS_DIR = LAB_DIR / "results"

FEATURE_SETS = {
    "minimalne": MINIMAL_FEATURES,
    "pelne": ALL_FEATURES,
}

METHODS = {
    "regresja_logistyczna": (
        LogisticRegression(max_iter=20000),
        {"C": [0.1, 1, 10, 100, 1000]},
    ),
    "kNN": (
        KNeighborsClassifier(),
        {"n_neighbors": [1, 3, 5, 7, 9, 15], "weights": ["uniform", "distance"]},
    ),
    "drzewo_decyzyjne": (
        DecisionTreeClassifier(random_state=0),
        {"max_depth": [3, 5, 8, 12, 20, None], "criterion": ["gini", "entropy"]},
    ),
    "SVM": (
        SVC(random_state=0),
        {"C": [0.1, 1, 10, 100], "kernel": ["linear", "rbf"], "gamma": ["scale", "auto"]},
    ),
    "xgboost": (
        XGBClassifier(random_state=0, eval_metric="mlogloss", verbosity=0),
        {"max_depth": [3, 6, 10], "learning_rate": [0.05, 0.1], "n_estimators": [100, 200]},
    ),
}


def load_split(split: str, names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    with (FEATURES_DIR / f"features_{split}.csv").open() as f:
        rows = list(csv.DictReader(f))
    X = np.array([[float(r[k]) for k in names] for r in rows])
    y = np.array([r["label"] for r in rows])
    return X, y


def load_all(names: list[str]):
    return [load_split(s, names) for s in ("1_trening", "2_walidacja", "3_test")]


def build(model, scaler=None):
    if scaler is False:
        return clone(model)
    return make_pipeline(scaler or MinMaxScaler(), clone(model))


def param_grid(grid: dict) -> list[dict]:
    combos = [{}]
    for key, values in grid.items():
        combos = [c | {key: v} for c in combos for v in values]
    return combos


def tune(model, grid: dict, train, val) -> tuple[dict, float, list]:
    history = []
    for params in param_grid(grid):
        pipe = build(clone(model).set_params(**params)).fit(*train)
        history.append((params, accuracy_score(val[1], pipe.predict(val[0]))))
    best_params, best_acc = max(history, key=lambda h: h[1])
    return best_params, best_acc, history


def metrics(y_true, y_pred) -> dict[str, float]:
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {"accuracy": accuracy_score(y_true, y_pred), "precision": p, "recall": r, "f1": f}


def evaluate(method: str, names: list[str]) -> dict:
    model, grid = METHODS[method]
    train, val, test = load_all(names)
    le = LabelEncoder()
    train_enc = (train[0], le.fit_transform(train[1]))
    val_enc = (val[0], le.transform(val[1]))
    y_test = test[1]
    best_params, val_acc, history = tune(model, grid, train_enc, val_enc)
    pipe = build(clone(model).set_params(**best_params)).fit(*train_enc)
    y_pred = le.inverse_transform(pipe.predict(test[0]))
    return {
        "params": best_params,
        "val_accuracy": val_acc,
        "history": history,
        "y_test": y_test,
        "y_pred": y_pred,
        **metrics(y_test, y_pred),
    }


def save(fig, name: str, dpi: int = 110) -> Path:
    out = PLOTS_DIR / name
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return out


def plot_confusion(y_true, y_pred, title: str, filename: str) -> Path:
    classes = sorted(set(y_true))
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues", vmin=0)
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("przewidziana klasa")
    ax.set_ylabel("prawdziwa klasa")
    ax.set_title(title)
    fig.colorbar(im, fraction=0.03)
    fig.tight_layout()
    return save(fig, filename)


def plot_comparison(summary: list[dict], feature_set: str) -> Path:
    names = [s["metoda"] for s in summary]
    keys = ["accuracy", "precision", "recall", "f1"]
    x = np.arange(len(names))
    width = 0.2
    fig, ax = plt.subplots(figsize=(max(10, 2 * len(names)), 5))
    for i, key in enumerate(keys):
        values = [s[f"test_{key}"] for s in summary]
        bars = ax.bar(x + (i - 1.5) * width, values, width, label=key)
        ax.bar_label(bars, fmt="%.2f", fontsize=8)
    ax.set_xticks(x, [n.replace("_", " ") for n in names])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("wartość (makro)")
    ax.set_title(f"Porównanie metod — cechy {feature_set}, zbiór testowy")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return save(fig, f"5_porownanie_metod_{feature_set}.png")


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def normalization_check(names: list[str]) -> None:
    train, val, _ = load_all(names)
    print("\nWpływ normalizacji (kNN k=1, walidacja):")
    knn = KNeighborsClassifier(1)
    for label, scaler in (("bez normalizacji", False), ("MinMax [0,1]", MinMaxScaler()),
                          ("standaryzacja", StandardScaler())):
        pipe = build(knn, scaler).fit(*train)
        print(f"  {label:18s} {pipe.score(*val):.3f}")


def run_feature_set(set_name: str, names: list[str]) -> list[dict]:
    summary = []
    per_class = []
    for method in METHODS:
        r = evaluate(method, names)
        report_dict = classification_report(r["y_test"], r["y_pred"], zero_division=0, output_dict=True)
        per_class += [
            {
                "cechy": set_name,
                "metoda": method,
                "klasa": cls,
                "precision": v["precision"],
                "recall": v["recall"],
                "f1": v["f1-score"],
                "support": v["support"],
            }
            for cls, v in report_dict.items()
            if cls in set(r["y_test"])
        ]
        print(f"\n=== {set_name} / {method} ===")
        print(f"  najlepsze: {r['params']}  (walidacja {r['val_accuracy']:.3f})")
        print("  test:      " + "  ".join(f"{k}={r[k]:.3f}" for k in ("accuracy", "precision", "recall", "f1")))
        report = classification_report(r["y_test"], r["y_pred"], zero_division=0, digits=3)
        (RESULTS_DIR / f"raport_{set_name}_{method}.txt").write_text(
            f"{method}, cechy={set_name}, parametry: {r['params']}\n\n{report}"
        )
        plot_confusion(
            r["y_test"],
            r["y_pred"],
            f"{method} ({set_name}) — accuracy={r['accuracy']:.3f}",
            f"6_macierz_{set_name}_{method}.png",
        )
        summary.append(
            {
                "cechy": set_name,
                "metoda": method,
                "parametry": str(r["params"]),
                "val_accuracy": r["val_accuracy"],
                **{f"test_{k}": r[k] for k in ("accuracy", "precision", "recall", "f1")},
            }
        )
    write_csv(per_class, RESULTS_DIR / f"wyniki_per_klasa_{set_name}.csv")
    plot_comparison(summary, set_name)
    return summary


def main() -> None:
    PLOTS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    all_summary = []
    for set_name, names in FEATURE_SETS.items():
        print(f"\n######## Cechy: {set_name} ({len(names)}) ########")
        normalization_check(names)
        all_summary.extend(run_feature_set(set_name, names))

    write_csv(all_summary, RESULTS_DIR / "podsumowanie.csv")
    print(f"\n-> {RESULTS_DIR.relative_to(LAB_DIR)}/podsumowanie.csv")


if __name__ == "__main__":
    main()
