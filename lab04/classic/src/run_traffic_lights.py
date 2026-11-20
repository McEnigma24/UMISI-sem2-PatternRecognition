"""Strojenie na train, porównanie M1/M2/M1+M2/M1+M2+M3 na val."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from dataset import LAB_DIR, RESULTS_DIR, load_image, load_index
from evaluate import EvalResult, evaluate_image_preds, nms
from tl_candidates import (
    TlParams,
    default_params,
    detect_color_blobs,
    image_has_tl_activity,
    learn_expansion_from_gt,
)
from tl_color import assign_colors
from tl_classifier import filter_by_svm, svm_scores, train_svm
from tl_housing import detect_housing, housing_activity

PLOTS_DIR = LAB_DIR / "plots"
PRED_DIR = RESULTS_DIR / "predictions"


def limit_top(boxes: list[dict], k: int) -> list[dict]:
    return sorted(boxes, key=lambda b: -float(b.get("score", 0)))[:k]


def detect_variant(
    img: np.ndarray,
    variant: str,
    p: TlParams,
    clf=None,
    svm_thr: float = 0.0,
) -> list[dict]:
    active = image_has_tl_activity(img, p)
    boxes: list[dict] = []
    if variant == "M1":
        if not active:
            return []
        boxes.extend(detect_color_blobs(img, p))
    elif variant == "M2":
        if housing_activity(img, p) or active:
            boxes.extend(detect_housing(img, p))
    else:
        if active:
            boxes.extend(detect_color_blobs(img, p))
        if housing_activity(img, p) or active:
            boxes.extend(detect_housing(img, p))
        if not boxes:
            return []
    boxes = nms(boxes, p.nms_iou)
    boxes = [b for b in boxes if float(b.get("score", 0)) >= p.min_score]
    boxes = assign_colors(img, boxes)
    boxes = limit_top(boxes, p.max_boxes)
    if variant == "M1+M2+M3" and clf is not None:
        boxes = svm_scores(clf, img, boxes)
        boxes = filter_by_svm(boxes, svm_thr)
    return boxes


def preds_for_split(
    objects_df: pd.DataFrame,
    images_df: pd.DataFrame,
    split: str,
    variant: str,
    p: TlParams,
    clf=None,
    svm_thr: float = 0.0,
    limit: int | None = None,
) -> dict[str, list[dict]]:
    frame = images_df[images_df["split"] == split]
    ids = frame["image_id"].unique()
    if limit:
        ids = ids[:limit]
    paths = dict(zip(images_df["image_id"], images_df["image_path"], strict=False))
    out: dict[str, list[dict]] = {}
    for image_id in tqdm(ids, desc=f"{variant} {split}", leave=False):
        path = paths.get(image_id)
        if not path:
            continue
        img = load_image(path)
        out[image_id] = detect_variant(img, variant, p, clf=clf, svm_thr=svm_thr)
    return out


def tune_svm_threshold(
    objects_df: pd.DataFrame,
    images_df: pd.DataFrame,
    p: TlParams,
    clf,
) -> float:
    ids = images_df[images_df["split"] == "train"]["image_id"].unique()[:120]
    paths = dict(zip(images_df["image_id"], images_df["image_path"], strict=False))
    best_thr = 0.5
    best_f1 = -1.0
    for thr in (-0.5, 0.0, 0.5, 1.0, 1.5, 2.0):
        preds: dict[str, list[dict]] = {}
        for image_id in ids:
            preds[image_id] = detect_variant(
                load_image(paths[image_id]), "M1+M2+M3", p, clf=clf, svm_thr=thr
            )
        ev = evaluate_image_preds(objects_df, preds, images_df=images_df)
        if ev.f1 > best_f1:
            best_f1 = ev.f1
            best_thr = thr
    print(f"SVM threshold: {best_thr} (F1={best_f1:.3f} na podzbiorze train)")
    return best_thr


def tune_params(objects_df: pd.DataFrame, images_df: pd.DataFrame) -> TlParams:
    p = default_params()
    p.expand = learn_expansion_from_gt(objects_df)
    best_p = p
    best_f1 = -1.0
    grid_s = [70, 90]
    grid_v = [100, 130]
    sample_ids = images_df[images_df["split"] == "train"]["image_id"].unique()[:180]

    for s_min in grid_s:
        for v_min in grid_v:
            trial = TlParams(
                roi_y_frac=p.roi_y_frac,
                s_min=s_min,
                v_min=v_min,
                morph_k=p.morph_k,
                min_area=p.min_area,
                max_area=p.max_area,
                min_circularity_small=p.min_circularity_small,
                min_circularity_large=p.min_circularity_large,
                tophat_k=p.tophat_k,
                tophat_v_thr=p.tophat_v_thr,
                nms_iou=p.nms_iou,
                expand=dict(p.expand),
            )
            preds: dict[str, list[dict]] = {}
            paths = dict(zip(images_df["image_id"], images_df["image_path"], strict=False))
            for image_id in sample_ids:
                path = paths[image_id]
                preds[image_id] = detect_variant(load_image(path), "M1+M2", trial)
            ev = evaluate_image_preds(objects_df, preds, images_df=images_df)
            if ev.f1 > best_f1:
                best_f1 = ev.f1
                best_p = trial
    print(f"Strojenie (podzbiór train n={len(sample_ids)}): najlepszy F1={best_f1:.3f}, s_min={best_p.s_min}, v_min={best_p.v_min}")
    return best_p


def plot_pr_curve(results: dict[str, EvalResult], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, ev in results.items():
        ax.scatter([ev.recall], [ev.precision], s=80, label=f"{name} F1={ev.f1:.3f}")
    ax.set_xlabel("Recall (traffic_light)")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.set_title("Porównanie wariantów — val")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_color_matrix(ev: EvalResult, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(ev.color_matrix.values, cmap="Blues")
    ax.set_xticks(range(len(ev.color_matrix.columns)))
    ax.set_yticks(range(len(ev.color_matrix.index)))
    ax.set_xticklabels([c.replace("traffic_light_", "") for c in ev.color_matrix.columns], rotation=45)
    ax.set_yticklabels([c.replace("traffic_light_", "") for c in ev.color_matrix.index])
    ax.set_title(f"Macierz koloru na TP (acc={ev.color_acc:.2f})")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_examples(
    objects_df: pd.DataFrame,
    images_df: pd.DataFrame,
    preds: dict[str, list[dict]],
    out_path: Path,
) -> None:
    from evaluate import match_boxes

    paths = dict(zip(images_df["image_id"], images_df["image_path"], strict=False))
    examples: list[tuple[str, str, str]] = []
    tls = objects_df[objects_df["class"].str.startswith("traffic_light")]
    for image_id, grp in tls.groupby("image_id"):
        if str(grp.iloc[0]["split"]) != "val":
            continue
        gts = grp.to_dict("records")
        pr = preds.get(image_id, [])
        pairs, fp_i, fn_i = match_boxes(pr, gts)
        for tag, idxs, pool in (
            ("TP", [p[0] for p in pairs[:1]], pr),
            ("FP", fp_i[:1], pr),
            ("FN", fn_i[:1], gts),
        ):
            if not idxs:
                continue
            i = idxs[0]
            examples.append((image_id, tag, paths[image_id]))
            if len(examples) >= 6:
                break
        if len(examples) >= 6:
            break

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, (image_id, tag, path) in zip(axes.flatten(), examples, strict=False):
        img = load_image(path)
        vis = img.copy()
        for p in preds.get(image_id, []):
            x1, y1, x2, y2 = map(int, (p["x1"], p["y1"], p["x2"], p["y2"]))
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        for _, row in tls[tls["image_id"] == image_id].iterrows():
            x1, y1, x2, y2 = map(int, (row["x1"], row["y1"], row["x2"], row["y2"]))
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
        ax.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        ax.set_title(f"{tag} {image_id[:12]}")
        ax.axis("off")
    for ax in axes.flatten()[len(examples) :]:
        ax.axis("off")
    fig.suptitle("Przykłady TP / FP / FN (val)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def save_predictions(preds: dict[str, list[dict]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {k: [{kk: float(vv) if isinstance(vv, (np.floating, float)) else vv for kk, vv in b.items()} for b in v] for k, v in preds.items()}
    path.write_text(json.dumps(serializable, indent=2))


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    images_df, objects_df = load_index()

    params = tune_params(objects_df, images_df)
    clf = train_svm(objects_df, force=False)
    svm_thr = tune_svm_threshold(objects_df, images_df, params, clf)

    variants = ["M1", "M2", "M1+M2", "M1+M2+M3"]
    results: dict[str, EvalResult] = {}
    best_name = variants[0]
    best_ev: EvalResult | None = None
    rows_csv = []
    best_preds: dict[str, list[dict]] = {}

    for variant in variants:
        preds = preds_for_split(
            objects_df,
            images_df,
            "val",
            variant,
            params,
            clf=clf if "M3" in variant else None,
            svm_thr=svm_thr,
        )
        ev = evaluate_image_preds(objects_df, preds, images_df=images_df)
        results[variant] = ev
        rows_csv.append(
            {
                "variant": variant,
                "precision": ev.precision,
                "recall": ev.recall,
                "f1": ev.f1,
                "color_acc": ev.color_acc,
                "tp": ev.tp,
                "fp": ev.fp,
                "fn": ev.fn,
            }
        )
        print(
            f"{variant}: P={ev.precision:.3f} R={ev.recall:.3f} F1={ev.f1:.3f} "
            f"kolor={ev.color_acc:.3f} (TP={ev.tp}, FP={ev.fp}, FN={ev.fn})"
        )
        if best_ev is None or ev.f1 > best_ev.f1:
            best_ev = ev
            best_name = variant
            best_preds = preds

    pd.DataFrame(rows_csv).to_csv(RESULTS_DIR / "tl_porownanie_val.csv", index=False)
    plot_pr_curve(results, PLOTS_DIR / "10_tl_pr_val.png")
    if best_ev:
        plot_color_matrix(best_ev, PLOTS_DIR / "12_tl_kolor.png")
        plot_examples(objects_df, images_df, best_preds, PLOTS_DIR / "11_tl_przyklady.png")
        save_predictions(best_preds, PRED_DIR / "tl_val.json")
        if not best_ev.by_size.empty:
            best_ev.by_size.to_csv(RESULTS_DIR / "tl_recall_by_size.csv", index=False)
        if not best_ev.by_timeofday.empty:
            best_ev.by_timeofday.to_csv(RESULTS_DIR / "tl_recall_by_timeofday.csv", index=False)

    with (RESULTS_DIR / "tl_wnioski.txt").open("w") as f:
        f.write("Lab04 zadanie 3 — detekcja sygnalizatorów\n\n")
        f.write(f"Wybrany wariant (najwyższy F1 na val): {best_name}\n\n")
        for v, ev in results.items():
            f.write(
                f"{v}: P={ev.precision:.3f} R={ev.recall:.3f} F1={ev.f1:.3f} color_acc={ev.color_acc:.3f}\n"
            )
        f.write("\nParametry: ")
        f.write(f"roi_y={params.roi_y_frac}, s_min={params.s_min}, v_min={params.v_min}\n")

    print(f"-> {RESULTS_DIR / 'tl_porownanie_val.csv'}")
    print(f"-> wykresy 10-12 w {PLOTS_DIR}")


if __name__ == "__main__":
    main()
