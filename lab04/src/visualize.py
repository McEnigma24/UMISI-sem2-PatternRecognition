"""Wizualizacja ramek referencyjnych (zadanie 1)."""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dataset import (
    LAB_DIR,
    RESULTS_DIR,
    TARGET_CLASSES,
    draw_boxes,
    load_image,
    load_index,
)

PLOTS_DIR = LAB_DIR / "plots"


def enrich_for_picks(objects_df: pd.DataFrame) -> pd.DataFrame:
    df = objects_df.copy()
    df["w"] = df["x2"] - df["x1"]
    df["h"] = df["y2"] - df["y1"]
    df["min_side"] = df[["w", "h"]].min(axis=1)
    return df


def build_image_meta(images_df: pd.DataFrame, objects_df: pd.DataFrame) -> pd.DataFrame:
    objs = enrich_for_picks(objects_df)
    agg = objs.groupby("image_id").agg(
        n=("class", "count"),
        min_side=("min_side", "min"),
        occl=("occluded", "any"),
        trunc=("truncated", "any"),
    )
    merged = images_df.merge(agg, left_on="image_id", right_index=True, how="inner")
    return merged[merged["n_targets"] > 0]


def pick_image(meta: pd.DataFrame, mask: pd.Series) -> str | None:
    sub = meta[mask]
    if sub.empty:
        return None
    return sub.sort_values("n", ascending=False).iloc[0]["image_path"]


def render_grid(images_df: pd.DataFrame, objects_df: pd.DataFrame) -> None:
    meta = build_image_meta(images_df, objects_df)
    picks: list[tuple[str, str | None]] = [
        ("dzień, pogoda clear", pick_image(meta, meta["weather"] == "clear")),
        ("noc", pick_image(meta, meta["timeofday"].str.contains("night", case=False, na=False))),
        ("deszcz", pick_image(meta, meta["weather"] == "rainy")),
        ("śnieg", pick_image(meta, meta["weather"] == "snowy")),
        ("mgła/overcast", pick_image(meta, meta["weather"].isin(["foggy", "overcast"]))),
        ("świt/zmierzch", pick_image(meta, meta["timeofday"] == "dawn/dusk")),
        (
            "tunel/parking",
            pick_image(
                meta,
                meta["scene"].str.contains("tunnel|parking", case=False, na=False),
            ),
        ),
        ("wiele obiektów", pick_image(meta, meta["n"] >= 6)),
        ("bardzo małe obiekty", pick_image(meta, meta["min_side"] < 12)),
        ("occluded/truncated", pick_image(meta, meta["occl"] | meta["trunc"])),
    ]
    paths = [(t, p) for t, p in picks if p]

    cols = 3
    rows = (len(paths) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 3.2))
    axes = np.atleast_2d(axes)
    for idx, (title, path) in enumerate(paths):
        r, c = divmod(idx, cols)
        img = load_image(path)
        image_id = Path(path).stem
        boxes = objects_df[objects_df["image_id"] == image_id]
        vis = cv2.cvtColor(draw_boxes(img, boxes), cv2.COLOR_BGR2RGB)
        axes[r, c].imshow(vis)
        axes[r, c].set_title(title, fontsize=10)
        axes[r, c].axis("off")
    for idx in range(len(paths), rows * cols):
        r, c = divmod(idx, cols)
        axes[r, c].axis("off")
    fig.suptitle("Ramki referencyjne — różne warunki")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "7_ramki_warunki.png", dpi=110)
    plt.close(fig)


def render_crops(objects_df: pd.DataFrame, per_class: int = 8) -> None:
    fig, axes = plt.subplots(len(TARGET_CLASSES), per_class, figsize=(per_class * 1.6, len(TARGET_CLASSES) * 1.8))
    objs = enrich_for_picks(objects_df)
    for i, cls in enumerate(TARGET_CLASSES):
        pool = objs[objs["class"] == cls]
        sub = pool.sample(n=min(per_class, len(pool)), random_state=42)
        for j in range(per_class):
            ax = axes[i, j]
            if j < len(sub):
                row = sub.iloc[j]
                img = load_image(row["image_path"])
                x1, y1, x2, y2 = map(int, (row["x1"], row["y1"], row["x2"], row["y2"]))
                pad = 4
                x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
                x2, y2 = min(img.shape[1], x2 + pad), min(img.shape[0], y2 + pad)
                crop = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
                ax.imshow(crop)
            ax.axis("off")
            if j == 0:
                ax.set_ylabel(cls.replace("traffic_", ""), fontsize=8)
    fig.suptitle("Wycinki obiektów docelowych")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "8_wycinki_per_klasa.png", dpi=120)
    plt.close(fig)


def render_anomalies() -> None:
    path = RESULTS_DIR / "anomalie.csv"
    if not path.is_file():
        return
    anom = pd.read_csv(path)
    box_anom = anom[anom["reason"].str.contains("box|duplicate", na=False, regex=True)].head(6)
    if box_anom.empty:
        return

    images_df, objects_df = load_index()
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes = axes.flatten()
    for ax, (_, row) in zip(axes, box_anom.iterrows(), strict=False):
        img_path = images_df.loc[images_df["image_id"] == row["image_id"], "image_path"].iloc[0]
        img = load_image(img_path)
        vis = draw_boxes(img, objects_df[objects_df["image_id"] == row["image_id"]])
        if pd.notna(row["x2"]):
            x1, y1, x2, y2 = map(int, (row["x1"], row["y1"], row["x2"], row["y2"]))
            cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 255), 3)
        ax.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        ax.set_title(str(row["reason"])[:28], fontsize=8)
        ax.axis("off")
    for ax in axes[len(box_anom) :]:
        ax.axis("off")
    fig.suptitle("Przykłady anomalii etykiet")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "9_anomalie_etykiet.png", dpi=110)
    plt.close(fig)


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    images_df, objects_df = load_index()
    render_grid(images_df, objects_df)
    render_crops(objects_df)
    render_anomalies()
    print(f"-> zapisano wizualizacje w {PLOTS_DIR.relative_to(LAB_DIR)}/")


if __name__ == "__main__":
    main()
