"""Analiza jakości BDD100K (zadanie 1): statystyki, wyjątki, artefakty."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from dataset import (
    IMG_HEIGHT,
    IMG_WIDTH,
    LAB_DIR,
    RESULTS_DIR,
    TARGET_CLASSES,
    box_iou,
    build_index,
    load_image,
)

PLOTS_DIR = LAB_DIR / "plots"
IMAGES_ROOT = LAB_DIR / "data" / "images" / "10k"


def enrich_objects(objects_df: pd.DataFrame) -> pd.DataFrame:
    df = objects_df.copy()
    df["w"] = df["x2"] - df["x1"]
    df["h"] = df["y2"] - df["y1"]
    df["area"] = df["w"] * df["h"]
    df["cx"] = (df["x1"] + df["x2"]) / 2
    df["cy"] = (df["y1"] + df["y2"]) / 2
    df["aspect"] = df["w"] / df["h"].replace(0, np.nan)
    return df


def find_label_anomalies(objects_df: pd.DataFrame, images_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for _, o in objects_df.iterrows():
        reasons: list[str] = []
        w, h = o["x2"] - o["x1"], o["y2"] - o["y1"]
        if w <= 0 or h <= 0:
            reasons.append("non_positive_box")
        if o["x1"] < 0 or o["y1"] < 0 or o["x2"] > IMG_WIDTH or o["y2"] > IMG_HEIGHT:
            reasons.append("out_of_bounds")
        if min(w, h) < 8:
            reasons.append("tiny_box")
        if w > 0 and h > 0 and (w / h > 8 or h / w > 8):
            reasons.append("extreme_aspect")
        if reasons:
            rows.append(
                {
                    "image_id": o["image_id"],
                    "class": o["class"],
                    "reason": ";".join(reasons),
                    "x1": o["x1"],
                    "y1": o["y1"],
                    "x2": o["x2"],
                    "y2": o["y2"],
                }
            )

    for image_id, grp in objects_df.groupby("image_id"):
        objs = grp.to_dict("records")
        for a, b in combinations(objs, 2):
            if a["class"] != b["class"]:
                continue
            if box_iou(a, b) > 0.9:
                rows.append(
                    {
                        "image_id": image_id,
                        "class": a["class"],
                        "reason": "duplicate_iou>0.9",
                        "x1": a["x1"],
                        "y1": a["y1"],
                        "x2": a["x2"],
                        "y2": a["y2"],
                    }
                )
                break

    empty = images_df.loc[images_df["n_targets"] == 0, "image_id"]
    for image_id in empty:
        rows.append(
            {
                "image_id": image_id,
                "class": "",
                "reason": "no_target_objects",
                "x1": np.nan,
                "y1": np.nan,
                "x2": np.nan,
                "y2": np.nan,
            }
        )

    return pd.DataFrame(rows)


def image_quality_row(path: str, meta: pd.Series) -> dict:
    img = cv2.imread(path)
    if img is None:
        return {"image_id": meta["image_id"], "readable": False}
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    brightness = float(gray.mean())
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    overexp = float((gray > 250).mean())
    td = str(meta["timeofday"]).lower()
    mismatch = False
    if "night" in td and brightness > 85:
        mismatch = True
    if "day" in td and brightness < 45:
        mismatch = True
    return {
        "image_id": meta["image_id"],
        "image_path": path,
        "split": meta["split"],
        "readable": True,
        "width": w,
        "height": h,
        "brightness": brightness,
        "blur_var": blur,
        "overexp_ratio": overexp,
        "timeofday": meta["timeofday"],
        "weather": meta["weather"],
        "timeofday_brightness_mismatch": mismatch,
    }


def compute_image_quality(images_df: pd.DataFrame, sample_for_mean: int = 800) -> pd.DataFrame:
    rows = []
    for _, meta in tqdm(images_df.iterrows(), total=len(images_df), desc="Jakość obrazów"):
        rows.append(image_quality_row(meta["image_path"], meta))
    return pd.DataFrame(rows)


def build_mean_image(images_df: pd.DataFrame, max_n: int = 800) -> np.ndarray:
    subset = images_df.sample(n=min(max_n, len(images_df)), random_state=42)
    acc = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.float64)
    count = 0
    for path in tqdm(subset["image_path"], desc="Średni obraz"):
        img = cv2.imread(path)
        if img is None or img.shape[:2] != (IMG_HEIGHT, IMG_WIDTH):
            continue
        acc += img.astype(np.float64)
        count += 1
    if count == 0:
        return np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
    return (acc / count).astype(np.uint8)


def plot_class_counts(objects_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    by_split = objects_df.groupby(["split", "class"]).size().unstack(fill_value=0)
    by_split.reindex(columns=TARGET_CLASSES, fill_value=0).plot(kind="bar", ax=axes[0])
    axes[0].set_title("Obiekty per klasa i split")
    axes[0].tick_params(axis="x", rotation=0)
    objects_df["class"].value_counts().reindex(TARGET_CLASSES).plot(kind="bar", ax=axes[1], color="steelblue")
    axes[1].set_title("Obiekty per klasa (łącznie)")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "1_licznosci_klas.png", dpi=120)
    plt.close(fig)


def plot_context_distributions(images_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, col in zip(axes, ("weather", "timeofday", "scene"), strict=True):
        images_df[col].value_counts().head(12).plot(kind="bar", ax=ax)
        ax.set_title(col)
        ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "2_kontekst_pogoda_czas_scena.png", dpi=120)
    plt.close(fig)


def plot_box_geometry(objects_df: pd.DataFrame) -> None:
    df = enrich_objects(objects_df)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for cls in TARGET_CLASSES:
        sub = df.loc[df["class"] == cls, "area"]
        sub = sub[sub > 0]
        if len(sub):
            axes[0, 0].hist(np.log10(sub), bins=40, alpha=0.5, label=cls.replace("traffic_", ""))
    axes[0, 0].set_title("log10(pole ramki)")
    axes[0, 0].legend(fontsize=7)

    df.boxplot(column="aspect", by="class", ax=axes[0, 1])
    axes[0, 1].set_title("Proporcje w/h")
    plt.suptitle("")

    lights = df[df["class"].str.startswith("traffic_light")]
    if len(lights):
        h2d, _, _, im = axes[1, 0].hist2d(
            lights["cx"],
            lights["cy"],
            bins=(32, 24),
            range=((0, IMG_WIDTH), (0, IMG_HEIGHT)),
        )
        axes[1, 0].set_title("Położenie środków — sygnalizatory")
        fig.colorbar(im, ax=axes[1, 0])

    signs = df[df["class"] == "traffic_sign"]
    if len(signs):
        h2d, _, _, im = axes[1, 1].hist2d(
            signs["cx"],
            signs["cy"],
            bins=(32, 24),
            range=((0, IMG_WIDTH), (0, IMG_HEIGHT)),
        )
        axes[1, 1].set_title("Położenie środków — znaki")
        fig.colorbar(im, ax=axes[1, 1])

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "3_geometria_ramek.png", dpi=120)
    plt.close(fig)


def plot_quality(quality_df: pd.DataFrame) -> None:
    ok = quality_df[quality_df["readable"]]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    axes[0].hist(ok["brightness"], bins=50, color="gray")
    axes[0].set_title("Jasność (średnia)")
    axes[1].hist(np.log10(ok["blur_var"].clip(lower=1)), bins=50, color="teal")
    axes[1].set_title("log10(wariancja Laplaciana)")
    axes[2].hist(ok["overexp_ratio"], bins=50, color="orange")
    axes[2].set_title("Udział pikseli >250")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "4_jakosc_obrazow.png", dpi=120)
    plt.close(fig)


def plot_extreme_examples(quality_df: pd.DataFrame) -> None:
    ok = quality_df[quality_df["readable"]].copy()
    if ok.empty:
        return
    picks = [
        ("najciemniejsze", ok.nsmallest(3, "brightness")),
        ("najrozmytsze", ok.nsmallest(3, "blur_var")),
        ("przeswietlone", ok.nlargest(3, "overexp_ratio")),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(11, 9))
    for row, (title, part) in enumerate(picks):
        for col, (_, r) in enumerate(part.iterrows()):
            img = cv2.cvtColor(load_image(r["image_path"]), cv2.COLOR_BGR2RGB)
            axes[row, col].imshow(img)
            axes[row, col].axis("off")
            if col == 0:
                axes[row, col].set_ylabel(title, fontsize=9)
    fig.suptitle("Skrajne przypadki jakości")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "5_skraine_jakosci.png", dpi=120)
    plt.close(fig)


def write_report(
    images_df: pd.DataFrame,
    objects_df: pd.DataFrame,
    anomalies_df: pd.DataFrame,
    quality_df: pd.DataFrame,
    val_only_images: int,
) -> None:
    lines = [
        "Lab04 — BDD100K (obrazy 10k, etykiety detekcji 100k)",
        "",
        "Parowanie: split wg folderu etykiet (train/val/test).",
        f"Obrazy z etykietami: {len(images_df)} (train {sum(images_df['split']=='train')}, "
        f"val {sum(images_df['split']=='val')}, test {sum(images_df['split']=='test')}).",
        f"Folder images/10k/val bez etykiet: {val_only_images} obrazów (pominięte).",
        f"Obiekty docelowe (ramki): {len(objects_df)}.",
        "",
        "Klasy (n):",
    ]
    for cls in TARGET_CLASSES:
        n = int((objects_df["class"] == cls).sum())
        lines.append(f"  {cls}: {n}")

    occl = float(objects_df["occluded"].mean() * 100)
    trunc = float(objects_df["truncated"].mean() * 100)
    unk = float((objects_df["class"] == "traffic_light_unknown").mean() * 100)
    lines.extend(
        [
            "",
            f"Udział occluded: {occl:.1f}%, truncated: {trunc:.1f}%.",
            f"Udział traffic_light_unknown wśród wszystkich obiektów: {unk:.1f}%.",
            "",
            f"Wyjątki etykiet (anomalie.csv): {len(anomalies_df)} wpisów.",
        ]
    )

    ok = quality_df[quality_df["readable"]]
    bad_res = int((ok["width"] != IMG_WIDTH).sum() + (ok["height"] != IMG_HEIGHT).sum())
    unread = int((~quality_df["readable"]).sum())
    mismatch = int(ok["timeofday_brightness_mismatch"].sum())
    lines.extend(
        [
            f"Nieczytelne obrazy: {unread}.",
            f"Nietypowa rozdzielczość: {bad_res} (oczekiwane {IMG_WIDTH}x{IMG_HEIGHT}).",
            f"Podejrzana zgodność timeofday vs jasność: {mismatch}.",
            "",
            "Wnioski:",
            "- Zbiór jest mocno niezbalansowany (znaki >> światła; mało yellow).",
            "- Sygnalizatory skupiają się w górnej części kadru — sensowny prior dla klasycznego detektora.",
            "- Trudne warunki (noc, deszcz, mgła) są reprezentowane, ale rzadziej niż dzień/clear.",
            "- Artefakt: średni obraz (plots/6_sredni_obraz.png) ujawnia maskę/deskę rozdzielczą u dołu kadru.",
            "- Małe i obcięte ramki oraz unknown koloru wymagają ostrożności przy metrykach IoU i klasyfikacji koloru.",
        ]
    )
    (RESULTS_DIR / "analiza_zbioru.txt").write_text("\n".join(lines) + "\n")


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    images_df, objects_df = build_index()
    objects_df = enrich_objects(objects_df)

    val_only = len(list((IMAGES_ROOT / "val").glob("*.jpg"))) if (IMAGES_ROOT / "val").is_dir() else 0

    anomalies_df = find_label_anomalies(objects_df, images_df)
    anomalies_df.to_csv(RESULTS_DIR / "anomalie.csv", index=False)

    quality_df = compute_image_quality(images_df)
    quality_df.to_csv(RESULTS_DIR / "jakosc_obrazow.csv", index=False)

    mean_img = build_mean_image(images_df)
    cv2.imwrite(str(PLOTS_DIR / "6_sredni_obraz.png"), mean_img)

    plot_class_counts(objects_df)
    plot_context_distributions(images_df)
    plot_box_geometry(objects_df)
    plot_quality(quality_df)
    plot_extreme_examples(quality_df)
    write_report(images_df, objects_df, anomalies_df, quality_df, val_only)

    print(f"-> {RESULTS_DIR / 'analiza_zbioru.txt'}")
    print(f"-> wykresy w {PLOTS_DIR.relative_to(LAB_DIR)}/")


if __name__ == "__main__":
    main()
