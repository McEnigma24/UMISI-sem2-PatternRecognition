"""BDD100K — indeks obrazów 10k + etykiet detekcji 100k (5 klas docelowych)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

LAB_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = LAB_DIR / "data"
IMAGES_ROOT = DATA_DIR / "images" / "10k"
LABELS_ROOT = DATA_DIR / "labels" / "100k"
RESULTS_DIR = LAB_DIR / "results"

TARGET_CLASSES = (
    "traffic_sign",
    "traffic_light_red",
    "traffic_light_yellow",
    "traffic_light_green",
    "traffic_light_unknown",
)

CLASS_COLORS_BGR = {
    "traffic_sign": (0, 215, 255),
    "traffic_light_red": (0, 0, 255),
    "traffic_light_yellow": (0, 255, 255),
    "traffic_light_green": (0, 255, 0),
    "traffic_light_unknown": (160, 160, 160),
}

IMG_WIDTH = 1280
IMG_HEIGHT = 720


def _map_label(category: str, attrs: dict) -> str | None:
    if category == "traffic sign":
        return "traffic_sign"
    if category != "traffic light":
        return None
    color = (attrs or {}).get("trafficLightColor", "none")
    mapping = {
        "red": "traffic_light_red",
        "yellow": "traffic_light_yellow",
        "green": "traffic_light_green",
    }
    return mapping.get(color, "traffic_light_unknown")


def build_image_lookup() -> dict[str, Path]:
    lookup: dict[str, Path] = {}
    for split in ("train", "val", "test"):
        folder = IMAGES_ROOT / split
        if not folder.is_dir():
            continue
        for path in folder.glob("*.jpg"):
            lookup[path.stem] = path
    return lookup


def parse_label_json(path: Path, split: str, image_path: Path) -> tuple[dict, list[dict]]:
    with path.open() as f:
        data = json.load(f)
    attrs = data.get("attributes") or {}
    frame = data["frames"][0]
    objects = frame.get("objects") or []
    rows: list[dict] = []
    for obj in objects:
        if "box2d" not in obj:
            continue
        cls = _map_label(obj.get("category", ""), obj.get("attributes") or {})
        if cls is None:
            continue
        box = obj["box2d"]
        x1, y1, x2, y2 = float(box["x1"]), float(box["y1"]), float(box["x2"]), float(box["y2"])
        oattrs = obj.get("attributes") or {}
        rows.append(
            {
                "image_id": data["name"],
                "split": split,
                "image_path": str(image_path),
                "class": cls,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "occluded": bool(oattrs.get("occluded", False)),
                "truncated": bool(oattrs.get("truncated", False)),
                "weather": attrs.get("weather", "unknown"),
                "scene": attrs.get("scene", "unknown"),
                "timeofday": attrs.get("timeofday", "unknown"),
            }
        )
    img_row = {
        "image_id": data["name"],
        "split": split,
        "image_path": str(image_path),
        "weather": attrs.get("weather", "unknown"),
        "scene": attrs.get("scene", "unknown"),
        "timeofday": attrs.get("timeofday", "unknown"),
        "n_targets": len(rows),
    }
    return img_row, rows


def build_index(force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    objects_path = RESULTS_DIR / "index_objects.csv"
    images_path = RESULTS_DIR / "index_images.csv"
    if not force and objects_path.is_file() and images_path.is_file():
        return load_index()

    lookup = build_image_lookup()
    image_rows: list[dict] = []
    object_rows: list[dict] = []
    without_labels: list[str] = []

    for stem, img_path in tqdm(sorted(lookup.items()), desc="Parowanie 10k z etykietami"):
        json_path = None
        split = None
        for label_split in ("train", "val", "test"):
            candidate = LABELS_ROOT / label_split / f"{stem}.json"
            if candidate.is_file():
                json_path = candidate
                split = label_split
                break
        if json_path is None or split is None:
            without_labels.append(stem)
            continue
        img_row, objs = parse_label_json(json_path, split, img_path)
        image_rows.append(img_row)
        object_rows.extend(objs)

    images_df = pd.DataFrame(image_rows)
    objects_df = pd.DataFrame(object_rows)
    images_df.to_csv(images_path, index=False)
    objects_df.to_csv(objects_path, index=False)

    miss_path = RESULTS_DIR / "images_10k_bez_etykiet.txt"
    miss_path.write_text("\n".join(without_labels) + ("\n" if without_labels else ""))
    print(
        f"Zindeksowano {len(images_df)} obrazów, {len(objects_df)} obiektów docelowych "
        f"(obrazy 10k bez etykiety detekcji: {len(without_labels)})."
    )
    return images_df, objects_df


def load_index() -> tuple[pd.DataFrame, pd.DataFrame]:
    images_path = RESULTS_DIR / "index_images.csv"
    objects_path = RESULTS_DIR / "index_objects.csv"
    if not images_path.is_file() or not objects_path.is_file():
        return build_index(force=True)
    return pd.read_csv(images_path), pd.read_csv(objects_path)


def load_image(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Nie można wczytać obrazu: {path}")
    return img


def draw_boxes(
    img: np.ndarray,
    boxes: pd.DataFrame,
    *,
    thickness: int = 2,
) -> np.ndarray:
    out = img.copy()
    for _, row in boxes.iterrows():
        cls = row["class"]
        color = CLASS_COLORS_BGR.get(cls, (255, 255, 255))
        x1, y1, x2, y2 = map(int, (row["x1"], row["y1"], row["x2"], row["y2"]))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        label = cls.replace("traffic_light_", "tl_").replace("traffic_sign", "sign")
        cv2.putText(
            out,
            label,
            (x1, max(y1 - 4, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    return out


def box_iou(a: dict, b: dict) -> float:
    ix1 = max(a["x1"], b["x1"])
    iy1 = max(a["y1"], b["y1"])
    ix2 = min(a["x2"], b["x2"])
    iy2 = min(a["y2"], b["y2"])
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a["x2"] - a["x1"]) * max(0.0, a["y2"] - a["y1"])
    area_b = max(0.0, b["x2"] - b["x1"]) * max(0.0, b["y2"] - b["y1"])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-index", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.build_index or args.force:
        build_index(force=True)
    else:
        images_df, objects_df = load_index()
        print(images_df["split"].value_counts())
        print(objects_df["class"].value_counts())


if __name__ == "__main__":
    main()
