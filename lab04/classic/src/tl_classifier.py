"""M3: HOG + histogram HSV -> SVM weryfikacja kandydatów."""

from __future__ import annotations

from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from skimage.feature import hog

from dataset import LAB_DIR, load_image
from evaluate import box_dict, box_iou
from tl_candidates import detect_color_blobs, default_params
from tl_housing import detect_housing

MODEL_PATH = LAB_DIR / "results" / "tl_svm.joblib"
HOG_SIZE = (16, 32)  # (szer., wys.) — pionowa obudowa


def crop_box(img: np.ndarray, box: dict, pad: float = 0.05) -> np.ndarray:
    h, w = img.shape[:2]
    bw = box["x2"] - box["x1"]
    bh = box["y2"] - box["y1"]
    x1 = int(max(0, box["x1"] - pad * bw))
    y1 = int(max(0, box["y1"] - pad * bh))
    x2 = int(min(w, box["x2"] + pad * bw))
    y2 = int(min(h, box["y2"] + pad * bh))
    if x2 <= x1 or y2 <= y1:
        return np.zeros((8, 8, 3), dtype=np.uint8)
    return img[y1:y2, x1:x2]


def feature_vector(bgr: np.ndarray) -> np.ndarray:
    if bgr.size == 0:
        bgr = np.zeros((8, 8, 3), dtype=np.uint8)
    patch = cv2.resize(bgr, HOG_SIZE)
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    hog_vec = hog(
        gray,
        orientations=8,
        pixels_per_cell=(8, 8),
        cells_per_block=(1, 1),
        feature_vector=True,
    )
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten()
    hist_s = cv2.calcHist([hsv], [1], None, [8], [0, 256]).flatten()
    hist_v = cv2.calcHist([hsv], [2], None, [8], [0, 256]).flatten()
    hist = np.concatenate([hist_h, hist_s, hist_v])
    hist = hist / (hist.sum() + 1e-6)
    return np.concatenate([hog_vec, hist]).astype(np.float32)


def build_training_set(
    objects_df,
    *,
    max_images: int = 900,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    tls = objects_df[objects_df["class"].str.startswith("traffic_light")]
    train = tls[tls["split"] == "train"]
    image_ids = train["image_id"].unique()
    rng = np.random.default_rng(seed)
    if len(image_ids) > max_images:
        image_ids = rng.choice(image_ids, size=max_images, replace=False)

    X_pos: list[np.ndarray] = []
    X_neg: list[np.ndarray] = []
    p = default_params()

    for image_id in image_ids:
        rows = train[train["image_id"] == image_id]
        path = rows.iloc[0]["image_path"]
        img = load_image(path)
        gts = rows.to_dict("records")
        gt_boxes = [box_dict(g) for g in gts]

        for g in gts:
            crop = crop_box(img, box_dict(g))
            X_pos.append(feature_vector(crop))

        cands = detect_color_blobs(img, p) + detect_housing(img, p)
        for c in cands:
            if all(box_iou(box_dict(c), gb) < 0.3 for gb in gt_boxes):
                crop = crop_box(img, c)
                X_neg.append(feature_vector(crop))

    if not X_pos or not X_neg:
        raise RuntimeError("Brak próbek do treningu SVM")
    # ogranicz negatywy
    if len(X_neg) > 3 * len(X_pos):
        idx = rng.choice(len(X_neg), size=3 * len(X_pos), replace=False)
        X_neg = [X_neg[i] for i in idx]

    X = np.vstack([np.stack(X_pos), np.stack(X_neg)])
    y = np.array([1] * len(X_pos) + [0] * len(X_neg))
    return X, y


def train_svm(objects_df, *, force: bool = False) -> Pipeline:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if MODEL_PATH.is_file() and not force:
        return joblib.load(MODEL_PATH)

    X, y = build_training_set(objects_df)
    clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svm", LinearSVC(class_weight="balanced", max_iter=5000)),
        ]
    )
    clf.fit(X, y)
    joblib.dump(clf, MODEL_PATH)
    return clf


def svm_scores(clf: Pipeline, img: np.ndarray, boxes: list[dict]) -> list[dict]:
    out = []
    for b in boxes:
        feat = feature_vector(crop_box(img, b))
        xs = clf.named_steps["scaler"].transform([feat])
        sc = float(clf.named_steps["svm"].decision_function(xs)[0])
        nb = dict(b)
        nb["score"] = sc
        out.append(nb)
    return out


def filter_by_svm(boxes: list[dict], thr: float = 0.0) -> list[dict]:
    return [b for b in boxes if float(b.get("score", 0)) >= thr]
