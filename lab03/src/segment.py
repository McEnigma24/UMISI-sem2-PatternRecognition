"""Segmentacja owocu od tła — metody klasyczne (progowanie, morfologia, GrabCut)."""

from __future__ import annotations

import cv2
import numpy as np
from skimage import measure, morphology
from skimage.segmentation import mark_boundaries, slic

WATERMARK_CROP = 0.88
MIN_FG_RATIO = 0.025
MAX_FG_RATIO = 0.78

# SLIC + filtr okrągłości (początkowa segmentacja owocu)
SLIC_COMPACTNESS = 12.0
MIN_CIRCULARITY = 0.55
MIN_ROUND_AREA_FRAC = 0.01


def _closing(mask: np.ndarray, radius: int) -> np.ndarray:
    return morphology.closing(mask, morphology.disk(radius))


def _opening(mask: np.ndarray, radius: int) -> np.ndarray:
    return morphology.opening(mask, morphology.disk(radius))


def crop_watermark_band(img: np.ndarray) -> tuple[np.ndarray, int]:
    """Obetnij dolny pasek (typowe watermarki stock). Zwraca obraz i wysokość roboczą."""
    h = img.shape[0]
    crop_h = max(1, int(h * WATERMARK_CROP))
    return img[:crop_h], crop_h


def mask_white_background(img: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    bg = (hsv[:, :, 1] < 40) & (hsv[:, :, 2] > 200)
    bg = _closing(bg, 5)
    fg = ~bg
    fg = morphology.remove_small_objects(fg, max_size=30)
    return fg


def mask_fruit_colors(img: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    red = ((h < 15) | (h > 165)) & (s > 35) & (v > 40)
    green = (h >= 25) & (h <= 95) & (s > 30) & (v > 35)
    yellow = (h >= 15) & (h < 35) & (s > 35) & (v > 40)
    fg = red | green | yellow
    fg = _closing(fg, 7)
    fg = _opening(fg, 3)
    fg = morphology.remove_small_objects(fg, max_size=80)
    return fg


def _border_touching(labels: np.ndarray, label_id: int, margin: int) -> bool:
    h, w = labels.shape
    my = max(1, margin)
    mx = max(1, margin)
    edge = np.concatenate(
        [
            labels[:my, :].ravel(),
            labels[-my:, :].ravel(),
            labels[:, :mx].ravel(),
            labels[:, -mx:].ravel(),
        ]
    )
    return label_id in edge


def _circularity(mask: np.ndarray) -> float:
    """1 = koło, mniejsze = bardziej wydłużone / poszarpane."""
    mask_u8 = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return 0.0
    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    peri = cv2.arcLength(contour, True)
    if area <= 0 or peri <= 0:
        return 0.0
    return float(4 * np.pi * area / (peri * peri))


def _slic_n_segments(h: int, w: int) -> int:
    """Liczba superpikseli ~ stała gęstość niezależnie od rozdzielczości."""
    return int(np.clip((h * w) ** 0.5 / 4.5, 120, 450))


def mask_superpixel_round(img: np.ndarray) -> np.ndarray:
    """
    SLIC → superpixele pierwszoplanowe (kolor vs tło z ramki) →
    spójne regiony → wybór najbardziej okrągłego (jabłko/pomidor w kadrze).
    """
    h, w = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    n_seg = _slic_n_segments(h, w)
    labels = slic(
        rgb,
        n_segments=n_seg,
        compactness=SLIC_COMPACTNESS,
        start_label=0,
        channel_axis=-1,
    )

    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    margin_y = max(2, int(h * 0.08))
    margin_x = max(2, int(w * 0.08))
    border = np.zeros((h, w), dtype=bool)
    border[:margin_y, :] = border[-margin_y:, :] = True
    border[:, :margin_x] = border[:, -margin_x:] = True

    border_pixels = lab_img[border]
    bg_color = np.median(border_pixels, axis=0)
    border_diff = np.linalg.norm(border_pixels - bg_color, axis=1)
    color_thr = float(np.percentile(border_diff, 92) + 6.0)

    fg_labels: set[int] = set()
    for label_id in np.unique(labels):
        if label_id < 0:
            continue
        sp_mask = labels == label_id
        if _border_touching(labels, int(label_id), margin_y):
            continue
        mean_color = lab_img[sp_mask].mean(axis=0)
        if np.linalg.norm(mean_color - bg_color) > color_thr:
            fg_labels.add(int(label_id))

    if not fg_labels:
        return np.zeros((h, w), dtype=bool)

    fg_seed = np.isin(labels, list(fg_labels))
    components = measure.label(fg_seed)
    if components.max() == 0:
        return fg_seed

    min_area = h * w * MIN_ROUND_AREA_FRAC
    best_mask = np.zeros((h, w), dtype=bool)
    best_score = -1.0

    for region in measure.regionprops(components):
        if region.area < min_area:
            continue
        comp_mask = components == region.label
        circ = _circularity(comp_mask)
        if circ < MIN_CIRCULARITY:
            continue
        # preferuj okrągłość, ale nie ignoruj całkiem większego owocu
        score = circ * (region.area**0.15)
        if score > best_score:
            best_score = score
            best_mask = comp_mask

    if best_mask.any():
        return best_mask

    # brak regionu powyżej progu okrągłości — weź najokrągalszy większy blob
    for region in sorted(measure.regionprops(components), key=lambda r: r.area, reverse=True):
        if region.area < min_area:
            continue
        comp_mask = components == region.label
        circ = _circularity(comp_mask)
        score = circ * (region.area**0.15)
        if score > best_score:
            best_score = score
            best_mask = comp_mask
    return best_mask


def mask_lab_distance(img: np.ndarray) -> np.ndarray:
    """Tło z ramki kadru — piksele daleko od mediany Lab na brzegu."""
    h, w = img.shape[:2]
    margin_y = max(2, int(h * 0.08))
    margin_x = max(2, int(w * 0.08))
    border = np.zeros((h, w), dtype=bool)
    border[:margin_y, :] = border[-margin_y:, :] = True
    border[:, :margin_x] = border[:, -margin_x:] = True

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    border_pixels = lab[border]
    bg_color = np.median(border_pixels, axis=0)
    diff = np.linalg.norm(lab - bg_color, axis=2)

    border_diff = diff[border]
    thr = float(np.percentile(border_diff, 95) + 8.0)
    fg = diff > thr
    fg = _closing(fg, 5)
    fg = _opening(fg, 3)
    fg = morphology.remove_small_objects(fg, max_size=150)
    fg = morphology.remove_small_holes(fg, max_size=400)
    return fg


def foreground_ratio(mask: np.ndarray) -> float:
    return float(mask.sum()) / mask.size


def keep_main_components(mask: np.ndarray, min_frac: float = 0.015, top: int = 6) -> np.ndarray:
    labeled = measure.label(mask)
    if labeled.max() == 0:
        return mask
    regions = sorted(measure.regionprops(labeled), key=lambda r: r.area, reverse=True)
    main_area = regions[0].area
    out = np.zeros_like(mask, dtype=bool)
    for r in regions[:top]:
        if r.area >= main_area * min_frac:
            out[labeled == r.label] = True
    return out


def refine_grabcut(img: np.ndarray, init: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    ys, xs = np.where(init)
    if len(xs) < 10:
        mx, my = int(w * 0.08), int(h * 0.08)
        rect = (mx, my, w - 2 * mx, h - 2 * my)
        gc = np.full((h, w), cv2.GC_BGD, dtype=np.uint8)
        gc[my : my + rect[3], mx : mx + rect[2]] = cv2.GC_PR_FGD
        mode = cv2.GC_INIT_WITH_RECT
    else:
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
        pad = 10
        rect = (
            max(0, x0 - pad),
            max(0, y0 - pad),
            min(w - 1, x1 - x0 + 2 * pad),
            min(h - 1, y1 - y0 + 2 * pad),
        )
        gc = np.where(init, cv2.GC_PR_FGD, cv2.GC_BGD).astype(np.uint8)
        mode = cv2.GC_INIT_WITH_MASK

    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(img, gc, rect, bgd, fgd, 5, mode)
        return np.isin(gc, (cv2.GC_FGD, cv2.GC_PR_FGD))
    except cv2.error:
        return init


def _pick_initial_mask(work: np.ndarray) -> np.ndarray:
    sp = keep_main_components(mask_superpixel_round(work))
    if MIN_FG_RATIO < foreground_ratio(sp) < MAX_FG_RATIO:
        return sp

    candidates: list[tuple[str, np.ndarray]] = [
        ("superpixel", sp),
        ("white", mask_white_background(work)),
        ("color", mask_fruit_colors(work)),
        ("lab", mask_lab_distance(work)),
    ]
    scored: list[tuple[float, np.ndarray]] = []
    for _, m in candidates:
        m = keep_main_components(m)
        ar = foreground_ratio(m)
        if MIN_FG_RATIO < ar < MAX_FG_RATIO:
            scored.append((abs(ar - 0.35), m))
    if scored:
        scored.sort(key=lambda t: t[0])
        return scored[0][1]
    # fallback: największa sensowna z kandydatów
    best = candidates[0][1]
    best_ar = -1.0
    for _, m in candidates:
        m = keep_main_components(m)
        ar = foreground_ratio(m)
        if MIN_FG_RATIO < ar and ar > best_ar:
            best_ar = ar
            best = m
    return best


def superpixel_round_debug(img_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Zwraca: etykiety SLIC, maskę wybranego okrągłego regionu, obraz z konturami SLIC (RGB)."""
    work, _ = crop_watermark_band(img_bgr)
    rgb = cv2.cvtColor(work, cv2.COLOR_BGR2RGB)
    h, w = work.shape[:2]
    labels = slic(
        rgb,
        n_segments=_slic_n_segments(h, w),
        compactness=SLIC_COMPACTNESS,
        start_label=0,
        channel_axis=-1,
    )
    mask = mask_superpixel_round(work)
    overlay = (mark_boundaries(rgb, labels, color=(1, 1, 0), mode="thick") * 255).astype(np.uint8)
    sel = rgb.copy()
    sel[~mask] = (sel[~mask] * 0.2).astype(np.uint8)
    return labels, mask, overlay


def segment_foreground(img_bgr: np.ndarray) -> tuple[np.ndarray, int]:
    """Maska bool (H×W) na pełnej wysokości obrazu; drugi element = wysokość po obcięciu watermarku."""
    work, crop_h = crop_watermark_band(img_bgr)
    init = _pick_initial_mask(work)
    refined = keep_main_components(refine_grabcut(work, init))
    refined = morphology.remove_small_holes(refined, max_size=300)

    full = np.zeros(img_bgr.shape[:2], dtype=bool)
    full[:crop_h] = refined
    return full, crop_h
