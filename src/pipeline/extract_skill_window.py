from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union, List

import cv2
import numpy as np

BBox = Tuple[int, int, int, int]  # x, y, w, h


# -----------------------------
# Result (frozen contract)
# -----------------------------
@dataclass
class CropResult:
    crop_bgr: Optional[np.ndarray]   # cropped (trimmed) skills window
    norm_bgr: Optional[np.ndarray]   # normalized version of crop_bgr
    bbox: Optional[BBox]             # bbox in original screenshot
    confidence: float
    margin: float
    reason: str
    scale: float                     # norm_width / crop_width (1.0 if unchanged)


# -----------------------------
# Public API (in-memory only)
# -----------------------------
def extract_skill_window(
    bgr: np.ndarray,
    *,
    norm_width: int = 1400,
    debug_dir: Union[str, Path, None] = None,
) -> CropResult:
    """
    Full screenshot (BGR) -> normalized skills window crop.

    Frozen pipeline stage.
    - No OCR.
    - Robust window detection + inner-panel trim + normalize-to-width.
    - Debug output is optional and minimal.
    """
    det = _detect_skills_window(bgr)

    if det["bbox"] is None:
        return CropResult(
            crop_bgr=None,
            norm_bgr=None,
            bbox=None,
            confidence=float(det["confidence"]),
            margin=float(det["margin"]),
            reason=det["reason"],
            scale=0.0,
        )

    x, y, w, h = det["bbox"]
    crop = bgr[y : y + h, x : x + w].copy()

    # Remove decorative border by snapping to inner dark panel
    crop = _trim_to_inner_panel(crop)

    # Normalize to fixed width
    norm, scale = _normalize_to_width(crop, target_width=norm_width)

    # Minimal debug artifacts (only if debug_dir is given)
    if debug_dir is not None:
        dbg = Path(debug_dir)
        dbg.mkdir(parents=True, exist_ok=True)

        vis = bgr.copy()
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.imwrite(str(dbg / "01_detected_bbox.png"), vis)
        cv2.imwrite(str(dbg / "02_crop_trimmed.png"), crop)
        cv2.imwrite(str(dbg / "03_crop_normalized.png"), norm)

    return CropResult(
        crop_bgr=crop,
        norm_bgr=norm,
        bbox=det["bbox"],
        confidence=float(det["confidence"]),
        margin=float(det["margin"]),
        reason=det["reason"],
        scale=float(scale),
    )


# -----------------------------
# Normalization
# -----------------------------
def _normalize_to_width(
    crop_bgr: np.ndarray,
    *,
    target_width: int = 1400,
    keep_aspect: bool = True,
    interp_down=cv2.INTER_AREA,
    interp_up=cv2.INTER_CUBIC,
) -> Tuple[np.ndarray, float]:
    if crop_bgr is None or crop_bgr.size == 0:
        raise ValueError("crop_bgr is empty")

    h, w = crop_bgr.shape[:2]
    if w == target_width:
        return crop_bgr, 1.0

    scale = target_width / float(w)
    new_w = target_width
    new_h = int(round(h * scale)) if keep_aspect else h

    interp = interp_down if scale < 1.0 else interp_up
    norm = cv2.resize(crop_bgr, (new_w, new_h), interpolation=interp)
    return norm, float(scale)


# -----------------------------
# Border trim (inner panel)
# -----------------------------
def _trim_to_inner_panel(bgr: np.ndarray) -> np.ndarray:
    """
    Remove decorative border by snapping to the largest dark inner panel.
    Safe: returns original if detection looks wrong.
    """
    if bgr is None or bgr.size == 0:
        return bgr

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Dark panel -> low luminance => invert + Otsu to get dark regions as white
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Clean up
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return bgr

    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)

    # Safety: require that this "inner panel" is most of the crop
    H, W = bgr.shape[:2]
    if w < 0.60 * W or h < 0.60 * H:
        return bgr

    # Small inward nudge to avoid including a 1px border from thresholding
    inset = max(1, int(0.002 * min(W, H)))
    x1 = min(W - 1, max(0, x + inset))
    y1 = min(H - 1, max(0, y + inset))
    x2 = max(x1 + 1, min(W, x + w - inset))
    y2 = max(y1 + 1, min(H, y + h - inset))

    return bgr[y1:y2, x1:x2]


# -----------------------------
# Detection core (frozen)
# -----------------------------
def _detect_skills_window(bgr: np.ndarray) -> dict:
    """
    Internal detector. Returns:
      { bbox, confidence, margin, reason }
    """
    if bgr is None or bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError("bgr must be an HxWx3 BGR image (uint8).")

    # Frozen parameters (these were your proven defaults)
    min_area_frac = 0.05
    max_area_frac = 0.90
    ar_min = 1.00
    ar_max = 2.20
    max_candidates = 25
    conf_threshold = 0.55
    margin_threshold = 0.08
    pad_frac = 0.0  # disabled; we trim inner panel instead

    H, W = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray_blur, 60, 160)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    candidates: List[BBox] = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)

        if w > int(0.95 * W) or h > int(0.95 * H):
            continue

        area_frac = (w * h) / float(W * H)
        ar = w / float(h + 1e-6)

        if not (min_area_frac <= area_frac <= max_area_frac):
            continue
        if not (ar_min <= ar <= ar_max):
            continue
        if w < int(0.20 * W) or h < int(0.15 * H):
            continue

        candidates.append((x, y, w, h))

    if not candidates:
        candidates = _fallback_candidates(gray, W, H, min_area_frac, max_area_frac, ar_min, ar_max)

    if candidates:
        candidates = sorted(candidates, key=lambda bb: bb[2] * bb[3], reverse=True)[:max_candidates]

    if not candidates:
        return {
            "bbox": None,
            "confidence": 0.0,
            "margin": 0.0,
            "reason": "no candidates after filtering",
        }

    scores: List[float] = [float(_score_candidate(gray, bb)) for bb in candidates]
    order = np.argsort(scores)[::-1]
    best_i = int(order[0])
    second_i = int(order[1]) if len(order) > 1 else -1

    best_bbox = candidates[best_i]
    best_score = float(scores[best_i])
    second_score = float(scores[second_i]) if second_i != -1 else 0.0
    margin = float(best_score - second_score)

    refined = _refine_bbox(gray, edges, best_bbox)
    refined = _pad_bbox(refined, W, H, pad_frac=pad_frac)

    refined_score = float(_score_candidate(gray, refined))
    confidence = float(max(best_score, refined_score))
    final_bbox = refined if refined_score >= best_score * 0.95 else best_bbox

    reasons = []
    if confidence < conf_threshold:
        reasons.append("low confidence")
    if margin < margin_threshold:
        reasons.append(f"low margin ({margin:.3f})")
    reason = "ok" if not reasons else "; ".join(reasons)

    return {
        "bbox": final_bbox,
        "confidence": confidence,
        "margin": margin,
        "reason": reason,
    }


# -----------------------------
# Scoring helpers (no OCR)
# -----------------------------
def _score_candidate(gray: np.ndarray, bbox: BBox) -> float:
    x, y, w, h = bbox
    roi = gray[y : y + h, x : x + w]
    if roi.size == 0:
        return 0.0

    roi_f = roi.astype(np.float32) / 255.0
    mean_lum = float(np.mean(roi_f))
    s_dark = _soft_step(0.45 - mean_lum, lo=0.00, hi=0.25)

    top_h = max(8, int(0.08 * h))
    s_topline = _score_horizontal_line(roi[:top_h, :])

    left_w = max(20, int(0.28 * w))
    s_left = _score_edge_density(roi[:, :left_w], target=0.06)

    right_w = max(20, int(0.28 * w))
    s_right = _score_horizontal_bariness(roi[:, w - right_w :])

    ar = w / float(h + 1e-6)
    s_ar = float(np.exp(-((ar - 1.45) ** 2) / (2 * (0.25**2))))

    score = 0.20 * s_dark + 0.25 * s_topline + 0.20 * s_left + 0.25 * s_right + 0.10 * s_ar
    return float(np.clip(score, 0.0, 1.0))


def _score_horizontal_line(strip: np.ndarray) -> float:
    if strip.size == 0:
        return 0.0
    strip_blur = cv2.GaussianBlur(strip, (3, 3), 0)
    gy = cv2.Sobel(strip_blur, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.abs(gy)
    row_energy = np.mean(mag, axis=1)
    if row_energy.size < 3:
        return 0.0
    peak = float(np.max(row_energy))
    median = float(np.median(row_energy) + 1e-6)
    ratio = peak / median
    peak_row = int(np.argmax(row_energy))
    pos_ok = 1.0 if peak_row < int(0.70 * strip.shape[0]) else 0.6
    s = _soft_step(ratio - 2.5, lo=0.0, hi=4.0) * pos_ok
    return float(np.clip(s, 0.0, 1.0))


def _score_edge_density(region: np.ndarray, target: float = 0.06) -> float:
    if region.size == 0:
        return 0.0
    r = cv2.GaussianBlur(region, (3, 3), 0)
    e = cv2.Canny(r, 60, 160)
    density = float(np.mean(e > 0))
    s = float(np.exp(-((density - target) ** 2) / (2 * (target * 0.8 + 1e-6) ** 2)))
    return float(np.clip(s, 0.0, 1.0))


def _score_horizontal_bariness(region: np.ndarray) -> float:
    if region.size == 0:
        return 0.0
    r = cv2.GaussianBlur(region, (3, 3), 0)
    gx = cv2.Sobel(r, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(r, cv2.CV_32F, 0, 1, ksize=3)
    hx = float(np.mean(np.abs(gy)))
    vx = float(np.mean(np.abs(gx)))
    ratio = hx / (vx + 1e-6)
    s = _soft_step(ratio - 1.2, lo=0.0, hi=1.5)
    return float(np.clip(s, 0.0, 1.0))


# -----------------------------
# Geometry / fallback / misc
# -----------------------------
def _refine_bbox(gray: np.ndarray, edges: np.ndarray, bbox: BBox) -> BBox:
    H, W = gray.shape[:2]
    x, y, w, h = bbox
    x2, y2 = x + w, y + h

    pad = int(0.03 * min(W, H))
    rx1 = max(0, x - pad)
    ry1 = max(0, y - pad)
    rx2 = min(W, x2 + pad)
    ry2 = min(H, y2 + pad)

    e = edges[ry1:ry2, rx1:rx2]
    if e.size == 0:
        return bbox

    col_sum = np.sum(e > 0, axis=0).astype(np.float32)
    row_sum = np.sum(e > 0, axis=1).astype(np.float32)

    def pick_edge(pos: int, vec: np.ndarray, win: int) -> int:
        a = max(0, pos - win)
        b = min(len(vec), pos + win)
        if b <= a + 1:
            return pos
        return int(a + np.argmax(vec[a:b]))

    winx = max(10, int(0.06 * e.shape[1]))
    winy = max(10, int(0.06 * e.shape[0]))

    ex1 = x - rx1
    ex2 = x2 - rx1
    ey1 = y - ry1
    ey2 = y2 - ry1

    nx1 = pick_edge(ex1, col_sum, winx)
    nx2 = pick_edge(ex2, col_sum, winx)
    ny1 = pick_edge(ey1, row_sum, winy)
    ny2 = pick_edge(ey2, row_sum, winy)

    fx1 = int(np.clip(rx1 + min(nx1, nx2), 0, W - 1))
    fx2 = int(np.clip(rx1 + max(nx1, nx2), 0, W - 1))
    fy1 = int(np.clip(ry1 + min(ny1, ny2), 0, H - 1))
    fy2 = int(np.clip(ry1 + max(ny1, ny2), 0, H - 1))

    return (fx1, fy1, max(1, fx2 - fx1), max(1, fy2 - fy1))


def _pad_bbox(bbox: BBox, W: int, H: int, pad_frac: float) -> BBox:
    if pad_frac <= 0:
        return bbox
    x, y, w, h = bbox
    pad = int(pad_frac * min(W, H))
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(W, x + w + pad)
    y2 = min(H, y + h + pad)
    return (x1, y1, max(1, x2 - x1), max(1, y2 - y1))


def _fallback_candidates(
    gray: np.ndarray,
    W: int,
    H: int,
    min_area_frac: float,
    max_area_frac: float,
    ar_min: float,
    ar_max: float,
) -> List[BBox]:
    inv = 255 - gray
    thr = cv2.adaptiveThreshold(inv, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 51, -5)
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=2)

    contours, _ = cv2.findContours(thr, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    out: List[BBox] = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > int(0.95 * W) or h > int(0.95 * H):
            continue
        area_frac = (w * h) / float(W * H)
        ar = w / float(h + 1e-6)
        if min_area_frac <= area_frac <= max_area_frac and ar_min <= ar <= ar_max:
            out.append((x, y, w, h))
    return sorted(out, key=lambda bb: bb[2] * bb[3], reverse=True)


def _soft_step(x: float, lo: float, hi: float) -> float:
    if x <= lo:
        return 0.0
    if x >= hi:
        return 1.0
    return float((x - lo) / (hi - lo + 1e-6))
