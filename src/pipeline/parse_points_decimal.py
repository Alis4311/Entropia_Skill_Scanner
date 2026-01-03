from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np


@dataclass(frozen=True)
class PointsDecimalResult:
    fraction: Optional[float]   # 0..1
    decimal: Optional[float]    # 0.00..1.00 (rounded to 2 decimals)
    confidence: float
    reason: str


# -----------------------------
# Fixed bar slice inside points_bgr (tune these once)
# -----------------------------

# X offsets (pixels) inside points_bgr
BAR_X1_OFF_PX = 12          # set this (e.g. 8, 12, 16) once you confirm
BAR_X2_OFF_PX = 15           # set as "trim from right" (e.g. 8, 10, 14)

# Y position inside points_bgr as fraction of height
BAR_Y_CENTER_FRAC = 0.75     # bar tends to sit around ~60-70% of row height
BAR_HEIGHT_PX = 7            # thin slice; 6–9 usually good

# Optional guardrail
ENABLE_TEAL_START_VALIDATION = True
TEAL_LEFT_STRIP_PX = 8
TEAL_MIN_LEFT_RATIO = 0.01   # how much teal must be present in left strip if bar has teal


def parse_points_decimal_from_bar(
    points_bgr: np.ndarray,
    *,
    sample_frac: float = 0.08,
    debug_dir: Union[str, Path, None] = None,
) -> PointsDecimalResult:
    """
    points_bgr -> decimal via progress bar fill ratio (LAB prototype distances).
    Now uses fixed-offset bar ROI (no detection), with optional teal-start validation.
    """
    if points_bgr is None or points_bgr.size == 0:
        return PointsDecimalResult(None, None, 0.0, "empty input")

    H, W = points_bgr.shape[:2]

    # Build bar ROI deterministically
    bx1 = int(np.clip(BAR_X1_OFF_PX, 0, W - 2))
    bx2 = int(np.clip(W - BAR_X2_OFF_PX, bx1 + 2, W))

    yc = int(round(BAR_Y_CENTER_FRAC * H))
    half = max(2, BAR_HEIGHT_PX // 2)
    by1 = int(np.clip(yc - half, 0, H - 2))
    by2 = int(np.clip(yc + half + 1, by1 + 1, H))

    bar_roi = (bx1, by1, bx2, by2)

    dbg = Path(debug_dir) if debug_dir else None
    if dbg:
        dbg.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dbg / "00_points_roi.png"), points_bgr)
        vis = points_bgr.copy()
        cv2.rectangle(vis, (bx1, by1), (bx2 - 1, by2 - 1), (0, 255, 255), 2)
        cv2.imwrite(str(dbg / "01_bar_roi_fixed.png"), vis)
        cv2.imwrite(str(dbg / "02_bar_crop.png"), points_bgr[by1:by2, bx1:bx2])

    # Optional teal-start validation (guardrail, not a detector)
    if ENABLE_TEAL_START_VALIDATION:
        ok, teal_meta = _validate_teal_start(points_bgr, bar_roi)
        if dbg:
            _write_teal_debug(points_bgr, bar_roi, dbg, teal_meta)

        if not ok:
            # Still attempt parse (sometimes 0.00 has no teal),
            # but mark low confidence and reason.
            # If you prefer hard-fail, return here instead.
            frac, dec = None, None
            try:
                frac_val = extract_decimal_from_bar(points_bgr, bar_roi, sample_frac=sample_frac, debug_dir=dbg)
                frac_val = float(np.clip(frac_val, 0.0, 1.0))
                frac, dec = frac_val, round(frac_val, 2)
            except Exception:
                pass

            return PointsDecimalResult(frac, dec, 0.25, f"teal-start failed: {teal_meta.get('reason','?')}")

    # Compute fill ratio using LAB prototype method
    try:
        frac = extract_decimal_from_bar(points_bgr, bar_roi, sample_frac=sample_frac, debug_dir=dbg)
    except Exception as e:
        return PointsDecimalResult(None, None, 0.0, f"bar parse failed: {e}")

    frac = float(np.clip(frac, 0.0, 1.0))
    dec = round(frac, 2)

    # Confidence: based on ROI width and teal validation (if enabled)
    bar_w = max(1, bx2 - bx1)
    conf = float(np.clip(bar_w / 200.0, 0.4, 1.0))

    return PointsDecimalResult(frac, dec, conf, "ok")


# -----------------------------
# Optional teal-start validation helpers
# -----------------------------
def _teal_mask(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([70, 60, 40], dtype=np.uint8)
    upper = np.array([105, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    # light cleanup
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    return mask


def _validate_teal_start(points_bgr: np.ndarray, bar_roi: Tuple[int, int, int, int]) -> Tuple[bool, dict]:
    x1, y1, x2, y2 = bar_roi
    crop = points_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return False, {"reason": "empty_bar_crop"}

    mask = _teal_mask(crop)
    teal_ratio = float((mask > 0).mean())

    # If there's basically no teal anywhere, bar might be empty (0.00) — don't fail hard.
    if teal_ratio < 0.01:
        return True, {"reason": "no_teal_detected_ok", "teal_ratio": teal_ratio}

    L = min(TEAL_LEFT_STRIP_PX, mask.shape[1])
    left = mask[:, :L]
    left_ratio = float((left > 0).mean())

    ok = left_ratio >= TEAL_MIN_LEFT_RATIO
    return ok, {
        "reason": "ok" if ok else "left_strip_low",
        "teal_ratio": teal_ratio,
        "left_ratio": left_ratio,
        "L": L,
    }


def _write_teal_debug(points_bgr: np.ndarray, bar_roi: Tuple[int, int, int, int], dbg: Path, meta: dict) -> None:
    x1, y1, x2, y2 = bar_roi
    crop = points_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return
    mask = _teal_mask(crop)
    cv2.imwrite(str(dbg / "03_teal_mask.png"), mask)
    vis = crop.copy()
    L = min(TEAL_LEFT_STRIP_PX, vis.shape[1])
    cv2.rectangle(vis, (0, 0), (L - 1, vis.shape[0] - 1), (0, 255, 255), 1)
    cv2.putText(vis, f"{meta}", (5, max(12, vis.shape[0] - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(dbg / "04_teal_start_vis.png"), vis)


# -----------------------------
# Your existing method stays the same
# -----------------------------
def extract_decimal_from_bar(
    image: np.ndarray,
    bar_roi: Tuple[int, int, int, int],
    *,
    sample_frac: float = 0.08,
    debug_dir: Optional[Path] = None,
) -> float:
    x1, y1, x2, y2 = bar_roi
    bar = image[y1:y2, x1:x2]

    h, w = bar.shape[:2]
    if w < 10 or h < 4:
        raise ValueError(f"Bar ROI too small: {bar.shape}")

    band_half = max(1, h // 6)
    cy = h // 2
    y0 = max(0, cy - band_half)
    y1b = min(h, cy + band_half + 1)
    band = bar[y0:y1b, :, :]

    lab = cv2.cvtColor(band, cv2.COLOR_BGR2LAB).astype(np.float32)
    ab = lab[:, :, 1:3]
    feat = ab.mean(axis=0)

    n = max(3, int(sample_frac * w))
    left_proto = feat[:n].mean(axis=0)
    right_proto = feat[-n:].mean(axis=0)

    d_left = np.linalg.norm(feat - left_proto, axis=1)
    d_right = np.linalg.norm(feat - right_proto, axis=1)

    filled_like = (d_left <= d_right).astype(np.uint8)

    k = max(3, (w // 50) * 2 + 1)
    kernel = np.ones((k,), dtype=np.uint8)
    filled_like = cv2.morphologyEx(filled_like.reshape(1, -1), cv2.MORPH_CLOSE, kernel).ravel()
    filled_like = cv2.morphologyEx(filled_like.reshape(1, -1), cv2.MORPH_OPEN, kernel).ravel()

    if filled_like.sum() == 0:
        fill_ratio = 0.0
    elif filled_like.sum() == w:
        fill_ratio = 1.0
    else:
        first_zero_after_left_run = int(np.argmax(filled_like == 0))
        if first_zero_after_left_run == 0:
            diff = np.diff(np.r_[0, filled_like, 0])
            starts = np.where(diff == 1)[0]
            ends = np.where(diff == -1)[0]
            lengths = ends - starts
            boundary = int(ends[np.argmax(lengths)])
        else:
            boundary = first_zero_after_left_run
        fill_ratio = boundary / float(w)

    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)
        vis = bar.copy()
        bx = int(round(fill_ratio * (w - 1)))
        cv2.line(vis, (bx, 0), (bx, h - 1), (0, 255, 0), 2)
        cv2.putText(
            vis,
            f"fill={fill_ratio:.3f}",
            (5, max(15, h - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.imwrite(str(debug_dir / "05_bar_fill_vis.png"), vis)

        fl = (filled_like * 255).astype(np.uint8).reshape(1, -1)
        fl = cv2.resize(fl, (w, 40), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(debug_dir / "06_filled_like.png"), fl)

    return float(fill_ratio)
