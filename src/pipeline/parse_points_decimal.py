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
BAR_X1_OFF_PX = 12
BAR_X2_OFF_PX = 15

BAR_Y_CENTER_FRAC = 0.80
BAR_HEIGHT_PX = 7


# -----------------------------
# Teal detection (HSV)
# -----------------------------
# These are intentionally conservative. Adjust H/S/V only if debug proves needed.
HSV_TEAL_LOWER = (70, 70, 40)
HSV_TEAL_UPPER = (105, 255, 255)

# Small cleanup (bar is only 7px tall, so keep kernels small)
MORPH_OPEN_K = (3, 3)
MORPH_CLOSE_K = (3, 3)


# -----------------------------
# Robust fill logic tuned for ~147x7 crops
# -----------------------------
# A column counts as "teal" if >= this many pixels are teal in that column.
# For a 7px tall crop, 2 pixels is a good "real signal vs speckle" threshold.
COL_TEAL_MIN_PX = 2

# Left anchor: require at least one teal column in the first N columns to consider non-empty.
LEFT_PROBE_PX = 10  # ~7% of 147

# Full bar: if >= this fraction of columns are teal, treat as full.
FULL_COL_FRAC = 0.99

# If teal exists but is not left-anchored, we treat it as noise/ambiguous.
# Optionally allow a tiny left-edge glitch.
LEFT_START_TOL_PX = 3

# Noise check: if we have teal but it’s extremely fragmented (many tiny CCs), fail.
MAX_CC_COUNT = 6
MIN_LARGEST_CC_PX = 25  # with 147x7, real fills create a CC bigger than this


def parse_points_decimal_from_bar(
    points_bgr: np.ndarray,
    *,
    debug_dir: Union[str, Path, None] = None,
) -> PointsDecimalResult:
    """
    points_bgr -> decimal via progress bar fill ratio using HSV teal mask and
    left-anchored column boundary. Designed for small bar crops (~147x7).
    """
    if points_bgr is None or points_bgr.size == 0:
        return PointsDecimalResult(None, None, 0.0, "empty input")

    H, W = points_bgr.shape[:2]

    # Deterministic bar ROI
    bx1 = int(np.clip(BAR_X1_OFF_PX, 0, W - 2))
    bx2 = int(np.clip(W - BAR_X2_OFF_PX, bx1 + 2, W))

    yc = int(round(BAR_Y_CENTER_FRAC * H))
    half = max(2, BAR_HEIGHT_PX // 2)
    by1 = int(np.clip(yc - half, 0, H - 2))
    by2 = int(np.clip(yc + half + 1, by1 + 1, H))

    bar_roi = (bx1, by1, bx2, by2)
    bar = points_bgr[by1:by2, bx1:bx2]
    if bar.size == 0:
        return PointsDecimalResult(None, None, 0.0, "empty_bar_crop")

    dbg = Path(debug_dir) if debug_dir else None
    if dbg:
        dbg.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dbg / "00_points_roi.png"), points_bgr)
        vis = points_bgr.copy()
        cv2.rectangle(vis, (bx1, by1), (bx2 - 1, by2 - 1), (0, 255, 255), 2)
        cv2.imwrite(str(dbg / "01_bar_roi_fixed.png"), vis)
        cv2.imwrite(str(dbg / "02_bar_crop.png"), bar)

    mask = _teal_mask(bar)

    # Connected components stats (for noise gating)
    cc_count, largest_cc = _cc_stats(mask)

    # Column-wise teal
    teal_per_col = (mask > 0).sum(axis=0)  # shape (w,)
    col_is_teal = teal_per_col >= COL_TEAL_MIN_PX
    w = col_is_teal.size

    teal_cols = int(col_is_teal.sum())
    teal_col_frac = teal_cols / max(1, w)

    # Debug saves
    if dbg:
        cv2.imwrite(str(dbg / "03_teal_mask.png"), mask)
        _write_fill_debug(dbg, bar, mask, col_is_teal, 0.0, reason="pre")

    # Hard zero: no teal columns at all
    if teal_cols == 0:
        if dbg:
            _write_fill_debug(dbg, bar, mask, col_is_teal, 0.0, reason="zero_no_teal")
        return PointsDecimalResult(0.0, 0.00, 1.0, "zero_no_teal")

    # Hard full: almost all columns teal
    if teal_col_frac >= FULL_COL_FRAC:
        if dbg:
            _write_fill_debug(dbg, bar, mask, col_is_teal, 0.99, reason="full_by_cols")
        return PointsDecimalResult(0.99, 0.99, 1.0, "full_by_cols")

    # Noise/fragmentation check: teal exists but looks like speckle
    if cc_count > MAX_CC_COUNT and largest_cc < MIN_LARGEST_CC_PX:
        if dbg:
            _write_fill_debug(dbg, bar, mask, col_is_teal, 0.0, reason=f"mismatch_noise_cc{cc_count}_lc{largest_cc}")
        return PointsDecimalResult(None, None, 0.0, "MISMATCH: teal_noise_fragmented")

    # Left anchoring: must see teal near the left, else likely false positives
    left_probe = min(LEFT_PROBE_PX, w)
    if not col_is_teal[:left_probe].any():
        if dbg:
            _write_fill_debug(dbg, bar, mask, col_is_teal, 0.0, reason="zero_not_left_anchored")
        # This is usually “empty bar + noise”. Safer to snap to 0.00 than guess.
        return PointsDecimalResult(0.0, 0.00, 0.9, "zero_not_left_anchored")

    # Find start of fill (allow tiny left-edge glitch)
    ones = np.where(col_is_teal)[0]
    start = int(ones[0])
    if start > LEFT_START_TOL_PX:
        if dbg:
            _write_fill_debug(dbg, bar, mask, col_is_teal, 0.0, reason=f"mismatch_start_{start}")
        return PointsDecimalResult(None, None, 0.0, "MISMATCH: teal_not_left_starting")

    # Boundary = first non-teal after left run (starting from `start`)
    tail = col_is_teal[start:]
    zeros = np.where(~tail)[0]
    boundary = (start + int(zeros[0])) if zeros.size > 0 else w

    fill_ratio = float(np.clip(boundary / max(1, w), 0.0, 1.0))
    fill_ratio = min(fill_ratio, 0.99)
    dec = round(fill_ratio, 2)

    # Simple confidence: anchored + non-noisy => high
    conf = 0.95
    reason = "ok"

    if dbg:
        _write_fill_debug(dbg, bar, mask, col_is_teal, fill_ratio, reason="ok")

    return PointsDecimalResult(fill_ratio, dec, conf, reason)


def _teal_mask(bar_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bar_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array(HSV_TEAL_LOWER, dtype=np.uint8)
    upper = np.array(HSV_TEAL_UPPER, dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    k1 = cv2.getStructuringElement(cv2.MORPH_RECT, MORPH_OPEN_K)
    k2 = cv2.getStructuringElement(cv2.MORPH_RECT, MORPH_CLOSE_K)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k1, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k2, iterations=1)

    return mask


def _cc_stats(mask: np.ndarray) -> Tuple[int, int]:
    # returns (component_count_excluding_bg, largest_component_area)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        return 0, 0
    areas = stats[1:, cv2.CC_STAT_AREA]
    return int(len(areas)), int(areas.max()) if len(areas) else 0


def _write_fill_debug(
    dbg: Path,
    bar: np.ndarray,
    mask: np.ndarray,
    col_is_teal: np.ndarray,
    fill_ratio: float,
    *,
    reason: str,
) -> None:
    h, w = bar.shape[:2]
    bx = int(round(fill_ratio * (w - 1)))

    vis = bar.copy()
    cv2.line(vis, (bx, 0), (bx, h - 1), (0, 255, 0), 2)
    cv2.putText(
        vis,
        f"fill={fill_ratio:.3f} {reason}",
        (2, max(12, h - 2)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(dbg / "05_bar_fill_vis.png"), vis)

    # Column signal visualization
    sig = (col_is_teal.astype(np.uint8) * 255).reshape(1, -1)
    sig = cv2.resize(sig, (w, 40), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(dbg / "06_col_is_teal.png"), sig)
