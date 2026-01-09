from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np

BBox = Tuple[int, int, int, int]  # x,y,w,h


@dataclass(frozen=True)
class FixedTableResult:
    table_bbox: BBox               # bbox in norm_bgr coordinates
    table_bgr: Optional[np.ndarray]  # cropped table ROI (None if empty)
    valid: bool
    density: float
    reason: str


# Fixed table bbox as fractions of normalized skills-window image (norm_bgr)
TABLE_X1_FRAC = 0.251
TABLE_Y1_FRAC = 0.091
TABLE_X2_FRAC = 0.981
TABLE_Y2_FRAC = 0.709


def extract_table(
    norm_bgr: np.ndarray,
    *,
    debug_dir: Union[str, Path, None] = None,
    min_density: float = 0.010,
) -> FixedTableResult:
    """
    Normalized skills window -> fixed table bbox + cropped ROI + quick validation.
    """
    bbox = _fixed_table_bbox(norm_bgr)
    table = _crop_bbox(norm_bgr, bbox)

    if table is None or table.size == 0:
        return FixedTableResult(bbox, None, False, 0.0, "empty table crop")

    valid, density, reason, horiz = _validate_table_roi(table, min_density=min_density)

    if debug_dir is not None:
        dbg = Path(debug_dir)
        dbg.mkdir(parents=True, exist_ok=True)

        vis = norm_bgr.copy()
        x, y, w, h = bbox
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0) if valid else (0, 0, 255), 3)
        cv2.putText(
            vis,
            f"fixed table valid={valid} dens={density:.4f}",
            (x, max(0, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0) if valid else (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imwrite(str(dbg / "01_fixed_table_bbox.png"), vis)
        cv2.imwrite(str(dbg / "02_table_roi.png"), table)
        if horiz is not None:
            cv2.imwrite(str(dbg / "03_table_horiz.png"), horiz)

    return FixedTableResult(bbox, table, valid, density, reason)


# -----------------------------
# Internals
# -----------------------------
def _fixed_table_bbox(norm_bgr: np.ndarray) -> BBox:
    """Return fixed table bbox (green rectangle) in normalized-image coordinates."""
    if norm_bgr is None or norm_bgr.size == 0:
        return (0, 0, 0, 0)

    H, W = norm_bgr.shape[:2]
    x1 = int(TABLE_X1_FRAC * W)
    y1 = int(TABLE_Y1_FRAC * H)
    x2 = int(TABLE_X2_FRAC * W)
    y2 = int(TABLE_Y2_FRAC * H)

    # clamp
    x1 = max(0, min(W - 2, x1))
    y1 = max(0, min(H - 2, y1))
    x2 = max(x1 + 10, min(W - 1, x2))
    y2 = max(y1 + 10, min(H - 1, y2))
    return (x1, y1, x2 - x1, y2 - y1)


def _crop_bbox(img: np.ndarray, bbox: BBox) -> Optional[np.ndarray]:
    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        return None
    roi = img[y : y + h, x : x + w]
    if roi.size == 0:
        return None
    return roi.copy()


def _validate_table_roi(table_bgr: np.ndarray, *, min_density: float) -> Tuple[bool, float, str, Optional[np.ndarray]]:
    """
    Quick structural validation:
    - In the right half of the table ROI, we should see horizontal-line evidence (bars/rows).
    Returns (valid, density, reason, horiz_mask).
    """
    gray = cv2.cvtColor(table_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, 60, 160)

    # emphasize horizontal lines
    k_w = max(25, table_bgr.shape[1] // 20)
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_w, 1))
    horiz = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horiz_kernel, iterations=1)
    horiz = cv2.dilate(horiz, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1)), iterations=1)

    rx1 = int(0.55 * horiz.shape[1])
    density = float(np.mean(horiz[:, rx1:] > 0))

    valid = density >= min_density
    reason = "ok" if valid else f"low horiz density ({density:.4f} < {min_density:.4f})"
    return valid, density, reason, horiz
