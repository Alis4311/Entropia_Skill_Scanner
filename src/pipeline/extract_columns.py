from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Union, Optional

import cv2
import numpy as np

BBox = Tuple[int, int, int, int]  # x,y,w,h


@dataclass(frozen=True)
class RowColumns:
    name_bbox: BBox
    rank_bbox: BBox
    points_bbox: BBox
    name_bgr: np.ndarray
    rank_bgr: np.ndarray
    points_bgr: np.ndarray


# ---- Frozen column geometry (calibrated) ----
NAME_X1_FRAC   = 0.00
NAME_X2_FRAC   = 0.50

RANK_X1_FRAC   = 0.50
RANK_X2_FRAC   = 0.82

POINTS_X1_FRAC = 0.82
POINTS_X2_FRAC = 0.99

ROW_Y1_FRAC = 0.00
ROW_Y2_FRAC = 1.00


def _frac_to_x(W: int, frac: float) -> int:
    return int(round(frac * W))


def extract_columns_from_row(
    row_bgr: np.ndarray,
    *,
    debug_dir: Union[str, Path, None] = None,
    row_index: Optional[int] = None,
) -> RowColumns:
    """
    Split a row ROI into (name, rank, points) crops using frozen fixed fractions.

    Uses contiguous boundaries to avoid tiny gaps/overlaps from rounding.
    """
    H, W = row_bgr.shape[:2]

    # Vertical slice (frozen)
    y1 = int(round(ROW_Y1_FRAC * H))
    y2 = int(round(ROW_Y2_FRAC * H))
    y1 = max(0, min(H - 1, y1))
    y2 = max(y1 + 1, min(H, y2))

    def fx(frac: float) -> int:
        return int(round(frac * W))

    # Contiguous x boundaries (single source of truth)
    nx1 = max(0, min(W - 1, fx(NAME_X1_FRAC)))
    nx2 = max(nx1 + 1, min(W, fx(NAME_X2_FRAC)))

    rx1 = nx2
    rx2 = max(rx1 + 1, min(W, fx(RANK_X2_FRAC)))

    px1 = rx2
    px2 = max(px1 + 1, min(W, fx(POINTS_X2_FRAC)))

    name_bbox: BBox = (nx1, y1, nx2 - nx1, y2 - y1)
    rank_bbox: BBox = (rx1, y1, rx2 - rx1, y2 - y1)
    points_bbox: BBox = (px1, y1, px2 - px1, y2 - y1)

    name_bgr = row_bgr[y1:y2, nx1:nx2].copy()
    rank_bgr = row_bgr[y1:y2, rx1:rx2].copy()
    points_bgr = row_bgr[y1:y2, px1:px2].copy()

    if debug_dir is not None:
        dbg = Path(debug_dir)
        dbg.mkdir(parents=True, exist_ok=True)

        vis = row_bgr.copy()
        cv2.rectangle(vis, (nx1, y1), (nx2 - 1, y2 - 1), (0, 255, 255), 2)   # name
        cv2.rectangle(vis, (rx1, y1), (rx2 - 1, y2 - 1), (255, 255, 0), 2)   # rank
        cv2.rectangle(vis, (px1, y1), (px2 - 1, y2 - 1), (0, 255, 0), 2)     # points

        tag = f"{row_index:02d}_" if row_index is not None else ""
        cv2.imwrite(str(dbg / f"{tag}row_columns_overlay.png"), vis)
        cv2.imwrite(str(dbg / f"{tag}name_roi.png"), name_bgr)
        cv2.imwrite(str(dbg / f"{tag}rank_roi.png"), rank_bgr)
        cv2.imwrite(str(dbg / f"{tag}points_roi.png"), points_bgr)

    return RowColumns(
        name_bbox=name_bbox,
        rank_bbox=rank_bbox,
        points_bbox=points_bbox,
        name_bgr=name_bgr,
        rank_bgr=rank_bgr,
        points_bgr=points_bgr,
    )


def extract_columns_from_rows(
    row_images: List[np.ndarray],
    *,
    debug_dir: Union[str, Path, None] = None,
) -> List[RowColumns]:
    out: List[RowColumns] = []
    for i, row in enumerate(row_images):
        dd = None
        if debug_dir is not None:
            dd = Path(debug_dir) / f"row_{i:02d}"
        out.append(extract_columns_from_row(row, debug_dir=dd, row_index=i))
    return out
