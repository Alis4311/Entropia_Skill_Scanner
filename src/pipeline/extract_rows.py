from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Union, Optional

import cv2
import numpy as np

BBox = Tuple[int, int, int, int]  # x, y, w, h


@dataclass(frozen=True)
class RowsResult:
    row_bboxes: List[BBox]            # in table ROI coordinates
    row_images: List[np.ndarray]      # BGR crops
    table_size: Tuple[int, int]       # (W, H)
    reason: str


# ---- Fixed geometry (CALIBRATED; do not change lightly) ----
TABLE_FIRST_ROW_Y = 55   # px from top of table ROI
TABLE_ROW_HEIGHT  = 39   # px
TABLE_NUM_ROWS    = 12   # visible rows infixed table ROI


def extract_rows(
    table_bgr: np.ndarray,
    *,
    first_row_y: int = TABLE_FIRST_ROW_Y,
    row_height: int = TABLE_ROW_HEIGHT,
    num_rows: int = TABLE_NUM_ROWS,
    debug_dir: Union[str, Path, None] = None,
) -> RowsResult:
    """
    Fixed row slicing from a fixed table ROI.

    Input:
      table_bgr: table ROI cropped from the normalized skills window (BGR)

    Output:
      RowsResult with row_bboxes and row_images (one per visible row)

    Debug (optional):
      debug_dir/
        rows_overlay.png
        row_roi_00.png ... row_roi_11.png
    """
    if table_bgr is None or table_bgr.size == 0:
        return RowsResult([], [], (0, 0), "empty input image")

    H, W = table_bgr.shape[:2]

    bboxes: List[BBox] = []
    crops: List[np.ndarray] = []

    for i in range(num_rows):
        y = int(first_row_y + i * row_height)
        h = int(row_height)

        if y < 0 or y + h > H:
            break

        bbox = (0, y, W, h)
        bboxes.append(bbox)
        crops.append(table_bgr[y : y + h, 0:W].copy())

    reason = "ok" if len(crops) == num_rows else f"cropped {len(crops)}/{num_rows} rows (table_h={H})"

    if debug_dir is not None:
        dbg = Path(debug_dir)
        dbg.mkdir(parents=True, exist_ok=True)

        # Save each row crop
        for i, crop in enumerate(crops):
            cv2.imwrite(str(dbg / f"row_roi_{i:02d}.png"), crop)

        # Save overlay
        vis = table_bgr.copy()
        for i, (x, y, w, h) in enumerate(bboxes):
            cv2.rectangle(vis, (x, y), (x + w - 1, y + h - 1), (0, 255, 255), 2)
            cv2.putText(
                vis,
                f"{i:02d}",
                (x + 6, y + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
        cv2.imwrite(str(dbg / "rows_overlay.png"), vis)

    return RowsResult(bboxes, crops, (W, H), reason)
