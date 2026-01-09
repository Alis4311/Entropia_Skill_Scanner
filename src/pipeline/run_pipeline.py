# src/pipeline/run_pipeline.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, List, Union

import numpy as np
from entropia_skillscanner.core import PipelineResult, PipelineRow

from .extract_skill_window import extract_skill_window
from .extract_table import extract_table
from .extract_rows import extract_rows

from .extract_columns import extract_columns_from_row
from .extract_skill_name import extract_skill_names_batched
from .parse_points_int import parse_points_int_batch
from .parse_points_decimal import parse_points_decimal_from_bar
from .stage_results import (
    PipelineStageError,
    WindowDetection,
    TableExtraction,
    RowExtraction,
    OcrBatchResult,
)


@dataclass(frozen=True)
class PipelineConfig:
    norm_width: int = 1400
    min_table_density: float = 0.010


def run_pipeline(
    cfg: PipelineConfig,
    screenshot_bgr: np.ndarray,
    *,
    debug: bool = False,
    debug_dir: Union[str, Path, None] = None,
    logger: Optional[Callable[[str], None]] = None,
) -> PipelineResult:
    out: List[PipelineRow] = []
    warnings: List[str] = []

    dbg_root = Path(debug_dir) if (debug and debug_dir is not None) else None
    if dbg_root:
        dbg_root.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        if logger:
            logger(msg)

    # ---- 1) Skills window ----
    log("detecting skills window")
    win_dbg = (dbg_root / "01_window") if dbg_root else None
    win_res = WindowDetection.from_crop_result(
        extract_skill_window(
            screenshot_bgr,
            norm_width=cfg.norm_width,
            debug_dir=win_dbg,
        )
    )
    try:
        win_res.require_ok()
    except PipelineStageError as e:
        return PipelineResult(rows=[], status=e.status, ok=False, warnings=())

    norm_bgr = win_res.norm_bgr  

    # ---- 2) Table ROI ----
    log("detecting table")
    table_dbg = (dbg_root / "02_table") if dbg_root else None
    table_res = TableExtraction.from_fixed_table(
        extract_table(
            norm_bgr,
            debug_dir=table_dbg,
            min_density=cfg.min_table_density,
        )
    )
    try:
        table_res.require_ok()
    except PipelineStageError as e:
        return PipelineResult(rows=[], status=e.status, ok=False, warnings=())

    table_bgr = table_res.table_bgr  

    # ---- 3) Rows ----
    log("extracting rows")
    rows_dbg = (dbg_root / "03_rows") if dbg_root else None
    rows_res = RowExtraction.from_rows_result(
        extract_rows(
            table_bgr,
            debug_dir=rows_dbg,
        )
    )
    try:
        rows_res.require_ok()
    except PipelineStageError as e:
        return PipelineResult(rows=[], status=e.status, ok=False, warnings=())
    row_images = rows_res.row_images

    # ---- 4) Columns per row (collect crops) ----
    log("extracting columns")
    cols_dbg = (dbg_root / "04_cols") if dbg_root else None

    cols_list = []
    name_crops: List[np.ndarray] = []
    points_rois: List[np.ndarray] = []

    for i, row_bgr in enumerate(row_images):
        row_dbg = (cols_dbg / f"row_{i:02d}") if cols_dbg else None
        cols = extract_columns_from_row(
            row_bgr,
            row_index=i if debug else None,
            debug_dir=row_dbg,
        )
        cols_list.append(cols)
        name_crops.append(cols.name_bgr)
        points_rois.append(cols.points_bgr)

    # ---- 5) Batched skill-name OCR ----
    log("batched: skill names")
    skill_names = extract_skill_names_batched(name_crops)
    mismatch_count = sum(1 for s in skill_names if isinstance(s, str) and s.startswith("MISMATCH:"))
    if mismatch_count:
        warnings.append(f"skill name mismatches: {mismatch_count}")

    # ---- 6) Batched integer OCR ----
    log("batched: points int")
    int_dbg = (dbg_root / "05_points_int_batch") if dbg_root else None
    int_batch = OcrBatchResult.from_points_batch(
        parse_points_int_batch(
            points_rois,
            upscale=3,
            blur=True,
            top_band_frac=0.55,
            debug_dir=int_dbg,
            min_confidence=0.50,
        ),
        expected_rows=len(row_images),
    )

    # ---- 7) Per-row decimals + assemble ----
    log("assembling")
    dec_dbg = (dbg_root / "06_points_dec") if dbg_root else None

    partial_int_status = ""
    if len(int_batch.points.results) < len(row_images):
        e = PipelineStageError(
            "ocr-int-batch",
            "missing OCR rows",
            context=f"results={len(int_batch.points.results)}, expected={len(row_images)}",
        )
        return PipelineResult(rows=[], status=e.status, ok=False, warnings=())

    if int_batch.points.ok_count == 0:
        e = PipelineStageError("ocr-int-batch", int_batch.points.reason, context=int_batch.failure_context)
        return PipelineResult(rows=[], status=e.status, ok=False, warnings=())

    if int_batch.points.ok_count < len(row_images):
        partial_int_status = f", int ok {int_batch.points.ok_count}/{len(row_images)}"
        warnings.append(f"integer OCR partial: {int_batch.points.ok_count}/{len(row_images)}")

    skipped_int = 0
    missing_dec = 0

    for i, cols in enumerate(cols_list):
        int_res = int_batch.points.results[i]
        if int_res.value is None:
            log(f"row {i}: int failed ({int_res.reason})")
            skipped_int += 1
            continue

        row_dec_dbg = (dec_dbg / f"row_{i:02d}") if dec_dbg else None
        dec_res = parse_points_decimal_from_bar(
            cols.points_bgr,
            debug_dir=row_dec_dbg,
        )

        if getattr(dec_res, "decimal", None) is None:
            missing_dec += 1

        dec_val = dec_res.decimal if dec_res.decimal is not None else 0.0
        value = float(int_res.value + dec_val)

        name = skill_names[i] if i < len(skill_names) else ""
        out.append(PipelineRow(name=name, value=f"{value:.2f}"))

    if skipped_int:
        warnings.append(f"rows skipped (int OCR failed): {skipped_int}")
    if missing_dec:
        warnings.append(f"rows with missing decimal (assumed .00): {missing_dec}")

    status = f"done (+{len(out)} rows{partial_int_status})"
    return PipelineResult(rows=out, status=status, ok=True, warnings=tuple(warnings))
