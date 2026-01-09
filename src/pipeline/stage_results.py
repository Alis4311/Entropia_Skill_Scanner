from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .extract_rows import RowsResult
from .extract_skill_window import CropResult
from .extract_table import FixedTableResult
from .parse_points_int import PointsIntBatchResult

BBox = Tuple[int, int, int, int]


class PipelineStageError(RuntimeError):
    """Structured pipeline failure used to short-circuit `run_pipeline`.

    Attributes
    ----------
    stage:
        Machine-readable stage identifier (e.g. "window-detection").
    reason:
        Failure summary suitable for a status string.
    context:
        Optional extra context that should accompany the reason when
        constructing status messages.
    """

    def __init__(self, stage: str, reason: str, *, context: Optional[str] = None) -> None:
        self.stage = stage
        self.reason = reason
        self.context = context
        super().__init__(self.status)

    @property
    def status(self) -> str:
        if self.context:
            return f"{self.stage}: {self.reason} ({self.context})"
        return f"{self.stage}: {self.reason}"


@dataclass(frozen=True)
class WindowDetection:
    """Outcome of skills-window detection.

    Invariants
    ----------
    * When ``ok`` is True, ``norm_bgr``, ``crop_bgr`` and ``bbox`` are non-None.
    * When ``ok`` is False, image fields are None and callers must not use them.
    """

    norm_bgr: Optional[np.ndarray]
    crop_bgr: Optional[np.ndarray]
    bbox: Optional[BBox]
    confidence: float
    margin: float
    reason: str
    scale: float

    @property
    def ok(self) -> bool:
        return self.norm_bgr is not None and self.crop_bgr is not None and self.bbox is not None

    @property
    def failure_context(self) -> str:
        return f"conf={self.confidence:.3f}, margin={self.margin:.3f}"

    @classmethod
    def from_crop_result(cls, res: CropResult) -> "WindowDetection":
        return cls(
            norm_bgr=res.norm_bgr,
            crop_bgr=res.crop_bgr,
            bbox=res.bbox,
            confidence=float(res.confidence),
            margin=float(res.margin),
            reason=res.reason,
            scale=float(res.scale),
        )

    def require_ok(self) -> None:
        if not self.ok:
            raise PipelineStageError("window-detection", self.reason, context=self.failure_context)


@dataclass(frozen=True)
class TableExtraction:
    """Fixed-table ROI extraction.

    Invariants
    ----------
    * ``table_bbox`` is always set.
    * When ``ok`` is True, ``table_bgr`` is a non-empty crop and ``valid`` is True.
    * When ``ok`` is False, ``table_bgr`` may be None or invalid and must not be used.
    """

    table_bbox: BBox
    table_bgr: Optional[np.ndarray]
    valid: bool
    density: float
    reason: str

    @property
    def ok(self) -> bool:
        return self.table_bgr is not None and self.valid

    @property
    def failure_context(self) -> str:
        return f"density={self.density:.4f}"

    @classmethod
    def from_fixed_table(cls, res: FixedTableResult) -> "TableExtraction":
        return cls(
            table_bbox=res.table_bbox,
            table_bgr=res.table_bgr,
            valid=res.valid,
            density=float(res.density),
            reason=res.reason,
        )

    def require_ok(self) -> None:
        if self.table_bgr is None or self.table_bgr.size == 0:
            raise PipelineStageError("table-extraction", "empty table crop", context=self.failure_context)
        if not self.valid:
            raise PipelineStageError("table-extraction", self.reason, context=self.failure_context)


@dataclass(frozen=True)
class RowExtraction:
    """Fixed row slicing from the table ROI.

    Invariants
    ----------
    * ``row_bboxes`` and ``row_images`` are always the same length.
    * When ``ok`` is True, at least one row crop is present.
    """

    row_bboxes: List[BBox]
    row_images: List[np.ndarray]
    table_size: Tuple[int, int]
    reason: str

    @property
    def ok(self) -> bool:
        return len(self.row_images) > 0 and len(self.row_images) == len(self.row_bboxes)

    @classmethod
    def from_rows_result(cls, res: RowsResult) -> "RowExtraction":
        return cls(
            row_bboxes=list(res.row_bboxes),
            row_images=list(res.row_images),
            table_size=res.table_size,
            reason=res.reason,
        )

    def require_ok(self) -> None:
        if not self.row_images:
            raise PipelineStageError("row-extraction", self.reason, context=f"table_size={self.table_size}")
        if len(self.row_images) != len(self.row_bboxes):
            raise PipelineStageError(
                "row-extraction",
                "bbox/image length mismatch",
                context=f"bboxes={len(self.row_bboxes)}, images={len(self.row_images)}",
            )


@dataclass(frozen=True)
class OcrBatchResult:
    """Aggregated OCR batch result for integer points.

    Invariants
    ----------
    * ``expected_rows`` expresses how many rows were sent for OCR.
    * ``points.results`` length should be at least ``expected_rows``.
    * ``ok`` is True only when OCR produced as many confident values as expected.
    """

    expected_rows: int
    points: PointsIntBatchResult

    @property
    def ok(self) -> bool:
        return self.points.ok_count == self.expected_rows

    @property
    def failure_context(self) -> str:
        return f"ok={self.points.ok_count}/{self.expected_rows}"

    @classmethod
    def from_points_batch(cls, points: PointsIntBatchResult, expected_rows: int) -> "OcrBatchResult":
        return cls(expected_rows=expected_rows, points=points)

    def require_ok(self) -> None:
        if len(self.points.results) < self.expected_rows:
            raise PipelineStageError(
                "ocr-int-batch",
                "missing OCR rows",
                context=f"results={len(self.points.results)}, expected={self.expected_rows}",
            )
        if not self.ok:
            raise PipelineStageError("ocr-int-batch", self.points.reason, context=self.failure_context)
