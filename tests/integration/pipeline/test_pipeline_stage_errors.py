from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from pipeline.extract_rows import RowsResult
from pipeline.extract_skill_window import CropResult
from pipeline.extract_table import FixedTableResult
from pipeline.parse_points_int import PointsIntBatchResult, PointsIntResult
from pipeline.run_pipeline import PipelineConfig, run_pipeline
from pipeline.stage_results import PipelineStageError, TableExtraction


@pytest.fixture
def sample_image() -> np.ndarray:
    return np.zeros((20, 20, 3), dtype=np.uint8)


@pytest.fixture(autouse=True)
def stub_downstream(monkeypatch):
    """Keep downstream OCR fast/stable during failure-path tests."""

    def fake_names(crops):
        return [f"skill-{i}" for i in range(len(crops))]

    monkeypatch.setattr("pipeline.run_pipeline.extract_skill_names_batched", fake_names)

    # Decimal parsing is not exercised in these tests, but stub to avoid CV/Tesseract.
    class _Dec:
        def __init__(self):
            self.decimal = 0.0

    monkeypatch.setattr("pipeline.run_pipeline.parse_points_decimal_from_bar", lambda roi, debug_dir=None: _Dec())


def _ok_window() -> CropResult:
    norm = np.zeros((20, 20, 3), dtype=np.uint8)
    return CropResult(
        crop_bgr=norm.copy(),
        norm_bgr=norm,
        bbox=(0, 0, 20, 20),
        confidence=0.9,
        margin=0.5,
        reason="ok",
        scale=1.0,
    )


def test_run_pipeline_window_failure(monkeypatch, sample_image):
    monkeypatch.setattr(
        "pipeline.run_pipeline.extract_skill_window",
        lambda *args, **kwargs: CropResult(
            crop_bgr=None,
            norm_bgr=None,
            bbox=None,
            confidence=0.1,
            margin=0.0,
            reason="no window",
            scale=0.0,
        ),
    )

    result = run_pipeline(PipelineConfig(), sample_image, debug=False)
    assert not result.ok
    assert result.status.startswith("window-detection: no window")


def test_run_pipeline_table_failure(monkeypatch, sample_image):
    monkeypatch.setattr("pipeline.run_pipeline.extract_skill_window", lambda *a, **k: _ok_window())
    monkeypatch.setattr(
        "pipeline.run_pipeline.extract_table",
        lambda *a, **k: FixedTableResult((0, 0, 10, 10), None, False, 0.001, "low density"),
    )

    result = run_pipeline(PipelineConfig(), sample_image, debug=False)
    assert not result.ok
    assert result.status.startswith("table-extraction:")
    assert "density=0.0010" in result.status


def test_run_pipeline_rows_failure(monkeypatch, sample_image):
    monkeypatch.setattr("pipeline.run_pipeline.extract_skill_window", lambda *a, **k: _ok_window())
    table_bgr = np.zeros((10, 10, 3), dtype=np.uint8)
    monkeypatch.setattr(
        "pipeline.run_pipeline.extract_table",
        lambda *a, **k: FixedTableResult((0, 0, 10, 10), table_bgr, True, 0.02, "ok"),
    )
    monkeypatch.setattr(
        "pipeline.run_pipeline.extract_rows",
        lambda *a, **k: RowsResult([], [], (10, 10), "no rows found"),
    )

    result = run_pipeline(PipelineConfig(), sample_image, debug=False)
    assert not result.ok
    assert result.status.startswith("row-extraction: no rows found")
    assert "table_size=(10, 10)" in result.status


def test_table_extraction_include_context_on_empty_crop():
    res = TableExtraction.from_fixed_table(FixedTableResult((1, 2, 3, 4), None, True, 0.5, "ok"))
    with pytest.raises(PipelineStageError) as exc:
        res.require_ok()

    assert exc.value.stage == "table-extraction"
    assert "empty table crop" in exc.value.reason or exc.value.reason.startswith("empty")
    assert "density=0.5000" in exc.value.status


def test_run_pipeline_ocr_failure(monkeypatch, sample_image):
    monkeypatch.setattr("pipeline.run_pipeline.extract_skill_window", lambda *a, **k: _ok_window())
    table_bgr = np.zeros((10, 10, 3), dtype=np.uint8)
    monkeypatch.setattr(
        "pipeline.run_pipeline.extract_table",
        lambda *a, **k: FixedTableResult((0, 0, 10, 10), table_bgr, True, 0.02, "ok"),
    )
    row_img = np.zeros((5, 5, 3), dtype=np.uint8)
    monkeypatch.setattr(
        "pipeline.run_pipeline.extract_rows",
        lambda *a, **k: RowsResult([(0, 0, 5, 5)], [row_img], (10, 10), "ok"),
    )
    monkeypatch.setattr(
        "pipeline.run_pipeline.parse_points_int_batch",
        lambda *a, **k: PointsIntBatchResult(
            values=[None],
            results=[PointsIntResult(None, "", 0.0, "no digits")],
            ok_count=0,
            reason="ok 0/1",
        ),
    )

    result = run_pipeline(PipelineConfig(), sample_image, debug=False)
    assert not result.ok
    assert result.status.startswith("ocr-int-batch: ok 0/1")
    assert "ok=0/1" in result.status


def test_run_pipeline_allows_partial_int_success(monkeypatch, sample_image):
    monkeypatch.setattr("pipeline.run_pipeline.extract_skill_window", lambda *a, **k: _ok_window())
    table_bgr = np.zeros((10, 10, 3), dtype=np.uint8)
    monkeypatch.setattr(
        "pipeline.run_pipeline.extract_table",
        lambda *a, **k: FixedTableResult((0, 0, 10, 10), table_bgr, True, 0.02, "ok"),
    )
    row_img = np.zeros((5, 5, 3), dtype=np.uint8)
    monkeypatch.setattr(
        "pipeline.run_pipeline.extract_rows",
        lambda *a, **k: RowsResult([(0, 0, 5, 5), (0, 0, 5, 5)], [row_img, row_img], (10, 10), "ok"),
    )
    monkeypatch.setattr(
        "pipeline.run_pipeline.parse_points_int_batch",
        lambda *a, **k: PointsIntBatchResult(
            values=[123, None],
            results=[PointsIntResult(123, "123", 0.9, "ok"), PointsIntResult(None, "", 0.1, "no digits")],
            ok_count=1,
            reason="ok 1/2",
        ),
    )

    result = run_pipeline(PipelineConfig(), sample_image, debug=False)
    assert result.ok
    assert len(result.rows) == 1
    assert result.rows[0].value == "123.00"
    assert "int ok 1/2" in result.status
