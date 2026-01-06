from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("cv2")

from entropia_skillscanner.cli import main as cli_main
from entropia_skillscanner.core import PipelineResult, PipelineRow


def _run_cli(argv):
    return cli_main(argv)


def test_cli_exits_2_when_no_inputs(tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    code = _run_cli(["scan", str(empty_dir)])
    assert code == 2


def test_cli_success_exit_code_and_json(monkeypatch, sample_screenshot_path, capsys):
    called = []

    def fake_run_pipeline(cfg, bgr, debug=False, debug_dir=None, logger=None):
        called.append((cfg, bgr.shape, debug, debug_dir))
        return PipelineResult(rows=[PipelineRow(name="Agility", value="10.00")], status="done (+1 rows)", ok=True)

    monkeypatch.setattr("entropia_skillscanner.api.run_pipeline_sync", fake_run_pipeline)

    code = _run_cli(["scan", "--json", str(sample_screenshot_path)])
    assert code == 0
    assert called, "pipeline was not invoked"
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload[0]["input"].endswith(Path(sample_screenshot_path).name)
    assert payload[0]["rows"] == [{"name": "Agility", "value": "10.00"}]


def test_cli_propagates_pipeline_failure(monkeypatch, sample_screenshot_path, capsys):
    def fake_run_pipeline(cfg, bgr, debug=False, debug_dir=None, logger=None):
        return PipelineResult(rows=[], status="error:table-detection", ok=False)

    monkeypatch.setattr("entropia_skillscanner.api.run_pipeline_sync", fake_run_pipeline)

    code = _run_cli(["scan", "--json", str(sample_screenshot_path)])
    captured = capsys.readouterr()
    assert code == 1
    data = json.loads(captured.out)
    assert data[0]["status"] == "error:table-detection"
    assert data[0]["rows"] == []


def test_cli_fail_on_empty_forces_error(monkeypatch, sample_screenshot_path, capsys):
    def fake_run_pipeline(cfg, bgr, debug=False, debug_dir=None, logger=None):
        return PipelineResult(rows=[], status="done (+0 rows)", ok=True)

    monkeypatch.setattr("entropia_skillscanner.api.run_pipeline_sync", fake_run_pipeline)

    code = _run_cli(["scan", "--json", "--fail-on-empty", str(sample_screenshot_path)])
    captured = capsys.readouterr()
    assert code == 1
    payload = json.loads(captured.out)
    assert payload[0]["status"] == "error:empty"
