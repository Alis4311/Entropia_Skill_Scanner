import json
from pathlib import Path

import pytest

from entropia_skillscanner.config import (
    AppConfig,
    DEFAULT_PROFESSIONS_LIST_PATH,
    DEFAULT_PROFESSIONS_WEIGHTS_PATH,
    load_app_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_load_app_config_reads_defaults_from_pyproject():
    cfg = load_app_config(project_root=PROJECT_ROOT)
    cfg.validate()

    assert cfg.export_schema.name == "OLD"
    assert cfg.professions_weights_path == PROJECT_ROOT / DEFAULT_PROFESSIONS_WEIGHTS_PATH
    assert cfg.professions_list_path == PROJECT_ROOT / DEFAULT_PROFESSIONS_LIST_PATH


def test_load_app_config_json_override(tmp_path: Path):
    override = tmp_path / "config.json"
    override.write_text(
        json.dumps(
            {
                "schema": "NEW",
                "debug_pipeline": True,
                "pipeline": {"norm_width": 1500},
                "paths": {"data_dir": str(PROJECT_ROOT / "data")},
            }
        ),
        encoding="utf-8",
    )

    cfg = load_app_config(project_root=PROJECT_ROOT, override_path=override)
    cfg.validate()

    assert cfg.pipeline_norm_width == 1500
    assert cfg.export_schema.name == "NEW"
    assert cfg.debug_pipeline is True


def test_app_config_validation_checks_required_paths(tmp_path: Path):
    cfg = AppConfig(
        data_dir=tmp_path / "missing_data",
        professions_weights_path=tmp_path / "missing_data" / "professions.json",
        professions_list_path=tmp_path / "missing_data" / "professions_list.json",
    )

    with pytest.raises(ValueError) as exc:
        cfg.validate()

    msg = str(exc.value)
    assert "professions_weights_path" in msg
    assert "professions_list_path" in msg
