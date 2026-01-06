from __future__ import annotations

from dataclasses import dataclass, replace
import time
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import cv2 as cv
import numpy as np

from entropia_skillscanner.config import AppConfig, load_app_config
from entropia_skillscanner.core import PipelineResult, PipelineRow, SkillRow
from entropia_skillscanner.exporter import (
    ExportResult,
    ProfessionServices,
    build_export,
)
from entropia_skillscanner.runtime import run_pipeline_sync

Pathish = Union[str, Path]


@dataclass(frozen=True)
class ScanRow:
    name: str
    value: str

    @classmethod
    def from_pipeline(cls, row: PipelineRow) -> "ScanRow":
        return cls(name=row.name, value=row.value)


@dataclass(frozen=True)
class ScanResult:
    input: str
    status: str
    ok: bool
    rows: Tuple[ScanRow, ...]
    logs: Tuple[str, ...]

    def as_json(self) -> dict:
        return {
            "input": self.input,
            "status": self.status,
            "ok": self.ok,
            "rows": [
                {
                    "name": row.name,
                    "value": row.value,
                }
                for row in self.rows
            ],
            "logs": list(self.logs),
        }


def _iter_inputs(paths: Sequence[Path], exts={".png", ".jpg", ".jpeg", ".bmp", ".webp"}) -> List[Path]:
    out: List[Path] = []
    for p in paths:
        if p.is_dir():
            out.extend(sorted([x for x in p.rglob("*") if x.suffix.lower() in exts and x.is_file()]))
        else:
            out.append(p)
    return out


def _load_bgr(path: Path) -> np.ndarray:
    bgr = cv.imread(str(path), cv.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Could not read image: {path}")
    return bgr


def _status_ok(result: PipelineResult) -> bool:
    if not result.ok:
        return False
    if result.status and result.status.lower().startswith("error"):
        return False
    return True


def _pipeline_cfg(app_cfg: AppConfig, *, norm_width: Optional[int], min_table_density: Optional[float]):
    cfg = app_cfg.pipeline_config
    if norm_width is not None:
        cfg = replace(cfg, norm_width=norm_width)
    if min_table_density is not None:
        cfg = replace(cfg, min_table_density=min_table_density)
    return cfg


def scan_paths(
    inputs: Sequence[Pathish],
    *,
    app_config: Optional[AppConfig] = None,
    config_path: Optional[Path] = None,
    debug_dir: Optional[Path] = None,
    norm_width: Optional[int] = None,
    min_table_density: Optional[float] = None,
    fail_on_empty: bool = False,
) -> List[ScanResult]:
    app_cfg = app_config or load_app_config(override_path=Path(config_path) if config_path else None)
    app_cfg.validate()

    input_paths = _iter_inputs([Path(x) for x in inputs])
    if not input_paths:
        raise ValueError("No input images found.")

    cfg = _pipeline_cfg(app_cfg, norm_width=norm_width, min_table_density=min_table_density)
    debug_root = debug_dir if debug_dir is not None else app_cfg.debug_dir

    results: List[ScanResult] = []

    for img_path in input_paths:
        logs: List[str] = []

        def logger(msg: str) -> None:
            logs.append(msg)

        try:
            bgr = _load_bgr(img_path)
            per_image_debug = Path(debug_root) / img_path.stem if debug_root else None
            result = run_pipeline_sync(
                cfg,
                bgr,
                debug=app_cfg.debug_pipeline,
                debug_dir=per_image_debug,
                logger=logger,
            )
        except Exception as e:  # pragma: no cover - defensive guard
            result = PipelineResult(rows=[], status=f"error:exception:{e}", ok=False)

        ok = _status_ok(result)
        status = result.status

        if fail_on_empty and not result.rows:
            ok = False
            if result.ok:
                status = "error:empty"

        results.append(
            ScanResult(
                input=str(img_path),
                status=status,
                ok=ok,
                rows=tuple(ScanRow.from_pipeline(r) for r in result.rows),
                logs=tuple(logs),
            )
        )

    return results


def scans_to_skill_rows(results: Iterable[ScanResult], *, added_label: Optional[str] = None) -> List[SkillRow]:
    added = added_label or time.strftime("%Y-%m-%d %H:%M:%S")
    out: List[SkillRow] = []
    for res in results:
        if not res.ok:
            continue
        for row in res.rows:
            try:
                value = float(row.value)
            except Exception:
                continue
            out.append(SkillRow(name=row.name, value=value, added=added))
    return out


def export_scan_results(
    results: Sequence[ScanResult],
    *,
    app_config: Optional[AppConfig] = None,
    schema=None,
    strict: bool = True,
    include_professions: Optional[bool] = None,
    professions_weights_path: Optional[Path] = None,
    professions_list_path: Optional[Path] = None,
    professions_strict: Optional[bool] = None,
    profession_services: Optional[ProfessionServices] = None,
) -> ExportResult:
    skills = scans_to_skill_rows(results)
    cfg = app_config

    return build_export(
        skills,
        app_config=cfg,
        schema=schema,
        strict=strict,
        include_professions=include_professions,
        professions_weights_path=professions_weights_path,
        professions_list_path=professions_list_path,
        professions_strict=professions_strict,
        profession_services=profession_services,
    )

