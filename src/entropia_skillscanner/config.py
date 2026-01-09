from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional

try:  # Python <3.11 fallback
    import tomllib  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - fallback path
    import tomli as tomllib  # type: ignore

from entropia_skillscanner.taxonomy import ExportSchema, SCHEMA_NEW, SCHEMA_OLD, validate_mappings

try:  
    import yaml  # type: ignore
except Exception:  # pragma: no cover - import guard
    yaml = None


DEFAULT_DATA_DIR = Path("data")
DEFAULT_PROFESSIONS_WEIGHTS_PATH = DEFAULT_DATA_DIR / "professions.json"
DEFAULT_PROFESSIONS_LIST_PATH = DEFAULT_DATA_DIR / "professions_list.json"
DEFAULT_SCHEMA = "OLD"

if TYPE_CHECKING:  
    from pipeline.run_pipeline import PipelineConfig


def _pipeline_defaults() -> tuple[int, float]:
    try:
        from pipeline.run_pipeline import PipelineConfig as _PipelineConfig  

        cfg = _PipelineConfig()
        return cfg.norm_width, cfg.min_table_density
    except Exception:
        return 1400, 0.010


_DEFAULT_NORM_WIDTH, _DEFAULT_MIN_TABLE_DENSITY = _pipeline_defaults()


def _load_pyproject_config(project_root: Path) -> Dict[str, Any]:
    pyproject_path = project_root / "pyproject.toml"
    if not pyproject_path.exists():
        return {}

    with pyproject_path.open("rb") as f:
        data = tomllib.load(f)
    return data.get("tool", {}).get("entropia_skillscanner", {}) or {}


def _ensure_mapping(obj: Any, ctx: str) -> Dict[str, Any]:
    if obj is None:
        return {}
    if not isinstance(obj, dict):
        raise ValueError(f"{ctx} must be a mapping/object")
    return obj


def _load_override_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config override not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".json":
        return _ensure_mapping(json.loads(path.read_text(encoding="utf-8")), "JSON config")
    if suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML is required for YAML config overrides")
        return _ensure_mapping(yaml.safe_load(path.read_text(encoding="utf-8")), "YAML config")
    if suffix == ".toml":
        with path.open("rb") as f:
            return _ensure_mapping(tomllib.load(f), "TOML config")

    raise ValueError(f"Unsupported config override format: {path}")


def _merge_section(base: Mapping[str, Any], override: Mapping[str, Any], key: str) -> Dict[str, Any]:
    merged = dict(_ensure_mapping(base.get(key), f"{key} section"))
    merged.update(_ensure_mapping(override.get(key), f"{key} section"))
    return merged


def _merge_top(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    merged.update(override)
    merged.pop("paths", None)
    merged.pop("pipeline", None)
    return merged


def _resolve_path(project_root: Path, candidate: Optional[object], default: Path) -> Path:
    from entropia_skillscanner.resources import resource_path, is_frozen, ensure_user_data_file 

    path = default if candidate is None else Path(str(candidate))

    if path.is_absolute():
        return path

    
    if is_frozen():
        rel = path.as_posix().replace("\\", "/")
        if rel.startswith("data/"):
            return ensure_user_data_file(rel)
        return resource_path(rel)
    return (project_root / path).resolve()




@dataclass(frozen=True)
class AppConfig:
    data_dir: Path = DEFAULT_DATA_DIR
    professions_weights_path: Path = DEFAULT_PROFESSIONS_WEIGHTS_PATH
    professions_list_path: Path = DEFAULT_PROFESSIONS_LIST_PATH
    schema: str = DEFAULT_SCHEMA
    include_professions: bool = True
    professions_strict: bool = False
    pipeline_norm_width: int = _DEFAULT_NORM_WIDTH
    pipeline_min_table_density: float = _DEFAULT_MIN_TABLE_DENSITY
    debug_pipeline: bool = False
    debug_dir: Optional[Path] = None
    enable_sfx: bool = True

    @property
    def export_schema(self) -> ExportSchema:
        schema_name = self.schema.upper()
        if schema_name == "OLD":
            return SCHEMA_OLD
        if schema_name == "NEW":
            return SCHEMA_NEW
        raise ValueError(f"Unknown export schema '{self.schema}'")

    @property
    def pipeline_config(self) -> "PipelineConfig":
        from pipeline.run_pipeline import PipelineConfig  # type: ignore

        return PipelineConfig(
            norm_width=self.pipeline_norm_width,
            min_table_density=self.pipeline_min_table_density,
        )

    def validate(self, *, strict: bool = True, require_paths: bool = True) -> Dict[str, str]:
        issues: Dict[str, str] = {}

        try:
            _ = self.export_schema
        except ValueError as e:
            issues["schema"] = str(e)

        for label, path in (
            ("professions_weights_path", self.professions_weights_path),
            ("professions_list_path", self.professions_list_path),
        ):
            if require_paths and not Path(path).exists():
                issues[label] = f"path does not exist: {path}"

        for idx, msg in enumerate(validate_mappings(strict=False)):
            issues[f"taxonomy_{idx}"] = msg

        if not isinstance(self.enable_sfx, bool):
            issues["enable_sfx"] = "enable_sfx must be a bool"

        if strict and issues:
            details = "\n- ".join(f"{k}: {v}" for k, v in issues.items())
            raise ValueError("Config validation failed:\n- " + details)
        return issues

    @classmethod
    def _from_maps(
        cls,
        *,
        project_root: Path,
        top: Mapping[str, Any],
        paths: Mapping[str, Any],
        pipeline: Mapping[str, Any],
    ) -> "AppConfig":
        data_dir = _resolve_path(project_root, paths.get("data_dir") or top.get("data_dir"), DEFAULT_DATA_DIR)

        prof_weights_default = data_dir / DEFAULT_PROFESSIONS_WEIGHTS_PATH.name
        prof_list_default = data_dir / DEFAULT_PROFESSIONS_LIST_PATH.name

        prof_weights = paths.get("professions_weights") or top.get("professions_weights_path") or top.get("professions_weights")
        prof_list = paths.get("professions_list") or top.get("professions_list_path") or top.get("professions_list")

        debug_dir_val = paths.get("debug_dir") or top.get("debug_dir")

        schema = str(top.get("schema", DEFAULT_SCHEMA)).upper()
        include_professions = top.get("include_professions")
        professions_strict = top.get("professions_strict")
        debug_pipeline = top.get("debug_pipeline")

        return cls(
            data_dir=data_dir,
            professions_weights_path=_resolve_path(project_root, prof_weights, prof_weights_default),
            professions_list_path=_resolve_path(project_root, prof_list, prof_list_default),
            schema=schema,
            include_professions=bool(True if include_professions is None else include_professions),
            professions_strict=bool(False if professions_strict is None else professions_strict),
            pipeline_norm_width=int(pipeline.get("norm_width", _DEFAULT_NORM_WIDTH)),
            pipeline_min_table_density=float(pipeline.get("min_table_density", _DEFAULT_MIN_TABLE_DENSITY)),
            debug_pipeline=bool(False if debug_pipeline is None else debug_pipeline),
            debug_dir=_resolve_path(project_root, debug_dir_val, Path(debug_dir_val)) if debug_dir_val else None,
        )


def load_app_config(*, project_root: Optional[Path] = None, override_path: Optional[Path] = None) -> AppConfig:
    root = Path(project_root) if project_root else Path.cwd()

    base = _load_pyproject_config(root)
    override = _load_override_file(override_path) if override_path else {}

    top = _merge_top(base, override)
    paths = _merge_section(base, override, "paths")
    pipeline = _merge_section(base, override, "pipeline")

    return AppConfig._from_maps(project_root=root, top=top, paths=paths, pipeline=pipeline)
