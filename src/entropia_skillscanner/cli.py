from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Iterable, List, Optional

from entropia_skillscanner import api
from entropia_skillscanner.config import load_app_config
from entropia_skillscanner.exporter import ExportResult, write_csv
from entropia_skillscanner.taxonomy import ExportSchema, SCHEMA_NEW, SCHEMA_OLD

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="entropia-skillscanner",
        description="Headless utilities for scanning and exporting Entropia skill screenshots.",
    )
    subparsers = ap.add_subparsers(dest="command", required=True)

    scan = subparsers.add_parser("scan", help="Run OCR pipeline on images/directories.")
    _add_scan_args(scan)
    scan.set_defaults(func=_cmd_scan)

    export = subparsers.add_parser("export", help="Generate export CSV/JSON from scan results.")
    _add_export_args(export)
    export.set_defaults(func=_cmd_export)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())


# ---------------- CLI subcommands ----------------


def _add_scan_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("inputs", nargs="+", help="Image files or directories (recursively scanned).")
    ap.add_argument("--config", default=None, help="Optional YAML/JSON config override (merged with pyproject.toml).")
    ap.add_argument("--debug-dir", default=None, help="If set, pipeline debug artifacts go here.")
    ap.add_argument("--json", action="store_true", help="Emit JSON (stable schema) instead of CSV.")
    ap.add_argument("--fail-on-empty", action="store_true", help="Fail if no rows are produced.")
    ap.add_argument("--norm-width", type=int, default=None)
    ap.add_argument("--min-table-density", type=float, default=None)
    ap.epilog = _SCAN_EPILOG


def _add_export_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--from-scan", required=True, help="Scan results JSON file (from `scan --json`). Use '-' for stdin.")
    ap.add_argument("--config", default=None, help="Optional YAML/JSON config override (merged with pyproject.toml).")
    ap.add_argument("--schema", choices=["OLD", "NEW"], default=None, help="Override export schema (default from config).")
    ap.add_argument("--no-professions", action="store_true", help="Skip profession calculations.")
    ap.add_argument("--output", "-o", default=None, help="Output path (CSV). If omitted, CSV is printed to stdout.")
    ap.add_argument("--json", action="store_true", help="Emit export JSON instead of CSV.")
    ap.epilog = _EXPORT_EPILOG


def _cmd_scan(args: argparse.Namespace) -> int:
    try:
        results = api.scan_paths(
            args.inputs,
            config_path=Path(args.config) if args.config else None,
            debug_dir=Path(args.debug_dir) if args.debug_dir else None,
            norm_width=args.norm_width,
            min_table_density=args.min_table_density,
            fail_on_empty=args.fail_on_empty,
        )
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 2

    if args.json:
        payload = [r.as_json() for r in results]
        print(json.dumps(payload, indent=2))
    else:
        _emit_scan_csv(results)

    overall_ok = all(r.ok for r in results)
    return 0 if overall_ok else 1


def _cmd_export(args: argparse.Namespace) -> int:
    app_cfg = load_app_config(override_path=Path(args.config) if args.config else None)
    app_cfg.validate()

    scan_results = _load_scan_results(Path(args.from_scan) if args.from_scan != "-" else None)
    schema = _resolve_schema(args.schema, app_cfg.export_schema)

    export_result = api.export_scan_results(
        scan_results,
        app_config=app_cfg,
        schema=schema,
        include_professions=not args.no_professions,
    )

    if args.json:
        print(json.dumps(_export_to_json(export_result), indent=2))
        return 0

    out_path = Path(args.output) if args.output else None
    if out_path:
        write_csv(export_result, out_path)
    else:
        _write_csv_stdout(export_result)
    return 0


def _emit_scan_csv(results: Iterable[api.ScanResult]) -> None:
    print("input,status,skill_name,skill_value")
    for r in results:
        if not r.rows:
            print(f"{r.input},{r.status},,")
            continue
        for row in r.rows:
            print(f"{r.input},{r.status},{row.name},{row.value}")


def _write_csv_stdout(result: ExportResult) -> None:
    out_lines: List[List[str]] = []

    out_lines.append(["[Skills]"])
    for s in result.skills:
        out_lines.append([s.name, format(s.value, ".2f"), s.category])
    out_lines.append([])

    if result.professions:
        out_lines.append(["[Professions]"])
        for p in result.professions:
            out_lines.append([p.name, format(p.value, ".2f"), p.category])
        out_lines.append([])

    out_lines.append(["[Totals]"])
    for t in result.totals:
        out_lines.append([t.category, str(int(t.total))])

    writer = csv.writer(sys.stdout)
    writer.writerows(out_lines)


def _load_scan_results(path: Optional[Path]) -> List[api.ScanResult]:
    raw = sys.stdin.read() if path is None else path.read_text(encoding="utf-8")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise SystemExit(f"Invalid scan JSON: {e}") from e

    out: List[api.ScanResult] = []
    for entry in data:
        out.append(
            api.ScanResult(
                input=str(entry.get("input", "")),
                status=str(entry.get("status", "")),
                ok=bool(entry.get("ok", False)),
                rows=tuple(api.ScanRow(name=r.get("name", ""), value=str(r.get("value", ""))) for r in entry.get("rows", [])),
                logs=tuple(entry.get("logs", [])),
            )
        )
    return out


def _export_to_json(result: ExportResult) -> dict:
    return {
        "skills": [
            {"name": s.name, "value": float(s.value), "category": s.category}
            for s in result.skills
        ],
        "professions": [
            {"name": p.name, "value": float(p.value), "category": p.category}
            for p in result.professions
        ],
        "totals": [
            {"category": t.category, "total": int(t.total)}
            for t in result.totals
        ],
        "warnings": list(result.warnings),
    }


def _resolve_schema(value: Optional[str], fallback: ExportSchema) -> ExportSchema:
    if value is None:
        return fallback
    upper = value.upper()
    if upper == "OLD":
        return SCHEMA_OLD
    if upper == "NEW":
        return SCHEMA_NEW
    raise SystemExit(f"Unknown schema: {value}")


_SCAN_EPILOG = """examples:
  entropia-skillscanner scan screenshot.png --json > scan.json
  entropia-skillscanner scan samples/ --debug-dir dbg/

stable JSON schema (scan):
  [
    {
      "input": "<path>",
      "status": "<string status>",
      "ok": true | false,
      "rows": [{"name": "<skill>", "value": "<string value>"}],
      "logs": ["<pipeline log>", ...]
    }
  ]
"""

_EXPORT_EPILOG = """examples:
  entropia-skillscanner export --from-scan scan.json --output skills.csv
  entropia-skillscanner export --from-scan - --json < scan.json
"""
