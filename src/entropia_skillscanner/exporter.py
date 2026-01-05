# exporter.py
from __future__ import annotations

import csv
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP, ROUND_DOWN
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from .models import ExportSkill, ExportTotals, SkillRow
from .taxonomy import ExportSchema, SCHEMA_OLD, validate_mappings, get_category, all_categories


# Quantize to 2 decimals everywhere (HALF_UP for human expectations)
_Q2 = Decimal("0.01")


@dataclass(frozen=True)
class ExportResult:
    skills: List[ExportSkill]
    totals: List[ExportTotals]
    # If you want auditable exports, you can surface warnings to UI later.
    warnings: Tuple[str, ...] = ()


class ExportError(Exception):
    """Raised when export preconditions fail (e.g., missing categories)."""


def _q2(x: Decimal) -> Decimal:
    return x.quantize(_Q2, rounding=ROUND_HALF_UP)


def _to_decimal_value(v: float) -> Decimal:
    # SkillRow.value is currently float-like in your codebase.
    # Use str() to avoid binary float surprises.
    return Decimal(str(v))


def _floor_points(x: Decimal) -> Decimal:
    # In-game totals appear to use full points only (truncate decimals).
    # ROUND_DOWN is correct for positive values.
    return x.quantize(Decimal("1"), rounding=ROUND_DOWN)


def _attach_categories(
    rows: Sequence[SkillRow],
    *,
    schema: ExportSchema,
    strict: bool,
    unknown_bucket: str = "Unknown Skills",
) -> Tuple[List[ExportSkill], Tuple[str, ...]]:
    out: List[ExportSkill] = []
    missing: List[str] = []
    warnings: List[str] = []

    for r in rows:
        cat = get_category(r.name, schema=schema)
        if not cat:
            if strict:
                missing.append(r.name)
                continue
            # non-strict: bucket + flag (for later surfacing)
            cat = unknown_bucket
            warnings.append(f"UNKNOWN_SKILL_CATEGORY:{r.name}")

        dec = _q2(_to_decimal_value(r.value))
        out.append(ExportSkill(name=r.name, value=dec, category=cat))

    if missing:
        uniq = ", ".join(sorted(set(missing)))
        raise ExportError(f"uncategorized skills ({schema.name}): {uniq}")

    return out, tuple(sorted(set(warnings)))


def _category_totals(
    skills: Iterable[ExportSkill],
    *,
    schema: ExportSchema,
    strict: bool,
    unknown_bucket: str = "Unknown Skills",
) -> List[ExportTotals]:
    # IMPORTANT: Totals must match in-game, which uses full points only.
    totals: Dict[str, Decimal] = {}
    for s in skills:
        iv = _floor_points(s.value)
        totals[s.category] = totals.get(s.category, Decimal("0")) + iv

    ordered: List[ExportTotals] = []

    # Deterministic ordering from schema, but also allow the unknown bucket (non-strict)
    schema_order = list(all_categories(schema=schema))
    if not strict and unknown_bucket not in schema_order and unknown_bucket in totals:
        schema_order = schema_order + [unknown_bucket]

    for cat in schema_order:
        if cat in totals:
            ordered.append(ExportTotals(category=cat, total=_floor_points(totals[cat])))

    return ordered


def build_export(
    rows: Sequence[SkillRow],
    *,
    schema: ExportSchema = SCHEMA_OLD,
    strict: bool = True,
) -> ExportResult:
    if not rows:
        raise ExportError("no rows to export")

    # Surface schema/config errors early (startup/export time).
    validate_mappings(strict=True)

    skills, warnings = _attach_categories(rows, schema=schema, strict=strict)

    # Stable ordering: schema category order, then name
    order = list(all_categories(schema=schema))
    idx = {c: i for i, c in enumerate(order)}
    if not strict and "Unknown Skills" not in idx:
        idx["Unknown Skills"] = len(idx) + 10

    skills = sorted(skills, key=lambda s: (idx.get(s.category, 10**9), s.category, s.name))

    totals = _category_totals(skills, schema=schema, strict=strict)

    # Overall total must also match in-game (full points only)
    overall_total = _floor_points(sum((_floor_points(s.value) for s in skills), Decimal("0")))
    totals.append(ExportTotals(category="Total", total=overall_total))

    return ExportResult(skills=skills, totals=totals, warnings=warnings)


def write_csv(result: ExportResult, path: Path) -> None:
    path = Path(path)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)

        # Skills section (keep 2 decimals)
        w.writerow(["[Skills]"])
        for s in result.skills:
            # locale-invariant decimal point
            w.writerow([s.name, format(s.value, ".2f"), s.category])

        w.writerow([])

        # Totals section (integer, in-game aligned)
        w.writerow(["[Totals]"])
        for t in result.totals:
            w.writerow([t.category, str(int(t.total))])

        # Optional: audit warnings at end (commented out for now)
        # if result.warnings:
        #     w.writerow([])
        #     w.writerow(["[Warnings]"])
        #     for msg in result.warnings:
        #         w.writerow([msg])
