# exporter.py
from __future__ import annotations

import csv
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from dataclasses import dataclass
from entropia_skillscanner.core import (
    ExportProfession,
    ExportResult,
    ExportSkill,
    ExportTotals,
    SkillRow,
)
from .taxonomy import ExportSchema, SCHEMA_OLD, validate_mappings, get_category, all_categories
from entropia_skillscanner.professions import (
    ProfessionValue,
    compute_professions,
    load_profession_categories,
    load_profession_weights,
)


# Quantize to 2 decimals everywhere (HALF_UP for human expectations)
_Q2 = Decimal("0.01")
_Q0 = Decimal("1")


class ExportError(Exception):
    """Raised when export preconditions fail (e.g., missing categories)."""


def _q2(x: Decimal) -> Decimal:
    return x.quantize(_Q2, rounding=ROUND_HALF_UP)


def _to_decimal_value(v: float) -> Decimal:
    # SkillRow.value is float-like in your codebase.
    # Use str() to avoid binary float surprises.
    try:
        return Decimal(str(v))
    except InvalidOperation as e:
        raise ExportError(f"invalid skill value: {v!r}") from e


def _round_points(x: Decimal) -> Decimal:
    """
    Integer totals derived from decimal skill values.
    Rule: sum full decimals, then ROUND_HALF_UP once at the end.
    """
    return x.quantize(_Q0, rounding=ROUND_HALF_UP)


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
    """
    Category totals = ROUND_HALF_UP(sum(skill.value)) per category.
    """
    totals: Dict[str, Decimal] = {}
    for s in skills:
        totals[s.category] = totals.get(s.category, Decimal("0")) + s.value

    ordered: List[ExportTotals] = []

    # Deterministic ordering from schema, but also allow the unknown bucket (non-strict)
    schema_order = list(all_categories(schema=schema))
    if not strict and unknown_bucket not in schema_order and unknown_bucket in totals:
        schema_order = schema_order + [unknown_bucket]

    for cat in schema_order:
        if cat in totals:
            ordered.append(ExportTotals(category=cat, total=_round_points(totals[cat])))

    return ordered


@dataclass(frozen=True)
class ProfessionServices:
    load_weights: Callable[[Path], Mapping[str, List[Dict[str, object]]]]
    load_categories: Callable[[Path], Mapping[str, str]]
    compute: Callable[..., Dict[str, ProfessionValue]]

    @classmethod
    def default(cls) -> "ProfessionServices":
        return cls(
            load_weights=load_profession_weights,
            load_categories=load_profession_categories,
            compute=compute_professions,
        )


def _build_professions(
    rows: Sequence[SkillRow],
    *,
    professions_weights_path: Path,
    professions_list_path: Path,
    strict: bool,
    unknown_bucket: str = "Miscellaneous",
    services: ProfessionServices,
) -> Tuple[List[ExportProfession], Tuple[str, ...]]:
    """
    Compute profession values from skills.
    Category comes from professions_list.json (activityName->category).
    """
    warnings: List[str] = []

    # Skill -> Decimal (full precision from SkillRow.value stringified)
    skills_map: Dict[str, Decimal] = {r.name: _to_decimal_value(r.value) for r in rows}

    weights = services.load_weights(professions_weights_path)
    cat_map = services.load_categories(professions_list_path)

    prof_vals: Dict[str, ProfessionValue] = services.compute(
        skills=skills_map,
        profession_weights=weights,
        strict=strict,
    )

    out: List[ExportProfession] = []
    for prof, pv in prof_vals.items():
        cat = cat_map.get(prof, unknown_bucket)
        if prof not in cat_map:
            warnings.append(f"UNKNOWN_PROF_CATEGORY:{prof}")

        # compute_professions already q2()'d, but we re-q2 defensively
        value = _q2(Decimal(pv.value))

        if pv.missing_skills:
            warnings.append(f"PROF_MISSING_SKILLS:{prof}:{','.join(pv.missing_skills)}")

        # informational; your validator notes it’s often not exactly 100
        if pv.pct_sum != Decimal("100"):
            warnings.append(f"PROF_PCT_SUM:{prof}:{pv.pct_sum}")

        out.append(ExportProfession(name=prof, value=value, category=cat))

    # Deterministic ordering: category then name
    out.sort(key=lambda p: (p.category, p.name))
    return out, tuple(sorted(set(warnings)))


def build_export(
    rows: Sequence[SkillRow],
    *,
    schema: ExportSchema = SCHEMA_OLD,
    strict: bool = True,
    include_professions: bool = True,
    professions_weights_path: Path = Path("data/professions.json"),
    professions_list_path: Path = Path("data/professions_list.json"),
    professions_strict: bool = False,
    profession_services: Optional[ProfessionServices] = None,
) -> ExportResult:
    if not rows:
        raise ExportError("no rows to export")

    # Surface schema/config errors early (startup/export time).
    validate_mappings(strict=True)

    skills, warnings_sk = _attach_categories(rows, schema=schema, strict=strict)

    # Stable ordering: schema category order, then name
    order = list(all_categories(schema=schema))
    idx = {c: i for i, c in enumerate(order)}
    if not strict and "Unknown Skills" not in idx:
        idx["Unknown Skills"] = len(idx) + 10

    skills = sorted(skills, key=lambda s: (idx.get(s.category, 10**9), s.category, s.name))

    # Totals (category order), but "Total" must be FIRST
    totals = _category_totals(skills, schema=schema, strict=strict)
    overall_total = _round_points(sum((s.value for s in skills), Decimal("0")))
    totals = [ExportTotals(category="Total", total=overall_total)] + totals

    professions: List[ExportProfession] = []
    warnings_prof: Tuple[str, ...] = ()
    if include_professions:
        services = profession_services or ProfessionServices.default()
        professions, warnings_prof = _build_professions(
            rows,
            professions_weights_path=professions_weights_path,
            professions_list_path=professions_list_path,
            strict=professions_strict,
            services=services,
        )

    warnings = tuple(sorted(set(warnings_sk) | set(warnings_prof)))
    return ExportResult(skills=skills, professions=professions, totals=totals, warnings=warnings)


def write_csv(result: ExportResult, path: Path) -> None:
    path = Path(path)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)

        # Skills section (keep 2 decimals)
        w.writerow(["[Skills]"])
        for s in result.skills:
            w.writerow([s.name, format(s.value, ".2f"), s.category])

        w.writerow([])

        # Professions section (keep 2 decimals)
        if result.professions:
            w.writerow(["[Professions]"])
            for p in result.professions:
                w.writerow([p.name, format(p.value, ".2f"), p.category])
            w.writerow([])

        # Totals section (integer)
        w.writerow(["[Totals]"])
        for t in result.totals:
            w.writerow([t.category, str(int(t.total))])

        # Optional: audit warnings at end (commented out for now)
        # if result.warnings:
        #     w.writerow([])
        #     w.writerow(["[Warnings]"])
        #     for msg in result.warnings:
        #         w.writerow([msg])
