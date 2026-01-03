from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from .models import ExportSkill, ExportTotals, SkillRow
from .taxonomy import CATEGORY_ORDER, SKILL_TO_CATEGORY, get_category


@dataclass(frozen=True)
class ExportResult:
    skills: List[ExportSkill]
    totals: List[ExportTotals]


class ExportError(Exception):
    """Raised when export preconditions fail (e.g., missing categories)."""


def _attach_categories(rows: Sequence[SkillRow]) -> List[ExportSkill]:
    out: List[ExportSkill] = []
    missing: List[str] = []
    for r in rows:
        cat = get_category(r.name)
        if not cat:
            missing.append(r.name)
            continue
        out.append(ExportSkill(name=r.name, value=r.value, category=cat))
    if missing:
        uniq = ", ".join(sorted(set(missing)))
        raise ExportError(f"uncategorized skills: {uniq}")
    return out


def _category_totals(skills: Iterable[ExportSkill]) -> List[ExportTotals]:
    totals: Dict[str, float] = {}
    for s in skills:
        totals[s.category] = totals.get(s.category, 0.0) + s.value
    ordered: List[ExportTotals] = []
    for cat in CATEGORY_ORDER:
        if cat in totals:
            ordered.append(ExportTotals(category=cat, total=totals[cat]))
    return ordered


def build_export(rows: Sequence[SkillRow]) -> ExportResult:
    if not rows:
        raise ExportError("no rows to export")
    skills = _attach_categories(rows)
    skills = sorted(
        skills,
        key=lambda s: (CATEGORY_ORDER.index(s.category) if s.category in CATEGORY_ORDER else len(CATEGORY_ORDER), s.name),
    )
    totals = _category_totals(skills)
    overall_total = sum(s.value for s in skills)
    totals.append(ExportTotals(category="Total", total=overall_total))
    return ExportResult(skills=skills, totals=totals)


def write_csv(result: ExportResult, path: Path) -> None:
    path = Path(path)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        # Skills section
        w.writerow(["[Skills]"])
        for s in result.skills:
            w.writerow([s.name, f"{s.value:.2f}", s.category])
        w.writerow([])  # blank line
        # Totals section
        w.writerow(["[Totals]"])
        for t in result.totals:
            w.writerow([t.category, f"{t.total:.2f}"])

