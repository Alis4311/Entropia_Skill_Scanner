# io/import_skills_csv.py
from __future__ import annotations

import csv
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import List, Tuple

from entropia_skillscanner.models import SkillRow

_SECTION_SKILLS = "Skills"
_Q2 = Decimal("0.01")


class ImportError(Exception):
    pass


@dataclass(frozen=True)
class ImportSkillsResult:
    rows: List[SkillRow]
    warnings: Tuple[str, ...] = ()


def _parse_decimal(s: str, *, ctx: str) -> Decimal:
    s = s.strip()
    try:
        return Decimal(s)
    except InvalidOperation as e:
        raise ImportError(f"Invalid decimal '{s}' in {ctx}") from e


def load_skill_rows_from_export_csv(
    path: Path,
    *,
    added_label: str = "imported",
    strict: bool = True,
) -> ImportSkillsResult:
    """
    Reads ONLY the [Skills] section from an export CSV and returns SkillRow list.

    Expected rows inside [Skills]:
      name,value,category

    Everything else (Professions/Totals/etc) is ignored on purpose.
    """
    if not path.exists():
        raise ImportError(f"File not found: {path}")

    rows: List[SkillRow] = []
    warnings: List[str] = []
    in_skills = False

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        for line_no, row in enumerate(reader, start=1):
            if not row or all(not cell.strip() for cell in row):
                continue

            first = row[0].strip()

            # Section header like [Skills]
            if first.startswith("[") and first.endswith("]") and len(row) == 1:
                section = first[1:-1].strip()
                in_skills = (section == _SECTION_SKILLS)
                continue

            if not in_skills:
                continue

            # name,value,category (category ignored here)
            if len(row) < 2:
                msg = f"Bad [Skills] row at line {line_no}: {row}"
                if strict:
                    raise ImportError(msg)
                warnings.append(msg)
                continue

            name = row[0].strip()
            if not name:
                msg = f"Empty skill name in [Skills] at line {line_no}"
                if strict:
                    raise ImportError(msg)
                warnings.append(msg)
                continue

            val_s = row[1].strip()
            d = _parse_decimal(val_s, ctx=f"[Skills] line {line_no}").quantize(_Q2, rounding=ROUND_HALF_UP)

            # SkillRow.value is float in your app; keep it consistent.
            value = float(d)

            rows.append(SkillRow(name=name, value=value, added=added_label))

    if not rows:
        msg = "No skills found (missing [Skills] section or empty section)."
        if strict:
            raise ImportError(msg)
        warnings.append(msg)

    return ImportSkillsResult(rows=rows, warnings=tuple(warnings))
