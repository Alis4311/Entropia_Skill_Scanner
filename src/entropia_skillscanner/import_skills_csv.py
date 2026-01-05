# io/import_skills_csv.py
from __future__ import annotations

import csv
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import List, Optional, Tuple


_SECTION_SKILLS = "Skills"


class ImportError(Exception):
    pass


@dataclass(frozen=True)
class ImportedSkill:
    name: str
    value: Decimal
    category: str  # imported, but you can ignore it if your taxonomy is source of truth


@dataclass(frozen=True)
class ImportSkillsResult:
    skills: List[ImportedSkill]
    warnings: Tuple[str, ...] = ()


def _parse_decimal(s: str, *, ctx: str) -> Decimal:
    s = s.strip()
    try:
        return Decimal(s)
    except InvalidOperation as e:
        raise ImportError(f"Invalid decimal '{s}' in {ctx}") from e


def load_skills_section(path: Path, *, strict: bool = True) -> ImportSkillsResult:
    """
    Reads ONLY the [Skills] section from an export CSV.

    Expected rows inside [Skills]:
      name,value,category

    Everything else is ignored (Totals/Professions/etc).
    """
    if not path.exists():
        raise ImportError(f"File not found: {path}")

    skills: List[ImportedSkill] = []
    warnings: List[str] = []

    in_skills: bool = False

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        for row_idx, row in enumerate(reader, start=1):
            if not row or all(not cell.strip() for cell in row):
                continue

            first = row[0].strip()

            # Section header?
            if first.startswith("[") and first.endswith("]") and len(row) == 1:
                sec = first[1:-1].strip()
                in_skills = (sec == _SECTION_SKILLS)
                continue

            if not in_skills:
                continue

            # Parse skill row
            if len(row) < 3:
                msg = f"Expected 3 columns (name,value,category) in [Skills] at line {row_idx}: {row}"
                if strict:
                    raise ImportError(msg)
                warnings.append(msg)
                continue

            name = row[0].strip()
            val_s = row[1].strip()
            category = row[2].strip()

            if not name:
                msg = f"Empty skill name in [Skills] at line {row_idx}"
                if strict:
                    raise ImportError(msg)
                warnings.append(msg)
                continue

            value = _parse_decimal(val_s, ctx=f"[Skills] line {row_idx}")
            skills.append(ImportedSkill(name=name, value=value, category=category))

    if not skills and strict:
        raise ImportError("No [Skills] rows found (or missing [Skills] section).")

    return ImportSkillsResult(skills=skills, warnings=tuple(warnings))
