# models.py
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True)
class SkillRow:
    """Normalized representation of a single extracted skill entry."""

    name: str
    value: float          # pipeline/UI-derived (float is fine here)
    added: str


@dataclass(frozen=True)
class ExportSkill:
    """Skill ready for CSV emission (with category attached)."""

    name: str
    value: Decimal        # export uses Decimal
    category: str


@dataclass(frozen=True)
class ExportTotals:
    category: str
    total: Decimal        # export uses Decimal
