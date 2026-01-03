from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SkillRow:
    """Normalized representation of a single extracted skill entry."""

    name: str
    value: float
    added: str


@dataclass(frozen=True)
class ExportSkill:
    """Skill ready for CSV emission (with category attached)."""

    name: str
    value: float
    category: str


@dataclass(frozen=True)
class ExportTotals:
    category: str
    total: float

