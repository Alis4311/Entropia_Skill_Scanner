from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class SkillRow:
    """Normalized representation of a single extracted skill entry."""

    name: str
    value: float          
    added: str


@dataclass(frozen=True)
class PipelineRow:
    """Single row emitted by the OCR pipeline."""

    name: str
    value: str            


@dataclass(frozen=True)
class PipelineResult:
    """Return type for run_pipeline."""
    rows: List[PipelineRow]
    status: str
    ok: bool
    warnings: Tuple[str, ...] = ()


@dataclass(frozen=True)
class ProfessionResult:
    """Profession computation output."""

    name: str
    value: Decimal
    category: Optional[str] = None
    missing_skills: Tuple[str, ...] = ()
    pct_sum: Optional[Decimal] = None


@dataclass(frozen=True)
class ExportSkill:
    """Skill ready for CSV emission (with category attached)."""

    name: str
    value: Decimal        
    category: str


@dataclass(frozen=True)
class ExportTotals:
    category: str
    total: Decimal       


@dataclass(frozen=True)
class ExportProfession:
    name: str
    value: Decimal
    category: str


@dataclass(frozen=True)
class ExportResult:
    skills: List[ExportSkill]
    professions: List[ExportProfession]
    totals: List[ExportTotals]
    warnings: Tuple[str, ...] = ()
