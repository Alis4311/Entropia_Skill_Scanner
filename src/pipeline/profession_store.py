# pipeline/profession_store.py
from __future__ import annotations

from pathlib import Path
from typing import Optional

from .professions import ProfessionWeights, load_profession_weights

_PROF_CACHE: Optional[ProfessionWeights] = None


def get_profession_weights(path: Path) -> ProfessionWeights:
    global _PROF_CACHE
    if _PROF_CACHE is None:
        _PROF_CACHE = load_profession_weights(path)
    return _PROF_CACHE
