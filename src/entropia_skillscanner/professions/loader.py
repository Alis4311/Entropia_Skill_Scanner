from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Any

from .compute import ProfessionWeights


def load_profession_weights(path: Path) -> ProfessionWeights:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("professions.json must be an object mapping profession -> weights list")

    out: ProfessionWeights = {}
    for prof, weights in data.items():
        if not isinstance(prof, str) or not isinstance(weights, list):
            raise ValueError(f"Invalid professions.json entry for {prof!r}")
        cleaned: List[Dict[str, Any]] = []
        for w in weights:
            if not isinstance(w, dict) or "skill" not in w or "pct" not in w:
                raise ValueError(f"Invalid weight item for {prof!r}: {w!r}")
            skill = w["skill"]
            pct = w["pct"]
            if not isinstance(skill, str):
                raise ValueError(f"Invalid skill name for {prof!r}: {skill!r}")
            cleaned.append({"skill": skill, "pct": pct})
        out[prof] = cleaned
    return out


def load_profession_categories(path: Path) -> Dict[str, str]:
    """
    data/professions_list.json: list of objects like:
      {"activityName": "...", "category": "...", ...}

    Returns: activityName -> category
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("professions_list.json must be a JSON list")

    out: Dict[str, str] = {}
    for item in raw:
        if not isinstance(item, dict):
            continue
        name = item.get("activityName")
        cat = item.get("category")
        if isinstance(name, str) and name.strip() and isinstance(cat, str) and cat.strip():
            out[name.strip()] = cat.strip()
    return out
