from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

from entropia_skillscanner.core import SkillRow
from entropia_skillscanner.taxonomy import SKILL_TO_CATEGORY


def _value_key(value: float) -> str:
    return f"{value:.2f}"


def _is_known_skill(name: str) -> bool:
    return name in SKILL_TO_CATEGORY


def merge_skill_rows(existing: Sequence[SkillRow], incoming: Iterable[SkillRow]) -> List[SkillRow]:
    merged = list(existing)
    name_to_index: Dict[str, int] = {row.name: idx for idx, row in enumerate(merged)}
    unknown_by_value: Dict[str, int] = {}

    for idx, row in enumerate(merged):
        if not _is_known_skill(row.name):
            unknown_by_value[_value_key(row.value)] = idx

    for row in incoming:
        if row.name in name_to_index:
            idx = name_to_index[row.name]
            merged[idx] = row
            if _is_known_skill(row.name):
                unknown_by_value = {k: v for k, v in unknown_by_value.items() if v != idx}
            else:
                unknown_by_value[_value_key(row.value)] = idx
            continue

        value_key = _value_key(row.value)
        if _is_known_skill(row.name) and value_key in unknown_by_value:
            idx = unknown_by_value[value_key]
            old_name = merged[idx].name
            merged[idx] = row
            name_to_index.pop(old_name, None)
            name_to_index[row.name] = idx
            unknown_by_value = {k: v for k, v in unknown_by_value.items() if v != idx}
            continue

        name_to_index[row.name] = len(merged)
        merged.append(row)
        if not _is_known_skill(row.name):
            unknown_by_value[value_key] = name_to_index[row.name]

    return merged


def dedupe_skill_rows(rows: Iterable[SkillRow]) -> List[SkillRow]:
    return merge_skill_rows([], rows)
