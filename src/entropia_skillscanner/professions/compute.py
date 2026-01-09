from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, Iterable, List, Mapping, Tuple, Any

from entropia_skillscanner.taxonomy import SKILL_TO_CATEGORY

_Q2 = Decimal("0.01")
ATTRIBUTE_FACTOR = Decimal("20")


def q2(x: Decimal) -> Decimal:
    return x.quantize(_Q2, rounding=ROUND_HALF_UP)


@dataclass(frozen=True)
class ProfessionValue:
    value: Decimal
    missing_skills: Tuple[str, ...] = ()
    pct_sum: Decimal = Decimal("0")  # useful for auditing weird weight sets


# Types: { "Profession": [ {"skill": "...", "pct": 5}, ... ] }
ProfessionWeights = Dict[str, List[Dict[str, Any]]]


def validate_profession_weights(
    *,
    profession_weights: Mapping[str, List[Dict[str, Any]]],
    known_skills: Iterable[str],
) -> List[str]:
    """
    Returns issues (strings). Does not raise.
    Checks:
      - referenced skills exist
      - pct sums (informational)
    """
    known = set(known_skills)
    issues: List[str] = []

    for prof, weights in profession_weights.items():
        missing = sorted({w["skill"] for w in weights if w["skill"] not in known})
        if missing:
            issues.append(f"{prof}: references unknown skills: {missing}")

        pct_sum = sum((Decimal(str(w["pct"])) for w in weights), Decimal("0"))
        # informational only
        if pct_sum != Decimal("100"):
            issues.append(f"{prof}: pct sum = {pct_sum}")

    return issues


def compute_professions(
    *,
    skills: Mapping[str, Decimal],  # canonical skill -> value (Decimal)
    profession_weights: Mapping[str, List[Dict[str, Any]]],
    strict: bool,
) -> Dict[str, ProfessionValue]:
    """
    value = Σ skill_value * (pct / 100)
    Missing skills:
      strict=True  -> raise
      strict=False -> treat as 0 and record missing_skills
    """
    out: Dict[str, ProfessionValue] = {}

    for prof, weights in profession_weights.items():
        total = Decimal("0")
        missing: List[str] = []
        pct_sum = Decimal("0")

        for w in weights:
            skill = w["skill"]
            pct = w["pct"] if isinstance(w["pct"], Decimal) else Decimal(str(w["pct"]))
            pct_sum += pct

            if pct == 0:
                continue

            if skill not in skills:
                if strict:
                    raise KeyError(f"Profession '{prof}' missing required skill '{skill}'")
                missing.append(skill)
                continue
            factor = ATTRIBUTE_FACTOR if SKILL_TO_CATEGORY.get(skill) == "Attributes" else Decimal("1")
            total += (skills[skill] * factor) * (pct / Decimal("100"))

        out[prof] = ProfessionValue(
            value=total / 100,
            missing_skills=tuple(sorted(set(missing))),
            pct_sum=pct_sum,
        )

    return out
