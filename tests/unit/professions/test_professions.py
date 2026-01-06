from decimal import Decimal
from pathlib import Path

import pytest

from entropia_skillscanner.professions import compute_professions, load_profession_categories, load_profession_weights


def test_compute_professions_applies_attribute_factor():
    skills = {"Agility": Decimal("10"), "Rifle": Decimal("10")}
    weights = {
        "Hunter": [
            {"skill": "Agility", "pct": 50},
            {"skill": "Rifle", "pct": 50},
        ]
    }

    result = compute_professions(skills=skills, profession_weights=weights, strict=True)
    hunter = result["Hunter"]

    assert hunter.value == Decimal("1.05")
    assert hunter.missing_skills == ()
    assert hunter.pct_sum == Decimal("100")


def test_compute_professions_handles_missing_and_pct_sum():
    skills = {"Rifle": Decimal("4")}
    weights = {
        "Crafter": [
            {"skill": "UnknownSkill", "pct": 75},
            {"skill": "Rifle", "pct": 25},
        ]
    }

    result = compute_professions(skills=skills, profession_weights=weights, strict=False)
    crafter = result["Crafter"]

    assert crafter.value == Decimal("0.01")
    assert crafter.missing_skills == ("UnknownSkill",)
    assert crafter.pct_sum == Decimal("100")


def test_loaders_read_json(tmp_path: Path):
    weights_path = tmp_path / "professions.json"
    weights_path.write_text(
        """
        {
            "Hunter": [{"skill": "Rifle", "pct": 10}]
        }
        """,
        encoding="utf-8",
    )

    categories_path = tmp_path / "professions_list.json"
    categories_path.write_text(
        """
        [
            {"activityName": "Hunter", "category": "Combat"}
        ]
        """,
        encoding="utf-8",
    )

    weights = load_profession_weights(weights_path)
    categories = load_profession_categories(categories_path)

    assert weights["Hunter"][0]["skill"] == "Rifle"
    assert categories["Hunter"] == "Combat"


def test_load_profession_weights_rejects_non_object(tmp_path: Path):
    path = tmp_path / "professions.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(ValueError):
        load_profession_weights(path)
