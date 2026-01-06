from pathlib import Path

import pytest

from entropia_skillscanner.exporter import ExportError, build_export, write_csv
from entropia_skillscanner.models import SkillRow
from entropia_skillscanner.taxonomy import SKILL_TO_CATEGORY, CATEGORY_ORDER


def test_build_export_orders_by_category_and_name(tmp_path: Path):
    rows = [
        SkillRow(name="Rifle", value=10.5, added="t1"),
        SkillRow(name="Agility", value=5.0, added="t1"),
        SkillRow(name="Evade", value=2.0, added="t1"),
    ]

    result = build_export(rows)

    # Category order: Attributes -> General -> Combat -> Defense...
    assert [s.name for s in result.skills] == ["Agility", "Rifle", "Evade"]
    assert result.totals[0].category == "Total"
    assert result.totals[0].total == 18


def test_build_export_fails_on_unknown_skill():
    rows = [SkillRow(name="MadeUpSkill", value=1.0, added="t1")]
    with pytest.raises(ExportError):
        build_export(rows)


def test_write_csv_round_trips(tmp_path: Path):
    rows = [
        SkillRow(name="Rifle", value=10.0, added="t1"),
        SkillRow(name="Handgun", value=2.345, added="t1"),
    ]
    result = build_export(rows)
    out = tmp_path / "skills.csv"
    write_csv(result, out)

    content = out.read_text(encoding="utf-8").strip().splitlines()
    assert content[0] == "[Skills]"
    assert content[1] == "Handgun,2.35,Combat"  # alphabetical inside category
    assert content[2] == "Rifle,10.00,Combat"
    assert "[Totals]" in content
