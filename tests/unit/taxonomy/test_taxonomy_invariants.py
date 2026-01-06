from __future__ import annotations

import pytest

from entropia_skillscanner import taxonomy


def test_validate_mappings_reports_no_issues():
    issues = taxonomy.validate_mappings(strict=False)
    assert issues == []


@pytest.mark.parametrize("schema", [taxonomy.SCHEMA_OLD, taxonomy.SCHEMA_NEW])
def test_all_skills_map_into_schema(schema):
    missing = []
    for skill in taxonomy.SKILL_TO_CATEGORY:
        cat = taxonomy.get_category(skill, schema=schema)
        if cat is None:
            missing.append(skill)
    assert not missing, f"skills missing in schema {schema.name}: {missing}"


def test_schemas_cover_all_canonical_categories():
    canonical = set(taxonomy.CANONICAL_CATEGORY_ORDER)
    for schema in (taxonomy.SCHEMA_OLD, taxonomy.SCHEMA_NEW):
        mapped = {schema.category_alias.get(cat, cat) for cat in canonical}
        assert mapped.issubset(set(schema.category_order)), f"{schema.name} missing categories: {mapped - set(schema.category_order)}"
