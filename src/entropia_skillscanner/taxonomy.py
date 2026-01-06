from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Set


# ---------------------------------------------------------------------------
# Canonical taxonomy (single source of truth)
#
# Skills are categorized ONCE using canonical category names.
# Export "OLD" vs "NEW" is handled by schema-specific:
#   1) category aliasing (rename)
#   2) category ordering
# ---------------------------------------------------------------------------

CANONICAL_CATEGORY_ORDER: List[str] = [
    "Attributes",
    "General",
    "Combat",
    "Defense",
    "Medical",
    "Science",
    "Information",
    "Mining",
    "Construction",
    "Design",      # canonical (NEW-style naming)
    "Mindforce",
    "Beauty",
    "Social",
]
CATEGORY_ORDER = CANONICAL_CATEGORY_ORDER

# Mapping of canonical skill name -> CANONICAL category.
# NOTE: Keys MUST match your snapped canonical vocab exactly.
SKILL_TO_CATEGORY: Mapping[str, str] = {
        # ------------------------------------------------------------------
    # Attributes
    # ------------------------------------------------------------------
    "Agility": "Attributes",
    "Health": "Attributes",
    "Intelligence": "Attributes",
    "Psyche": "Attributes",
    "Stamina": "Attributes",
    "Strength": "Attributes",

    # ------------------------------------------------------------------
    # General
    # ------------------------------------------------------------------
    "Alertness": "General",
    "Athletics": "General",
    "Bravado": "General",
    "Coolness": "General",
    "Courage": "General",
    "Dexterity": "General",
    "Perception": "General",
    "Serendipity": "General",
    "Intuition": "General",
    "Quickness": "General",

    # ------------------------------------------------------------------
    # Combat
    # ------------------------------------------------------------------
    "Aim": "Combat",
    "Clubs": "Combat",
    "Combat Reflexes": "Combat",
    "Combat Sense": "Combat",
    "Commando": "Combat",
    "Handgun": "Combat",
    "Heavy Melee Weapons": "Combat",
    "Heavy Weapons": "Combat",
    "Inflict Melee Damage": "Combat",
    "Inflict Ranged Damage": "Combat",
    "Kill Strike": "Combat",
    "Light Melee Weapons": "Combat",
    "Longblades": "Combat",
    "Marksmanship": "Combat",
    "Martial Arts": "Combat",
    "Melee Combat": "Combat",
    "Melee Damage Assessment": "Combat",
    "Mining Laser Operator": "Combat",
    "Power Fist": "Combat",
    "Ranged Damage Assessment": "Combat",
    "Rifle": "Combat",
    "Shortblades": "Combat",
    "Support Weapon Systems": "Combat",
    "Weapons Handling": "Combat",
    "Whip": "Combat",
    "Wounding": "Combat",

    # ------------------------------------------------------------------
    # Defense
    # ------------------------------------------------------------------
    "Avoidance": "Defense",
    "Dispense Decoy": "Defense",
    "Dodge": "Defense",
    "Evade": "Defense",

    # ------------------------------------------------------------------
    # Medical
    # ------------------------------------------------------------------
    "Anatomy": "Medical",
    "Diagnosis": "Medical",
    "First Aid": "Medical",
    "Medicine": "Medical",
    "Treatment": "Medical",

    # ------------------------------------------------------------------
    # Science
    # ------------------------------------------------------------------
    "Analysis": "Science",
    "Animal Lore": "Science",
    "Animal Taming": "Science",
    "Biology": "Science",
    "Botany": "Science",
    "Computer": "Science",
    "Deep Space Knowledge": "Science",
    "Electronics": "Science",
    "Engineering": "Science",
    "Genetics": "Science",
    "Mechanics": "Science",
    "Robotology": "Science",
    "Scientist": "Science",
    "Spacecraft Pilot": "Science",
    "Spacecraft Systems": "Science",
    "Xenobiology": "Science",
    "Zoology": "Science",

    # ------------------------------------------------------------------
    # Information
    # ------------------------------------------------------------------
    "Mentor": "Information",
    "Probing": "Information",
    "Scan Animal": "Information",
    "Scan Human": "Information",
    "Scan Mutant": "Information",
    "Scan Robot": "Information",
    "Scan Technology": "Information",
    "Reclaiming": "Information",
    "Scourging": "Information",
    "Skinning": "Information",
    "Butchering": "Information",
    "Fragmentating": "Information",
    "Reaping": "Information",
    "Salvaging": "Information",
    "Scavenging": "Information",

    # ------------------------------------------------------------------
    # Mining
    # ------------------------------------------------------------------
    "Ground Assessment": "Mining",
    "Drilling": "Mining",
    "Extraction": "Mining",
    "Geology": "Mining",
    "Metallurgy": "Mining",
    "Mining": "Mining",
    "Prospecting": "Mining",
    "Surveying": "Mining",
    "Artefact Preservation": "Mining",
    "Precision Artefact Extraction": "Mining",
    "Treasure Sense": "Mining",
    "Resource Gathering": "Mining",
    "Archaeological Lore": "Mining",

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    "Armor Technology": "Construction",
    "Attachments Technology": "Construction",
    "BLP Weaponry Technology": "Construction",
    "Blueprint Comprehension": "Construction",
    "Carpentry": "Construction",
    "Chemistry": "Construction",
    "Enhancer Technology": "Construction",
    "Explosive Projectile Weaponry Technology": "Construction",
    "Gauss Weaponry Technology": "Construction",
    "Laser Weaponry Technology": "Construction",
    "Machinery": "Construction",
    "Manufacture Armor": "Construction",
    "Manufacture Attachments": "Construction",
    "Manufacture Electronic Equipment": "Construction",
    "Manufacture Enhancers": "Construction",
    "Manufacture Mechanical Equipment": "Construction",
    "Manufacture Metal Equipment": "Construction",
    "Manufacture Tools": "Construction",
    "Manufacture Vehicle": "Construction",
    "Manufacture Weapons": "Construction",
    "Material Extraction Methodology": "Construction",
    "Mining Laser Technology": "Construction",
    "Particle Beamer Technology": "Construction",
    "Plasma Weaponry Technology": "Construction",
    "Spacecraft Engineering": "Construction",
    "Spacecraft Weaponry": "Construction",
    "Texture Engineering": "Construction",
    "Tier Upgrading": "Construction",
    "Tools Technology": "Construction",
    "Vehicle Repairing": "Construction",
    "Vehicle Technology": "Construction",
    "Weapon Technology": "Construction",
    "Wood Carving": "Construction",
    "Wood Processing": "Construction",

    # ------------------------------------------------------------------
    # Design (canonical; OLD export aliases to "Tailoring")
    # ------------------------------------------------------------------
    "Color Matching": "Design",
    "Coloring": "Design",
    "Fashion Design": "Design",
    "Make Textile": "Design",
    "Tailoring": "Design",
    "Texture Pattern Matching": "Design",

    # ------------------------------------------------------------------
    # Mindforce
    # ------------------------------------------------------------------
    "Bioregenesis": "Mindforce",
    "Concentration": "Mindforce",
    "Cryogenics": "Mindforce",
    "Electrokinesis": "Mindforce",
    "Force Merge": "Mindforce",
    "Jamming": "Mindforce",
    "Mindforce Harmony": "Mindforce",
    "Power Catalyst": "Mindforce",
    "Pyrokinesis": "Mindforce",
    "Sweat Gatherer": "Mindforce",
    "Telepathy": "Mindforce",
    "Translocation": "Mindforce",

    # ------------------------------------------------------------------
    # Beauty
    # ------------------------------------------------------------------
    "Body Sculpting": "Beauty",
    "Face Sculpting": "Beauty",
    "Hair Stylist": "Beauty",
    "Plastic Surgery": "Beauty",

    # ------------------------------------------------------------------
    # Social
    # ------------------------------------------------------------------
    "Promoter Rating": "Social",
    "Reputation": "Social",
}


# ---------------------------------------------------------------------------
# Export schemas (views)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExportSchema:
    name: str
    category_order: Sequence[str]
    category_alias: Mapping[str, str]  # canonical -> schema label


SCHEMA_OLD = ExportSchema(
    name="OLD",
    category_order=[
        "Attributes",
        "General",
        "Combat",
        "Defense",
        "Medical",
        "Science",
        "Information",
        "Mining",
        "Construction",
        "Tailoring",   # OLD label for canonical "Design"
        "Mindforce",
        "Beauty",
        "Social",
    ],
    category_alias={
        "Design": "Tailoring",
    },
)

SCHEMA_NEW = ExportSchema(
    name="NEW",
    category_order=[
        "Attributes",
        "General",
        "Combat",
        "Defense",
        "Medical",
        "Science",
        "Information",
        "Mining",
        "Construction",
        "Design",
        "Mindforce",
        "Beauty",
        "Social",
    ],
    category_alias={},  # identity mapping
)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def all_categories(*, schema: ExportSchema = SCHEMA_OLD) -> List[str]:
    """Categories in deterministic order for a given export schema."""
    return list(schema.category_order)


def get_category(skill: str, *, schema: ExportSchema = SCHEMA_OLD) -> Optional[str]:
    """
    Return the schema-category for a canonical skill name.
    Returns None if the skill is unknown or schema cannot represent the category.
    """
    canon = SKILL_TO_CATEGORY.get(skill)
    if canon is None:
        return None

    out = schema.category_alias.get(canon, canon)
    if out not in schema.category_order:
        # Schema cannot represent this category (misconfigured schema)
        return None
    return out


def get_category_old(skill: str) -> str:
    """Back-compat convenience wrapper (returns '' when unknown, like the old version)."""
    return get_category(skill, schema=SCHEMA_OLD) or ""


def get_category_new(skill: str) -> str:
    """New schema wrapper (returns '' when unknown, like the old version)."""
    return get_category(skill, schema=SCHEMA_NEW) or ""


# ---------------------------------------------------------------------------
# Validation (recommended to run at startup / export time)
# ---------------------------------------------------------------------------

def validate_mappings(*, strict: bool = True) -> List[str]:
    """
    Validate that:
      - canonical categories used by SKILL_TO_CATEGORY are known
      - schemas can represent all canonical categories used
    Returns a list of human-readable issues. If strict=True and issues exist, raises ValueError.
    """
    issues: List[str] = []

    canonical_set: Set[str] = set(CANONICAL_CATEGORY_ORDER)
    used_canonical: Set[str] = set(SKILL_TO_CATEGORY.values())

    unknown_canonical = sorted(used_canonical - canonical_set)
    if unknown_canonical:
        issues.append(f"SKILL_TO_CATEGORY uses unknown canonical categories: {unknown_canonical}")

    for schema in (SCHEMA_OLD, SCHEMA_NEW):
        schema_set = set(schema.category_order)
        for canon_cat in sorted(used_canonical):
            schema_cat = schema.category_alias.get(canon_cat, canon_cat)
            if schema_cat not in schema_set:
                issues.append(
                    f"Schema {schema.name} cannot represent canonical category '{canon_cat}' "
                    f"(maps to '{schema_cat}' not in category_order)"
                )

    if strict and issues:
        raise ValueError("Category mapping validation failed:\n- " + "\n- ".join(issues))
    return issues
