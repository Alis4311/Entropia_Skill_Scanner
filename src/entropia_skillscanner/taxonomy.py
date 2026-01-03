from __future__ import annotations

from typing import List, Mapping

# Canonical category order for deterministic CSV output
CATEGORY_ORDER: List[str] = [
    "Attributes",
    "General",
    "Combat",
    "Defense",
    "Medical",
    "Science",
    "Mining",
    "Manufacturing",
    "Tailoring",
    "Beauty",
    "Mindforce",
    "Taming",
    "Vehicle",
    "Space",
]


# Mapping of normalized skill name -> category.
# NOTE: Keep skill names normalized to match the vocabulary used by OCR snapping.
SKILL_TO_CATEGORY: Mapping[str, str] = {
    # Attributes
    "Agility": "Attributes",
    "Dexterity": "Attributes",
    "Intelligence": "Attributes",
    "Quickness": "Attributes",
    "Strength": "Attributes",

    # General
    "Alertness": "General",
    "Anatomy": "General",
    "Athletics": "General",
    "Perception": "General",

    # Combat
    "Bravado": "Combat",
    "Combat Reflexes": "Combat",
    "Combat Sense": "Combat",
    "Coolness": "Combat",
    "Handgun": "Combat",
    "Heavy Melee Weapons": "Combat",
    "Heavy Weapons Handling": "Combat",
    "Inflict Melee Damage": "Combat",
    "Inflict Ranged Critical Hit": "Combat",
    "Inflict Ranged Damage": "Combat",
    "Laser Weaponry Technology": "Combat",
    "Longblades": "Combat",
    "Marksmanship": "Combat",
    "Martial Arts": "Combat",
    "Melee Combat": "Combat",
    "Melee Weaponry Technology": "Combat",
    "Power Fist": "Combat",
    "Rifle": "Combat",
    "Shortblades": "Combat",
    "Sniper (Hit)": "Combat",
    "Support Weapon Systems": "Combat",
    "Weapons Handling": "Combat",
    "Explosive Projectile Weaponry": "Combat",

    # Defense
    "Avoidance": "Defense",
    "Evade": "Defense",
    "Jamming": "Defense",

    # Medical
    "Diagnoser": "Medical",
    "First Aid": "Medical",
    "Medicine": "Medical",
    "Treatment": "Medical",

    # Science
    "Biology": "Science",
    "Botany": "Science",
    "Botany Technician": "Science",
    "Computer Engineering": "Science",
    "Physics": "Science",
    "Robotology": "Science",
    "Science": "Science",
    "Zoology": "Science",

    # Mining
    "Drilling": "Mining",
    "Geology": "Mining",
    "Mining": "Mining",
    "Mining Engineering": "Mining",
    "Mining Expertise": "Mining",

    # Manufacturing / crafting
    "Electronics": "Manufacturing",
    "Engineering": "Manufacturing",
    "Mechanics": "Manufacturing",
    "Plastics Engineering": "Manufacturing",
    "Technology Design": "Manufacturing",

    # Tailoring
    "Tailoring": "Tailoring",
    "Textile Fabrication": "Tailoring",

    # Beauty
    "Beauty Sense": "Beauty",

    # Mindforce (grouped here, but kept in category list)
    "Mindforce Harmonics": "Mindforce",

    # Taming
    # (placeholder in case future skills added)

    # Vehicle
    "Vehicle Technology": "Vehicle",

    # Space
    # (placeholder; add concrete mappings if applicable)
}


def all_categories() -> List[str]:
    return list(CATEGORY_ORDER)


def get_category(skill: str) -> str:
    return SKILL_TO_CATEGORY.get(skill, "")
