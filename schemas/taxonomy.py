"""Single source of truth for every label used in the project.

The same definitions are rendered into:
  * the Pydantic enums (validation),
  * all three classification prompts (identical definitions => fair comparison),
  * the human annotation guide and annotation UI.

To extend the taxonomy, add a member to the enum AND a definition to the
matching ``*_DEFINITIONS`` dict; ``tests/test_schemas.py`` checks they stay in sync.
"""

from __future__ import annotations

from enum import Enum


class OverallSentiment(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"


class AspectSentiment(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class AspectCategory(str, Enum):
    PERFORMANCE = "performance"
    QUALITY_DURABILITY = "quality_durability"
    DESIGN_BUILD = "design_build"
    EASE_OF_USE = "ease_of_use"
    INSTALLATION_SETUP = "installation_setup"
    NOISE = "noise"
    SIZE_CAPACITY = "size_capacity"
    COMPATIBILITY = "compatibility"
    PRICE_VALUE = "price_value"
    CUSTOMER_SERVICE = "customer_service"
    SHIPPING_PACKAGING = "shipping_packaging"
    SAFETY = "safety"
    OTHER = "other"


class ComplaintCategory(str, Enum):
    PRODUCT_DEFECT = "product_defect"
    POOR_DURABILITY = "poor_durability"
    PERFORMANCE_ISSUE = "performance_issue"
    NOISE_ISSUE = "noise_issue"
    INSTALLATION_DIFFICULTY = "installation_difficulty"
    USABILITY_ISSUE = "usability_issue"
    COMPATIBILITY_FIT_ISSUE = "compatibility_fit_issue"
    POOR_VALUE = "poor_value"
    SHIPPING_DAMAGE = "shipping_damage"
    MISSING_PARTS = "missing_parts"
    NOT_AS_DESCRIBED = "not_as_described"
    CUSTOMER_SERVICE_ISSUE = "customer_service_issue"
    SAFETY_CONCERN = "safety_concern"
    OTHER = "other"


OVERALL_SENTIMENT_DEFINITIONS: dict[OverallSentiment, str] = {
    OverallSentiment.POSITIVE: "The reviewer is satisfied overall; any criticism is absent or trivial.",
    OverallSentiment.NEGATIVE: "The reviewer is dissatisfied overall; any praise is absent or trivial.",
    OverallSentiment.NEUTRAL: "No clear opinion: factual, descriptive, or too vague to judge.",
    OverallSentiment.MIXED: "Substantial praise AND substantial criticism (e.g. 'works great but very loud').",
}

ASPECT_SENTIMENT_DEFINITIONS: dict[AspectSentiment, str] = {
    AspectSentiment.POSITIVE: "The text praises this aspect.",
    AspectSentiment.NEGATIVE: "The text criticises this aspect.",
    AspectSentiment.NEUTRAL: "The aspect is mentioned factually without evaluation.",
}

ASPECT_CATEGORY_DEFINITIONS: dict[AspectCategory, str] = {
    AspectCategory.PERFORMANCE: "How well the product does its core job (cooling, heating, filtering, cleaning, power, speed, results).",
    AspectCategory.QUALITY_DURABILITY: "Build quality, reliability, how long it lasts, breaking or failing over time.",
    AspectCategory.DESIGN_BUILD: "Appearance, materials, finish, physical construction and design choices.",
    AspectCategory.EASE_OF_USE: "Controls, operation, cleaning/maintenance effort, convenience, instructions for use.",
    AspectCategory.INSTALLATION_SETUP: "Installing, assembling, mounting or initial setup, including installation instructions.",
    AspectCategory.NOISE: "Sound level, vibration, rattling, humming.",
    AspectCategory.SIZE_CAPACITY: "Physical dimensions, capacity, footprint, weight.",
    AspectCategory.COMPATIBILITY: "Whether it fits / works with the reviewer's model or appliance (parts, filters, replacements).",
    AspectCategory.PRICE_VALUE: "Price, value for money, cost compared with alternatives or OEM parts.",
    AspectCategory.CUSTOMER_SERVICE: "Seller or manufacturer support, warranty, returns, refunds, replacements.",
    AspectCategory.SHIPPING_PACKAGING: "Delivery speed, packaging, condition on arrival.",
    AspectCategory.SAFETY: "Burning smells, smoke, overheating, leaks causing hazard, electric shock, unsafe materials.",
    AspectCategory.OTHER: "A product aspect that clearly fits none of the categories above.",
}

COMPLAINT_CATEGORY_DEFINITIONS: dict[ComplaintCategory, str] = {
    ComplaintCategory.PRODUCT_DEFECT: "Arrived broken, dead on arrival or with a manufacturing defect.",
    ComplaintCategory.POOR_DURABILITY: "Worked at first but broke, wore out or stopped working too soon.",
    ComplaintCategory.PERFORMANCE_ISSUE: "Works but performs poorly at its core job (weak, slow, ineffective).",
    ComplaintCategory.NOISE_ISSUE: "Too loud, rattles, vibrates, hums.",
    ComplaintCategory.INSTALLATION_DIFFICULTY: "Hard to install/assemble, poor installation instructions.",
    ComplaintCategory.USABILITY_ISSUE: "Hard or inconvenient to operate, clean or maintain.",
    ComplaintCategory.COMPATIBILITY_FIT_ISSUE: "Did not fit or work with the stated/expected model; wrong size for its slot.",
    ComplaintCategory.POOR_VALUE: "Overpriced or not worth the money.",
    ComplaintCategory.SHIPPING_DAMAGE: "Damaged in transit, poor packaging, late or lost delivery.",
    ComplaintCategory.MISSING_PARTS: "Parts, accessories or hardware missing from the package.",
    ComplaintCategory.NOT_AS_DESCRIBED: "Differs from the listing (counterfeit, different item, misleading description).",
    ComplaintCategory.CUSTOMER_SERVICE_ISSUE: "Bad experience with seller/manufacturer support, warranty or returns.",
    ComplaintCategory.SAFETY_CONCERN: "Hazard: smoke, burning smell, overheating, sparks, dangerous leak.",
    ComplaintCategory.OTHER: "A genuine complaint that fits none of the categories above.",
}


def render_definitions(definitions: dict, title: str) -> str:
    lines = [f"{title}:"]
    lines += [f"  - {member.value}: {text}" for member, text in definitions.items()]
    return "\n".join(lines)


def taxonomy_prompt_block() -> str:
    """The exact label definitions shared by every classification prompt."""
    return "\n\n".join(
        [
            render_definitions(OVERALL_SENTIMENT_DEFINITIONS, "OVERALL SENTIMENT (choose exactly one)"),
            render_definitions(ASPECT_CATEGORY_DEFINITIONS, "ASPECT CATEGORIES"),
            render_definitions(ASPECT_SENTIMENT_DEFINITIONS, "ASPECT SENTIMENT"),
            render_definitions(COMPLAINT_CATEGORY_DEFINITIONS, "COMPLAINT CATEGORIES (multi-label; empty list if no complaint)"),
        ]
    )


def values(enum_cls: type[Enum]) -> list[str]:
    return [m.value for m in enum_cls]
