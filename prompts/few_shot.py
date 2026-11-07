"""Strategy B — Few-shot: same instructions + labelled demonstrations.

The demonstrations below are SYNTHETIC reviews written by the project author
for illustration. They are not taken from the dataset, so they cannot leak into
the 300-review evaluation set. The same examples, in the same order, are used
for every few-shot run (consistency across experiments).

Chosen to cover: multi-aspect positive, negative with several complaints,
mixed sentiment, compatibility issue, and a review with no product aspects.
"""

from __future__ import annotations

import json

from langchain_core.prompts import ChatPromptTemplate

from prompts.common import EXTRACTION_RULES, SYSTEM_ROLE

FEW_SHOT_EXAMPLES: list[dict] = [
    {
        "review": "Installed this ice maker in about 20 minutes, the instructions were clear. It makes plenty of ice and is much quieter than my old one.",
        "output": {
            "overall_sentiment": "positive",
            "aspects": [
                {"aspect_name": "installation", "aspect_category": "installation_setup", "sentiment": "positive", "evidence": "Installed this ice maker in about 20 minutes, the instructions were clear", "confidence": 0.95},
                {"aspect_name": "ice production", "aspect_category": "performance", "sentiment": "positive", "evidence": "It makes plenty of ice", "confidence": 0.9},
                {"aspect_name": "noise level", "aspect_category": "noise", "sentiment": "positive", "evidence": "much quieter than my old one", "confidence": 0.9},
            ],
            "complaint_categories": [],
            "confidence": 0.92,
            "rationale": "Praise for installation, output and noise; no criticism.",
        },
    },
    {
        "review": "The heating element burned out after two weeks. Seller never answered my emails about a replacement. Total waste of money.",
        "output": {
            "overall_sentiment": "negative",
            "aspects": [
                {"aspect_name": "heating element", "aspect_category": "quality_durability", "sentiment": "negative", "evidence": "The heating element burned out after two weeks", "confidence": 0.95},
                {"aspect_name": "seller support", "aspect_category": "customer_service", "sentiment": "negative", "evidence": "Seller never answered my emails about a replacement", "confidence": 0.93},
                {"aspect_name": "value", "aspect_category": "price_value", "sentiment": "negative", "evidence": "Total waste of money", "confidence": 0.85},
            ],
            "complaint_categories": ["poor_durability", "customer_service_issue", "poor_value"],
            "confidence": 0.93,
            "rationale": "Early failure, unresponsive seller and poor value.",
        },
    },
    {
        "review": "Cups brew a strong, tasty coffee and they are cheap. However the lids are flimsy and some popped off and made a mess in my machine.",
        "output": {
            "overall_sentiment": "mixed",
            "aspects": [
                {"aspect_name": "coffee taste", "aspect_category": "performance", "sentiment": "positive", "evidence": "Cups brew a strong, tasty coffee", "confidence": 0.9},
                {"aspect_name": "price", "aspect_category": "price_value", "sentiment": "positive", "evidence": "they are cheap", "confidence": 0.8},
                {"aspect_name": "lids", "aspect_category": "design_build", "sentiment": "negative", "evidence": "the lids are flimsy and some popped off and made a mess in my machine", "confidence": 0.9},
            ],
            "complaint_categories": ["product_defect", "usability_issue"],
            "confidence": 0.85,
            "rationale": "Good taste and price, but flimsy lids cause messes.",
        },
    },
    {
        "review": "Listing says it fits my Whirlpool model but the connector is a different shape. Had to send it back.",
        "output": {
            "overall_sentiment": "negative",
            "aspects": [
                {"aspect_name": "fit with model", "aspect_category": "compatibility", "sentiment": "negative", "evidence": "Listing says it fits my Whirlpool model but the connector is a different shape", "confidence": 0.95},
            ],
            "complaint_categories": ["compatibility_fit_issue", "not_as_described"],
            "confidence": 0.9,
            "rationale": "Part did not fit despite the listing's claim.",
        },
    },
    {
        "review": "Bought this as a gift for my brother. He hasn't opened it yet.",
        "output": {
            "overall_sentiment": "neutral",
            "aspects": [],
            "complaint_categories": [],
            "confidence": 0.85,
            "rationale": "No opinion about the product is expressed.",
        },
    },
]


def render_examples() -> str:
    blocks = []
    for i, ex in enumerate(FEW_SHOT_EXAMPLES, 1):
        blocks.append(f"EXAMPLE {i}\nREVIEW:\n\"\"\"\n{ex['review']}\n\"\"\"\nOUTPUT:\n{json.dumps(ex['output'], ensure_ascii=False)}")
    return "\n\n".join(blocks)


TEMPLATE = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_ROLE + "\n\n" + EXTRACTION_RULES + "\n\n{taxonomy}\n\nOUTPUT FORMAT:\n{format_instructions}"),
        ("human", "Here are labelled examples of correct outputs:\n\n{examples}\n\nNow analyse this review and return the JSON object.\n\nREVIEW:\n\"\"\"\n{review_text}\n\"\"\""),
    ]
)
