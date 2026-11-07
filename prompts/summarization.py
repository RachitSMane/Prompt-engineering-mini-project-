"""Stage 3 prompt — SUMMARISE aggregated statistics + evidence into a product summary."""

from __future__ import annotations

import json
from functools import lru_cache

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate

from schemas.evaluation_schema import ProductAggregate, ProductSummaryContent

SUMMARY_PROMPT_VERSION = "v1"

SUMMARY_RULES = """RULES:
1. Use ONLY the STATISTICS and EVIDENCE below. Do not use outside knowledge about the product or brand.
2. Never calculate or invent numbers. If you mention a number or percentage, copy it exactly from STATISTICS.
3. Every item in "appreciated" and "complaints" must cite 1-3 supporting review ids taken from the EVIDENCE list.
4. Only discuss aspects and complaints that appear in STATISTICS or EVIDENCE.
5. Reflect the balance of the statistics: do not overstate rare complaints or ignore frequent ones.
6. Improvement areas and recommendations must follow directly from the complaints / negative aspects.
7. If there is little evidence, say so instead of guessing.
8. Respond with ONE JSON object only — no markdown, no extra text."""

TEMPLATE = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a product insights analyst writing an evidence-grounded summary of customer reviews "
            "for a product manager.\n\n" + SUMMARY_RULES + "\n\nOUTPUT FORMAT:\n{format_instructions}",
        ),
        ("human", "PRODUCT: {product_title} ({product_id})\n\nSTATISTICS (computed from {total_reviews} validated review extractions):\n{statistics}\n\nEVIDENCE (verbatim review quotes):\n{evidence}\n\nWrite the product summary JSON."),
    ]
)


@lru_cache(maxsize=1)
def summary_format_instructions() -> str:
    return PydanticOutputParser(pydantic_object=ProductSummaryContent).get_format_instructions()


def statistics_block(agg: ProductAggregate) -> str:
    data = {
        "total_reviews": agg.total_reviews,
        "overall_sentiment_percentages": agg.sentiment_percentages,
        "overall_sentiment_counts": agg.sentiment_counts,
        "average_star_rating": agg.average_rating,
        "reviews_with_complaints": agg.reviews_with_complaints,
        "aspects_by_mentions": [
            {"aspect": s.aspect_category, "mentions": s.mentions, "positive": s.positive, "negative": s.negative, "neutral": s.neutral, "net_sentiment": s.net_sentiment, "examples": s.top_aspect_names}
            for s in agg.aspect_stats
        ],
        "complaint_counts": agg.complaint_counts,
        "complaint_percentages_of_reviews": agg.complaint_percentages,
    }
    return json.dumps(data, indent=1)


def evidence_block(agg: ProductAggregate) -> str:
    lines = [f'- [{q.review_id}] ({q.sentiment}, {q.aspect_category}) "{q.quote}"' for q in agg.positive_evidence + agg.negative_evidence]
    return "\n".join(lines) or "(no aspect evidence available)"


def allowed_review_ids(agg: ProductAggregate) -> set[str]:
    return {q.review_id for q in agg.positive_evidence + agg.negative_evidence}
