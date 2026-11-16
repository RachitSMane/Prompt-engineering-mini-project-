"""Deterministic claim decomposition of a structured product summary (judge v4). No LLM.

Every claim's text is taken verbatim from the summary; nothing is paraphrased or invented:

* ``overall_opinion``  -> one claim per sentence (kind ``opinion_sentence``; evidence = the product statistics);
* ``appreciated`` / ``complaints`` -> one claim per point with the point's own cited review ids (kind ``point``);
* ``frequent_aspects`` -> one claim for the list (kind ``derived``);
* ``improvement_areas`` / ``recommendations`` -> one claim per item (kind ``derived``; they imply a problem, so they
  need negative evidence for the aspect they target).

Claim ids are stable: ``<summary_id>:<field>:<index>``.
"""

from __future__ import annotations

import re

from evaluation.claim_checks import ASPECT_TO_CATEGORY, COUNT_RX, PCT_RX, AVG_RX, aspects_in
from schemas.claim_schema import Claim
from schemas.evaluation_schema import ProductSummary, ProductSummaryContent

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\"'])")
_SENTIMENT_WORDS = re.compile(r"\b(sentiment|positive|negative|mixed|well[- ]received|reception|feedback|star rating|satisf\w*)\b", re.I)


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split((text or "").strip()) if s.strip()]


def _is_numeric(text: str) -> bool:
    return bool(PCT_RX.search(text) or COUNT_RX.search(text) or AVG_RX.search(text))


def categorise(text: str, kind: str) -> str:
    if kind == "opinion_sentence" and _is_numeric(text):
        return "quantitative_statistic"
    labs = aspects_in(text)
    if labs:
        return ASPECT_TO_CATEGORY.get(labs[0], "other")
    if kind == "opinion_sentence" and _SENTIMENT_WORDS.search(text):
        return "overall_sentiment"
    if _is_numeric(text):
        return "quantitative_statistic"
    return "other"


def decompose(summary_id: str, product_id: str, content: ProductSummaryContent) -> list[Claim]:
    claims: list[Claim] = []

    def add(field, idx, kind, text, polarity="neutral", cited=None):
        claims.append(Claim(claim_id=f"{summary_id}:{field}:{idx}", summary_id=summary_id, product_id=product_id,
                            source_field=field, index=idx, kind=kind, text=text, polarity=polarity,
                            category=categorise(text, kind), is_numeric=_is_numeric(text), cited_review_ids=list(cited or [])))

    for i, s in enumerate(split_sentences(content.overall_opinion)):
        add("overall_opinion", i, "opinion_sentence", s)
    for i, p in enumerate(content.appreciated):
        add("appreciated", i, "point", p.point, "positive", p.supporting_review_ids)
    for i, p in enumerate(content.complaints):
        add("complaints", i, "point", p.point, "negative", p.supporting_review_ids)
    if content.frequent_aspects:
        add("frequent_aspects", 0, "derived", "Frequently mentioned aspects: " + ", ".join(content.frequent_aspects))
    for i, s in enumerate(content.improvement_areas):
        add("improvement_areas", i, "derived", s, "negative")
    for i, s in enumerate(content.recommendations):
        add("recommendations", i, "derived", s, "negative")
    return claims


def decompose_summary(summary: ProductSummary) -> list[Claim]:
    return decompose(summary.summary_id, summary.product_id, summary.content)
