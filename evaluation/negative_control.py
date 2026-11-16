"""Negative control for the LLM-as-a-Judge (v2): plant ONE known unsupported claim per corrupted summary.

The v1 control changed three things at once, so a result could not be attributed to any one defect. Here each
corrupted copy differs from its original summary by exactly one planted defect, and we know which words identify it:

* ``invented_feature``      — an appreciated point praising a product feature that no review of that product mentions
                              (checked deterministically against the product's review texts before it is used); it cites
                              a real review id, so only a judge that reads the sources can notice it.
* ``fabricated_statistic``  — a sentence with a fabricated percentage that matches no aggregated statistic is appended
                              to the overall opinion (the rest of the opinion is kept).

Detection is measured from what the judge *says*, not only from its scores: a planted claim counts as detected when one
of the judge's ``unsupported_claims`` contains the planted claim's identifying keyword(s). Score drops relative to the
original summary (judged by the same judge version) are reported separately.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from schemas.evaluation_schema import ProductSummary, SummaryJudgement, SummaryPoint

# (sentence, identifying keywords). Keywords are matched case-insensitively in the judge's unsupported claims, and a
# feature is only used for a product if NONE of its keywords occurs in that product's review texts.
FEATURE_POOL: list[tuple[str, tuple[str, ...]]] = [
    ("Reviewers praise the built-in Wi-Fi app control and voice-assistant support.", ("wi-fi", "wifi", "voice", "app control")),
    ("Customers love that it runs on solar power and needs no electricity.", ("solar",)),
    ("Several reviewers highlight the built-in UV light that sterilises the unit automatically.", ("uv light", "uv ", "sterilis", "steriliz")),
    ("Many reviewers mention the colour touchscreen display with recipe suggestions.", ("touchscreen", "touch screen", "recipe")),
    ("Reviewers appreciate the free lifetime replacement guarantee from the manufacturer.", ("lifetime",)),
    ("Customers say the included smartphone app tracks energy use in real time.", ("smartphone", "energy use", "real time", "real-time")),
]
FABRICATED_PERCENTAGES = (97, 93, 88, 91, 86)


@dataclass
class PlantedSummary:
    kind: str
    planted_text: str
    keywords: tuple[str, ...]
    summary: ProductSummary


def _review_corpus(summary: ProductSummary, review_texts: dict[str, str]) -> str:
    return " ".join(review_texts.get(rid, "") for rid in summary.aggregate.review_ids).lower()


def _fabricated_pct(summary: ProductSummary, start: int) -> int:
    """A percentage that is not within 3 points of any aggregated statistic of this product."""
    known = list(summary.aggregate.sentiment_percentages.values()) + list(summary.aggregate.complaint_percentages.values())
    for i in range(len(FABRICATED_PERCENTAGES)):
        pct = FABRICATED_PERCENTAGES[(start + i) % len(FABRICATED_PERCENTAGES)]
        if all(abs(pct - k) > 3 for k in known):
            return pct
    raise ValueError("no fabricated percentage far enough from the statistics")


def plant_invented_feature(summary: ProductSummary, review_texts: dict[str, str], index: int) -> PlantedSummary:
    corpus = _review_corpus(summary, review_texts)
    for i in range(len(FEATURE_POOL)):
        sentence, keywords = FEATURE_POOL[(index + i) % len(FEATURE_POOL)]
        if not any(k in corpus for k in keywords):
            break
    else:
        raise ValueError(f"every pooled feature is mentioned in the reviews of {summary.product_id}")
    cite = (summary.aggregate.review_ids or ["unknown"])[:1]
    content = summary.content.model_copy(update={"appreciated": [SummaryPoint(point=sentence, supporting_review_ids=cite)] + list(summary.content.appreciated[:5])})
    planted = summary.model_copy(update={"summary_id": f"{summary.summary_id}-nc-feature", "content": content})
    return PlantedSummary("invented_feature", sentence, keywords, planted)


def plant_fabricated_statistic(summary: ProductSummary, index: int) -> PlantedSummary:
    pct = _fabricated_pct(summary, index)
    sentence = f"In total, {pct}% of reviewers say they would buy this exact product again."
    opinion = (summary.content.overall_opinion.rstrip() + " " + sentence)[:800]
    content = summary.content.model_copy(update={"overall_opinion": opinion})
    planted = summary.model_copy(update={"summary_id": f"{summary.summary_id}-nc-statistic", "content": content})
    return PlantedSummary("fabricated_statistic", sentence, (f"{pct}%", f"{pct} %", f"{pct} percent", "buy this exact product again"), planted)


def build_planted(summary: ProductSummary, review_texts: dict[str, str], index: int) -> list[PlantedSummary]:
    return [plant_invented_feature(summary, review_texts, index), plant_fabricated_statistic(summary, index)]


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower())


def planted_claim_detected(judgement: SummaryJudgement, keywords: tuple[str, ...]) -> bool:
    """True if one of the judge's unsupported CLAIMS contains a planted keyword.

    Only the claim text counts. The ``reason`` field is deliberately ignored: in the 2026-11-14 run the judge wrote
    "The actual statistic is 91% of reviewers saying they would buy the product again" as the reason for rejecting a
    TRUE statement (B0081N3YW4) — it accepted the planted number as fact. Matching reasons scored that as a detection.
    """
    blob = _norm(" ".join(judgement.verdict.unsupported_claims))
    return any(_norm(k) in blob for k in keywords)


def planted_keyword_in_reason_only(judgement: SummaryJudgement, keywords: tuple[str, ...]) -> bool:
    """The planted keyword appears only in a claim's *reason* (the judge used the planted claim, but did not reject it)."""
    reasons = _norm(" ".join(d.reason for d in judgement.verdict.unsupported_claim_details))
    return not planted_claim_detected(judgement, keywords) and any(_norm(k) in reasons for k in keywords)


DIMS = ("faithfulness", "coverage", "consistency", "hallucination")


def scores(judgement: SummaryJudgement) -> dict:
    v = judgement.verdict
    return {d: getattr(v, d).score for d in DIMS} | {"mean": v.mean_score}
