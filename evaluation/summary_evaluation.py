"""LLM-as-a-Judge evaluation of product summaries + deterministic sanity checks.

Judge scores are model-based assessments, NOT objective human evaluation. To
make them auditable we (1) store the judge's justifications and raw output,
(2) run cheap deterministic checks alongside, and (3) export a small
side-by-side sample for manual inspection (``export_inspection_sample``).
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from config import Settings
from llm.base import LLMClient
from pipeline.repair import StructuredResult, generate_structured
from prompts.common import render
from prompts.judge import JUDGE_PROMPT_VERSION, TEMPLATE, judge_format_instructions
from prompts.summarization import statistics_block
from schemas.evaluation_schema import JudgeVerdict, ProductSummary, SummaryJudgement, SummaryPoint
from schemas.review_schema import ReviewRecord
from utils.storage import append_jsonl, log_activity, read_jsonl

JUDGE_TASK_DESCRIPTION = "Evaluate a product summary against source reviews, extractions and statistics using a 1-5 rubric for faithfulness, coverage, consistency and hallucination."
MAX_REVIEW_WORDS = 80
_PCT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")


def summary_text(summary: ProductSummary) -> str:
    c = summary.content
    parts = [c.overall_opinion]
    parts += [p.point for p in c.appreciated + c.complaints]
    parts += c.frequent_aspects + c.improvement_areas + c.recommendations
    return "\n".join(parts)


def summary_for_judge(summary: ProductSummary) -> str:
    return json.dumps(summary.content.model_dump(mode="json"), indent=1, ensure_ascii=False)


def _truncate(text: str, n: int = MAX_REVIEW_WORDS) -> str:
    words = text.split()
    return text if len(words) <= n else " ".join(words[:n]) + " …"


def deterministic_checks(summary: ProductSummary) -> dict:
    """Cheap, transparent checks that complement the judge."""
    agg = summary.aggregate
    text = summary_text(summary)
    known = list(agg.sentiment_percentages.values()) + list(agg.complaint_percentages.values())
    found = [float(x) for x in _PCT_RE.findall(text)]
    unmatched = [x for x in found if not any(abs(x - k) <= 0.5 for k in known)]
    top = [s.aspect_category for s in agg.aspect_stats[:3]]
    lower = text.lower()
    covered = [cat for cat in top if any(w in lower for w in cat.replace("_", " ").split()) or any(n in lower for n in next(s.top_aspect_names for s in agg.aspect_stats if s.aspect_category == cat))]
    top_complaints = list(agg.complaint_counts)[:3]
    complaint_covered = [c for c in top_complaints if any(w in lower for w in c.replace("_issue", "").replace("_", " ").split())]
    cited = {rid for p in summary.content.appreciated + summary.content.complaints for rid in p.supporting_review_ids}
    return {
        "percentages_in_summary": found,
        "percentages_not_in_statistics": unmatched,
        "top_aspects": top,
        "top_aspects_mentioned": covered,
        "top_aspect_coverage": round(len(covered) / len(top), 2) if top else None,
        "top_complaints": top_complaints,
        "top_complaints_mentioned": complaint_covered,
        "cited_review_ids": len(cited),
        "cited_ids_all_in_product": cited <= set(agg.review_ids),
    }


def judge_summary(client: LLMClient, summary: ProductSummary, records: list[ReviewRecord], settings: Settings) -> tuple[SummaryJudgement | None, StructuredResult]:
    records = [r for r in records if r.product_id == summary.product_id]
    reviews = "\n".join(f"[{r.review_id}] {_truncate(r.review_text)}" for r in records)
    extractions = "\n".join(
        f"[{r.review_id}] {r.overall_sentiment.value} | "
        + (", ".join(f"{a.aspect_category.value}:{a.sentiment.value}" for a in r.aspects) or "-")
        + " | " + (", ".join(c.value for c in r.complaint_categories) or "-")
        for r in records
    )
    stats = statistics_block(summary.aggregate)
    summ = summary_for_judge(summary)
    messages = render(TEMPLATE, format_instructions=judge_format_instructions(), reviews=reviews, extractions=extractions, statistics=stats, summary=summ)
    result = generate_structured(
        client,
        messages,
        JudgeVerdict,
        task="judge",
        task_description=JUDGE_TASK_DESCRIPTION,
        format_instructions=judge_format_instructions(),
        payload={"summary": summary.content.model_dump(mode="json"), "allowed_review_ids": sorted(summary.aggregate.review_ids)},
        max_repairs=settings.max_repair_attempts,
    )
    if not result.success:
        return None, result
    jid = hashlib.sha1(f"{summary.summary_id}|{client.model_name}|{result.attempts[-1].raw_output}".encode()).hexdigest()[:12]
    judgement = SummaryJudgement(
        judgement_id=jid,
        summary_id=summary.summary_id,
        product_id=summary.product_id,
        verdict=result.value,  # type: ignore[arg-type]
        deterministic_checks=deterministic_checks(summary),
        judge_provider=client.provider,
        judge_model=client.model_name,
        is_mock=client.is_mock,
        prompt_version=JUDGE_PROMPT_VERSION,
        repair_attempts=result.repair_attempts,
        latency_s=result.total_latency_s,
    )
    log_activity(settings.paths.activity_log, "judge_summary", client.is_mock, product_id=summary.product_id, summary_id=summary.summary_id)
    return judgement, result


CORRUPTION_FABRICATED_FEATURE = "Reviewers praise the built-in Wi-Fi app control and voice-assistant support."


def corrupt_summary(summary: ProductSummary) -> ProductSummary:
    """A deliberately unfaithful copy of ``summary`` used as a negative control for the judge.

    Three known defects are injected: (1) the overall opinion is replaced by one with the OPPOSITE tone of the
    aggregated statistics and a fabricated percentage, (2) an appreciated point about a feature no review
    mentions is added (citing a real review id, so only a careful judge notices), (3) all complaints are dropped.
    A judge that discriminates should score the copy lower than the original, above all on faithfulness,
    consistency and hallucination.
    """
    pct = summary.aggregate.sentiment_percentages
    mostly_positive = pct.get("positive", 0) >= max(pct.get("negative", 0), pct.get("mixed", 0))
    opinion = (
        "Customers are overwhelmingly dissatisfied: 92% of reviewers say the product failed within a week and would not buy it again."
        if mostly_positive else
        "Customers are overwhelmingly delighted: 98% of reviewers rate it as perfect and report no problems of any kind."
    )
    cite = (summary.aggregate.review_ids or ["unknown"])[:1]
    content = summary.content.model_copy(
        update={
            "overall_opinion": opinion,
            "appreciated": [SummaryPoint(point=CORRUPTION_FABRICATED_FEATURE, supporting_review_ids=cite)] + list(summary.content.appreciated[:5]),
            "complaints": [],
        }
    )
    return summary.model_copy(update={"summary_id": f"{summary.summary_id}-corrupted", "content": content})


class JudgementStore:
    def __init__(self, path: Path):
        self.path = path

    @classmethod
    def for_mode(cls, settings: Settings, is_mock: bool) -> "JudgementStore":
        return cls(settings.paths.mode_dir(is_mock) / "judgements" / "judgements.jsonl")

    def add(self, j: SummaryJudgement) -> None:
        append_jsonl(self.path, [j.model_dump(mode="json")])

    def all(self) -> list[SummaryJudgement]:
        return [SummaryJudgement.model_validate(r) for r in read_jsonl(self.path)]

    def latest_by_summary(self) -> dict[str, SummaryJudgement]:
        return {j.summary_id: j for j in self.all()}


def judgements_table(judgements: list[SummaryJudgement], summaries: dict[str, ProductSummary]) -> list[dict]:
    rows = []
    for j in judgements:
        s = summaries.get(j.summary_id)
        v = j.verdict
        rows.append(
            {
                "product_id": j.product_id,
                "product_title": (s.product_title if s else "") or "",
                "summary_id": j.summary_id,
                "faithfulness": v.faithfulness.score,
                "coverage": v.coverage.score,
                "consistency": v.consistency.score,
                "hallucination": v.hallucination.score,
                "mean_score": v.mean_score,
                "unsupported_claims": len(v.unsupported_claims),
                "pct_not_in_stats": len(j.deterministic_checks.get("percentages_not_in_statistics", [])),
                "top_aspect_coverage": j.deterministic_checks.get("top_aspect_coverage"),
                "judge_model": j.judge_model,
                "is_mock": j.is_mock,
            }
        )
    return rows


def export_inspection_sample(judgements: list[SummaryJudgement], summaries: dict[str, ProductSummary], n: int = 3) -> str:
    """Markdown with side-by-side material so a human can audit the judge."""
    lines = ["# LLM-as-a-Judge — Manual Inspection Sample", "",
             "Judge scores are model-generated and are **not** human evaluation. For each example, read the evidence "
             "and statistics, then record whether you agree with each score.", ""]
    if any(j.is_mock for j in judgements):
        lines += ["> **MOCK / OFFLINE DATA — not real summaries or judgements.**", ""]
    for j in judgements[:n]:
        s = summaries.get(j.summary_id)
        if not s:
            continue
        v = j.verdict
        lines += [f"## {s.product_title or s.product_id} (`{s.product_id}`)", "", "### Aggregated statistics", "", "```json", statistics_block(s.aggregate), "```", "",
                  "### Evidence quotes", ""]
        lines += [f'- [{q.review_id}] ({q.sentiment}, {q.aspect_category}) "{q.quote}"' for q in s.aggregate.positive_evidence + s.aggregate.negative_evidence]
        lines += ["", "### Generated summary", "", "```json", summary_for_judge(s), "```", "", "### Judge verdict", "",
                  "| dimension | score | justification | human agrees? (y/n) |", "| --- | --- | --- | --- |"]
        for dim in ("faithfulness", "coverage", "consistency", "hallucination"):
            d = getattr(v, dim)
            lines.append(f"| {dim} | {d.score} | {d.justification} |  |")
        lines += ["", f"Unsupported claims flagged: {v.unsupported_claims or 'none'}", "",
                  f"Deterministic checks: `{json.dumps(j.deterministic_checks)}`", ""]
    return "\n".join(lines)
