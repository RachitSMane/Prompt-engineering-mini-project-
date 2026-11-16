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
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from config import Settings
from llm.base import LLMClient, LLMError
from pipeline.repair import StructuredResult, generate_structured
from pipeline.validation import normalize_judge_v2
from prompts.common import render
from prompts.judge import (
    CURRENT_JUDGE_VERSION,
    JUDGE_PROMPT_VERSION,
    JUDGE_PROMPT_VERSION_V2,
    TEMPLATE,
    TEMPLATE_V2,
    V2_FORMAT_INSTRUCTIONS,
    judge_format_instructions,
    judge_v2_json_schema,
)
from prompts.summarization import statistics_block
from schemas.evaluation_schema import JudgeOutputV2, JudgeVerdict, ProductSummary, SummaryJudgement, SummaryPoint
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


def _judge_sources(summary: ProductSummary, records: list[ReviewRecord]) -> dict[str, str]:
    records = [r for r in records if r.product_id == summary.product_id]
    reviews = "\n".join(f"[{r.review_id}] {_truncate(r.review_text)}" for r in records)
    extractions = "\n".join(
        f"[{r.review_id}] {r.overall_sentiment.value} | "
        + (", ".join(f"{a.aspect_category.value}:{a.sentiment.value}" for a in r.aspects) or "-")
        + " | " + (", ".join(c.value for c in r.complaint_categories) or "-")
        for r in records
    )
    return {"reviews": reviews, "extractions": extractions, "statistics": statistics_block(summary.aggregate), "summary": summary_for_judge(summary)}


def judge_summary(
    client: LLMClient,
    summary: ProductSummary,
    records: list[ReviewRecord],
    settings: Settings,
    *,
    version: str = JUDGE_PROMPT_VERSION,
    raise_provider_errors: bool = False,
) -> tuple[SummaryJudgement | None, StructuredResult]:
    """Judge one summary.

    ``version``: "v1" (original nested schema, kept so stored results stay reproducible) or "v2" (flat schema,
    structural normaliser, rubric-consistency rule; create the client with ``get_judge_client(..., version="v2")``).
    Output that is still invalid after the bounded repairs is returned as a failure (``None``); no score is invented.
    """
    src = _judge_sources(summary, records)
    payload = {"summary": summary.content.model_dump(mode="json"), "allowed_review_ids": sorted(summary.aggregate.review_ids)}
    if version == JUDGE_PROMPT_VERSION_V2:
        messages = render(TEMPLATE_V2, **src)
        result = generate_structured(
            client, messages, JudgeOutputV2, task="judge_v2", task_description=JUDGE_TASK_DESCRIPTION,
            format_instructions=V2_FORMAT_INSTRUCTIONS, payload=payload, max_repairs=settings.max_repair_attempts,
            normalizer=normalize_judge_v2, raise_provider_errors=raise_provider_errors,
        )
        verdict = JudgeVerdict.from_v2(result.value) if result.success else None  # type: ignore[arg-type]
    elif version == JUDGE_PROMPT_VERSION:
        messages = render(TEMPLATE, format_instructions=judge_format_instructions(), **src)
        result = generate_structured(
            client, messages, JudgeVerdict, task="judge", task_description=JUDGE_TASK_DESCRIPTION,
            format_instructions=judge_format_instructions(), payload=payload, max_repairs=settings.max_repair_attempts,
            raise_provider_errors=raise_provider_errors,
        )
        verdict = result.value
    else:
        raise ValueError(f"unknown judge version {version!r}")
    if verdict is None:
        return None, result
    jid = hashlib.sha1(f"{summary.summary_id}|{client.model_name}|{version}|{result.attempts[-1].raw_output}".encode()).hexdigest()[:12]
    checks = deterministic_checks(summary)
    checks.update(claim_traceability(summary, verdict))
    judgement = SummaryJudgement(
        judgement_id=jid,
        summary_id=summary.summary_id,
        product_id=summary.product_id,
        verdict=verdict,  # type: ignore[arg-type]
        deterministic_checks=checks,
        judge_provider=client.provider,
        judge_model=client.model_name,
        is_mock=client.is_mock,
        prompt_version=version,
        repair_attempts=result.repair_attempts,
        latency_s=result.total_latency_s,
    )
    log_activity(settings.paths.activity_log, "judge_summary", client.is_mock, product_id=summary.product_id, summary_id=summary.summary_id, judge_version=version)
    return judgement, result


def get_judge_client(settings: Settings, *, use_mock: bool, version: str = CURRENT_JUDGE_VERSION) -> LLMClient:
    """Judge client. For v2 on Ollama it uses constrained decoding with the v2 JSON schema (shape only; still validated)."""
    from llm.factory import get_llm_client

    schema = judge_v2_json_schema() if version == JUDGE_PROMPT_VERSION_V2 else None
    return get_llm_client(settings, use_mock=use_mock, role="judge", response_schema=schema)


# --------------------------------------------------------------------------- claim traceability (deterministic)
_WORD_RE = re.compile(r"[a-z0-9%]+")
_STOP = {"the", "a", "an", "and", "or", "of", "to", "is", "are", "it", "in", "on", "for", "with", "that", "this", "as", "be", "by", "at", "its", "their"}


def _tokens(text: str) -> set[str]:
    return {w for w in _WORD_RE.findall(text.lower()) if w not in _STOP}


def claim_in_text(claim: str, text: str, threshold: float = 0.6) -> bool:
    """True if most content words of ``claim`` occur in ``text`` (does a flagged claim really come from the summary?)."""
    toks = _tokens(claim)
    return bool(toks) and len(toks & _tokens(text)) / len(toks) >= threshold


def claim_traceability(summary: ProductSummary, verdict: JudgeVerdict) -> dict:
    """How many of the judge's 'unsupported claims' can be traced back to the summary text (no LLM)."""
    text = summary_text(summary)
    claims = verdict.unsupported_claims
    allowed = set(summary.aggregate.review_ids)
    cited_by_judge = {rid for d in verdict.unsupported_claim_details for rid in d.evidence_review_ids}
    return {
        "unsupported_claims_flagged": len(claims),
        "unsupported_claims_traceable_to_summary": sum(claim_in_text(c, text) for c in claims),
        "judge_evidence_ids_valid": sorted(cited_by_judge & allowed),
        "judge_evidence_ids_unknown": sorted(cited_by_judge - allowed),
    }


# --------------------------------------------------------------------------- resumable batch judging
@dataclass
class JudgeBatchResult:
    judged: list[SummaryJudgement] = field(default_factory=list)
    reused: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    skipped_stored_failure: list[str] = field(default_factory=list)
    new_calls: int = 0


class JudgeProviderStopped(RuntimeError):
    """The judge provider failed repeatedly (outage/timeout). Nothing was recorded for that summary; re-run to resume."""


def failure_record(summary: ProductSummary, client: LLMClient, version: str, result: StructuredResult, source: str) -> dict:
    """A stored judge failure WITH the raw output and validation errors of every attempt (v1 failures lacked them)."""
    return {
        "summary_id": summary.summary_id,
        "product_id": summary.product_id,
        "judge_provider": client.provider,
        "judge_model": client.model_name,
        "is_mock": client.is_mock,
        "prompt_version": version,
        "error": result.error,
        "attempts": [a.model_dump(mode="json") for a in result.attempts],
        "latency_s": result.total_latency_s,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": source,
    }


def judge_summaries(
    client: LLMClient,
    summaries: list[ProductSummary],
    records: list[ReviewRecord],
    settings: Settings,
    store: "JudgementStore",
    *,
    version: str = CURRENT_JUDGE_VERSION,
    retry_failed: bool = False,
    provider_retries: int = 2,
    retry_wait_s: float = 15.0,
    before_call: Callable[[ProductSummary], None] | None = None,
    progress: Callable[[str, ProductSummary, SummaryJudgement | None, str], None] | None = None,
    source: str = "batch",
) -> JudgeBatchResult:
    """Judge many summaries, resumably.

    * A stored judgement from the same judge model and version is reused, never repeated.
    * A stored invalid-output failure for the same model/version is final (temperature 0 would reproduce it) unless
      ``retry_failed`` is set explicitly.
    * Provider errors (timeouts, server down) are retried at most ``provider_retries`` times; if they persist the batch
      stops with ``JudgeProviderStopped`` WITHOUT recording a failure, so re-running resumes at that summary.
    * Each summary gets at most 1 + ``settings.max_repair_attempts`` judge calls for invalid output.
    """
    out = JudgeBatchResult()
    key = (client.model_name, version, client.is_mock)
    stored = {sid: j for sid, j in store.latest_by_summary().items() if (j.judge_model, j.prompt_version, j.is_mock) == key}
    stored_fail = {r["summary_id"] for r in store.failures() if (r.get("judge_model"), r.get("prompt_version"), r.get("is_mock")) == key}
    for summ in summaries:
        if summ.summary_id in stored:
            out.reused.append(summ.summary_id)
            out.judged.append(stored[summ.summary_id])
            if progress:
                progress("reused", summ, stored[summ.summary_id], "")
            continue
        if summ.summary_id in stored_fail and not retry_failed:
            out.skipped_stored_failure.append(summ.summary_id)
            if progress:
                progress("stored_failure", summ, None, "")
            continue
        j, res = None, None
        for attempt in range(provider_retries + 1):
            if before_call:
                before_call(summ)
            try:
                j, res = judge_summary(client, summ, records, settings, version=version, raise_provider_errors=True)
                out.new_calls += 1
                break
            except LLMError as exc:
                if attempt == provider_retries:
                    raise JudgeProviderStopped(f"judge provider failed {provider_retries + 1}x for {summ.product_id}: {exc}") from exc
                time.sleep(retry_wait_s * (attempt + 1))
        if j:
            store.add(j)
            out.judged.append(j)
            if progress:
                progress("judged", summ, j, "")
        else:
            store.add_failure(failure_record(summ, client, version, res, source))
            out.failed.append(summ.summary_id)
            if progress:
                progress("failed", summ, None, res.error or "")
    return out


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
    """Append-only JSONL of judgements, with ``judge_failures.jsonl`` beside it.

    v1 (the original run) lives in ``judgements/``. Every later judge version gets its own sub-folder
    (``judgements/v2/``), so the original results are never mixed with or overwritten by a new judge run.
    """

    def __init__(self, path: Path):
        self.path = path

    @classmethod
    def for_mode(cls, settings: Settings, is_mock: bool, version: str = JUDGE_PROMPT_VERSION) -> "JudgementStore":
        base = settings.paths.mode_dir(is_mock) / "judgements"
        return cls((base if version == JUDGE_PROMPT_VERSION else base / version) / "judgements.jsonl")

    @property
    def failures_path(self) -> Path:
        return self.path.parent / "judge_failures.jsonl"

    def add(self, j: SummaryJudgement) -> None:
        append_jsonl(self.path, [j.model_dump(mode="json")])

    def add_failure(self, record: dict) -> None:
        append_jsonl(self.failures_path, [record])

    def all(self) -> list[SummaryJudgement]:
        return [SummaryJudgement.model_validate(r) for r in read_jsonl(self.path)]

    def latest_by_summary(self) -> dict[str, SummaryJudgement]:
        return {j.summary_id: j for j in self.all()}

    def failures(self) -> list[dict]:
        return read_jsonl(self.failures_path) if self.failures_path.exists() else []

    def open_failures(self) -> dict[str, dict]:
        """Latest failure per summary that has no successful judgement in this store."""
        judged = self.latest_by_summary()
        return {r["summary_id"]: r for r in self.failures() if r["summary_id"] not in judged}


def available_judge_versions(settings: Settings, is_mock: bool) -> list[str]:
    """Judge versions with stored results, newest first (e.g. ["v2", "v1"])."""
    found = []
    for v in (JUDGE_PROMPT_VERSION_V2, JUDGE_PROMPT_VERSION):
        st = JudgementStore.for_mode(settings, is_mock, v)
        if st.path.exists() or st.failures_path.exists():
            found.append(v)
    return found


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
