"""LLM-as-a-Judge v3: one call per summary, raw response always stored, no repair loop, claim-consistent scores.

Why v3 exists (see reports/D3_JUDGE_EVALUATION.md): in v2 a cross-field validator rejected every verdict whose
hallucination score was below 5 without listed claims, and the repair loop re-asked the judge until it produced
claims. For 7 of 15 summaries the claim list only exists because of that pressure. v3 removes the pressure:

* Same rubric text as v1/v2 (``prompts.judge.RUBRIC``), same flat JSON fields, Ollama constrained decoding for the
  shape only. The claim list is requested BEFORE the hallucination score (field order), so the score is written
  after the claims it should reflect.
* Exactly ONE judge call per summary. The raw first (and only) response is stored for every call, valid or not.
* No repair and no consistency validator. Output that fails JSON parsing or the schema is a validation failure,
  stored separately (``failures.jsonl``) with the raw text; no score is invented.
* Scores consistent with claims, without forcing claims: the judge's own hallucination score is always kept as
  ``hallucination_score_raw``. A hallucination score is REPORTED only when it agrees with the judge's own claim list
  (5 with no claims, or 1-4 with at least one claim). An inconsistent verdict (e.g. "several unsupported claims" in
  the justification, an empty list and score 4) gets no reported hallucination score and is counted as
  ``inconsistent_excluded``: neither 'no hallucination' nor invented claims are attributed to the summary.
* Corrected copies, original summaries and negative-control corrupted copies are separate ``evaluated_object``
  kinds and are never mixed; no score is ever transferred between summary ids.
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from config import Settings
from evaluation.summary_evaluation import _judge_sources, claim_in_text, summary_text
from llm.base import LLMClient, LLMError
from prompts.common import render
from prompts.judge import RUBRIC
from schemas.evaluation_schema import ProductSummary, UnsupportedClaim
from schemas.review_schema import ReviewRecord
from utils.storage import append_jsonl, read_jsonl

JUDGE_PROMPT_VERSION_V3 = "v3"
OBJECT_KINDS = ("original", "corrected", "nc_corrupted")

_SCORE = dict(ge=1, le=5)
_JUST = dict(min_length=5, max_length=800)


class JudgeOutputV3(BaseModel):
    """What the v3 judge returns. Like v2, but claims come before the hallucination score and there is NO cross-field
    validator: an inconsistent verdict is stored as the judge wrote it and reconciled transparently afterwards."""

    model_config = ConfigDict(extra="forbid")

    faithfulness_score: int = Field(..., **_SCORE)
    faithfulness_justification: str = Field(..., **_JUST)
    coverage_score: int = Field(..., **_SCORE)
    coverage_justification: str = Field(..., **_JUST)
    consistency_score: int = Field(..., **_SCORE)
    consistency_justification: str = Field(..., **_JUST)
    unsupported_claims: list[UnsupportedClaim] = Field(default_factory=list, max_length=15)
    hallucination_score: int = Field(..., **_SCORE)
    hallucination_justification: str = Field(..., **_JUST)
    overall_comment: str = Field(..., min_length=5, max_length=800)

    @field_validator("faithfulness_score", "coverage_score", "consistency_score", "hallucination_score", mode="before")
    @classmethod
    def _integer_scores_only(cls, v):
        if isinstance(v, bool):
            raise ValueError("score must be an integer 1-5, not a boolean")
        if isinstance(v, float) and not v.is_integer():
            raise ValueError(f"score must be a whole number 1-5 (got {v})")
        return v


V3_SKELETON = """{
  "faithfulness_score": <integer 1-5>,
  "faithfulness_justification": "<1-3 sentences citing review ids or statistics>",
  "coverage_score": <integer 1-5>,
  "coverage_justification": "<1-3 sentences>",
  "consistency_score": <integer 1-5>,
  "consistency_justification": "<1-3 sentences>",
  "unsupported_claims": [
    {"claim": "<statement copied from the summary>", "reason": "<what the sources say instead>", "evidence_review_ids": ["<review id>"]}
  ],
  "hallucination_score": <integer 1-5>,
  "hallucination_justification": "<1-3 sentences>",
  "overall_comment": "<1-2 sentences>"
}"""

V3_RULES = """OUTPUT RULES:
- Return exactly ONE flat JSON object with exactly the keys shown in the skeleton. No markdown, no text before or after it.
- Every *_score is a whole number from 1 to 5. Every *_justification is a non-empty string.
- Check EVERY sentence of the summary against the sources. A feature, number, percentage or opinion that no review,
  extraction or statistic supports is an unsupported claim, even if it cites a review id.
- Copy each unsupported claim from the summary. List ONLY statements that really appear in the summary and that the
  sources do not support. Do not list claims just to fill the field: if every statement is supported, return [].
- Write "unsupported_claims" first, then give hallucination_score from the rubric: no unsupported claims means 5.
- "evidence_review_ids" lists real review ids from SOURCE 1 that you checked for that claim ([] if none)."""

SYSTEM_V3 = (
    "You are a strict, impartial evaluator of review summaries. Judge ONLY against the source material "
    "provided; do not reward fluent writing that is not supported.\n\n"
    + RUBRIC.replace('List every unsupported claim verbatim in "unsupported_claims" (empty list if none).\n', "")
    + "\n\n" + V3_RULES + "\n\nJSON SKELETON (replace every <...>):\n" + V3_SKELETON
)
HUMAN_V3 = (
    "SOURCE 1 — ORIGINAL REVIEWS (id: text):\n{reviews}\n\n"
    "SOURCE 2 — STRUCTURED EXTRACTIONS (id: overall | aspects | complaints):\n{extractions}\n\n"
    "SOURCE 3 — AGGREGATED STATISTICS:\n{statistics}\n\n"
    "SUMMARY TO EVALUATE:\n{summary}\n\nReturn the evaluation as one flat JSON object following the skeleton."
)
TEMPLATE_V3 = ChatPromptTemplate.from_messages(
    [("system", SYSTEM_V3.replace("{", "{{").replace("}", "}}")), ("human", HUMAN_V3)]
)


def prompt_sha256() -> str:
    return hashlib.sha256((SYSTEM_V3 + "\n----\n" + HUMAN_V3).encode("utf-8")).hexdigest()


@lru_cache(maxsize=1)
def judge_v3_json_schema() -> dict:
    return JudgeOutputV3.model_json_schema()


def get_judge_v3_client(settings: Settings, *, use_mock: bool) -> LLMClient:
    from llm.factory import get_llm_client

    return get_llm_client(settings, use_mock=use_mock, role="judge", response_schema=judge_v3_json_schema())


def run_config(client: LLMClient, settings: Settings) -> dict:
    return {
        "judge_version": JUDGE_PROMPT_VERSION_V3,
        "judge_provider": client.provider,
        "judge_model": client.model_name,
        "is_mock": client.is_mock,
        "constrained_decoding": bool(getattr(client, "constrained_decoding", False)),
        "temperature": settings.llm_temperature,
        "num_ctx": settings.ollama_num_ctx,
        "num_predict": settings.llm_max_output_tokens,
        "timeout_s": settings.llm_timeout_seconds,
        "calls_per_summary": 1,
        "repair_attempts": 0,
        "prompt_sha256": prompt_sha256(),
        "rubric": "prompts.judge.RUBRIC (unchanged from v1/v2)",
        "hallucination_reconciliation": "reported only if consistent with the claim list (5 <-> no claims); otherwise None, raw kept",
    }


# --------------------------------------------------------------------------- deterministic post-processing
def reconcile_hallucination(raw_score: int, n_claims: int) -> tuple[int | None, str]:
    """Reported hallucination score: the raw score if it agrees with the judge's own claim list, else None.

    Never adds or removes claims and never changes a score: an inconsistent verdict is excluded, not repaired."""
    if (n_claims == 0) == (raw_score == 5):
        return raw_score, "consistent"
    return None, ("inconsistent_score_below_5_without_claims" if n_claims == 0 else "inconsistent_score_5_with_claims")


def rubric_note(score: int, n_claims: int) -> str | None:
    """Flags the rubric mismatch seen in v2: score 4 ('one minor unsupported detail') with several claims."""
    if score == 4 and n_claims >= 2:
        return "score 4 with >= 2 claims (rubric: 4 = one minor detail)"
    return None


def parse_v3(raw: str) -> tuple[JudgeOutputV3 | None, str | None]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON: {exc}"
    try:
        return JudgeOutputV3.model_validate(data), None
    except ValidationError as exc:
        return None, "schema: " + "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()[:8])


def build_record(summary: ProductSummary, out: JudgeOutputV3, *, kind: str, raw: str, latency_s: float, config: dict,
                 run_id: str, extra: dict | None = None) -> dict:
    claims = list(out.unsupported_claims)
    text = summary_text(summary)
    allowed = set(summary.aggregate.review_ids)
    cited = {rid for c in claims for rid in c.evidence_review_ids}
    h, action = reconcile_hallucination(out.hallucination_score, len(claims))
    scores = {"faithfulness": out.faithfulness_score, "coverage": out.coverage_score, "consistency": out.consistency_score,
              "hallucination": h}
    scores["mean"] = round(sum(scores.values()) / 4, 2) if h is not None else None
    rec = {
        "record_id": hashlib.sha1(f"{summary.summary_id}|{config['judge_model']}|v3|{raw}".encode()).hexdigest()[:12],
        "run_id": run_id,
        "evaluated_object": kind,
        "summary_id": summary.summary_id,
        "product_id": summary.product_id,
        "valid": True,
        "scores": scores,
        "hallucination_score_raw": out.hallucination_score,
        "hallucination_reconciliation": action,
        "rubric_note": rubric_note(out.hallucination_score, len(claims)),
        "unsupported_claims": [c.model_dump() for c in claims],
        "claims_found_in_summary": sum(claim_in_text(c.claim, text) for c in claims),
        "judge_evidence_ids_valid": sorted(cited & allowed),
        "judge_evidence_ids_unknown": sorted(cited - allowed),
        "justifications": {"faithfulness": out.faithfulness_justification, "coverage": out.coverage_justification,
                           "consistency": out.consistency_justification, "hallucination": out.hallucination_justification},
        "overall_comment": out.overall_comment,
        "raw_response": raw,
        "latency_s": round(latency_s, 2),
        "config": config,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if extra:
        rec.update(extra)
    return rec


class V3Store:
    """Append-only v3 store: judgements and validation failures in separate files; resumable by (kind, summary_id)."""

    def __init__(self, folder: Path):
        self.folder = folder
        self.path = folder / "judgements.jsonl"
        self.failures_path = folder / "failures.jsonl"

    def done(self) -> set[tuple[str, str]]:
        keys = set()
        for p in (self.path, self.failures_path):
            for r in read_jsonl(p):
                keys.add((r["evaluated_object"], r["summary_id"]))
        return keys

    def judgements(self) -> list[dict]:
        return read_jsonl(self.path)

    def failures(self) -> list[dict]:
        return read_jsonl(self.failures_path)


def judge_once(client: LLMClient, summary: ProductSummary, records: list[ReviewRecord], *, kind: str, config: dict,
               run_id: str, extra: dict | None = None) -> tuple[bool, dict]:
    """ONE judge call. Returns (valid, record). Provider errors (network, Ollama down) raise LLMError and record nothing."""
    if kind not in OBJECT_KINDS:
        raise ValueError(f"unknown evaluated_object {kind!r}")
    messages = render(TEMPLATE_V3, **_judge_sources(summary, records))
    t0 = time.perf_counter()
    resp = client.invoke(messages, task="judge_v3", payload={"summary": summary.content.model_dump(mode="json"),
                                                              "allowed_review_ids": sorted(summary.aggregate.review_ids)})
    latency = resp.latency_s or (time.perf_counter() - t0)
    out, err = parse_v3(resp.text)
    if out is None:
        rec = {"run_id": run_id, "evaluated_object": kind, "summary_id": summary.summary_id, "product_id": summary.product_id,
               "valid": False, "validation_error": err, "raw_response": resp.text, "latency_s": round(latency, 2),
               "config": config, "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        if extra:
            rec.update(extra)
        return False, rec
    return True, build_record(summary, out, kind=kind, raw=resp.text, latency_s=latency, config=config, run_id=run_id, extra=extra)


def judge_items(client: LLMClient, items: list[tuple[str, ProductSummary, dict | None]], records: list[ReviewRecord],
                store: V3Store, *, config: dict, run_id: str, before_call=None, progress=None) -> dict:
    """Judge (kind, summary, extra) items once each; skip every (kind, summary_id) already stored (valid or failed)."""
    done = store.done()
    stats = {"judged": 0, "failed": 0, "skipped_stored": 0}
    for kind, summ, extra in items:
        if (kind, summ.summary_id) in done:
            stats["skipped_stored"] += 1
            continue
        if before_call:
            before_call(summ)
        try:
            ok, rec = judge_once(client, summ, records, kind=kind, config=config, run_id=run_id, extra=extra)
        except LLMError:
            raise  # nothing recorded; the same command resumes
        append_jsonl(store.path if ok else store.failures_path, [rec])
        stats["judged" if ok else "failed"] += 1
        if progress:
            progress(kind, summ, ok, rec)
    return stats


# --------------------------------------------------------------------------- analysis (no LLM)
def summarise_v3(judgements: list[dict], failures: list[dict], kind: str) -> dict:
    js = [j for j in judgements if j["evaluated_object"] == kind]
    fs = [f for f in failures if f["evaluated_object"] == kind]
    n = len(js)
    claims = sum(len(j["unsupported_claims"]) for j in js)

    def dist(dim):
        d: dict = {}
        for j in js:
            k = str(j["scores"][dim])
            d[k] = d.get(k, 0) + 1
        return dict(sorted(d.items()))

    def mean(dim):
        vals = [j["scores"][dim] for j in js if j["scores"][dim] is not None]
        return round(sum(vals) / len(vals), 2) if vals else None

    return {
        "evaluated_object": kind,
        "attempted": n + len(fs),
        "valid": n,
        "validation_failures": len(fs),
        "mean_scores": {d: mean(d) for d in ("faithfulness", "coverage", "consistency", "hallucination", "mean")},
        "mean_scores_note": "hallucination and mean exclude verdicts whose raw score contradicts their own claim list",
        "hallucination_consistent": sum(1 for j in js if j["scores"]["hallucination"] is not None),
        "hallucination_distribution": dist("hallucination") if n else {},
        "hallucination_raw_distribution": dict(sorted({s: sum(1 for j in js if j["hallucination_score_raw"] == s)
                                                       for s in {j["hallucination_score_raw"] for j in js}}.items())),
        "reconciliation": {a: sum(1 for j in js if j["hallucination_reconciliation"] == a)
                           for a in sorted({j["hallucination_reconciliation"] for j in js})},
        "summaries_with_claims": sum(1 for j in js if j["unsupported_claims"]),
        "claims_total": claims,
        "claims_found_in_summary": sum(j["claims_found_in_summary"] for j in js),
        "rubric_notes": sum(1 for j in js if j["rubric_note"]),
        "judgements_citing_unknown_ids": sum(1 for j in js if j["judge_evidence_ids_unknown"]),
    }
