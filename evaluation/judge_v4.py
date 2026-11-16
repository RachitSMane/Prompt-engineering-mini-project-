"""Judge v4: claim-level faithfulness evaluation with derived scores.

Why (reports/proposed_revisions_2026-11-15/D3_INVESTIGATION_AND_COMPLETION_PLAN.md): v1-v3 asked a 3B model to extract
claims, search 9-42 reviews, verify and write four holistic scores in one generation; the scores were independent of
the claim list (37/47 v3 verdicts contradict their own list) and did not separate corrupted from original summaries.

v4 design:
1. ``evaluation.claims.decompose`` splits the summary into claims (verbatim, no LLM).
2. ``evaluation.claim_checks`` verifies numbers, counts, denominators, majorities, aspect statements, citations and
   frequent aspects deterministically (no LLM).
3. The local LLM verifies ONE claim at a time against ONLY that claim's evidence (the full text of its cited reviews;
   for an opinion sentence no deterministic rule covers, the product statistics). It returns {evidence_quote,
   explanation, status} under constrained decoding. The quote must be an exact (normalised) substring of the evidence,
   otherwise its verdict is replaced by UNVERIFIABLE (the support or contradiction was not demonstrated).
4. ``combine`` merges the deterministic and LLM verdicts: a failed deterministic check can never disappear (the worse
   verdict wins and UNVERIFIABLE never overrides a deterministic finding).
5. ``schemas.claim_schema.derive_scores`` computes every score from the claim statuses; ``SummaryEvaluationV4``
   re-derives them on validation. The LLM never writes a score.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import pandas as pd
from pydantic import ValidationError

from evaluation.claim_checks import (
    Stats, aspects_in, citation_evidence, derived_claim_check, frequent_aspects_check, number_checks_in_texts, numeric_checks_statistics,
    qualitative_checks, quote_in_texts, rating_polarity_finding,
)
from evaluation.claims import decompose
from llm.base import ChatMessage, LLMClient, LLMError
from schemas.claim_schema import (
    SEVERITY, Claim, ClaimEvaluation, ClaimStatus, DeterministicFinding, EvidenceExcerpt, LLMClaimVerdict, NumericCheck,
    SummaryEvaluationV4, derive_scores,
)
from utils.storage import append_jsonl, read_jsonl

JUDGE_VERSION_V4 = "v4"
OBJECT_KINDS = ("original", "corrected", "nc_corrupted", "nc_clean", "repro")
REVIEW_WORD_LIMIT = 350
V4_NUM_CTX = 4096
V4_NUM_PREDICT = 384

SYSTEM_V4 = """You are a careful fact-checker. You check ONE claim taken from a summary of customer reviews against the EVIDENCE given below it. Use ONLY the evidence; ignore anything you believe about the product.

Choose exactly one status:
SUPPORTED - the evidence directly says what the claim says. Reasonable paraphrase is fine. A claim about "users" or "customers" is supported if the cited reviews say it.
PARTIALLY_SUPPORTED - part of the claim is supported, but another part is missing, or the claim is broader, stronger or more specific than the evidence (for example "all users" when one review says it).
UNSUPPORTED - the evidence does not mention what the claim says.
CONTRADICTED - the evidence says the opposite of the claim, or gives a different value.
UNVERIFIABLE - the evidence is too unclear or garbled to decide.

Rules:
1. "evidence_quote": first find the sentence in the EVIDENCE that is most relevant to the claim and copy it word for word, exactly as written (same words, same order). Never leave it empty. Write NONE only if no sentence in the evidence is about the topic of the claim.
2. "explanation": one or two sentences saying what the evidence says about the claim.
3. "status": one of SUPPORTED, PARTIALLY_SUPPORTED, UNSUPPORTED, CONTRADICTED, UNVERIFIABLE.
4. If the claim has several parts (for example joined by "and"), check every part. If one part is not in the evidence, the status is at most PARTIALLY_SUPPORTED.
Return one JSON object: {"evidence_quote": "...", "explanation": "...", "status": "..."}

Examples (a different product):
CLAIM: The toaster browns bread evenly.
EVIDENCE: [R1] (rating 5/5) Toast comes out evenly browned every time. Love it.
{"evidence_quote": "Toast comes out evenly browned every time", "explanation": "The review says the toast is evenly browned.", "status": "SUPPORTED"}

CLAIM: The toaster is easy to clean.
EVIDENCE: [R2] (rating 2/5) Cleaning the crumb tray is a nightmare and it rattles.
{"evidence_quote": "Cleaning the crumb tray is a nightmare", "explanation": "The review says cleaning is very hard, the opposite of the claim.", "status": "CONTRADICTED"}

CLAIM: The toaster comes with a three-year warranty.
EVIDENCE: [R3] (rating 4/5) Works fine and arrived quickly.
{"evidence_quote": "NONE", "explanation": "The review does not mention any warranty.", "status": "UNSUPPORTED"}

CLAIM: All users say the toaster is quiet and fast.
EVIDENCE: [R4] (rating 4/5) It is very quiet in the morning.
{"evidence_quote": "It is very quiet in the morning", "explanation": "One review says it is quiet; nothing about speed or about all users.", "status": "PARTIALLY_SUPPORTED"}"""

HUMAN_V4 = """CLAIM ({role}): {claim}

EVIDENCE ({evidence_kind}):
{evidence}

Return the JSON object for this claim."""


def prompt_sha256() -> str:
    return hashlib.sha256((SYSTEM_V4 + "\n----\n" + HUMAN_V4).encode("utf-8")).hexdigest()


@lru_cache(maxsize=1)
def v4_json_schema() -> dict:
    return LLMClaimVerdict.model_json_schema()


def v4_settings(settings):
    """Shorter context and output budget for one-claim prompts (prompts are < 1.5k tokens)."""
    return dataclasses.replace(settings, ollama_num_ctx=V4_NUM_CTX, llm_max_output_tokens=V4_NUM_PREDICT)


def get_judge_v4_client(settings, *, use_mock: bool) -> LLMClient:
    from llm.factory import get_llm_client

    return get_llm_client(v4_settings(settings), use_mock=use_mock, role="judge", response_schema=v4_json_schema())


def run_config(client: LLMClient, settings) -> dict:
    s = v4_settings(settings)
    return {"judge_version": JUDGE_VERSION_V4, "judge_provider": client.provider, "judge_model": client.model_name,
            "is_mock": client.is_mock, "constrained_decoding": bool(getattr(client, "constrained_decoding", False)),
            "temperature": s.llm_temperature, "num_ctx": s.ollama_num_ctx, "num_predict": s.llm_max_output_tokens,
            "timeout_s": s.llm_timeout_seconds, "calls_per_claim": 1, "repair_attempts": 0, "prompt_sha256": prompt_sha256(),
            "review_word_limit": REVIEW_WORD_LIMIT}


# --------------------------------------------------------------------------- data
def load_review_index(reviews_csv: Path) -> dict[str, dict]:
    df = pd.read_csv(reviews_csv, dtype={"review_id": str, "product_id": str}, keep_default_na=False)
    out = {}
    for r in df.itertuples(index=False):
        try:
            rating = float(r.rating)
        except (TypeError, ValueError):
            rating = None
        out[r.review_id] = {"product_id": r.product_id, "rating": rating, "title": str(getattr(r, "review_title", "") or ""),
                            "text": str(r.review_text or "")}
    return out


def review_texts(ids: list[str], reviews: dict[str, dict], product_id: str) -> list[str]:
    return [f"{reviews[r]['title']}\n{reviews[r]['text']}" for r in ids if r in reviews and reviews[r]["product_id"] == product_id]


def render_review_evidence(ids: list[str], reviews: dict[str, dict], product_id: str) -> str:
    blocks = []
    for rid in ids:
        r = reviews.get(rid)
        if not r or r["product_id"] != product_id:
            continue
        words = r["text"].split()
        text = " ".join(words[:REVIEW_WORD_LIMIT]) + (" [...]" if len(words) > REVIEW_WORD_LIMIT else "")
        rating = f"{r['rating']:.0f}/5" if r["rating"] is not None else "unknown"
        blocks.append(f"[{rid}] (rating {rating}) {r['title']}. {text}")
    return "\n\n".join(blocks)


# --------------------------------------------------------------------------- deterministic stage
@dataclasses.dataclass
class DetResult:
    status: ClaimStatus | None  # None: no deterministic verdict (point claims always go to the LLM)
    findings: list[DeterministicFinding]
    numeric: list[NumericCheck]
    evidence: list[EvidenceExcerpt]
    needs_llm: bool
    evidence_mode: str  # "reviews" | "statistics" | "none"


def _caps(findings: list[DeterministicFinding], numeric: list[NumericCheck]) -> list[ClaimStatus]:
    caps = [f.status_cap for f in findings if f.status_cap is not None]
    for n in numeric:
        if n.severity == "error":
            caps.append(ClaimStatus.CONTRADICTED)
        elif n.severity == "misleading":
            caps.append(ClaimStatus.PARTIALLY_SUPPORTED)
    return caps


def _worst(caps: list[ClaimStatus]) -> ClaimStatus | None:
    return max(caps, key=lambda s: SEVERITY[s]) if caps else None


def numeric_findings(numeric: list[NumericCheck]) -> list[DeterministicFinding]:
    """Numeric checks expressed as findings with caps, so the schema can enforce that they never disappear."""
    out = []
    for n in numeric:
        if n.severity == "error":
            out.append(DeterministicFinding(check=f"numeric_{n.kind}", outcome="fail", status_cap=ClaimStatus.CONTRADICTED,
                                            detail=f"'{n.mention}': {n.calculation}"))
        elif n.severity == "misleading":
            out.append(DeterministicFinding(check=f"numeric_{n.kind}", outcome="misleading", status_cap=ClaimStatus.PARTIALLY_SUPPORTED,
                                            detail=f"'{n.mention}': {n.calculation}"))
    return out


def deterministic_stage(claim: Claim, st: Stats, reviews: dict[str, dict], frequent_list: list[str] | None = None,
                        verified_negative: dict[str, str] | None = None) -> DetResult:
    if claim.kind == "opinion_sentence":
        numeric = numeric_checks_statistics(claim.text, st)
        findings = qualitative_checks(claim.text, st) + numeric_findings(numeric)
        definite = [f for f in findings if f.outcome != "info"] or numeric
        if definite:
            return DetResult(_worst(_caps(findings, [])) or ClaimStatus.SUPPORTED, findings, numeric, [], False, "statistics")
        return DetResult(None, findings, numeric, [], True, "statistics")
    if claim.kind == "point":
        ev, findings = citation_evidence(claim.cited_review_ids, claim.product_id, reviews)
        texts = review_texts(claim.cited_review_ids, reviews, claim.product_id)
        numeric = number_checks_in_texts(claim.text, texts)
        findings += qualitative_checks(claim.text, st, polarity_hint=claim.polarity, aspect_statements=False)
        findings += numeric_findings(numeric)
        rp = rating_polarity_finding(claim.polarity, ev)
        if rp:
            findings.append(rp)
        return DetResult(_worst(_caps(findings, [])), findings, numeric, ev, bool(texts), "reviews")
    # derived
    if claim.source_field == "frequent_aspects":
        findings = frequent_aspects_check(frequent_list or [], st)
        return DetResult(_worst(_caps(findings, [])) or ClaimStatus.SUPPORTED, findings, [], [], False, "statistics")
    status, findings = derived_claim_check(claim.text, st, verified_negative)
    return DetResult(status or ClaimStatus.UNVERIFIABLE, findings, [], [], False, "statistics" if status else "none")


# --------------------------------------------------------------------------- LLM stage
def build_messages(claim: Claim, det: DetResult, st: Stats, reviews: dict[str, dict]) -> tuple[list[ChatMessage], list[str]]:
    """Messages for one claim and the evidence texts its quote must come from."""
    if det.evidence_mode == "reviews":
        evidence = render_review_evidence(claim.cited_review_ids, reviews, claim.product_id)
        texts = review_texts(claim.cited_review_ids, reviews, claim.product_id) + [evidence]
        kind = "the full text of the customer reviews this claim cites"
    else:
        evidence = st.render()
        texts = [evidence]
        kind = "statistics computed from all reviews of this product"
    role = {"positive": "something customers praise", "negative": "a complaint or problem", "neutral": "an overall statement"}[claim.polarity]
    human = HUMAN_V4.format(role=role, claim=claim.text, evidence_kind=kind, evidence=evidence)
    return [ChatMessage("system", SYSTEM_V4), ChatMessage("user", human)], texts


def parse_v4(raw: str) -> tuple[LLMClaimVerdict | None, str | None]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON: {exc}"
    try:
        return LLMClaimVerdict.model_validate(data), None
    except ValidationError as exc:
        return None, "schema: " + "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()[:6])


def call_key(kind: str, summary_id: str, claim_id: str) -> tuple[str, str, str]:
    return (kind, summary_id, claim_id)


class V4Store:
    """Append-only store of LLM claim verdicts (valid and failed), resumable by (kind, summary_id, claim_id)."""

    def __init__(self, folder: Path):
        self.folder = folder
        self.path = folder / "claim_verdicts.jsonl"
        self.failures_path = folder / "failures.jsonl"

    def records(self) -> list[dict]:
        return read_jsonl(self.path)

    def failures(self) -> list[dict]:
        return read_jsonl(self.failures_path)

    def index(self) -> dict[tuple, dict]:
        out = {}
        for r in self.failures() + self.records():  # a valid record wins over an earlier failure
            out[call_key(r["evaluated_object"], r["summary_id"], r["claim_id"])] = r
        return out


def judge_claim(client: LLMClient, claim: Claim, det: DetResult, st: Stats, reviews: dict[str, dict], *, kind: str, config: dict,
                run_id: str) -> tuple[bool, dict]:
    """ONE LLM call for one claim. Provider errors raise ``LLMError`` and record nothing (the run resumes)."""
    messages, texts = build_messages(claim, det, st, reviews)
    input_sha = hashlib.sha256("\n".join(m.content for m in messages).encode("utf-8")).hexdigest()
    t0 = time.perf_counter()
    resp = client.invoke(messages, task="judge_v4", payload={"claim": claim.text, "evidence_texts": texts, "polarity": claim.polarity})
    latency = resp.latency_s or (time.perf_counter() - t0)
    out, err = parse_v4(resp.text)
    rec = {"record_id": hashlib.sha1(f"{kind}|{claim.claim_id}|{run_id}|{resp.text}".encode()).hexdigest()[:12],
           "run_id": run_id, "evaluated_object": kind, "summary_id": claim.summary_id, "product_id": claim.product_id,
           "claim_id": claim.claim_id, "claim_text": claim.text, "evidence_mode": det.evidence_mode,
           "evidence_review_ids": claim.cited_review_ids if det.evidence_mode == "reviews" else [],
           "input_sha256": input_sha, "raw_response": resp.text, "valid": out is not None, "latency_s": round(latency, 2),
           "input_tokens": resp.input_tokens, "output_tokens": resp.output_tokens, "config": config,
           "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    if out is None:
        rec["validation_error"] = err
        return False, rec
    rec.update({"status": out.status.value, "evidence_quote": out.evidence_quote, "explanation": out.explanation,
                "quote_verified": quote_in_texts(out.evidence_quote, texts)})
    return True, rec


# --------------------------------------------------------------------------- combination
def effective_llm_status(rec: dict | None) -> ClaimStatus | None:
    if not rec or not rec.get("valid"):
        return None
    s = ClaimStatus(rec["status"])
    if s in (ClaimStatus.SUPPORTED, ClaimStatus.PARTIALLY_SUPPORTED, ClaimStatus.CONTRADICTED) and not rec.get("quote_verified"):
        return ClaimStatus.UNVERIFIABLE
    return s


def combine(det_status: ClaimStatus | None, llm_eff: ClaimStatus | None, needs_llm: bool) -> tuple[ClaimStatus, str]:
    """Final status. Deterministic findings are never overridden by a better or an UNVERIFIABLE LLM verdict."""
    if not needs_llm:
        return (det_status or ClaimStatus.UNVERIFIABLE), "deterministic"
    if det_status is None:
        return (llm_eff or ClaimStatus.UNVERIFIABLE), ("llm" if llm_eff else "none")
    if llm_eff is None or llm_eff == ClaimStatus.UNVERIFIABLE:
        return det_status, "deterministic+llm"
    return max((det_status, llm_eff), key=lambda s: SEVERITY[s]), "deterministic+llm"


def evaluate_claim(claim: Claim, det: DetResult, rec: dict | None) -> ClaimEvaluation:
    llm_eff = effective_llm_status(rec) if det.needs_llm else None
    status, method = combine(det.status, llm_eff, det.needs_llm)
    parts = [f.detail for f in det.findings if f.outcome in ("fail", "misleading")]
    if rec and rec.get("valid"):
        q = "" if rec.get("quote_verified") or not rec.get("evidence_quote") else " [quote NOT found in the evidence: verdict treated as UNVERIFIABLE]"
        parts.append(f"LLM ({rec['status']}): {rec['explanation']}{q}")
    elif det.needs_llm:
        parts.append("LLM verdict unavailable (" + (rec.get("validation_error", "not run") if rec else "not run") + ")")
    if not parts:
        parts = [f.detail for f in det.findings if f.outcome == "pass"][:3] or ["no deterministic rule applies and no evidence was available"]
    ev = list(det.evidence)
    if rec and rec.get("valid") and rec.get("evidence_quote") and det.evidence_mode == "reviews":
        ev = [e.model_copy(update={"excerpt": rec["evidence_quote"], "excerpt_verified": bool(rec.get("quote_verified"))}) for e in ev]
    return ClaimEvaluation(
        claim=claim, status=status, explanation=" | ".join(parts)[:2000], method=method, deterministic_status=det.status,
        llm_status=ClaimStatus(rec["status"]) if rec and rec.get("valid") else None, llm_status_effective=llm_eff,
        llm_quote=rec.get("evidence_quote") if rec else None, llm_quote_verified=rec.get("quote_verified") if rec else None,
        llm_explanation=rec.get("explanation") if rec else None, llm_record_id=rec.get("record_id") if rec else None,
        evidence=ev, numeric_checks=det.numeric, findings=det.findings)


def build_summary_evaluation(*, kind: str, summary_id: str, product_id: str, claim_evals: list[ClaimEvaluation], run_id: str,
                             config: dict, original_summary_id: str | None = None) -> SummaryEvaluationV4:
    d = derive_scores([e.status for e in claim_evals])
    numeric = [e for e in claim_evals if e.numeric_checks]
    failed = [e.claim.claim_id for e in numeric if any(not n.passed for n in e.numeric_checks)]

    def ids(s):
        return [e.claim.claim_id for e in claim_evals if e.status == s]

    return SummaryEvaluationV4(
        evaluation_id=hashlib.sha1(f"{kind}|{summary_id}|{run_id}".encode()).hexdigest()[:12], run_id=run_id, evaluated_object=kind,
        summary_id=summary_id, product_id=product_id, original_summary_id=original_summary_id, total_claims=d["total_claims"],
        counts=d["counts"], verifiable_claims=d["verifiable_claims"], evidence_coverage=d["evidence_coverage"],
        numeric_claims=len(numeric), numeric_claims_passed=len(numeric) - len(failed), numeric_claims_failed=len(failed),
        faithfulness_score=d["faithfulness_score"], faithfulness_band_1_5=d["faithfulness_band_1_5"], validity_status=d["validity_status"],
        unsupported_claims=ids(ClaimStatus.UNSUPPORTED), contradicted_claims=ids(ClaimStatus.CONTRADICTED),
        partially_supported_claims=ids(ClaimStatus.PARTIALLY_SUPPORTED), unverifiable_claims=ids(ClaimStatus.UNVERIFIABLE),
        failed_numeric_claims=failed, claim_evaluations=claim_evals, config=config,
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))


# --------------------------------------------------------------------------- orchestration
@dataclasses.dataclass
class Item:
    """One object to evaluate: its claims plus the context needed to check them."""

    kind: str
    summary_id: str
    product_id: str
    claims: list[Claim]
    stats: Stats
    frequent_list: list[str]
    original_summary_id: str | None = None
    extra: dict = dataclasses.field(default_factory=dict)


def item_from_summary(kind: str, summary, records=None, original_summary_id: str | None = None, extra: dict | None = None) -> Item:
    return Item(kind=kind, summary_id=summary.summary_id, product_id=summary.product_id,
                claims=decompose(summary.summary_id, summary.product_id, summary.content),
                stats=Stats.from_aggregate(summary.aggregate, records), frequent_list=list(summary.content.frequent_aspects),
                original_summary_id=original_summary_id, extra=extra or {})


def run_llm(client: LLMClient, items: list[Item], reviews: dict[str, dict], store: V4Store, *, config: dict, run_id: str,
            before_call=None, progress=None, limit: int | None = None) -> dict:
    """Sequential: one LLM call per claim that needs one and is not yet stored (valid or failed)."""
    done = store.index()
    stats = {"called": 0, "valid": 0, "failed": 0, "skipped_stored": 0, "deterministic_only": 0}
    for it in items:
        for c in it.claims:
            det = deterministic_stage(c, it.stats, reviews, it.frequent_list)
            if not det.needs_llm:
                stats["deterministic_only"] += 1
                continue
            if call_key(it.kind, it.summary_id, c.claim_id) in done:
                stats["skipped_stored"] += 1
                continue
            if limit is not None and stats["called"] >= limit:
                return stats
            if before_call:
                before_call()
            try:
                ok, rec = judge_claim(client, c, det, it.stats, reviews, kind=it.kind, config=config, run_id=run_id)
            except LLMError:
                raise
            store.folder.mkdir(parents=True, exist_ok=True)
            append_jsonl(store.path if ok else store.failures_path, [rec])
            stats["called"] += 1
            stats["valid" if ok else "failed"] += 1
            if progress:
                progress(it, c, ok, rec)
    return stats


def evaluate_items(items: list[Item], reviews: dict[str, dict], store: V4Store, *, run_id: str, config: dict) -> list[SummaryEvaluationV4]:
    """Pure post-processing (no LLM): combine stored LLM verdicts with deterministic checks and derive the scores."""
    idx = store.index()
    out = []
    for it in items:
        by_id: dict[str, ClaimEvaluation] = {}
        for c in (c for c in it.claims if c.kind != "derived"):
            det = deterministic_stage(c, it.stats, reviews, it.frequent_list)
            rec = idx.get(call_key(it.kind, it.summary_id, c.claim_id)) if det.needs_llm else None
            by_id[c.claim_id] = evaluate_claim(c, det, rec)
        verified_negative = {}
        for e in by_id.values():
            if e.claim.source_field in ("complaints", "planted") and e.claim.polarity == "negative" and                     e.status in (ClaimStatus.SUPPORTED, ClaimStatus.PARTIALLY_SUPPORTED):
                for a in aspects_in(e.claim.text):
                    verified_negative.setdefault(a, e.claim.claim_id)
        for c in (c for c in it.claims if c.kind == "derived"):
            det = deterministic_stage(c, it.stats, reviews, it.frequent_list, verified_negative)
            by_id[c.claim_id] = evaluate_claim(c, det, None)
        evs = [by_id[c.claim_id] for c in it.claims]
        out.append(build_summary_evaluation(kind=it.kind, summary_id=it.summary_id, product_id=it.product_id, claim_evals=evs,
                                            run_id=run_id, config=config, original_summary_id=it.original_summary_id))
    return out
