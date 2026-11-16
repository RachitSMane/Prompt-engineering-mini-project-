"""Schemas for the claim-level D3 evaluation (judge v4).

A product summary is decomposed into CLAIMS (``evaluation/claims.py``). Every claim receives exactly one
``ClaimStatus``. Scores are never written by an LLM: ``derive_scores`` computes them from the claim statuses with a
fixed, documented formula, and ``SummaryEvaluationV4`` re-derives and checks every count, list and score when it is
validated, so a stored evaluation whose score contradicts its own claims cannot be loaded.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ClaimStatus(str, Enum):
    SUPPORTED = "SUPPORTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    UNSUPPORTED = "UNSUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    UNVERIFIABLE = "UNVERIFIABLE"


#: Severity used when two verdicts (deterministic and LLM) are combined: the worse one wins. UNVERIFIABLE is handled
#: separately (it never overrides a definite deterministic finding; see ``evaluation.judge_v4.combine``).
SEVERITY = {ClaimStatus.SUPPORTED: 0, ClaimStatus.PARTIALLY_SUPPORTED: 1, ClaimStatus.UNSUPPORTED: 3, ClaimStatus.CONTRADICTED: 4}

CLAIM_CATEGORIES = (
    "product_performance", "quality_durability", "design_build", "ease_of_use", "installation", "compatibility",
    "price_value", "customer_service", "shipping_packaging", "quantitative_statistic", "overall_sentiment", "other",
)
ClaimCategory = Literal[
    "product_performance", "quality_durability", "design_build", "ease_of_use", "installation", "compatibility",
    "price_value", "customer_service", "shipping_packaging", "quantitative_statistic", "overall_sentiment", "other",
]
SourceField = Literal["overall_opinion", "appreciated", "complaints", "frequent_aspects", "improvement_areas", "recommendations",
                      "planted"]
ClaimKind = Literal["opinion_sentence", "point", "derived"]


class Claim(BaseModel):
    """One factual statement taken verbatim from a summary (never invented)."""

    model_config = ConfigDict(extra="forbid")

    claim_id: str
    summary_id: str
    product_id: str
    source_field: SourceField
    index: int = Field(..., ge=0)
    kind: ClaimKind
    text: str = Field(..., min_length=1)
    polarity: Literal["positive", "negative", "neutral"] = "neutral"
    category: ClaimCategory = "other"
    is_numeric: bool = False
    cited_review_ids: list[str] = Field(default_factory=list)


class EvidenceExcerpt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    review_id: str
    exists: bool
    belongs_to_product: bool
    rating: float | None = None
    excerpt: str = ""
    excerpt_verified: bool = False  # exact (normalised) substring of the original review text


class NumericCheck(BaseModel):
    """One number in a claim, re-computed from the data (never by an LLM)."""

    model_config = ConfigDict(extra="forbid")

    mention: str
    kind: Literal["percentage", "count", "average_rating", "number_in_cited_text"]
    extracted_value: float
    expected_value: float | None = None
    denominator: int | None = None
    calculation: str
    passed: bool
    severity: Literal["ok", "misleading", "error"] = "ok"


class DeterministicFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    check: str
    outcome: Literal["pass", "misleading", "fail", "info"]
    status_cap: ClaimStatus | None = None  # the best status the claim can still get because of this finding
    detail: str


class LLMClaimVerdict(BaseModel):
    """What the v4 claim verifier (local LLM) must return for ONE claim. Constrained decoding enforces this shape."""

    model_config = ConfigDict(extra="forbid")

    evidence_quote: str = Field(..., min_length=1, max_length=600)
    explanation: str = Field(..., min_length=3, max_length=600)
    status: ClaimStatus


class ClaimEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim: Claim
    status: ClaimStatus
    explanation: str
    method: Literal["deterministic", "llm", "deterministic+llm", "none"]
    deterministic_status: ClaimStatus | None = None
    llm_status: ClaimStatus | None = None
    llm_status_effective: ClaimStatus | None = None  # after the evidence-quote check
    llm_quote: str | None = None
    llm_quote_verified: bool | None = None
    llm_explanation: str | None = None
    llm_record_id: str | None = None
    evidence: list[EvidenceExcerpt] = Field(default_factory=list)
    numeric_checks: list[NumericCheck] = Field(default_factory=list)
    findings: list[DeterministicFinding] = Field(default_factory=list)

    @model_validator(mode="after")
    def _deterministic_failures_cannot_disappear(self) -> "ClaimEvaluation":
        if any(n.severity == "error" for n in self.numeric_checks) and self.status == ClaimStatus.SUPPORTED:
            raise ValueError(f"{self.claim.claim_id}: a failed numeric check cannot end as SUPPORTED")
        caps = [f.status_cap for f in self.findings if f.status_cap is not None]
        if caps:
            worst = max(caps, key=lambda s: SEVERITY[s])
            if self.status == ClaimStatus.UNVERIFIABLE or SEVERITY[self.status] < SEVERITY[worst]:
                raise ValueError(f"{self.claim.claim_id}: status {self.status.value} is better than deterministic cap {worst.value}")
        return self


# --------------------------------------------------------------------------- scoring
#: Fixed weights of the faithfulness score (rationale in reports/final/D3_Final_Report.md, "Scoring methodology").
WEIGHTS = {ClaimStatus.SUPPORTED: 1.0, ClaimStatus.PARTIALLY_SUPPORTED: 0.5, ClaimStatus.UNSUPPORTED: 0.0,
           ClaimStatus.CONTRADICTED: -1.0}
SCORING_FORMULA = ("faithfulness = 100 * max(0, (S + 0.5*P + 0*U - 1*C) / (S + P + U + C)); UNVERIFIABLE claims are excluded "
                   "from the denominator and reported as (1 - evidence_coverage). Caps: any CONTRADICTED claim -> score <= 60; "
                   "any UNSUPPORTED claim -> score <= 80; 'perfect' (100) only if every verifiable claim is SUPPORTED.")
MIN_COVERAGE = 0.5
CONTRADICTED_CAP = 60.0
UNSUPPORTED_CAP = 80.0


def status_counts(statuses: list[ClaimStatus]) -> dict[str, int]:
    return {s.value: sum(1 for x in statuses if x == s) for s in ClaimStatus}


def derive_scores(statuses: list[ClaimStatus], weights: dict[ClaimStatus, float] | None = None, *, caps: bool = True) -> dict:
    """Pure function: claim statuses -> counts, coverage, faithfulness score (0-100), 1-5 band and validity status."""
    w = weights or WEIGHTS
    c = status_counts(statuses)
    total = len(statuses)
    verifiable = total - c["UNVERIFIABLE"]
    coverage = round(verifiable / total, 4) if total else 0.0
    if verifiable == 0:
        score = None
    else:
        raw = sum(w[s] for s in statuses if s != ClaimStatus.UNVERIFIABLE) / verifiable
        score = 100.0 * max(0.0, raw)
        if caps and c["CONTRADICTED"]:
            score = min(score, CONTRADICTED_CAP)
        if caps and c["UNSUPPORTED"]:
            score = min(score, UNSUPPORTED_CAP)
        score = round(score, 1)
    if total == 0 or coverage < MIN_COVERAGE:
        validity = "INSUFFICIENT_EVIDENCE"
    elif c["CONTRADICTED"]:
        validity = "CONTRADICTED_CLAIMS"
    elif c["UNSUPPORTED"]:
        validity = "UNSUPPORTED_CLAIMS"
    elif c["PARTIALLY_SUPPORTED"]:
        validity = "MINOR_ISSUES"
    else:
        validity = "FAITHFUL"
    band = None if score is None else (5 if score >= 95 else 4 if score >= 80 else 3 if score >= 60 else 2 if score >= 40 else 1)
    return {"counts": c, "total_claims": total, "verifiable_claims": verifiable, "evidence_coverage": coverage,
            "faithfulness_score": score, "faithfulness_band_1_5": band, "validity_status": validity}


class SummaryEvaluationV4(BaseModel):
    """Claim-level evaluation of one summary. Every aggregate field is re-derived from ``claim_evaluations`` on
    validation; a mismatch raises, so a score/claim contradiction cannot be stored or loaded."""

    model_config = ConfigDict(extra="forbid")

    evaluation_id: str
    run_id: str
    evaluated_object: Literal["original", "corrected", "nc_corrupted", "nc_clean"]
    summary_id: str
    product_id: str
    original_summary_id: str | None = None
    total_claims: int
    counts: dict[str, int]
    verifiable_claims: int
    evidence_coverage: float
    numeric_claims: int
    numeric_claims_passed: int
    numeric_claims_failed: int
    faithfulness_score: float | None
    faithfulness_band_1_5: int | None
    validity_status: str
    unsupported_claims: list[str]
    contradicted_claims: list[str]
    partially_supported_claims: list[str]
    unverifiable_claims: list[str]
    failed_numeric_claims: list[str]
    claim_evaluations: list[ClaimEvaluation]
    scoring_formula: str = SCORING_FORMULA
    config: dict = Field(default_factory=dict)
    created_at: str

    @model_validator(mode="after")
    def _consistent_with_claims(self) -> "SummaryEvaluationV4":
        evs = self.claim_evaluations
        statuses = [e.status for e in evs]
        d = derive_scores(statuses)
        problems = []
        for k in ("total_claims", "counts", "verifiable_claims", "evidence_coverage", "faithfulness_score",
                  "faithfulness_band_1_5", "validity_status"):
            if getattr(self, k) != d[k]:
                problems.append(f"{k}={getattr(self, k)!r} but claims give {d[k]!r}")

        def ids(status):
            return [e.claim.claim_id for e in evs if e.status == status]

        for field, status in (("unsupported_claims", ClaimStatus.UNSUPPORTED), ("contradicted_claims", ClaimStatus.CONTRADICTED),
                              ("partially_supported_claims", ClaimStatus.PARTIALLY_SUPPORTED),
                              ("unverifiable_claims", ClaimStatus.UNVERIFIABLE)):
            if getattr(self, field) != ids(status):
                problems.append(f"{field} does not list exactly the {status.value} claims")
        numeric = [e for e in evs if e.numeric_checks]
        failed = [e.claim.claim_id for e in numeric if any(not n.passed for n in e.numeric_checks)]
        if (self.numeric_claims, self.numeric_claims_failed, self.numeric_claims_passed) != (len(numeric), len(failed), len(numeric) - len(failed)):
            problems.append("numeric claim counts do not match the numeric checks")
        if self.failed_numeric_claims != failed:
            problems.append("failed_numeric_claims does not list exactly the claims with a failed numeric check")
        if self.faithfulness_score == 100.0 and any(s != ClaimStatus.SUPPORTED for s in statuses if s != ClaimStatus.UNVERIFIABLE):
            problems.append("perfect score with a claim that is not SUPPORTED")
        if problems:
            raise ValueError("inconsistent v4 evaluation: " + "; ".join(problems))
        return self
