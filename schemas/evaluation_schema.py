"""Schemas for annotations, aggregation, product summaries and LLM-as-a-Judge."""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator

from schemas.taxonomy import AspectCategory, AspectSentiment, ComplaintCategory, OverallSentiment


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------- annotation
class AnnotatedAspect(BaseModel):
    """Human label: one sentiment per aspect category per review."""

    model_config = ConfigDict(extra="forbid")

    aspect_category: AspectCategory
    sentiment: AspectSentiment
    aspect_name: str | None = None
    evidence: str | None = None


class Annotation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    review_id: str
    product_id: str
    overall_sentiment: OverallSentiment
    aspects: list[AnnotatedAspect] = Field(default_factory=list)
    complaint_categories: list[ComplaintCategory] = Field(default_factory=list)
    annotator: str = Field(..., min_length=1)
    notes: str = ""
    status: str = Field("complete", pattern="^(complete|flagged)$")
    annotated_at: str = Field(default_factory=_now)

    @model_validator(mode="after")
    def _one_label_per_category(self) -> "Annotation":
        cats = [a.aspect_category for a in self.aspects]
        if len(cats) != len(set(cats)):
            raise ValueError("each aspect category may be labelled only once per review")
        self.complaint_categories = list(dict.fromkeys(self.complaint_categories))
        return self


# --------------------------------------------------------------------------- aggregation
class AspectStat(BaseModel):
    aspect_category: str
    mentions: int
    positive: int
    negative: int
    neutral: int
    net_sentiment: float = Field(..., description="(positive - negative) / (positive + negative + neutral), in [-1, 1]")
    top_aspect_names: list[str] = Field(default_factory=list)


class EvidenceQuote(BaseModel):
    review_id: str
    aspect_category: str
    sentiment: str
    quote: str


class ProductAggregate(BaseModel):
    product_id: str
    product_title: str | None = None
    total_reviews: int
    sentiment_counts: dict[str, int]
    sentiment_percentages: dict[str, float]
    average_rating: float | None = None
    aspect_stats: list[AspectStat]
    complaint_counts: dict[str, int]
    complaint_percentages: dict[str, float]
    reviews_with_complaints: int
    aspect_trend: list[dict] = Field(default_factory=list, description="Rows: period, aspect_category, mentions, net_sentiment")
    positive_evidence: list[EvidenceQuote] = Field(default_factory=list)
    negative_evidence: list[EvidenceQuote] = Field(default_factory=list)
    review_ids: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- summary
class SummaryPoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    point: str = Field(..., min_length=3, max_length=400)
    supporting_review_ids: list[str] = Field(..., min_length=1, description="IDs of reviews that support this point (from the evidence provided).")

    @field_validator("supporting_review_ids")
    @classmethod
    def _ids_exist(cls, v: list[str], info: ValidationInfo) -> list[str]:
        allowed = (info.context or {}).get("allowed_review_ids")
        if allowed is not None:
            unknown = [rid for rid in v if rid not in allowed]
            if unknown:
                raise ValueError(f"unknown review ids {unknown[:5]}; cite only ids that appear in the evidence")
        return list(dict.fromkeys(v))


class ProductSummaryContent(BaseModel):
    """What the summarisation LLM must return."""

    model_config = ConfigDict(extra="forbid")

    overall_opinion: str = Field(..., min_length=10, max_length=800)
    appreciated: list[SummaryPoint] = Field(default_factory=list, max_length=6)
    complaints: list[SummaryPoint] = Field(default_factory=list, max_length=6)
    frequent_aspects: list[str] = Field(default_factory=list, max_length=8)
    improvement_areas: list[str] = Field(default_factory=list, max_length=6)
    recommendations: list[str] = Field(default_factory=list, max_length=6)


class ProductSummary(BaseModel):
    summary_id: str
    product_id: str
    product_title: str | None = None
    content: ProductSummaryContent
    aggregate: ProductAggregate
    source_extraction: str = Field(..., description="Name of the extraction result set used.")
    provider: str
    model_name: str
    is_mock: bool
    prompt_version: str
    repair_attempts: int = 0
    latency_s: float = 0.0
    input_tokens: int | None = None
    output_tokens: int | None = None
    created_at: str = Field(default_factory=_now)


# --------------------------------------------------------------------------- judge
class DimensionScore(BaseModel):
    model_config = ConfigDict(extra="forbid")

    score: int = Field(..., ge=1, le=5)
    justification: str = Field(..., min_length=5, max_length=800)


class UnsupportedClaim(BaseModel):
    """One claim from the summary that the judge could not find support for in the sources."""

    model_config = ConfigDict(extra="forbid")

    claim: str = Field(..., min_length=3, max_length=400, description="The unsupported statement, quoted from the summary.")
    reason: str = Field("", max_length=400, description="Why it is unsupported (what the sources say instead).")
    evidence_review_ids: list[str] = Field(default_factory=list, description="Review ids the judge checked for this claim, if any.")


class JudgeVerdict(BaseModel):
    """Structured LLM-as-a-Judge verdict as stored (rubric in prompts/judge.py).

    v1 judgements were produced directly in this nested shape. v2 judgements are produced in the flat
    ``JudgeOutputV2`` shape (easier for small local models) and converted with ``from_v2``; they also
    carry ``unsupported_claim_details``. The field defaults keep every stored v1 judgement loadable.
    """

    model_config = ConfigDict(extra="forbid")

    faithfulness: DimensionScore
    coverage: DimensionScore
    consistency: DimensionScore
    hallucination: DimensionScore = Field(..., description="5 = no unsupported claims, 1 = many/severe unsupported claims")
    unsupported_claims: list[str] = Field(default_factory=list)
    overall_comment: str = Field(..., min_length=5, max_length=800)
    unsupported_claim_details: list[UnsupportedClaim] = Field(default_factory=list)

    @classmethod
    def from_v2(cls, out: "JudgeOutputV2") -> "JudgeVerdict":
        return cls(
            faithfulness=DimensionScore(score=out.faithfulness_score, justification=out.faithfulness_justification),
            coverage=DimensionScore(score=out.coverage_score, justification=out.coverage_justification),
            consistency=DimensionScore(score=out.consistency_score, justification=out.consistency_justification),
            hallucination=DimensionScore(score=out.hallucination_score, justification=out.hallucination_justification),
            unsupported_claims=[c.claim for c in out.unsupported_claims],
            unsupported_claim_details=list(out.unsupported_claims),
            overall_comment=out.overall_comment,
        )

    @property
    def mean_score(self) -> float:
        return round((self.faithfulness.score + self.coverage.score + self.consistency.score + self.hallucination.score) / 4, 2)


_SCORE = dict(ge=1, le=5, description="Integer 1-5")
_JUST = dict(min_length=5, max_length=800)


class JudgeOutputV2(BaseModel):
    """What the v2 judge LLM must return: one flat object, one score + justification per dimension.

    Root cause of the v1 failures: the nested v1 schema (dimension -> {score, justification}) led llama3.2 to copy
    ``unsupported_claims`` into every dimension object or to return bare integers, which ``extra="forbid"`` rejected.
    A flat object has no nesting to get wrong. The cross-field rule below makes the verdict internally consistent:
    a hallucination score below 5 must name the unsupported claims, and named claims rule out a perfect score.
    """

    model_config = ConfigDict(extra="forbid")

    faithfulness_score: int = Field(..., **_SCORE)
    faithfulness_justification: str = Field(..., **_JUST)
    coverage_score: int = Field(..., **_SCORE)
    coverage_justification: str = Field(..., **_JUST)
    consistency_score: int = Field(..., **_SCORE)
    consistency_justification: str = Field(..., **_JUST)
    hallucination_score: int = Field(..., ge=1, le=5, description="5 = no unsupported claims, 1 = fabricated statistics/features")
    hallucination_justification: str = Field(..., **_JUST)
    unsupported_claims: list[UnsupportedClaim] = Field(default_factory=list, max_length=15)
    overall_comment: str = Field(..., min_length=5, max_length=800)

    @field_validator("faithfulness_score", "coverage_score", "consistency_score", "hallucination_score", mode="before")
    @classmethod
    def _integer_scores_only(cls, v):
        # Reject 3.5 / "4/5" / True instead of letting them be coerced or truncated; accept 4, 4.0 and "4".
        if isinstance(v, bool):
            raise ValueError("score must be an integer 1-5, not a boolean")
        if isinstance(v, float) and not v.is_integer():
            raise ValueError(f"score must be a whole number 1-5 (got {v})")
        return v

    @model_validator(mode="after")
    def _hallucination_consistent_with_claims(self) -> "JudgeOutputV2":
        if self.hallucination_score < 5 and not self.unsupported_claims:
            raise ValueError(
                f"hallucination_score is {self.hallucination_score} but unsupported_claims is empty: list the unsupported claims, "
                "or use hallucination_score 5 if there are none"
            )
        if self.hallucination_score == 5 and self.unsupported_claims:
            raise ValueError("hallucination_score 5 means no unsupported claims, but unsupported_claims is not empty")
        return self


class SummaryJudgement(BaseModel):
    judgement_id: str
    summary_id: str
    product_id: str
    verdict: JudgeVerdict
    deterministic_checks: dict = Field(default_factory=dict)
    judge_provider: str
    judge_model: str
    is_mock: bool
    prompt_version: str
    repair_attempts: int = 0
    latency_s: float = 0.0
    created_at: str = Field(default_factory=_now)
