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


class JudgeVerdict(BaseModel):
    """Structured LLM-as-a-Judge output (rubric in prompts/judge.py)."""

    model_config = ConfigDict(extra="forbid")

    faithfulness: DimensionScore
    coverage: DimensionScore
    consistency: DimensionScore
    hallucination: DimensionScore = Field(..., description="5 = no unsupported claims, 1 = many/severe unsupported claims")
    unsupported_claims: list[str] = Field(default_factory=list)
    overall_comment: str = Field(..., min_length=5, max_length=800)

    @property
    def mean_score(self) -> float:
        return round((self.faithfulness.score + self.coverage.score + self.consistency.score + self.hallucination.score) / 4, 2)


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
