"""LLM-as-a-Judge prompt and rubric for product summaries.

The judge's scores are an automated, model-based assessment. They are NOT
objective human evaluation and are reported as such.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate

from schemas.evaluation_schema import JudgeVerdict

JUDGE_PROMPT_VERSION = "v1"

RUBRIC = """RUBRIC (integer scores 1-5):

FAITHFULNESS — are the summary's statements supported by the reviews and extractions?
  5 = every statement is directly supported; 4 = all supported, minor imprecise wording;
  3 = mostly supported but some over-generalisation; 2 = several statements weakly supported or misattributed;
  1 = many statements unsupported or contradicted by the reviews.

COVERAGE — does the summary cover the important aspects and complaints?
  5 = covers all frequently mentioned aspects and the main complaints; 4 = misses one minor theme;
  3 = misses one major theme or several minor ones; 2 = covers only a few themes; 1 = misses most important themes.

CONSISTENCY — does the summary agree with the aggregated statistics?
  5 = overall tone, emphasis and any numbers match the statistics exactly; 4 = small emphasis mismatch;
  3 = noticeable mismatch (e.g. calls a mostly negative product "well liked"); 2 = several mismatches or wrong numbers;
  1 = contradicts the statistics.

HALLUCINATION — does it introduce unsupported claims? (higher = better, i.e. fewer hallucinations)
  5 = no unsupported claims; 4 = one minor unsupported detail; 3 = a few minor or one notable unsupported claim;
  2 = several unsupported claims; 1 = fabricated statistics, features or opinions.

List every unsupported claim verbatim in "unsupported_claims" (empty list if none).
Justify each score in 1-3 sentences referring to specific review ids or statistics."""

TEMPLATE = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a strict, impartial evaluator of review summaries. Judge ONLY against the source material "
            "provided; do not reward fluent writing that is not supported.\n\n" + RUBRIC +
            "\n\nRespond with ONE JSON object only.\n\nOUTPUT FORMAT:\n{format_instructions}",
        ),
        (
            "human",
            "SOURCE 1 — ORIGINAL REVIEWS (id: text):\n{reviews}\n\n"
            "SOURCE 2 — STRUCTURED EXTRACTIONS (id: overall | aspects | complaints):\n{extractions}\n\n"
            "SOURCE 3 — AGGREGATED STATISTICS:\n{statistics}\n\n"
            "SUMMARY TO EVALUATE:\n{summary}\n\nReturn the evaluation JSON.",
        ),
    ]
)


@lru_cache(maxsize=1)
def judge_format_instructions() -> str:
    return PydanticOutputParser(pydantic_object=JudgeVerdict).get_format_instructions()


# =========================================================================== v2
# v2 fixes the v1 structured-output failures (see schemas.evaluation_schema.JudgeOutputV2): a FLAT output object,
# a literal JSON skeleton instead of a generated JSON-schema dump, explicit rules for the claim list, and (for Ollama)
# constrained decoding with the same JSON schema. Scores and justifications are still entirely the judge's own.
JUDGE_PROMPT_VERSION_V2 = "v2"
CURRENT_JUDGE_VERSION = JUDGE_PROMPT_VERSION_V2

V2_SKELETON = """{
  "faithfulness_score": <integer 1-5>,
  "faithfulness_justification": "<1-3 sentences citing review ids or statistics>",
  "coverage_score": <integer 1-5>,
  "coverage_justification": "<1-3 sentences>",
  "consistency_score": <integer 1-5>,
  "consistency_justification": "<1-3 sentences>",
  "hallucination_score": <integer 1-5>,
  "hallucination_justification": "<1-3 sentences>",
  "unsupported_claims": [
    {"claim": "<statement copied from the summary>", "reason": "<what the sources say instead>", "evidence_review_ids": ["<review id>"]}
  ],
  "overall_comment": "<1-2 sentences>"
}"""

V2_RULES = """OUTPUT RULES:
- Return exactly ONE flat JSON object with exactly the keys shown in the skeleton. No markdown, no text before or after it.
- Every *_score is a whole number from 1 to 5. Every *_justification is a non-empty string.
- "unsupported_claims" appears ONCE, at the top level. Do NOT put it inside any other field.
- Check EVERY sentence of the summary against the sources. A feature, number, percentage or opinion that no review,
  extraction or statistic supports is an unsupported claim, even if it cites a review id.
- If you list at least one unsupported claim, hallucination_score must be 4 or lower.
- If there are no unsupported claims, return "unsupported_claims": [] and hallucination_score 5.
- "evidence_review_ids" lists the review ids you checked for that claim ([] if none)."""

TEMPLATE_V2 = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a strict, impartial evaluator of review summaries. Judge ONLY against the source material "
            "provided; do not reward fluent writing that is not supported.\n\n" + RUBRIC.replace(
                'List every unsupported claim verbatim in "unsupported_claims" (empty list if none).\n', ""
            ) + "\n\n" + V2_RULES + "\n\nJSON SKELETON (replace every <...>):\n" + V2_SKELETON.replace("{", "{{").replace("}", "}}"),
        ),
        (
            "human",
            "SOURCE 1 — ORIGINAL REVIEWS (id: text):\n{reviews}\n\n"
            "SOURCE 2 — STRUCTURED EXTRACTIONS (id: overall | aspects | complaints):\n{extractions}\n\n"
            "SOURCE 3 — AGGREGATED STATISTICS:\n{statistics}\n\n"
            "SUMMARY TO EVALUATE:\n{summary}\n\nReturn the evaluation as one flat JSON object following the skeleton.",
        ),
    ]
)

V2_FORMAT_INSTRUCTIONS = V2_RULES + "\n\nJSON SKELETON (replace every <...>):\n" + V2_SKELETON


@lru_cache(maxsize=1)
def judge_v2_json_schema() -> dict:
    """JSON schema passed to Ollama's constrained decoding (``format=``). Pydantic still validates the result."""
    from schemas.evaluation_schema import JudgeOutputV2

    return JudgeOutputV2.model_json_schema()
