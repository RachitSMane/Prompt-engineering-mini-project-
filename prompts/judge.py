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
