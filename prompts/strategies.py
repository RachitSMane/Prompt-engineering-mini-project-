"""Common interface for prompting strategies.

The extraction pipeline only calls ``strategy.build_messages(review_text)``, so
strategies can be swapped without touching pipeline code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from langchain_core.prompts import ChatPromptTemplate

from llm.base import ChatMessage
from prompts import cot, few_shot, zero_shot
from prompts.common import PROMPT_VERSION, extraction_format_instructions, render, taxonomy


@dataclass(frozen=True)
class PromptStrategy:
    name: str
    label: str
    description: str
    template: ChatPromptTemplate
    extra_vars: Callable[[], dict] = lambda: {}
    version: str = PROMPT_VERSION

    def build_messages(self, review_text: str) -> list[ChatMessage]:
        return render(
            self.template,
            taxonomy=taxonomy(),
            format_instructions=extraction_format_instructions(),
            review_text=review_text,
            **self.extra_vars(),
        )


STRATEGIES: dict[str, PromptStrategy] = {
    "zero_shot": PromptStrategy(
        name="zero_shot",
        label="Zero-shot",
        description="Task instructions, label definitions and the review. No examples.",
        template=zero_shot.TEMPLATE,
    ),
    "few_shot": PromptStrategy(
        name="few_shot",
        label="Few-shot",
        description=f"Same instructions plus {len(few_shot.FEW_SHOT_EXAMPLES)} fixed synthetic labelled examples (not from the evaluation set).",
        template=few_shot.TEMPLATE,
        extra_vars=lambda: {"examples": few_shot.render_examples()},
    ),
    "cot": PromptStrategy(
        name="cot",
        label="Chain-of-Thought",
        description="Same instructions plus an explicit step-by-step analysis procedure performed silently; returns only JSON + concise rationale.",
        template=cot.TEMPLATE,
    ),
}


def get_strategy(name: str) -> PromptStrategy:
    try:
        return STRATEGIES[name]
    except KeyError as exc:
        raise ValueError(f"Unknown strategy '{name}'. Choose from {list(STRATEGIES)}") from exc
