"""Generic "generate -> parse -> validate -> repair" loop for any Pydantic schema.

Used for review extraction, product summaries and judge verdicts alike.

Algorithm
---------
1. Call the LLM with the task prompt; keep the raw response.
2. Extract/parse JSON (with logged deterministic fixes).
3. Validate against the Pydantic model (with context, e.g. review text for grounding).
4. On failure, collect readable errors and send a *repair prompt* containing the
   original output + errors; repeat up to ``max_repairs`` times.
5. Every attempt is logged (raw output, errors, tokens, latency).
6. If all attempts fail, return a controlled failure — invalid data is never accepted.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from pydantic import BaseModel

from llm.base import ChatMessage, LLMClient, LLMError
from pipeline.validation import Normalizer, identity_normalizer, parse_json, validate_payload
from prompts.common import render
from prompts.repair import REPAIR_TEMPLATE
from schemas.review_schema import AttemptLog

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


@dataclass
class StructuredResult(Generic[T]):
    value: T | None
    attempts: list[AttemptLog] = field(default_factory=list)
    error: str | None = None

    @property
    def success(self) -> bool:
        return self.value is not None

    @property
    def repair_attempts(self) -> int:
        return sum(a.kind == "llm_repair" for a in self.attempts)

    @property
    def local_repairs(self) -> int:
        return sum(len(a.local_fixes) for a in self.attempts)

    @property
    def total_latency_s(self) -> float:
        return round(sum(a.latency_s for a in self.attempts), 3)

    def _sum(self, attr: str) -> int | None:
        vals = [getattr(a, attr) for a in self.attempts if getattr(a, attr) is not None]
        return sum(vals) if vals else None

    @property
    def input_tokens(self) -> int | None:
        return self._sum("input_tokens")

    @property
    def output_tokens(self) -> int | None:
        return self._sum("output_tokens")


def _process(raw: str, model_cls: type[T], context: dict | None, normalizer: Normalizer) -> tuple[T | None, list[str], list[str]]:
    obj, fixes, parse_error = parse_json(raw)
    if parse_error:
        return None, [parse_error], fixes
    obj, norm_notes = normalizer(obj)
    value, errors = validate_payload(obj, model_cls, context)
    return value, errors, fixes + norm_notes  # type: ignore[return-value]


def generate_structured(
    client: LLMClient,
    messages: list[ChatMessage],
    model_cls: type[T],
    *,
    task: str,
    task_description: str,
    format_instructions: str,
    payload: dict[str, Any] | None = None,
    context: dict | None = None,
    source_text: str | None = None,
    max_repairs: int = 2,
    normalizer: Normalizer = identity_normalizer,
) -> StructuredResult[T]:
    result: StructuredResult[T] = StructuredResult(value=None)
    current_messages, current_task = messages, task
    current_payload: dict[str, Any] = payload or {}

    for attempt in range(max_repairs + 1):
        kind = "initial" if attempt == 0 else "llm_repair"
        try:
            response = client.invoke(current_messages, task=current_task, payload=current_payload)
        except LLMError as exc:
            logger.warning("LLM call failed on attempt %d (%s): %s", attempt, task, exc)
            result.attempts.append(AttemptLog(attempt=attempt, kind="provider_error", raw_output="", success=False, errors=[str(exc)]))
            result.error = f"LLM provider error: {exc}"
            return result

        value, errors, fixes = _process(response.text, model_cls, context, normalizer)
        result.attempts.append(
            AttemptLog(
                attempt=attempt,
                kind=kind,
                raw_output=response.text,
                success=value is not None,
                errors=errors,
                local_fixes=fixes,
                latency_s=response.latency_s,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
            )
        )
        if value is not None:
            if attempt or fixes:
                logger.info("%s output valid after attempt %d (local fixes: %s)", task, attempt, fixes or "none")
            result.value = value
            return result

        logger.info("%s attempt %d invalid: %s", task, attempt, "; ".join(errors)[:300])
        if attempt == max_repairs:
            break
        source_block = f"SOURCE MATERIAL (the only allowed facts):\n<<<\n{source_text}\n>>>\n\n" if source_text else ""
        current_messages = render(
            REPAIR_TEMPLATE,
            task_description=task_description,
            format_instructions=format_instructions,
            source_block=source_block,
            original_output=response.text[:6000],
            errors="\n".join(f"- {e}" for e in errors),
        )
        current_task = "repair"
        current_payload = {"target_task": task, "original_payload": payload or {}, "errors": errors}

    last_errors = result.attempts[-1].errors if result.attempts else []
    result.error = f"Output still invalid after {max_repairs} repair attempt(s): " + "; ".join(last_errors)[:500]
    logger.warning("%s failed: %s", task, result.error)
    return result
