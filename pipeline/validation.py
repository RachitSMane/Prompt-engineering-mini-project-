"""JSON extraction, deterministic clean-up and Pydantic validation.

Order of operations for every LLM response:
  1. ``extract_json_candidate`` – pull the JSON object out of surrounding prose / fences.
  2. ``apply_local_fixes``      – cheap deterministic syntax fixes (trailing commas, smart quotes...).
  3. ``json.loads``
  4. ``normalizer``             – conservative value normalisation (e.g. "Positive" -> "positive").
  5. Pydantic validation (with context for evidence grounding / allowed IDs).
Every fix is recorded so nothing is changed silently.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable

from pydantic import BaseModel, ValidationError

from schemas.taxonomy import AspectCategory, ComplaintCategory

_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")


def extract_json_candidate(raw: str) -> tuple[str | None, list[str]]:
    """Return the most likely JSON object substring and notes on what was stripped."""
    notes: list[str] = []
    if raw is None:
        return None, ["empty response"]
    text = raw.strip().lstrip("﻿")
    fence = _FENCE_RE.search(text)
    if fence:
        text = fence.group(1).strip()
        notes.append("removed markdown code fence")
    start = text.find("{")
    if start == -1:
        return None, notes + ["no JSON object found in response"]
    if start > 0:
        notes.append("removed text before JSON")
    depth, in_str, escape = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                if text[i + 1 :].strip():
                    notes.append("removed text after JSON")
                return text[start : i + 1], notes
    return text[start:], notes + ["JSON object appears truncated (unbalanced braces)"]


def apply_local_fixes(json_text: str) -> tuple[str, list[str]]:
    notes: list[str] = []
    fixed = json_text.replace("“", '"').replace("”", '"')
    if fixed != json_text:
        notes.append("replaced smart double quotes")
    new = _TRAILING_COMMA_RE.sub(r"\1", fixed)
    if new != fixed:
        notes.append("removed trailing commas")
    return new, notes


def parse_json(raw: str) -> tuple[Any | None, list[str], str | None]:
    """Return (parsed_object | None, local_fix_notes, error_message | None)."""
    candidate, notes = extract_json_candidate(raw)
    if candidate is None:
        return None, notes, notes[-1] if notes else "no JSON found"
    try:
        return json.loads(candidate), notes, None
    except json.JSONDecodeError:
        pass
    fixed, fix_notes = apply_local_fixes(candidate)
    try:
        return json.loads(fixed), notes + fix_notes, None
    except json.JSONDecodeError as exc:
        return None, notes + fix_notes, f"invalid JSON: {exc.msg} at line {exc.lineno} column {exc.colno}"


def format_validation_errors(exc: ValidationError) -> list[str]:
    errors = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"]) or "(root)"
        msg = err["msg"].removeprefix("Value error, ")
        inp = err.get("input")
        detail = f" (got {inp!r})" if err["type"] in {"enum", "literal_error", "float_parsing", "int_parsing", "list_type", "string_type"} and inp is not None else ""
        errors.append(f"{loc}: {msg}{detail}"[:400])
    return errors


def validate_payload(obj: Any, model_cls: type[BaseModel], context: dict | None = None) -> tuple[BaseModel | None, list[str]]:
    if not isinstance(obj, dict):
        return None, [f"(root): expected a JSON object, got {type(obj).__name__}"]
    try:
        return model_cls.model_validate(obj, context=context or {}), []
    except ValidationError as exc:
        return None, format_validation_errors(exc)


# --------------------------------------------------------------------------- normalisers
def _norm_label(value: Any) -> Any:
    if isinstance(value, str):
        return re.sub(r"[\s\-/]+", "_", value.strip().lower())
    return value


def unwrap_object(obj: Any, expected_keys: set[str]) -> tuple[Any, list[str]]:
    """Unwrap ``[{...}]`` or ``{"result": {...}}`` wrappers around the real object."""
    notes = []
    if isinstance(obj, list) and len(obj) == 1 and isinstance(obj[0], dict):
        obj = obj[0]
        notes.append("unwrapped single-element list")
    if isinstance(obj, dict) and len(obj) == 1:
        (inner,) = obj.values()
        if isinstance(inner, dict) and expected_keys & set(inner):
            obj = inner
            notes.append("unwrapped nested wrapper object")
    return obj, notes


def normalize_extraction(obj: Any) -> tuple[Any, list[str]]:
    """Conservative normalisation of extraction payloads. Only case/format changes —
    semantically invalid values (e.g. 'very good') are left for validation to reject."""
    obj, notes = unwrap_object(obj, {"overall_sentiment", "aspects"})
    if not isinstance(obj, dict):
        return obj, notes
    obj = dict(obj)
    if "overall_sentiment" in obj:
        new = _norm_label(obj["overall_sentiment"])
        if new != obj["overall_sentiment"]:
            notes.append("normalised overall_sentiment casing")
            obj["overall_sentiment"] = new
    cc = obj.get("complaint_categories")
    if isinstance(cc, list):
        new_cc = [_norm_label(c) for c in cc]
        if new_cc != cc:
            notes.append("normalised complaint category casing")
        obj["complaint_categories"] = new_cc
    elif cc is None and "complaint_categories" in obj:
        obj["complaint_categories"] = []
        notes.append("replaced null complaint_categories with []")
    if obj.get("aspects") is None and "aspects" in obj:
        obj["aspects"] = []
        notes.append("replaced null aspects with []")
    if isinstance(obj.get("aspects"), list):
        new_aspects = []
        for a in obj["aspects"]:
            if isinstance(a, dict):
                a = dict(a)
                for key in ("aspect_category", "sentiment"):
                    if key in a:
                        n = _norm_label(a[key])
                        if n != a[key]:
                            notes.append(f"normalised aspect {key} casing")
                            a[key] = n
            new_aspects.append(a)
        obj["aspects"] = new_aspects
    return obj, list(dict.fromkeys(notes))


def identity_normalizer(obj: Any) -> tuple[Any, list[str]]:
    return obj, []


Normalizer = Callable[[Any], tuple[Any, list[str]]]

VALID_ASPECT_CATEGORIES = {c.value for c in AspectCategory}
VALID_COMPLAINTS = {c.value for c in ComplaintCategory}


# --------------------------------------------------------------------------- judge (v2) normaliser
JUDGE_DIMENSIONS = ("faithfulness", "coverage", "consistency", "hallucination")
_V2_KEYS = {f"{d}_{k}" for d in JUDGE_DIMENSIONS for k in ("score", "justification")} | {"unsupported_claims", "overall_comment"}


def normalize_judge_v2(obj: Any) -> tuple[Any, list[str]]:
    """Purely structural, logged clean-up of judge output toward the flat ``JudgeOutputV2`` shape.

    It only MOVES or RENAMES values the judge itself produced; it never creates a score, a justification or a claim:
      * unwraps ``[{...}]`` / ``{"evaluation": {...}}`` wrappers;
      * flattens the v1 nested shape ``{"faithfulness": {"score": 4, "justification": "..."}}`` to
        ``faithfulness_score`` / ``faithfulness_justification``;
      * moves ``unsupported_claims`` lists that were put inside a dimension object to the single top-level list
        (de-duplicated) — the dominant v1 failure;
      * turns plain-string claims into ``{"claim": "..."}`` objects and ``null`` claim lists into ``[]``.
    Missing or invalid values (e.g. a missing justification, score 7, score "4/5") are left for validation to reject.
    """
    obj, notes = unwrap_object(obj, _V2_KEYS | set(JUDGE_DIMENSIONS))
    if not isinstance(obj, dict):
        return obj, notes
    obj = dict(obj)
    moved_claims: list[Any] = []
    for dim in JUDGE_DIMENSIONS:
        inner = obj.get(dim)
        if isinstance(inner, dict):
            inner = dict(inner)
            nested = inner.pop("unsupported_claims", None)
            if isinstance(nested, list):
                moved_claims += nested
                notes.append("moved nested unsupported_claims to the top level")
            for key in ("score", "justification"):
                if key in inner and f"{dim}_{key}" not in obj:
                    obj[f"{dim}_{key}"] = inner.pop(key)
            if not inner:
                del obj[dim]
                notes.append("flattened nested dimension objects")
            else:
                obj[dim] = inner  # leftover unknown keys: keep them so validation reports them
    if "unsupported_claims" in obj and obj["unsupported_claims"] is None:
        obj["unsupported_claims"] = []
        notes.append("replaced null unsupported_claims with []")
    if moved_claims:
        top = obj.get("unsupported_claims") if isinstance(obj.get("unsupported_claims"), list) else []
        obj["unsupported_claims"] = top + moved_claims
    claims = obj.get("unsupported_claims")
    if isinstance(claims, list):
        out, seen = [], set()
        for c in claims:
            if isinstance(c, str):
                c = {"claim": c}
                notes.append("converted plain-string claims to claim objects")
            key = json.dumps(c, sort_keys=True) if isinstance(c, dict) else repr(c)
            if key in seen:
                notes.append("removed duplicate unsupported claims")
                continue
            seen.add(key)
            out.append(c)
        obj["unsupported_claims"] = out
    return obj, list(dict.fromkeys(notes))
