"""OFFLINE MOCK LLM — development and testing only.

This is NOT a language model. It is a deterministic, keyword-lexicon fixture
generator that returns JSON in the same shape a real LLM must return, so the
whole application (parsing, repair, aggregation, summaries, judge, UI) can be
developed and demonstrated without an API key.

* Everything it produces is tagged ``is_mock=True`` and written under
  ``outputs/mock/`` — never mixed with real results.
* It returns the SAME prediction for every prompting strategy; mock "metrics"
  therefore say nothing about zero-shot vs few-shot vs CoT.
* A configurable percentage of extraction responses is deliberately corrupted
  (prose around JSON, trailing commas, bad enum values, missing fields...) so
  that the malformed-output repair pipeline is exercised offline.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from config import MOCK_MODEL_NAME
from llm.base import ChatMessage, LLMResponse

MOCK_NOTICE = "MOCK OUTPUT (offline deterministic fixture, not an LLM result)"

ASPECT_LEXICON: dict[str, list[str]] = {
    "noise": ["noise", "noisy", "loud", "quiet", "silent", "rattle", "rattling", "hum", "humming", "buzz", "squeak"],
    "installation_setup": ["install", "installed", "installation", "installing", "setup", "set up", "assemble", "assembly", "mount", "instructions", "swap", "replace it", "replaced"],
    "ease_of_use": ["easy to use", "easy to clean", "simple to use", "convenient", "user friendly", "complicated", "confusing", "hard to use", "easy to fill", "messy"],
    "quality_durability": ["broke", "broken", "stopped working", "quit working", "durable", "sturdy", "flimsy", "lasted", "last long", "died", "cheaply made", "well made", "quality", "cracked"],
    "performance": ["works", "worked", "working", "cools", "heats", "heating", "cleans", "ice", "temperature", "powerful", "performance", "fast", "slow", "dries", "drying", "taste", "brew", "filter"],
    "design_build": ["design", "looks", "color", "finish", "plastic", "metal", "chrome", "build", "handle", "lid"],
    "size_capacity": ["size", "small", "large", "capacity", "compact", "big", "tiny", "holds"],
    "compatibility": ["fit", "fits", "fitted", "compatible", "oem", "genuine", "exact replacement", "direct replacement", "perfect match", "model"],
    "price_value": ["price", "value", "worth", "expensive", "cheap", "money", "cost", "deal", "bargain", "affordable"],
    "customer_service": ["customer service", "support", "warranty", "refund", "return", "returned", "seller", "replacement sent"],
    "shipping_packaging": ["shipping", "shipped", "arrived", "package", "packaging", "packaged", "box", "delivery", "delivered", "came quickly"],
    "safety": ["smoke", "burning", "burnt", "fire", "sparks", "overheat", "shock", "hazard", "melted"],
}

POSITIVE_WORDS = {
    "great", "good", "excellent", "perfect", "perfectly", "love", "loved", "works", "worked", "easy", "fantastic", "awesome",
    "happy", "recommend", "nice", "best", "quick", "quickly", "well", "sturdy", "quiet", "fine", "exactly", "amazing",
    "satisfied", "pleased", "solid", "affordable", "bargain", "durable", "fast", "helpful", "beautiful", "glad", "fits",
}
NEGATIVE_WORDS = {
    "bad", "poor", "terrible", "awful", "broke", "broken", "cheap", "flimsy", "loud", "noisy", "leak", "leaks", "leaking",
    "disappointed", "disappointing", "waste", "useless", "junk", "worst", "hate", "return", "returned", "defective", "died",
    "stopped", "failed", "fail", "wrong", "difficult", "hard", "smell", "melted", "cracked", "expensive", "slow", "mess",
    "messy", "damaged", "missing", "rattle", "horrible", "garbage", "problem", "issue", "doesn't", "didn't", "wouldn't",
}
NEGATIONS = {"not", "no", "never", "don't", "dont", "isn't", "wasn't", "didn't", "doesn't", "won't", "hardly"}

NEG_COMPLAINT_MAP = {
    "noise": "noise_issue",
    "installation_setup": "installation_difficulty",
    "ease_of_use": "usability_issue",
    "quality_durability": "poor_durability",
    "performance": "performance_issue",
    "compatibility": "compatibility_fit_issue",
    "price_value": "poor_value",
    "customer_service": "customer_service_issue",
    "shipping_packaging": "shipping_damage",
    "safety": "safety_concern",
    "design_build": "product_defect",
    "size_capacity": "compatibility_fit_issue",
}
KEYWORD_COMPLAINTS = {
    "missing": "missing_parts",
    "not as described": "not_as_described",
    "counterfeit": "not_as_described",
    "fake": "not_as_described",
    "dead on arrival": "product_defect",
    "defective": "product_defect",
}

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_WORD = re.compile(r"[a-z']+")


def _stable_int(*parts: str) -> int:
    return int(hashlib.sha256("|".join(parts).encode()).hexdigest()[:8], 16)


def _sentence_score(sentence: str) -> int:
    words = _WORD.findall(sentence.lower())
    score = 0
    for i, w in enumerate(words):
        polarity = 1 if w in POSITIVE_WORDS else -1 if w in NEGATIVE_WORDS else 0
        if polarity and any(prev in NEGATIONS for prev in words[max(0, i - 3) : i]):
            polarity = -polarity
        score += polarity
    return score


def _find_category(sentence_lower: str) -> list[tuple[str, str]]:
    found = []
    for category, keywords in ASPECT_LEXICON.items():
        for kw in keywords:
            if re.search(rf"\b{re.escape(kw)}\b", sentence_lower):
                found.append((category, kw))
                break
    return found


def lexicon_extract(review_text: str) -> dict[str, Any]:
    """Deterministic rule-based 'extraction' used as the mock fixture."""
    sentences = [s.strip() for s in _SENT_SPLIT.split(review_text) if s.strip()]
    aspects: list[dict[str, Any]] = []
    seen: set[str] = set()
    total_score = 0
    for sentence in sentences:
        score = _sentence_score(sentence)
        total_score += score
        for category, keyword in _find_category(sentence.lower()):
            if category in seen:
                continue
            seen.add(category)
            sentiment = "positive" if score > 0 else "negative" if score < 0 else "neutral"
            aspects.append(
                {
                    "aspect_name": keyword,
                    "aspect_category": category,
                    "sentiment": sentiment,
                    "evidence": sentence[:280],
                    "confidence": 0.6 if sentiment != "neutral" else 0.4,
                }
            )
    sentiments = {a["sentiment"] for a in aspects}
    if {"positive", "negative"} <= sentiments:
        overall = "mixed"
    elif total_score > 0:
        overall = "positive"
    elif total_score < 0:
        overall = "negative"
    else:
        overall = "neutral"
    complaints = [NEG_COMPLAINT_MAP[a["aspect_category"]] for a in aspects if a["sentiment"] == "negative" and a["aspect_category"] in NEG_COMPLAINT_MAP]
    lower = review_text.lower()
    complaints += [c for kw, c in KEYWORD_COMPLAINTS.items() if kw in lower]
    return {
        "overall_sentiment": overall,
        "aspects": aspects,
        "complaint_categories": list(dict.fromkeys(complaints)),
        "confidence": 0.5,
        "rationale": f"{MOCK_NOTICE}: keyword lexicon match.",
    }


CORRUPTIONS = ["prose_wrapper", "markdown_fence", "trailing_comma", "bad_enum", "missing_field", "wrong_type", "truncated"]


def corrupt(payload: dict[str, Any], kind: str) -> str:
    """Produce a realistic malformed LLM response of the given kind."""
    text = json.dumps(payload, indent=2)
    if kind == "prose_wrapper":
        return f"Sure! Here is the analysis of the review:\n{text}\nLet me know if you need anything else."
    if kind == "markdown_fence":
        return f"```json\n{text}\n```"
    if kind == "trailing_comma":
        return text[:-1].rstrip() + ",\n}"
    if kind == "bad_enum":
        bad = dict(payload, overall_sentiment="very good" if payload["overall_sentiment"] == "positive" else "somewhat bad")
        return json.dumps(bad)
    if kind == "missing_field":
        bad = {k: v for k, v in payload.items() if k != "confidence"}
        return json.dumps(bad)
    if kind == "wrong_type":
        bad = dict(payload, confidence="high", complaint_categories=", ".join(payload["complaint_categories"]) or "none")
        return json.dumps(bad)
    if kind == "truncated":
        return text[: max(10, len(text) // 2)]
    return text


class MockLLMClient:
    provider = "mock"
    model_name = MOCK_MODEL_NAME
    is_mock = True

    def __init__(self, malformed_rate: int = 15):
        self.malformed_rate = max(0, min(100, malformed_rate))
        self.calls: list[dict[str, Any]] = []

    def invoke(self, messages: list[ChatMessage], *, task: str, payload: dict[str, Any] | None = None) -> LLMResponse:
        payload = payload or {}
        self.calls.append({"task": task, "payload_keys": sorted(payload)})
        if task == "extraction":
            text = self._extraction(payload)
        elif task == "repair":
            text = self._repair(payload)
        elif task == "summary":
            text = json.dumps(mock_summary(payload["aggregate"]))
        elif task == "judge":
            text = json.dumps(mock_judge(payload))
        else:
            text = json.dumps({"error": f"mock has no fixture for task '{task}'"})
        prompt_chars = sum(len(m.content) for m in messages)
        return LLMResponse(
            text=text,
            model_name=self.model_name,
            provider=self.provider,
            latency_s=0.0,
            input_tokens=prompt_chars // 4,  # rough character-based estimate, labelled mock
            output_tokens=len(text) // 4,
            is_mock=True,
        )

    def _extraction(self, payload: dict[str, Any]) -> str:
        result = lexicon_extract(payload["review_text"])
        key = payload.get("review_id", payload["review_text"])
        if _stable_int(key, "corrupt") % 100 < self.malformed_rate:
            kind = CORRUPTIONS[_stable_int(key, "kind") % len(CORRUPTIONS)]
            return corrupt(result, kind)
        return json.dumps(result)

    def _repair(self, payload: dict[str, Any]) -> str:
        target = payload.get("target_task", "extraction")
        original = payload.get("original_payload", {})
        if target == "extraction":
            return json.dumps(lexicon_extract(original["review_text"]))
        if target == "summary":
            return json.dumps(mock_summary(original["aggregate"]))
        if target == "judge":
            return json.dumps(mock_judge(original))
        return "{}"


# --------------------------------------------------------------------------- summary / judge fixtures
def mock_summary(aggregate: dict[str, Any]) -> dict[str, Any]:
    pct = aggregate.get("sentiment_percentages", {})
    stats = aggregate.get("aspect_stats", [])
    pos_ev = aggregate.get("positive_evidence", [])
    neg_ev = aggregate.get("negative_evidence", [])

    def points(evidence: list[dict], sentiment: str) -> list[dict]:
        by_cat: dict[str, list[str]] = {}
        for e in evidence:
            by_cat.setdefault(e["aspect_category"], []).append(e["review_id"])
        return [
            {"point": f"[MOCK] Reviewers were {sentiment} about {cat.replace('_', ' ')}.", "supporting_review_ids": ids[:3]}
            for cat, ids in list(by_cat.items())[:4]
        ]

    top = [s["aspect_category"].replace("_", " ") for s in stats[:5]]
    weak = [s["aspect_category"].replace("_", " ") for s in stats if s["net_sentiment"] < 0][:3]
    return {
        "overall_opinion": (
            f"[MOCK TEMPLATE] {pct.get('positive', 0)}% positive, {pct.get('negative', 0)}% negative, "
            f"{pct.get('mixed', 0)}% mixed and {pct.get('neutral', 0)}% neutral reviews (from computed statistics)."
        ),
        "appreciated": points(pos_ev, "positive"),
        "complaints": points(neg_ev, "negative"),
        "frequent_aspects": top,
        "improvement_areas": [f"[MOCK] Improve {w}." for w in weak],
        "recommendations": [f"[MOCK] Investigate negative feedback on {w}." for w in weak] or ["[MOCK] No major issue detected in the computed statistics."],
    }


def mock_judge(payload: dict[str, Any]) -> dict[str, Any]:
    summary = payload.get("summary", {})
    allowed = set(payload.get("allowed_review_ids", []))
    cited = [rid for key in ("appreciated", "complaints") for p in summary.get(key, []) for rid in p.get("supporting_review_ids", [])]
    valid_ratio = (sum(rid in allowed for rid in cited) / len(cited)) if cited else 0.0
    score = 1 + round(4 * valid_ratio)
    note = f"{MOCK_NOTICE}: score derived only from the share of cited review IDs that exist ({valid_ratio:.0%})."
    dim = {"score": score, "justification": note}
    return {
        "faithfulness": dim,
        "coverage": {"score": 3, "justification": f"{MOCK_NOTICE}: fixed placeholder value."},
        "consistency": {"score": 3, "justification": f"{MOCK_NOTICE}: fixed placeholder value."},
        "hallucination": dim,
        "unsupported_claims": [],
        "overall_comment": note,
    }
