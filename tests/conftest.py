"""Shared test fixtures. No test makes a network/LLM call.

``ScriptedLLM`` returns pre-written responses in order, letting us simulate
exactly how a real model might misbehave (invalid JSON, wrong enums, ...).
All review texts used in tests are synthetic.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import Paths, load_settings  # noqa: E402
from llm.base import LLMResponse  # noqa: E402
from schemas.review_schema import ReviewInput  # noqa: E402


class ScriptedLLM:
    provider = "scripted"
    model_name = "scripted-test-model"
    is_mock = True

    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def invoke(self, messages, *, task, payload=None):
        self.calls.append({"task": task, "messages": messages, "payload": payload})
        if not self.responses:
            raise AssertionError("ScriptedLLM ran out of responses")
        return LLMResponse(text=self.responses.pop(0), model_name=self.model_name, provider=self.provider, latency_s=0.01, input_tokens=100, output_tokens=50, is_mock=True)


REVIEW_TEXT = "The ice maker works great and installation was easy. But it is very loud at night."


def valid_extraction(**overrides) -> dict:
    data = {
        "overall_sentiment": "mixed",
        "aspects": [
            {"aspect_name": "ice maker", "aspect_category": "performance", "sentiment": "positive", "evidence": "The ice maker works great", "confidence": 0.9},
            {"aspect_name": "installation", "aspect_category": "installation_setup", "sentiment": "positive", "evidence": "installation was easy", "confidence": 0.9},
            {"aspect_name": "noise", "aspect_category": "noise", "sentiment": "negative", "evidence": "it is very loud at night", "confidence": 0.9},
        ],
        "complaint_categories": ["noise_issue"],
        "confidence": 0.88,
        "rationale": "Praise for performance and setup, complaint about noise.",
    }
    data.update(overrides)
    return data


@pytest.fixture
def review() -> ReviewInput:
    return ReviewInput(review_id="T1", product_id="P1", review_text=REVIEW_TEXT)


@pytest.fixture
def settings(tmp_path):
    paths = Paths(
        root=tmp_path,
        data_raw=tmp_path / "data" / "raw",
        data_processed=tmp_path / "data" / "processed",
        annotations=tmp_path / "data" / "annotations",
        outputs=tmp_path / "outputs",
        reports=tmp_path / "reports",
    )
    return replace(load_settings(), paths=paths, max_repair_attempts=2, strict_evidence_grounding=True, mock_malformed_rate=0)


@pytest.fixture
def valid_json() -> str:
    return json.dumps(valid_extraction())
