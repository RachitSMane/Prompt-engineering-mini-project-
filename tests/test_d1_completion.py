"""D1 gap tests: incorrect types, invalid enums, failure diagnostics and JSON/CSV export consistency.
Offline: scripted / mock clients only, no network or model calls."""

from __future__ import annotations

import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import pandas as pd

from evaluation.prompt_comparison import run_experiment
from llm.mock import MockLLMClient
from pipeline.extraction import ReviewExtractor
from prompts.strategies import get_strategy
from schemas.review_schema import ExtractionFailure, ReviewInput, ReviewRecord
from tests.conftest import ScriptedLLM, valid_extraction
from utils.export import FAILURE_CSV_COLUMNS, failures_to_dataframe, failures_to_json, records_to_dataframe, records_to_json

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _extract(settings, review, responses):
    client = ScriptedLLM(responses)
    return ReviewExtractor(client, get_strategy("zero_shot"), settings).extract(review), client


# ------------------------------------------------------------------ malformed outputs
def test_markdown_fence_and_extra_text_are_stripped(settings, review):
    out, client = _extract(settings, review, ["Here is the JSON:\n```json\n" + json.dumps(valid_extraction()) + "\n```\nLet me know!"])
    assert isinstance(out, ReviewRecord) and len(client.calls) == 1
    assert {"removed markdown code fence"} <= set(out.extraction_metadata.attempts[0].local_fixes)


def test_aspects_wrong_type_then_repaired(settings, review):
    bad = valid_extraction(aspects="performance: positive")  # string instead of list
    out, client = _extract(settings, review, [json.dumps(bad), json.dumps(valid_extraction())])
    assert isinstance(out, ReviewRecord) and out.extraction_metadata.repair_attempts == 1
    assert any(e.startswith("aspects") for e in out.extraction_metadata.attempts[0].errors)
    assert client.calls[1]["task"] == "repair"


def test_invalid_complaint_enum_and_out_of_range_confidence_rejected(settings, review):
    bad = valid_extraction(complaint_categories=["too_expensive_lol"], confidence=1.7)
    out, _ = _extract(settings, review, [json.dumps(bad), json.dumps(valid_extraction())])
    errs = " ".join(out.extraction_metadata.attempts[0].errors)
    assert "complaint_categories.0" in errs and "confidence" in errs
    assert isinstance(out, ReviewRecord)


def test_invalid_aspect_category_is_never_coerced(settings, review):
    bad = valid_extraction()
    bad["aspects"][0]["aspect_category"] = "vibes"
    out, _ = _extract(settings, review, [json.dumps(bad)] * 3)
    assert isinstance(out, ExtractionFailure)  # an invalid category is not silently mapped to a valid one
    assert "aspects.0.aspect_category" in out.error


# ------------------------------------------------------------------ persistent failure diagnostics
def test_persistent_failure_keeps_every_attempt(settings, review):
    out, client = _extract(settings, review, ["not json", '{"overall_sentiment": "great"}', "{broken"])
    assert isinstance(out, ExtractionFailure) and len(client.calls) == 3
    attempts = out.extraction_metadata.attempts
    assert [a.kind for a in attempts] == ["initial", "llm_repair", "llm_repair"]
    assert [a.raw_output for a in attempts] == ["not json", '{"overall_sentiment": "great"}', "{broken"]
    assert all(a.errors and not a.success for a in attempts)
    assert out.extraction_metadata.validation_success is False


def test_failures_csv_and_json_carry_diagnostics(settings, review):
    out, _ = _extract(settings, review, ["not json", '{"overall_sentiment": "great"}', "{broken"])
    df = failures_to_dataframe([out])
    assert list(df.columns) == FAILURE_CSV_COLUMNS
    row = df.iloc[0]
    assert row["failure_type"] == "invalid_output" and row["attempts"] == 3
    assert row["attempt_kinds"] == "initial; llm_repair; llm_repair"
    assert "no JSON object" in row["first_attempt_errors"] and row["last_raw_output_excerpt"] == "{broken"
    full = json.loads(failures_to_json([out]))
    assert full[0]["extraction_metadata"]["attempts"][1]["raw_output"] == '{"overall_sentiment": "great"}'


# ------------------------------------------------------------------ export consistency
def test_records_json_and_csv_agree(settings, review):
    out, _ = _extract(settings, review, [json.dumps(valid_extraction())])
    data = json.loads(records_to_json([out]))
    df = records_to_dataframe([out])
    assert len(data) == len(df) == 1
    assert df.iloc[0]["overall_sentiment"] == data[0]["overall_sentiment"]
    assert json.loads(df.iloc[0]["aspects_json"]) == data[0]["aspects"]
    assert df.iloc[0]["complaint_categories"] == "; ".join(data[0]["complaint_categories"])


def test_verify_d1_exports_passes_on_fresh_export_and_detects_tampering(settings, tmp_path):
    reviews = [ReviewInput(review_id=f"E{i}", product_id="P", review_text=t) for i, t in
               enumerate(["Works great and very quiet.", "Broke after a week, total junk.", "Fits my fridge perfectly, good price.", "Loud and broke quickly."])]
    s = replace(settings, max_repair_attempts=0)
    run_dir = run_experiment(s, MockLLMClient(malformed_rate=50), ["zero_shot"], reviews, [], required_gold=4)
    d1_dir = tmp_path / "d1" / run_dir.name
    _load("export_d1").export_strategy(run_dir, "zero_shot", [r.review_id for r in reviews], d1_dir)
    verify = _load("verify_d1_exports")
    res = verify.verify_strategy(run_dir, d1_dir, "zero_shot")
    assert res["passed"], res
    assert res["records"] + res["failures"] == 4
    assert (d1_dir / "zero_shot" / "failures.json").exists()

    csv_path = d1_dir / "zero_shot" / "records.csv"
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    if len(df):
        df.loc[0, "overall_sentiment"] = "neutral" if df.loc[0, "overall_sentiment"] != "neutral" else "positive"
        df.to_csv(csv_path, index=False)
        assert verify.verify_strategy(run_dir, d1_dir, "zero_shot")["checks"]["csv_values_match_json"] is False
