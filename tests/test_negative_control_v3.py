"""Negative control v3: generation, pre-registration, detection rules and metrics. No LLM."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.negative_control_v3 import (
    ERROR_TYPES, EXPECTED, STAT_TYPES, confusion, dataset_sha256, flagged, generate, to_claim, wilson,
)

ROOT = Path(__file__).resolve().parents[1]
SRC = "dataset__few_shot__qwen2.5-3b__v1"
NC = ROOT / "outputs" / "live" / "d3" / "negative_control_v3" / SRC


def test_twelve_error_types_with_expected_labels():
    assert len(ERROR_TYPES) == 12 and set(EXPECTED) == set(ERROR_TYPES)
    assert EXPECTED["misleading_paraphrase"] == "PARTIALLY_SUPPORTED"


def test_flagged_rules():
    assert flagged("UNSUPPORTED", "flipped_sentiment") and flagged("CONTRADICTED", "incorrect_percentage")
    assert not flagged("PARTIALLY_SUPPORTED", "flipped_sentiment")
    assert flagged("PARTIALLY_SUPPORTED", "misleading_paraphrase")      # its expected label is PARTIALLY_SUPPORTED
    assert flagged("UNVERIFIABLE", "flipped_sentiment", "lenient") and not flagged("UNVERIFIABLE", "flipped_sentiment", "strict")
    assert flagged(None, "flipped_sentiment") is None


def test_confusion_metrics_by_hand():
    rows = [{"error_type": "flipped_sentiment", "corrupted_s": "CONTRADICTED", "clean_s": "SUPPORTED"},
            {"error_type": "flipped_sentiment", "corrupted_s": "SUPPORTED", "clean_s": "SUPPORTED"},
            {"error_type": "fabricated_warranty", "corrupted_s": "UNSUPPORTED", "clean_s": "UNSUPPORTED"},
            {"error_type": "misleading_paraphrase", "corrupted_s": "PARTIALLY_SUPPORTED", "clean_s": "SUPPORTED"}]
    m = confusion(rows, "s")
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (3, 1, 1, 3)
    assert m["precision"] == 0.75 and m["recall"] == 0.75 and m["specificity"] == 0.75 and m["f1"] == 0.75
    assert m["false_positive_rate"] == 0.25


def test_wilson_interval():
    assert wilson(0, 0) is None
    lo, hi = wilson(5, 30)
    assert 0.07 <= lo <= 0.08 and 0.33 <= hi <= 0.35  # matches the v3 report's [0.07, 0.34]


def _live():
    from config import load_settings
    from evaluation.judge_v4 import load_review_index
    from scripts.run_judge import load_inputs
    from utils.storage import read_json

    s = load_settings()
    if not (s.paths.mode_dir(False) / "summaries" / "summaries.jsonl").exists():
        pytest.skip("live data not present")
    summ, recs = load_inputs(s, False, SRC)
    human = read_json(s.paths.annotations / "human_annotations.json", default={}).get("labels", {})
    return summ, recs, human, load_review_index(s.paths.reviews_csv)


def test_generation_is_reproducible_and_matches_frozen_dataset():
    summ, recs, human, reviews = _live()
    a = generate(summ, human, reviews, recs)
    b = generate(summ, human, reviews, recs)
    assert dataset_sha256(a) == dataset_sha256(b)
    if (NC / "manifest.json").exists():
        man = json.loads((NC / "manifest.json").read_text(encoding="utf-8"))
        assert dataset_sha256(a) == man["dataset_sha256"], "the generator no longer reproduces the frozen nc-v3 dataset"


def test_every_case_is_complete_and_types_covered():
    summ, recs, human, reviews = _live()
    cases = generate(summ, human, reviews, recs)
    assert {c["error_type"] for c in cases} == set(ERROR_TYPES)
    for c in cases:
        for k in ("original_content", "corrupted_content", "clean_content", "planted_change", "expected_detection", "source_evidence"):
            assert c[k]
        assert c["corrupted_claim"]["text"] != c["clean_claim"]["text"] or c["corrupted_claim"]["cited_review_ids"] != c["clean_claim"]["cited_review_ids"]
        assert c["corrupted_content"] != c["original_content"] and c["clean_content"] != c["original_content"]
        for which in ("corrupted_claim", "clean_claim"):
            for rid in c[which]["cited_review_ids"]:
                assert reviews[rid]["product_id"] == c["product_id"]


def test_statistic_types_are_caught_deterministically_and_clean_ones_pass():
    from evaluation.claim_checks import Stats
    from evaluation.judge_v4 import deterministic_stage

    summ, recs, human, reviews = _live()
    by = {x.product_id: x for x in summ}
    for c in generate(summ, human, reviews, recs):
        if c["error_type"] not in STAT_TYPES:
            continue
        st = Stats.from_aggregate(by[c["product_id"]].aggregate, recs)
        assert deterministic_stage(to_claim(c["corrupted_claim"]), st, reviews).status.value == "CONTRADICTED", c["case_id"]
        assert deterministic_stage(to_claim(c["clean_claim"]), st, reviews).status.value == "SUPPORTED", c["case_id"]


def test_previous_negative_control_untouched():
    p = ROOT / "outputs" / "live" / "d3" / "judge_negative_control_v2" / SRC / "pairs.jsonl"
    if not p.exists():
        pytest.skip("nc-v2 not present")
    rows = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
    assert len(rows) == 30 and {r["kind"] for r in rows} == {"invented_feature", "fabricated_statistic"}
