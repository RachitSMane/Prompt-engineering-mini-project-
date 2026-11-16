"""Approval of the existing reference annotations: originals untouched, provenance kept, idempotent, flagged never
bulk-approved, stale after a source change, read-only safe, evaluation on identical ids with a stdlib cross-check."""

from __future__ import annotations

import hashlib
import json

import pandas as pd
import pytest

from annotation.reference_adjudication import AdjudicationError, ReferenceAdjudicationStore, load_reference
from evaluation.reference_evaluation import evaluate_reference
from utils.readonly import ReadOnlyModeError


def _ann(rid, pid="P1", sentiment="positive", status="complete", note="AI-assisted label (test)"):
    return {"review_id": rid, "product_id": pid, "overall_sentiment": sentiment,
            "aspects": [{"aspect_category": "performance", "sentiment": "positive", "aspect_name": None, "evidence": None}],
            "complaint_categories": [], "annotator": "x", "notes": note, "status": status, "annotated_at": "2026-11-10T12:34:06+00:00"}


@pytest.fixture
def files(tmp_path):
    recs = {"R0": _ann("R0", note=""), "R1": _ann("R1", sentiment="negative"), "R2": _ann("R2", sentiment="mixed"),
            "R3": _ann("R3", sentiment="neutral", status="flagged")}
    ann = tmp_path / "annotations.json"
    ann.write_text(json.dumps({"meta": {"annotation_method": {"human_labels": ["R0"], "approved_by": "mentor"}}, "annotations": recs}), encoding="utf-8")
    sample = tmp_path / "sample.csv"
    pd.DataFrame({"review_id": list(recs), "product_id": ["P1"] * 4}).to_csv(sample, index=False)
    return ann, sample


def _store(tmp_path, files, required=4):
    ann, sample = files
    return ReferenceAdjudicationStore(tmp_path / "reference_adjudication.json", ann, sample, required=required)


def test_validation_counts_ids_fields_and_provenance(files):
    ref = load_reference(*files)
    assert ref["valid"] and len(ref["records"]) == 4
    assert ref["provenance"] == {"R0": "human_created", "R1": "ai_assisted", "R2": "ai_assisted", "R3": "ai_assisted"}


def test_validation_detects_missing_wrong_product_and_invalid_fields(tmp_path, files):
    ann, sample = files
    d = json.loads(ann.read_text())
    d["annotations"]["R1"]["product_id"] = "P9"
    d["annotations"]["R2"]["overall_sentiment"] = "ecstatic"
    del d["annotations"]["R3"]
    ann.write_text(json.dumps(d))
    ref = load_reference(ann, sample)
    assert not ref["valid"] and set(ref["invalid"]) == {"R1", "R2"} and ref["missing"] == ["R3"]


def test_bulk_approval_is_idempotent_keeps_originals_and_skips_flagged(tmp_path, files):
    store = _store(tmp_path, files)
    before = hashlib.sha256(files[0].read_bytes()).hexdigest()
    s1 = store.bulk_approve("Mentor A", confirmed=True)
    assert s1 == {"approved_now": 3, "already_decided": 0, "skipped_flagged": 1}
    s2 = store.bulk_approve("Mentor A", confirmed=True)
    assert s2 == {"approved_now": 0, "already_decided": 3, "skipped_flagged": 1}
    assert hashlib.sha256(files[0].read_bytes()).hexdigest() == before  # annotations.json untouched
    dec = store.decisions()
    assert dec["R1"]["original_provenance"] == "ai_assisted" and dec["R0"]["original_provenance"] == "human_created"
    assert dec["R1"]["reviewer"] == "Mentor A" and dec["R1"]["method"] == "bulk" and len(dec["R1"]["history"]) == 1
    p = store.progress()
    assert p["accepted"] == 3 and p["awaiting_review"] == 1 and not p["ready"]
    assert p["accepted_by_original_provenance"] == {"human_created": 1, "ai_assisted": 2}


def test_bulk_approval_needs_reviewer_and_confirmation(tmp_path, files):
    store = _store(tmp_path, files)
    with pytest.raises(AdjudicationError):
        store.bulk_approve("", confirmed=True)
    with pytest.raises(AdjudicationError):
        store.bulk_approve("Mentor A", confirmed=False)
    assert not store.path.exists()


def test_individual_decisions_modify_flag_and_complete_the_set(tmp_path, files):
    store = _store(tmp_path, files)
    store.bulk_approve("Mentor A", confirmed=True)
    with pytest.raises(AdjudicationError):
        store.decide("R3", "excluded", "Mentor A")  # reason required
    store.decide("R3", "approved", "Mentor A")
    assert store.progress()["ready"]
    store.decide("R2", "modified", "Mentor B", correction={"overall_sentiment": "negative"}, reason="clearly negative")
    gold = {g.review_id: g.overall_sentiment.value for g in store.accepted_gold()}
    assert gold["R2"] == "negative" and json.loads(files[0].read_text())["annotations"]["R2"]["overall_sentiment"] == "mixed"
    hist = store.decisions()["R2"]["history"]
    assert [h["action"] for h in hist] == ["bulk_approved", "modified"] and hist[-1]["previous"] == "approved"
    store.decide("R1", "excluded", "Mentor B", reason="duplicate text")
    p = store.progress()
    assert p["excluded"] == 1 and p["accepted"] == 3 and not p["ready"]  # 3 accepted < required 4


def test_change_in_source_label_makes_decision_stale(tmp_path, files):
    store = _store(tmp_path, files)
    store.bulk_approve("Mentor A", confirmed=True)
    d = json.loads(files[0].read_text())
    d["annotations"]["R1"]["overall_sentiment"] = "positive"
    files[0].write_text(json.dumps(d))
    p = store.progress()
    assert p["stale_decisions"] == ["R1"] and p["accepted"] == 2
    assert store.bulk_approve("Mentor A", confirmed=True)["approved_now"] == 1  # re-approval of the changed record only


def test_readonly_mode_blocks_every_write(tmp_path, files, monkeypatch):
    monkeypatch.setenv("REVIEWIQ_DEMO_READONLY", "true")
    store = _store(tmp_path, files)
    with pytest.raises(ReadOnlyModeError):
        store.bulk_approve("Mentor A", confirmed=True)
    assert not store.path.exists()


def test_reference_evaluation_same_ids_crosscheck_and_separate_folder(tmp_path, settings, files):
    from llm.mock import MockLLMClient
    from evaluation.prompt_comparison import run_experiment
    from schemas.review_schema import ReviewInput

    reviews = [ReviewInput(review_id=r, product_id="P1", review_text=t) for r, t in
               [("R0", "Great, works perfectly."), ("R1", "Broke after a week, terrible."), ("R2", "Good price but loud."), ("R3", "ok")]]
    store = _store(tmp_path, files)
    gold_all = [g for g in store.accepted_gold()]
    run_dir = run_experiment(settings, MockLLMClient(malformed_rate=0), ["zero_shot", "few_shot"], reviews[:3], gold_all, required_gold=3)
    original_metrics = (run_dir / "metrics.json").read_bytes()
    store.bulk_approve("Mentor A", confirmed=True)
    prog = store.progress()
    res = evaluate_reference(run_dir, store.accepted_gold(), progress=prog, provenance=prog["accepted_by_original_provenance"])
    assert res["status"] == "incomplete_reference"  # R3 still awaiting review
    store.decide("R3", "approved", "Mentor A")
    prog = store.progress()
    res = evaluate_reference(run_dir, store.accepted_gold(), progress=prog, provenance=prog["accepted_by_original_provenance"])
    assert res["status"] == "needs_inference" and res["needs_inference"]["zero_shot"] == ["R3"]
    assert res["n_evaluated_common"] == 3 and res["independent_check_passed"]
    assert (run_dir / "approved_reference" / "metrics.json").exists() and (run_dir / "metrics.json").read_bytes() == original_metrics
    assert "NOT independent human ground truth" in res["reference"]


def test_export_contains_only_accepted_rows_with_provenance_and_reviewer(tmp_path, files):
    from annotation.reference_adjudication import export_reference

    store = _store(tmp_path, files)
    store.bulk_approve("Mentor A", confirmed=True)
    store.decide("R1", "modified", "Mentor B", correction={"overall_sentiment": "mixed"}, reason="mixed tone")
    res = export_reference(store, tmp_path / "out")
    df = pd.read_csv(res["csv"])
    assert res["rows"] == 3 and sorted(df["review_id"]) == ["R0", "R1", "R2"] and df["review_id"].is_unique
    row = df.set_index("review_id")
    assert row.loc["R1", "overall_sentiment"] == "mixed" and row.loc["R1", "decision"] == "modified"
    assert row.loc["R0", "original_provenance"] == "human_created" and row.loc["R2", "original_provenance"] == "ai_assisted"
    assert set(df["reviewer"]) == {"Mentor A", "Mentor B"}
