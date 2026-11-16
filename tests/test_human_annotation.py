"""D2 human annotation workflow: provenance, blind labelling, verification, save/resume, export, evaluation eligibility.
Offline: no LLM calls. All review texts are synthetic."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from annotation.annotation_manager import AI_ASSISTED_MARKER, AnnotationStore
from annotation.human_annotation import REQUIRED_HUMAN_LABELS, HumanAnnotationStore, HumanLabel, migrate_legacy_human_labels
from evaluation.human_evaluation import evaluate_against_human, independent_sentiment_metrics
from evaluation.classification_metrics import single_label_metrics
from evaluation.prompt_comparison import run_experiment
from llm.mock import MockLLMClient
from schemas.evaluation_schema import AnnotatedAspect, Annotation
from schemas.review_schema import ReviewInput
from utils.readonly import ReadOnlyModeError

QUEUE = [f"R{i}" for i in range(6)]


@pytest.fixture
def store(tmp_path) -> HumanAnnotationStore:
    return HumanAnnotationStore(tmp_path / "human_annotations.json")


def _label(store, rid="R0", who="alice", sentiment="positive", **kw):
    return store.save_blind_label(rid, "P1", annotator=who, overall_sentiment=sentiment, **kw)


# ------------------------------------------------------------------ provenance rules
def test_blind_label_is_human_created_completed(store):
    lab = _label(store, aspects=[AnnotatedAspect(aspect_category="noise", sentiment="negative")], complaint_categories=["noise_issue"])
    assert (lab.provenance, lab.status, lab.blind) == ("human_created", "completed", True)
    assert lab.history[-1]["action"] == "labelled" and lab.history[-1]["by"] == "alice"
    assert not lab.eligible  # not verified yet


def test_ai_assisted_labels_cannot_enter_the_human_store():
    with pytest.raises(ValidationError, match="AI-assisted labels are not stored"):
        HumanLabel(review_id="R", product_id="P", status="completed", provenance="ai_assisted", overall_sentiment="positive", annotator="x", blind=False)
    with pytest.raises(ValidationError, match="AI-assisted marker"):
        HumanLabel(review_id="R", product_id="P", status="completed", provenance="human_created", overall_sentiment="positive",
                   annotator="x", blind=True, notes=f"{AI_ASSISTED_MARKER} (Claude)")


def test_human_created_must_be_blind_and_human_verified_must_not():
    with pytest.raises(ValidationError, match="must be blind"):
        HumanLabel(review_id="R", product_id="P", status="completed", provenance="human_created", overall_sentiment="positive", annotator="x", blind=False)
    with pytest.raises(ValidationError, match="blind must be False"):
        HumanLabel(review_id="R", product_id="P", status="completed", provenance="human_verified", overall_sentiment="positive", annotator="x", blind=True)


def test_checked_ai_label_is_human_verified_and_never_replaces_blind_label(store):
    lab = store.save_checked_ai_label("R1", "P1", annotator="bob", overall_sentiment="negative", ai_label={"overall_sentiment": "mixed"})
    assert (lab.provenance, lab.blind, lab.status) == ("human_verified", False, "completed")
    assert lab.history[-1]["ai_label"] == {"overall_sentiment": "mixed"}  # what was shown is recorded
    _label(store, "R0")
    with pytest.raises(ValueError, match="already has a blind human label"):
        store.save_checked_ai_label("R0", "P1", annotator="bob", overall_sentiment="negative")


def test_legacy_store_refuses_to_turn_ai_label_into_human_label(tmp_path):
    """The bug found in the audit: re-saving a pre-filled AI-assisted label without its marker made it 'human'."""
    legacy = AnnotationStore(tmp_path / "annotations.json")
    legacy.save(Annotation(review_id="R0", product_id="P", overall_sentiment="positive", annotator="ai", notes=f"{AI_ASSISTED_MARKER} (x)"))
    with pytest.raises(ValueError, match="AI-assisted label"):
        legacy.save(Annotation(review_id="R0", product_id="P", overall_sentiment="positive", annotator="someone", notes=""))
    res = legacy.import_csv(pd.DataFrame([{"review_id": "R0", "product_id": "P", "overall_sentiment": "negative", "aspects": "",
                                           "complaint_categories": "", "annotator": "someone", "notes": "", "status": ""}]))
    assert res["imported"] == 0 and res["errors"]
    assert AnnotationStore(tmp_path / "annotations.json").get("R0").notes.startswith(AI_ASSISTED_MARKER)


# ------------------------------------------------------------------ verification
def test_verification_requires_a_different_person(store):
    _label(store, "R0", who="alice")
    with pytest.raises(ValueError, match="cannot verify your own"):
        store.verify("R0", reviewer="Alice ")
    lab = store.verify("R0", reviewer="bob")
    assert lab.status == "verified" and lab.verified_by == "bob" and lab.eligible and lab.independent_verification
    assert lab.history[-1]["action"] == "verified"


def test_verification_with_corrections_is_logged(store):
    _label(store, "R0", who="alice", sentiment="positive")
    lab = store.verify("R0", reviewer="bob", corrections={"overall_sentiment": "mixed"})
    assert lab.overall_sentiment.value == "mixed"
    assert lab.history[-1]["action"] == "verified_with_corrections" and lab.history[-1]["previous"]["overall_sentiment"] == "positive"


def test_only_completed_labels_can_be_verified(store):
    with pytest.raises(ValueError, match="only completed"):
        store.verify("R5", reviewer="bob")
    store.flag("R1", "P1", annotator="alice", reason="not about the product")
    with pytest.raises(ValueError, match="only completed"):
        store.verify("R1", reviewer="bob")


def test_relabelling_a_verified_label_requires_reverification(store):
    _label(store, "R0", who="alice")
    store.verify("R0", reviewer="bob")
    lab = _label(store, "R0", who="alice", sentiment="negative")
    assert lab.status == "completed" and lab.verified_by is None and not lab.eligible
    assert [h["action"] for h in lab.history] == ["labelled", "verified", "relabelled"]


# ------------------------------------------------------------------ statuses, save & resume
def test_status_tracking_and_progress(store):
    _label(store, "R0", who="alice")
    _label(store, "R1", who="alice")
    store.verify("R1", reviewer="bob")
    store.flag("R2", "P1", annotator="alice", reason="spam")
    assert [store.status_of(r) for r in QUEUE[:4]] == ["completed", "verified", "flagged", "pending"]
    p = store.progress(QUEUE)
    assert (p["pending"], p["completed"], p["verified"], p["flagged"], p["eligible"]) == (3, 1, 1, 1, 1)
    assert p["required"] == REQUIRED_HUMAN_LABELS and p["remaining_to_required"] == REQUIRED_HUMAN_LABELS - 1


def test_flag_requires_reason(store):
    with pytest.raises(ValueError, match="reason"):
        store.flag("R0", "P1", annotator="alice", reason=" ")


def test_progress_is_saved_after_every_action_and_resumes(store, tmp_path):
    _label(store, "R0")
    _label(store, "R1")
    reopened = HumanAnnotationStore(tmp_path / "human_annotations.json")  # e.g. after closing the app
    assert reopened.status_of("R0") == "completed" and reopened.next_pending(QUEUE) == "R2"
    assert reopened.next_pending(QUEUE, after="R4") == "R5"
    assert reopened.next_pending(QUEUE, after="R5") == "R2"  # wraps around


def test_reset_keeps_an_audit_copy(store, tmp_path):
    _label(store, "R0")
    store.reset("R0", by="alice")
    reopened = HumanAnnotationStore(tmp_path / "human_annotations.json")
    assert reopened.status_of("R0") == "pending" and reopened.meta["removed"][0]["record"]["review_id"] == "R0"


def test_reserve_reviews_replace_flagged_ones_reproducibly(tmp_path):
    reviews = pd.DataFrame({"review_id": [f"X{i}" for i in range(20)]})
    a = HumanAnnotationStore(tmp_path / "a.json").add_reserve(reviews, exclude={"X0", "X1"}, n=2, seed=42)
    b = HumanAnnotationStore(tmp_path / "b.json").add_reserve(reviews, exclude={"X0", "X1"}, n=2, seed=42)
    assert a == b and not {"X0", "X1"} & set(a)
    st = HumanAnnotationStore(tmp_path / "a.json")
    assert st.queue(["X0", "X1"]) == ["X0", "X1", *a]


# ------------------------------------------------------------------ gold / export / import
def test_gold_contains_only_verified_human_labels(store):
    _label(store, "R0", who="alice")
    _label(store, "R1", who="alice")
    store.verify("R1", reviewer="bob")
    store.flag("R2", "P1", annotator="alice", reason="spam")
    assert [g.review_id for g in store.gold(QUEUE)] == ["R1"]
    assert sorted(g.review_id for g in store.gold(QUEUE, include_unverified=True)) == ["R0", "R1"]
    with pytest.raises(ValueError, match="only verified human labels"):
        store.get("R0").to_annotation()


def test_export_reference_contains_only_eligible_labels(store, tmp_path):
    _label(store, "R0", who="alice")
    _label(store, "R1", who="alice")
    store.verify("R1", reviewer="bob")
    paths = store.export_reference(tmp_path / "out")
    doc = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert doc["count"] == 1 and doc["meets_requirement"] is False and doc["provenance"] == {"human_created": 1, "human_verified": 0}
    csv = pd.read_csv(paths["csv"])
    assert list(csv["review_id"]) == ["R1"] and set(csv["status"]) == {"verified"}


def test_import_blind_csv_rejects_ai_rows_and_unknown_ids(store):
    df = pd.DataFrame([
        {"review_id": "R0", "overall_sentiment": "positive", "aspects": "noise:negative", "complaint_categories": "noise_issue", "annotator": "alice", "notes": ""},
        {"review_id": "R1", "overall_sentiment": "negative", "aspects": "", "complaint_categories": "", "annotator": "alice", "notes": f"{AI_ASSISTED_MARKER} copy"},
        {"review_id": "R2", "overall_sentiment": "negative", "aspects": "", "complaint_categories": "", "annotator": "alice", "notes": "", "provenance": "ai_assisted"},
        {"review_id": "ZZ", "overall_sentiment": "negative", "aspects": "", "complaint_categories": "", "annotator": "alice", "notes": ""},
        {"review_id": "R3", "overall_sentiment": "", "aspects": "", "complaint_categories": "", "annotator": "", "notes": ""},
    ]).fillna("")
    res = store.import_blind_csv(df, {r: "P1" for r in QUEUE})
    assert res["imported"] == 1 and res["skipped_unlabelled"] == 1 and len(res["errors"]) == 3
    assert store.get("R0").provenance == "human_created" and store.get("R1") is None and store.get("R2") is None


def test_migration_copies_only_documented_human_labels(store, tmp_path):
    legacy = tmp_path / "annotations.json"
    legacy.write_text(json.dumps({"meta": {"annotation_method": {"human_labels": ["H1"]}}, "annotations": {
        "H1": {"review_id": "H1", "product_id": "P", "overall_sentiment": "positive", "aspects": [], "complaint_categories": [], "annotator": "akash", "notes": "", "status": "complete", "annotated_at": "t"},
        "A1": {"review_id": "A1", "product_id": "P", "overall_sentiment": "negative", "aspects": [], "complaint_categories": [], "annotator": "akash", "notes": f"{AI_ASSISTED_MARKER} (x)", "status": "complete", "annotated_at": "t"},
    }}), encoding="utf-8")
    assert migrate_legacy_human_labels(store, legacy) == ["H1"]
    assert migrate_legacy_human_labels(store, legacy) == []  # idempotent
    lab = store.get("H1")
    assert lab.provenance == "human_created" and lab.status == "completed" and lab.history[-1]["action"] == "migrated_from_annotations_json"
    assert store.get("A1") is None


def test_readonly_mode_blocks_every_human_store_write(store, monkeypatch, tmp_path):
    _label(store, "R0")
    monkeypatch.setenv("REVIEWIQ_DEMO_READONLY", "true")
    before = store.path.read_bytes()
    for action in (lambda: _label(store, "R1"), lambda: store.verify("R0", reviewer="bob"),
                   lambda: store.flag("R2", "P1", annotator="a", reason="x"), lambda: store.reset("R0", by="a"),
                   lambda: store.export_reference(tmp_path / "exp"),
                   lambda: store.add_reserve(pd.DataFrame({"review_id": ["Z"]}), set(), 1, 1)):
        with pytest.raises(ReadOnlyModeError):
            action()
    assert store.path.read_bytes() == before and not (tmp_path / "exp").exists()


# ------------------------------------------------------------------ D2 evaluation against human labels
def _reviews(n=4):
    texts = ["Works great and very quiet.", "Broke after a week, total junk.", "Fits my fridge perfectly, good price.", "Loud but it works, okay value."]
    return [ReviewInput(review_id=f"E{i}", product_id="P", review_text=texts[i % 4]) for i in range(n)]


def test_independent_metrics_match_sklearn():
    y_true = ["positive", "negative", "mixed", "neutral", "positive", "negative", "mixed", "positive"]
    y_pred = ["positive", "mixed", "mixed", "positive", "__invalid__", "negative", "negative", "neutral"]
    sk = single_label_metrics(y_true, y_pred, ["positive", "negative", "neutral", "mixed"])
    ind = independent_sentiment_metrics(y_true, y_pred)
    assert abs(ind["macro_f1"] - sk["macro_f1"]) < 1e-4 and abs(ind["accuracy"] - sk["accuracy"]) < 1e-4
    assert ind["confusion_matrix"] == sk["confusion_matrix"] and ind["invalid_predictions"] == 1


def test_human_evaluation_same_review_set_and_status(settings, store):
    reviews = _reviews(4)
    run_dir = run_experiment(settings, MockLLMClient(malformed_rate=0), ["zero_shot", "few_shot", "cot"], reviews[:3], [], required_gold=3)
    for i, sent in enumerate(["positive", "negative", "positive", "mixed"]):
        store.save_blind_label(f"E{i}", "P", annotator="alice", overall_sentiment=sent)
        store.verify(f"E{i}", reviewer="bob")
    gold = store.gold([f"E{i}" for i in range(4)])
    res = evaluate_against_human(run_dir, gold, required=4)
    assert res["n_human_labels"] == 4 and res["n_evaluated_common"] == 3  # E3 has no predictions
    assert res["status"] == "incomplete" and res["needs_inference"]["few_shot"] == ["E3"]
    assert {e["classification"]["overall_sentiment"]["n"] for e in res["strategies"].values()} == {3}  # identical set
    assert res["independent_check_passed"]
    assert (run_dir / "human_reference" / "metrics.json").exists() and (run_dir / "metrics.json").exists()

    res2 = evaluate_against_human(run_dir, gold[:3], required=3, write=False)
    assert res2["status"] == "complete"
    res3 = evaluate_against_human(run_dir, gold[:3], required=300, write=False)
    assert res3["status"] == "provisional"
    assert evaluate_against_human(run_dir, [], write=False)["status"] == "awaiting_human_labels"


def test_failed_predictions_count_as_wrong(settings, store):
    from dataclasses import replace

    reviews = _reviews(4)
    s = replace(settings, max_repair_attempts=0)
    run_dir = run_experiment(s, MockLLMClient(malformed_rate=100), ["zero_shot"], reviews, [], required_gold=4)
    for i in range(4):
        store.save_blind_label(f"E{i}", "P", annotator="alice", overall_sentiment="positive")
        store.verify(f"E{i}", reviewer="bob")
    res = evaluate_against_human(run_dir, store.gold(), required=4, write=False)
    sent = res["strategies"]["zero_shot"]["classification"]["overall_sentiment"]
    assert sent["n"] == 4 and sent["invalid_predictions"] == res["strategies"]["zero_shot"]["classification"]["n_failed_extractions"]
    assert sent["accuracy"] <= (4 - sent["invalid_predictions"]) / 4


def test_unverified_inclusion_is_never_complete(settings, store):
    run_dir = run_experiment(settings, MockLLMClient(malformed_rate=0), ["zero_shot"], _reviews(2), [], required_gold=2)
    for i in range(2):
        store.save_blind_label(f"E{i}", "P", annotator="alice", overall_sentiment="positive")
    res = evaluate_against_human(run_dir, store.gold(include_unverified=True), required=2, write=False, unverified_included=True)
    assert res["status"] == "provisional" and "UNVERIFIED" in res["reference"]


# ------------------------------------------------------------------ conflicts: nothing is silently overwritten
from annotation.human_annotation import AnnotationConflict  # noqa: E402


def test_another_annotator_cannot_overwrite_flag_or_reset_a_label(store):
    _label(store, "R0", who="alice", sentiment="positive")
    with pytest.raises(AnnotationConflict, match="alice"):
        _label(store, "R0", who="bob", sentiment="negative")
    with pytest.raises(AnnotationConflict):
        store.flag("R0", "P1", annotator="bob", reason="ambiguous")
    with pytest.raises(AnnotationConflict):
        store.reset("R0", by="bob")
    with pytest.raises(AnnotationConflict):
        store.save_checked_ai_label("R0", "P1", annotator="bob", overall_sentiment="negative")
    lab = store.get("R0")
    assert lab.annotator == "alice" and lab.overall_sentiment.value == "positive" and [h["action"] for h in lab.history] == ["labelled"]


def test_annotator_names_compare_case_insensitively(store):
    _label(store, "R0", who="Alice")
    assert _label(store, "R0", who=" alice ", sentiment="negative").overall_sentiment.value == "negative"


def test_verified_label_cannot_be_flagged_or_verified_twice(store):
    _label(store, "R0", who="alice")
    store.verify("R0", reviewer="bob")
    with pytest.raises(AnnotationConflict, match="verified"):
        store.flag("R0", "P1", annotator="alice", reason="changed my mind")
    with pytest.raises(AnnotationConflict, match="already verified"):
        store.verify("R0", reviewer="carol")
    assert store.get("R0").verified_by == "bob"


def test_two_sessions_never_erase_each_others_labels(tmp_path):
    """Two app sessions open the store, then each saves a label: both labels must survive (no lost update)."""
    path = tmp_path / "human_annotations.json"
    a, b = HumanAnnotationStore(path), HumanAnnotationStore(path)
    a.save_blind_label("R0", "P1", annotator="alice", overall_sentiment="positive")
    b.save_blind_label("R1", "P1", annotator="bob", overall_sentiment="negative")  # b's in-memory copy predates R0
    fresh = HumanAnnotationStore(path)
    assert fresh.status_of("R0") == "completed" and fresh.status_of("R1") == "completed"
    with pytest.raises(AnnotationConflict):  # b now sees alice's label and cannot overwrite it
        b.save_blind_label("R0", "P1", annotator="bob", overall_sentiment="negative")
    a.verify("R1", reviewer="alice")
    with pytest.raises(AnnotationConflict, match="already verified"):  # stale session tries a second verification
        b.verify("R1", reviewer="carol")
    assert not (tmp_path / "human_annotations.json.lock").exists()


def test_locked_store_raises_conflict_and_stale_lock_is_recovered(tmp_path):
    import os
    import time as _t
    path = tmp_path / "human_annotations.json"
    s = HumanAnnotationStore(path)
    lock = tmp_path / "human_annotations.json.lock"
    lock.write_text("other")
    with pytest.raises(AnnotationConflict, match="locked"):
        with s._transaction(timeout_s=0.2):
            pass
    old = _t.time() - 120
    os.utime(lock, (old, old))  # crashed session left the lock behind
    s.save_blind_label("R0", "P1", annotator="alice", overall_sentiment="neutral")
    assert s.status_of("R0") == "completed" and not lock.exists()


def test_import_never_overwrites_existing_or_duplicate_rows(store):
    _label(store, "R0", who="alice", sentiment="positive")
    df = pd.DataFrame([
        {"review_id": "R0", "overall_sentiment": "negative", "annotator": "alice"},
        {"review_id": "R1", "overall_sentiment": "negative", "annotator": "bob"},
        {"review_id": "R1", "overall_sentiment": "positive", "annotator": "bob"},
    ])
    res = store.import_blind_csv(df, {r: "P1" for r in QUEUE})
    assert res["imported"] == 1 and len(res["errors"]) == 2 and all("overwrite" in e for e in res["errors"])
    assert store.get("R0").overall_sentiment.value == "positive" and store.get("R1").overall_sentiment.value == "negative"


def test_writes_outside_a_transaction_are_refused(store):
    _label(store, "R0")
    store._items["R0"].notes = "edited without the lock"
    with pytest.raises(RuntimeError, match="_transaction"):
        store._flush()
    assert HumanAnnotationStore(store.path).get("R0").notes == ""  # nothing reached the file


def test_import_never_relabels_even_the_same_annotators_label(store):
    _label(store, "R0", who="alice", sentiment="positive")
    with pytest.raises(AnnotationConflict, match="never overwrites"):
        store.save_blind_label("R0", "P1", annotator="alice", overall_sentiment="negative", allow_relabel=False)
    assert store.get("R0").overall_sentiment.value == "positive"


def test_migration_rerun_preserves_history_and_never_overwrites(store, tmp_path):
    legacy = tmp_path / "annotations.json"
    legacy.write_text(json.dumps({"meta": {"annotation_method": {"human_labels": ["H1"]}}, "annotations": {
        "H1": {"review_id": "H1", "product_id": "P", "overall_sentiment": "positive", "aspects": [], "complaint_categories": [],
               "annotator": "akash", "notes": "", "status": "complete", "annotated_at": "2026-11-10T12:20:30+00:00"}}}), encoding="utf-8")
    migrate_legacy_human_labels(store, legacy)
    store.verify("H1", reviewer="bob")
    before = HumanAnnotationStore(store.path).get("H1").model_dump()
    assert migrate_legacy_human_labels(store, legacy) == []  # rerun: no change
    after = HumanAnnotationStore(store.path).get("H1")
    assert after.model_dump() == before and after.status == "verified"
    assert [h["action"] for h in after.history] == ["migrated_from_annotations_json", "verified"]
    assert after.history[0]["original_annotated_at"] == "2026-11-10T12:20:30+00:00"


def test_ai_assisted_label_has_no_path_to_verified_human_label(store, tmp_path):
    """Neither verification nor migration nor import can turn an AI-assisted label into an eligible human label."""
    with pytest.raises(ValueError, match="only completed"):  # nothing to verify: AI labels are never in the human store
        store.verify("A1", reviewer="bob")
    legacy = tmp_path / "annotations.json"
    legacy.write_text(json.dumps({"meta": {"annotation_method": {"human_labels": []}}, "annotations": {
        "A1": {"review_id": "A1", "product_id": "P", "overall_sentiment": "negative", "aspects": [], "complaint_categories": [],
               "annotator": "akash", "notes": f"{AI_ASSISTED_MARKER} (x)", "status": "complete", "annotated_at": "t"}}}), encoding="utf-8")
    assert migrate_legacy_human_labels(store, legacy) == []
    res = store.import_blind_csv(pd.DataFrame([{"review_id": "A1", "overall_sentiment": "negative", "annotator": "akash",
                                                "notes": f"{AI_ASSISTED_MARKER} (x)"}]), {"A1": "P"})
    assert res["imported"] == 0 and store.get("A1") is None and store.gold() == []
    with pytest.raises(ValidationError):
        HumanLabel(review_id="A1", product_id="P", status="verified", provenance="ai_assisted", overall_sentiment="negative",
                   annotator="model", blind=False, verified_by="bob")


def test_progress_report_shows_completion_without_verification(store):
    from scripts.human_annotation import progress_markdown

    queue = ["R0", "R1", "R2"]
    _label(store, "R0", who="alice")
    text = progress_markdown(store, queue)
    assert "**1 / 3**" in text and "`R1`" in text and "in progress: 2 review(s)" in text
    assert "AI-assisted labels" in text and "never" in text and "not** been independently verified" in text
    _label(store, "R1", who="bob")
    store.flag("R2", "P1", annotator="alice", reason="spam")
    text = progress_markdown(store, queue)
    assert "**2 / 3**" in text and "complete: all 2 independent human annotations are done" in text


def test_completion_does_not_require_second_person_verification(store):
    queue = ["R0", "R1", "R2"]
    for rid, who in zip(queue, ("alice", "bob", "carol")):
        _label(store, rid, who=who)
    p = store.progress(queue)
    assert (p["labelled"], p["completed"], p["pending"], p["flagged"], p["verified"]) == (3, 3, 0, 0, 0)
    assert p["annotation_complete"] is True and p["human_created"] == 3
    assert all(store.get(r).verified_by is None for r in queue)  # completion never marks a label as verified
    p = store.progress(queue + ["R3"])
    assert p["annotation_complete"] is False and p["pending"] == 1


def test_finalise_refuses_and_writes_nothing_below_the_requirement(tmp_path, monkeypatch, capsys):
    import scripts.human_annotation as cli

    class FakeSettings:
        class paths:
            annotation_sample_csv = tmp_path / "sample.csv"
            annotations = tmp_path / "ann"
    pd.DataFrame({"review_id": ["R0", "R1"]}).to_csv(FakeSettings.paths.annotation_sample_csv, index=False)
    hs = HumanAnnotationStore(tmp_path / "human_annotations.json")
    _label(hs, "R0", who="alice")
    monkeypatch.setattr(cli, "load_settings", lambda: FakeSettings)
    monkeypatch.setattr(cli.HumanAnnotationStore, "default", classmethod(lambda cls, s: hs))
    monkeypatch.setattr(cli, "evaluate_against_human", lambda *a, **k: pytest.fail("must not evaluate below the requirement"))
    monkeypatch.setattr("sys.argv", ["human_annotation.py", "finalise", "--run", "X"])
    before = (tmp_path / "human_annotations.json").read_bytes()
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2 and "NOT READY" in capsys.readouterr().out
    assert not (tmp_path / "ann").exists() and (tmp_path / "human_annotations.json").read_bytes() == before
