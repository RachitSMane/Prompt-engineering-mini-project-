"""Primary D2 evaluation against the 300 blind human-created labels (evaluation/d2_human_eval.py).

Covers: label completeness / uniqueness / provenance, prediction coverage (same 300 reviews for every strategy),
metric correctness (independent recomputation), reproducibility, refusal on inconsistent data, labels never changed,
historical outputs preserved, stale detection, and the Prompt Experiments dashboard (read-only demo mode).
All mutation tests work on temporary copies; the real files are only read.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from evaluation.d2_human_eval import (
    OUT_DIR, D2ValidationError, evaluate, human_gold, prediction_coverage, stored_inputs_match, validate_labels,
)

ROOT = Path(__file__).resolve().parents[1]
STORE = ROOT / "data" / "annotations" / "human_annotations.json"
SAMPLE = ROOT / "data" / "annotations" / "annotation_sample.csv"
LEGACY = ROOT / "data" / "annotations" / "annotations.json"
RUN = ROOT / "outputs" / "live" / "experiments" / "20261111-030912_qwen2.5-3b"
SAVED = RUN / OUT_DIR / "metrics.json"
STRATS = ("zero_shot", "few_shot", "cot")

pytestmark = pytest.mark.skipif(not (STORE.exists() and RUN.exists()), reason="real human labels / D2 run not present")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


@pytest.fixture()
def tmp_data(tmp_path):
    """Copies of the label store, sample and the run's config + prediction files."""
    ann = tmp_path / "annotations"
    ann.mkdir()
    shutil.copy(STORE, ann / "human_annotations.json")
    shutil.copy(SAMPLE, ann / "annotation_sample.csv")
    shutil.copy(LEGACY, ann / "annotations.json")
    run = tmp_path / "run"
    (run / "supplementary").mkdir(parents=True)
    shutil.copy(RUN / "config.json", run / "config.json")
    for n in STRATS:
        shutil.copy(RUN / f"predictions_{n}.jsonl", run / f"predictions_{n}.jsonl")
        shutil.copy(RUN / "supplementary" / f"predictions_{n}.jsonl", run / "supplementary" / f"predictions_{n}.jsonl")
    return {"store": ann / "human_annotations.json", "sample": ann / "annotation_sample.csv", "legacy": ann / "annotations.json", "run": run}


def _edit_store(path: Path, fn) -> None:
    raw = json.loads(path.read_text(encoding="utf-8"))
    fn(raw)
    path.write_text(json.dumps(raw), encoding="utf-8")


# ------------------------------------------------------------------------------------------- completeness / provenance
def test_real_labels_are_300_unique_blind_human_created_with_recorded_annotators():
    v = validate_labels(STORE, SAMPLE)
    assert v["valid"], v["problems"]
    assert v["n_labels"] == 300 and v["n_sample"] == 300
    assert v["provenance"] == {"human_created": 300} and v["blind"] == 300 and v["status"] == {"completed": 300}
    # actual allocation and identities are preserved, not redistributed
    assert v["annotators"] == {"Rachit": 100, "Akash": 83, "p.chinni krishna reddy": 62, "pranav menon": 55}
    assert sum(v["overall_sentiment"].values()) == 300
    raw = json.loads(STORE.read_text(encoding="utf-8"))["labels"]
    assert all(rec["history"] and all(h["by"] == rec["annotator"] for h in rec["history"]) for rec in raw.values())


def test_gold_copies_labels_unchanged_in_sample_order():
    gold = human_gold(STORE, SAMPLE)
    raw = json.loads(STORE.read_text(encoding="utf-8"))["labels"]
    assert len(gold) == 300 and len({g.review_id for g in gold}) == 300
    import pandas as pd
    assert [g.review_id for g in gold] == pd.read_csv(SAMPLE, dtype=str)["review_id"].tolist()
    assert all(g.overall_sentiment.value == raw[g.review_id]["overall_sentiment"] and g.annotator == raw[g.review_id]["annotator"] for g in gold)


@pytest.mark.parametrize("mutate, needle", [
    (lambda r: r["labels"].pop(next(iter(r["labels"]))), "without a human label"),
    (lambda r: r["labels"].__setitem__("Rnotinsample01", dict(next(iter(r["labels"].values())), review_id="Rnotinsample01")), "outside the sample"),
    (lambda r: next(iter(r["labels"].values())).update(provenance="human_verified", blind=False), "expected blind human_created"),
    (lambda r: next(iter(r["labels"].values())).update(notes="AI-assisted label (Claude) x"), "invalid record"),  # the schema itself rejects it
    (lambda r: next(iter(r["labels"].values())).update(overall_sentiment="very good"), "invalid record"),
    (lambda r: next(iter(r["labels"].values())).update(product_id="BXXXXXXXXX"), "differs from the sample"),
    (lambda r: next(iter(r["labels"].values()))["history"].append({"at": "x", "action": "relabelled", "by": "someone else"}), "someone other than the annotator"),
    (lambda r: next(iter(r["labels"].values())).update(status="flagged"), "expected 'completed'"),
])
def test_invalid_label_sets_are_reported_and_evaluation_refused(tmp_data, mutate, needle):
    _edit_store(tmp_data["store"], mutate)
    before = _sha(tmp_data["store"])
    v = validate_labels(tmp_data["store"], tmp_data["sample"])
    assert not v["valid"] and any(needle in p for p in v["problems"]), v["problems"]
    with pytest.raises(D2ValidationError):
        evaluate(tmp_data["run"], tmp_data["store"], tmp_data["sample"])
    assert not (tmp_data["run"] / OUT_DIR).exists()  # nothing written
    assert _sha(tmp_data["store"]) == before  # nothing "fixed"


def test_duplicate_review_id_in_sample_is_reported(tmp_data):
    lines = tmp_data["sample"].read_text(encoding="utf-8").splitlines()
    tmp_data["sample"].write_text("\n".join(lines + [lines[1]]) + "\n", encoding="utf-8")
    v = validate_labels(tmp_data["store"], tmp_data["sample"])
    assert not v["valid"] and any("duplicate review ids" in p for p in v["problems"])


# ------------------------------------------------------------------------------------------- prediction coverage
def test_all_strategies_have_predictions_for_exactly_the_same_300_reviews():
    ids = [g.review_id for g in human_gold(STORE, SAMPLE)]
    cov = prediction_coverage(RUN, ids)
    assert cov["problems"] == [] and cov["same_review_set_for_all_strategies"]
    for n in STRATS:
        e = cov["strategies"][n]
        assert e["valid"] + e["failed"] == 300 and not e["missing"] and not e["duplicates"]
        assert {k: f["rows"] for k, f in e["files"].items()} == {f"predictions_{n}.jsonl": 299, f"supplementary/predictions_{n}.jsonl": 1}


def test_missing_or_duplicate_predictions_stop_the_evaluation(tmp_data):
    f = tmp_data["run"] / "supplementary" / "predictions_cot.jsonl"
    f.write_text("", encoding="utf-8")  # cot loses its prediction for the supplementary review
    with pytest.raises(D2ValidationError, match="cot: no usable prediction"):
        evaluate(tmp_data["run"], tmp_data["store"], tmp_data["sample"])
    shutil.copy(RUN / "supplementary" / "predictions_cot.jsonl", f)
    p = tmp_data["run"] / "predictions_few_shot.jsonl"
    p.write_text(p.read_text(encoding="utf-8") + p.read_text(encoding="utf-8").splitlines()[0] + "\n", encoding="utf-8")
    with pytest.raises(D2ValidationError, match="few_shot: duplicate prediction rows"):
        evaluate(tmp_data["run"], tmp_data["store"], tmp_data["sample"])
    assert not (tmp_data["run"] / OUT_DIR).exists()


# ------------------------------------------------------------------------------------------- metrics
@pytest.mark.skipif(not SAVED.exists(), reason="human-labelled evaluation not generated")
def test_saved_metrics_are_complete_current_and_independently_reproduced():
    from scripts.d2_human_eval import independent_check

    m = json.loads(SAVED.read_text(encoding="utf-8"))
    assert m["status"] == "complete" and m["independent_check_passed"] and m["fresh_inference"] is False
    assert m["n_evaluated"] == 300 and m["reviews_from_supplementary_predictions"] == ["R5cfb12ef75b4"]
    assert stored_inputs_match(RUN, STORE, SAMPLE)["current"]
    assert independent_check(RUN, STORE, SAMPLE, m) == []
    for n in STRATS:
        s = m["strategies"][n]["classification"]["overall_sentiment"]
        cm = s["confusion_matrix"]
        failed = m["strategies"][n]["invalid_predictions"]
        assert s["n"] == 300 and sum(map(sum, cm)) + failed == 300  # failures are in no column but count as wrong
        assert abs(sum(cm[i][i] for i in range(4)) / 300 - s["accuracy"]) < 1e-4
        assert sum(s["per_class"][c]["support"] for c in s["labels"]) == 300
        assert all(sum(row) <= s["per_class"][c]["support"] for c, row in zip(s["labels"], cm))  # the rest are failures
        assert abs(sum(s["per_class"][c]["f1-score"] for c in s["labels"]) / 4 - s["macro_f1"]) < 1e-3
    assert not m["multilabel_validity"]["valid"]  # aspects/complaints were almost never annotated -> not reported


def test_evaluation_is_reproducible_and_matches_the_saved_result(tmp_data):
    a = evaluate(tmp_data["run"], tmp_data["store"], tmp_data["sample"], legacy_path=tmp_data["legacy"], write=False)
    b = evaluate(tmp_data["run"], tmp_data["store"], tmp_data["sample"], legacy_path=tmp_data["legacy"], write=False)
    assert a["content_sha256"] == b["content_sha256"]
    if SAVED.exists():
        saved = json.loads(SAVED.read_text(encoding="utf-8"))
        for n in STRATS:
            assert a["strategies"][n]["classification"]["overall_sentiment"] == saved["strategies"][n]["classification"]["overall_sentiment"]


def test_evaluation_writes_only_its_own_folder_and_never_changes_labels(tmp_data):
    run = tmp_data["run"]
    before = {p: _sha(p) for p in [tmp_data["store"], tmp_data["sample"], tmp_data["legacy"], *run.rglob("*.json*")]}
    evaluate(run, tmp_data["store"], tmp_data["sample"], legacy_path=tmp_data["legacy"])
    assert {p: _sha(p) for p in before} == before
    assert sorted(p.name for p in (run / OUT_DIR).iterdir()) == ["manifest.json", "metrics.json", "reference_labels.csv", "report.md"]
    assert stored_inputs_match(run, tmp_data["store"], tmp_data["sample"])["current"]
    # a later human change to a label makes the saved evaluation stale
    _edit_store(tmp_data["store"], lambda r: next(iter(r["labels"].values())).update(notes="changed by a human"))
    cur = stored_inputs_match(run, tmp_data["store"], tmp_data["sample"])
    assert not cur["current"] and cur["changed"] == ["human_annotations.json"]


def test_historical_d2_results_are_preserved():
    m = json.loads((RUN / "metrics.json").read_text(encoding="utf-8"))
    assert [m["strategies"][n]["classification"]["overall_sentiment"]["macro_f1"] for n in STRATS] == [0.438, 0.6025, 0.4232]
    assert m["n_gold_available"] == 299
    raw = json.loads(LEGACY.read_text(encoding="utf-8"))
    assert raw["meta"]["annotation_method"]["ai_assisted_labels"] == 299 and len(raw["annotations"]) == 300
    assert (ROOT / "reports" / "D2_FINAL_REPORT.md").exists() and (RUN / "approved_reference" / "metrics.json").exists()


@pytest.mark.skipif(not SAVED.exists(), reason="human-labelled evaluation not generated")
def test_report_matches_saved_metrics():
    from scripts.d2_human_eval import REPORT, build_report

    m = json.loads(SAVED.read_text(encoding="utf-8"))
    text = REPORT.read_text(encoding="utf-8")
    assert text == build_report(m, RUN)
    assert "re-evaluation of stored predictions" in text and "not verified by a second person" in text
    for n in STRATS:
        assert f"{m['strategies'][n]['classification']['overall_sentiment']['macro_f1']:.4f}" in text


# ------------------------------------------------------------------------------------------- dashboard
@pytest.mark.skipif(not SAVED.exists(), reason="human-labelled evaluation not generated")
def test_prompt_experiments_shows_human_labelled_primary_and_historical_view(monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("REVIEWIQ_DEMO_READONLY", "true")
    monkeypatch.chdir(ROOT)
    m = json.loads(SAVED.read_text(encoding="utf-8"))
    at = AppTest.from_file(str(ROOT / "ui" / "prompt_comparison.py"), default_timeout=180)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    html = " ".join(x.value for x in at.markdown)
    assert at.segmented_control(key="pc_reference_view").value.startswith("Human-labelled")
    assert "Primary D2 evaluation" in html and "stored predictions" in html and "Label quality" in html
    assert f"{m['strategies']['few_shot']['classification']['overall_sentiment']['macro_f1']:.3f}" in html
    assert "Not valid for the human labels" in html
    at.segmented_control(key="pc_reference_view").set_value("AI-assisted reference · historical").run()
    assert not at.exception, [e.value for e in at.exception]
    html = " ".join(x.value for x in at.markdown)
    assert "Historical evaluation" in html and "0.603" in html  # historical few-shot macro F1 0.6025
