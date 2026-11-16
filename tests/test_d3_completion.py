"""D3 completion: derived rubric, single-annotator human claim validation, completion status, dashboard and UI."""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

import pytest

from evaluation.claim_checks import Stats
from evaluation.claim_gold import (
    LABELS, AnnotationInputError, ClaimValidationStore, DuplicateAnnotationError, cohen_kappa, compare, export_csv_bytes,
    export_rows, load_template, sha256_file, validate_store, write_template,
)
from evaluation.d3_status import badge_text, compute, load_evaluations
from evaluation.judge_v4 import build_summary_evaluation
from evaluation.rubric_v4 import (
    check_rubric_consistent, consistency, coverage, hallucination, important_themes, rubric_for,
)
from schemas.claim_schema import (
    Claim, ClaimEvaluation, ClaimStatus, DeterministicFinding, NumericCheck,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "dataset__few_shot__qwen2.5-3b__v1"
LIVE = ROOT / "outputs" / "live"
V4 = LIVE / "d3" / "judge_v4" / SOURCE
HAVE_V4 = (V4 / "evaluations.jsonl").exists() and (V4 / "report.json").exists()
TEMPLATE = LIVE / "d3" / "claim_gold" / SOURCE / "claim_gold_template.csv"
HAVE_TEMPLATE = TEMPLATE.exists()


# --------------------------------------------------------------------------- helpers
def stats(aspects=None, complaints=None) -> Stats:
    aspects = aspects or {"performance": 6, "compatibility": 4, "price_value": 3, "other": 9}
    complaints = complaints or {"poor_durability": 5, "installation_difficulty": 3}
    return Stats(total=10, sentiment_counts={"positive": 4, "negative": 6, "neutral": 0, "mixed": 0},
                 sentiment_pct={"positive": 40.0, "negative": 60.0, "neutral": 0.0, "mixed": 0.0}, average_rating=2.9,
                 reviews_with_complaints=6, complaint_counts=complaints, complaint_pct={k: 10.0 * v for k, v in complaints.items()},
                 aspects={a: {"mentions": n, "positive": n // 2, "negative": n - n // 2, "neutral": 0} for a, n in aspects.items()})


def ce(i, status, text="The product works well.", field="appreciated", polarity="positive", findings=(), numeric=()):
    kind = "opinion_sentence" if field == "overall_opinion" else ("derived" if field == "frequent_aspects" else "point")
    claim = Claim(claim_id=f"s1:{field}:{i}", summary_id="s1", product_id="P1", source_field=field, index=i, kind=kind, text=text,
                  polarity=polarity, cited_review_ids=["R000000000001"] if kind == "point" else [])
    return ClaimEvaluation(claim=claim, status=status, explanation="test", method="deterministic", findings=list(findings),
                           numeric_checks=list(numeric))


def ev(claims):
    return build_summary_evaluation(kind="original", summary_id="s1", product_id="P1", claim_evals=claims, run_id="t", config={})


WRONG = NumericCheck(mention="80%", kind="percentage", extracted_value=80, expected_value=86.7, calculation="x", passed=False, severity="error")
MISLEAD = NumericCheck(mention="8 out of 15", kind="count", extracted_value=8, expected_value=8, denominator=19, calculation="x",
                       passed=False, severity="misleading")
FAIL = DeterministicFinding(check="sentiment_majority", outcome="fail", status_cap=ClaimStatus.CONTRADICTED, detail="x")
MIS = DeterministicFinding(check="numeric_count", outcome="misleading", status_cap=ClaimStatus.PARTIALLY_SUPPORTED, detail="x")


# --------------------------------------------------------------------------- rubric
def test_hallucination_anchors():
    assert hallucination(ev([ce(0, ClaimStatus.SUPPORTED)]))["score"] == 5
    assert hallucination(ev([ce(0, ClaimStatus.SUPPORTED), ce(1, ClaimStatus.UNSUPPORTED)]))["score"] == 4
    assert hallucination(ev([ce(0, ClaimStatus.CONTRADICTED)]))["score"] == 3
    assert hallucination(ev([ce(i, ClaimStatus.UNSUPPORTED) for i in range(5)]))["score"] == 2
    assert hallucination(ev([ce(i, ClaimStatus.UNSUPPORTED) for i in range(7)]))["score"] == 1
    wrong = ce(0, ClaimStatus.CONTRADICTED, "80% are positive.", "overall_opinion", "neutral", [FAIL], [WRONG])
    assert hallucination(ev([wrong]))["score"] == 1


def test_misleading_denominator_is_not_a_fabricated_number():
    c = ce(0, ClaimStatus.PARTIALLY_SUPPORTED, "8 out of 15 reviews mention durability.", "overall_opinion", "neutral", [MIS], [MISLEAD])
    e = ev([c])
    assert hallucination(e)["score"] == 5 and hallucination(e)["misleading_numeric_claims"] == ["s1:overall_opinion:0"]
    assert consistency(e)["score"] == 4


def test_consistency_anchors():
    op = lambda i, f, n=(): ce(i, ClaimStatus.CONTRADICTED if f is FAIL else ClaimStatus.PARTIALLY_SUPPORTED, "Mostly positive.",
                               "overall_opinion", "neutral", [f], n)
    assert consistency(ev([ce(0, ClaimStatus.SUPPORTED, "Mixed reviews.", "overall_opinion", "neutral")]))["score"] == 5
    assert consistency(ev([op(0, MIS)]))["score"] == 4
    assert consistency(ev([op(0, MIS), op(1, MIS)]))["score"] == 3
    assert consistency(ev([op(0, FAIL)]))["score"] == 3
    assert consistency(ev([op(0, FAIL, [WRONG])]))["score"] == 2
    assert consistency(ev([op(0, FAIL), op(1, FAIL)]))["score"] == 2
    assert consistency(ev([op(i, FAIL) for i in range(3)]))["score"] == 1
    # findings on cited points are review checks, not statistic consistency
    assert consistency(ev([ce(0, ClaimStatus.CONTRADICTED, findings=[FAIL])]))["score"] == 5


def test_coverage_themes_and_anchors():
    st = stats()
    themes = important_themes(st)
    assert [t.label for t in themes] == ["performance", "compatibility", "price_value", "poor_durability", "installation_difficulty"]
    assert "other" not in [t.label for t in themes]
    assert [t.major for t in themes] == [True, False, False, True, False]
    full = [ce(0, ClaimStatus.SUPPORTED, "It performs well and fits, at a fair price."),
            ce(1, ClaimStatus.SUPPORTED, "It broke after a month; poor durability.", "complaints", "negative"),
            ce(2, ClaimStatus.SUPPORTED, "Installation was difficult.", "complaints", "negative")]
    assert coverage(ev(full), st)["score"] == 5
    assert coverage(ev(full[:2]), st)["score"] == 4  # misses one minor complaint
    assert coverage(ev([full[0]]), st)["score"] == 3  # misses the major complaint, 3/5 mentioned
    assert coverage(ev([ce(0, ClaimStatus.SUPPORTED, "Nice colour.")]), st)["score"] == 1
    # a complaint theme only counts when stated in a negative role, not as praise
    praise = ce(0, ClaimStatus.SUPPORTED, "Installation was easy and it lasted well.")
    assert coverage(ev([praise]), st)["missed"] == ["aspect:performance", "aspect:compatibility", "aspect:price_value",
                                                     "complaint:poor_durability", "complaint:installation_difficulty"]


def test_rubric_never_contradicts_claims_for_random_claim_sets():
    rng = random.Random(7)
    statuses = list(ClaimStatus)
    for _ in range(300):
        claims = []
        for i in range(rng.randint(1, 12)):
            s = rng.choice(statuses)
            if rng.random() < 0.2:
                claims.append(ce(i, ClaimStatus.CONTRADICTED, "80% positive.", "overall_opinion", "neutral", [FAIL], [WRONG]))
            else:
                claims.append(ce(i, s))
        e = ev(claims)
        row = rubric_for(e, stats())
        assert check_rubric_consistent(row, e) == []
        assert all(1 <= row[d] <= 5 for d in ("coverage", "consistency", "hallucination"))


@pytest.mark.skipif(not HAVE_V4, reason="judge v4 results not present")
def test_real_rubric_rows_are_consistent_and_corrections_fix_statistics():
    rub = json.loads((V4 / "rubric" / "rubric.json").read_text(encoding="utf-8"))
    assert rub["invariant_violations"] == [] and rub["llm_calls"] == 0
    rows = {r["summary_id"]: r for r in rub["rows"]}
    originals = [r for r in rub["rows"] if r["evaluated_object"] == "original"]
    assert len(originals) == 15
    # the misattributed statistics are reflected in consistency and hallucination
    for sid in ("a61df84f4901", "2fb2110754ba", "f336d1f67313"):  # B0BTY63SFY, B00DM8K1NS, B01FZJXITU
        assert rows[sid]["consistency"] <= 2 and rows[sid]["hallucination"] == 1
    for sid in ("a61df84f4901-corr2", "2fb2110754ba-corr2", "f336d1f67313-corr2"):
        assert rows[sid]["consistency"] == 5
    # the stored rubric equals a fresh derivation from the stored evaluations
    from config import load_settings
    from scripts.run_judge import load_inputs
    summaries, records = load_inputs(load_settings(), False, SOURCE)
    st = {x.product_id: Stats.from_aggregate(x.aggregate, records) for x in summaries}
    for e in load_evaluations(V4 / "evaluations.jsonl"):
        fresh = rubric_for(e, st[e.product_id])
        for d in ("faithfulness", "coverage", "consistency", "hallucination"):
            assert fresh[d] == rows[e.summary_id][d], (e.summary_id, d)


# --------------------------------------------------------------------------- single-annotator store
def mini_template(tmp_path: Path) -> Path:
    rows = [{"claim_id": f"s1:appreciated:{i}", "product_id": "P1", "summary_id": "s1", "claim_field": "appreciated",
             "claim_text": f"Claim {i} text.", "cited_review_ids": "R000000000001",
             "evidence": f"[R000000000001] (rating 5/5) Review text {i}.", "annotator": "", "label": "", "note": ""} for i in range(3)]
    rows.append({"claim_id": "s1:overall_opinion:0", "product_id": "P1", "summary_id": "s1", "claim_field": "overall_opinion",
                 "claim_text": "Mostly positive.", "cited_review_ids": "", "evidence": "STATISTICS\nTotal reviews: 3",
                 "annotator": "", "label": "", "note": ""})
    path = tmp_path / "claim_gold_template.csv"
    write_template(path, rows)
    return path


def judge_for(template_path: Path, statuses: list[str]) -> dict:
    ids = [r["claim_id"] for r in load_template(template_path)]
    return {c: {"status": s, "method": "llm" if "appreciated" in c else "deterministic", "explanation": "j"} for c, s in zip(ids, statuses)}


def test_template_export_is_frozen(tmp_path):
    path = mini_template(tmp_path)
    before = sha256_file(path)
    with pytest.raises(FileExistsError):
        write_template(path, [])
    assert sha256_file(path) == before


def test_annotation_creation_and_persistence(tmp_path):
    tpl = mini_template(tmp_path)
    store_path = tmp_path / "ann" / "d3_claim_validation.json"
    s = ClaimValidationStore(store_path, tpl)
    assert s.progress() == {"total": 4, "completed": 0, "remaining": 4, "next_unreviewed": "s1:appreciated:0", "complete": False}
    rec = s.add("s1:appreciated:0", "Partially Supported", "  only one review  ", "Asha")
    assert rec["label"] == "partially_supported" and rec["explanation"] == "only one review" and rec["interface"] == "streamlit"
    # a NEW store object (= Streamlit restarted) sees the label: persistence + resume
    s2 = ClaimValidationStore(store_path, tpl)
    assert s2.annotator() == "Asha" and set(s2.labels()) == {"s1:appreciated:0"}
    assert s2.progress()["next_unreviewed"] == "s1:appreciated:1" and s2.progress()["completed"] == 1
    data = s2.data()
    assert data["methodology"] == "single_human_annotator" and data["template_sha256"] == sha256_file(tpl)
    assert validate_store(data, load_template(tpl), sha256_file(tpl))["valid"]


def test_duplicate_prevention_and_explicit_revision(tmp_path):
    tpl = mini_template(tmp_path)
    s = ClaimValidationStore(tmp_path / "v.json", tpl)
    s.add("s1:appreciated:0", "supported", "", "Asha")
    with pytest.raises(DuplicateAnnotationError):
        s.add("s1:appreciated:0", "unsupported", "", "Asha")
    assert len(s.data()["labels"]) == 1 and s.labels()["s1:appreciated:0"]["label"] == "supported"
    s.add("s1:appreciated:0", "supported", "", "Asha", revise=True)  # unchanged revision: no history entry
    assert s.data()["revisions"] == []
    s.add("s1:appreciated:0", "unsupported", "re-read", "Asha", revise=True)
    d = s.data()
    assert [r["claim_id"] for r in d["labels"]] == ["s1:appreciated:0"] and d["labels"][0]["revision"] == 1
    assert d["revisions"][0]["label"] == "supported" and "superseded_at" in d["revisions"][0]


def test_invalid_input_is_rejected_and_nothing_written(tmp_path):
    tpl = mini_template(tmp_path)
    path = tmp_path / "v.json"
    s = ClaimValidationStore(path, tpl)
    for args in [("s1:appreciated:0", "true", "", "Asha"), ("s1:appreciated:0", "SUPPORTED!", "", "Asha"),
                 ("nope", "supported", "", "Asha"), ("s1:appreciated:0", "supported", "", ""),
                 ("s1:appreciated:0", "supported", "", "Claude"), ("s1:appreciated:0", "supported", "", "auto-labeller"),
                 ("s1:appreciated:0", "supported", "x" * 2001, "Asha")]:
        with pytest.raises(AnnotationInputError):
            s.add(*args)
    assert not path.exists()
    s.add("s1:appreciated:0", "supported", "", "Asha")
    with pytest.raises(AnnotationInputError, match="single annotator"):
        s.add("s1:appreciated:1", "supported", "", "Ravi")


def test_store_validation_detects_incomplete_duplicate_tampered_and_invalid(tmp_path):
    tpl = mini_template(tmp_path)
    tmpl, tsha = load_template(tpl), sha256_file(tpl)
    s = ClaimValidationStore(tmp_path / "v.json", tpl)
    for c in [r["claim_id"] for r in tmpl][:3]:
        s.add(c, "supported", "", "Asha")
    v = validate_store(s.data(), tmpl, tsha)
    assert v["valid"] and not v["complete"] and v["labelled"] == 3
    s.add("s1:overall_opinion:0", "contradicted", "wrong direction", "Asha")
    assert validate_store(s.data(), tmpl, tsha)["complete"]
    d = s.data()
    dup = json.loads(json.dumps(d)); dup["labels"].append(dict(dup["labels"][0]))
    assert any("duplicate" in x for x in validate_store(dup, tmpl, tsha)["problems"])
    bad = json.loads(json.dumps(d)); bad["labels"][0]["label"] = "maybe"
    assert any("invalid label" in x for x in validate_store(bad, tmpl, tsha)["problems"])
    miss = json.loads(json.dumps(d)); del miss["labels"][1]["evidence_sha256"]
    assert any("missing fields" in x for x in validate_store(miss, tmpl, tsha)["problems"])
    changed = [dict(r) for r in tmpl]; changed[0]["evidence"] = "edited evidence"
    assert any("changed since it was labelled" in x for x in validate_store(d, changed, tsha)["problems"])
    other = json.loads(json.dumps(d)); other["labels"][2]["annotator"] = "Ravi"
    assert any("differs from the store annotator" in x for x in validate_store(other, tmpl, tsha)["problems"])
    assert any("judge v4" in x for x in validate_store(d, tmpl, tsha, {"x"})["problems"])
    for bad_v in (dup, bad, miss, other):
        assert not validate_store(bad_v, tmpl, tsha)["complete"]


def test_export_matches_store_and_contains_no_judge_output(tmp_path):
    import csv
    import io
    tpl = mini_template(tmp_path)
    s = ClaimValidationStore(tmp_path / "v.json", tpl)
    s.add("s1:appreciated:2", "unsupported", "not in review", "Asha")
    s.add("s1:appreciated:0", "supported", "", "Asha")
    rows = export_rows(s.data(), s.template)
    assert [r["claim_id"] for r in rows] == ["s1:appreciated:0", "s1:appreciated:2"]  # template order
    back = list(csv.DictReader(io.StringIO(export_csv_bytes(rows).decode("utf-8-sig"))))
    store_labels = s.labels()
    assert [(r["claim_id"], r["label"], r["explanation"], r["annotator"]) for r in back] == \
        [(c, store_labels[c]["label"], store_labels[c]["explanation"], "Asha") for c in ("s1:appreciated:0", "s1:appreciated:2")]
    assert not any("judge" in k or "status" in k for k in back[0])


def test_cohen_kappa_by_hand():
    a = ["S", "S", "U", "U", "S", "C"]
    b = ["S", "U", "U", "U", "S", "S"]
    assert cohen_kappa(a, b) == round((4 / 6 - 15 / 36) / (1 - 15 / 36), 4)
    assert cohen_kappa([], []) is None


def test_comparison_only_when_complete_and_metrics_by_hand(tmp_path):
    tpl = mini_template(tmp_path)
    s = ClaimValidationStore(tmp_path / "v.json", tpl)
    judge = judge_for(tpl, ["SUPPORTED", "UNSUPPORTED", "SUPPORTED", "CONTRADICTED"])
    s.add("s1:appreciated:0", "supported", "", "Asha")
    res = compare(s.data(), s.template, judge, store_sha256="x", template_sha256=s.template_sha256)
    assert res["status"] == "incomplete" and "confusion_matrix" not in res and "exact_agreement" not in res
    s.add("s1:appreciated:1", "unsupported", "", "Asha")
    s.add("s1:appreciated:2", "unsupported", "only the title", "Asha")
    s.add("s1:overall_opinion:0", "contradicted", "", "Asha")
    res = compare(s.data(), s.template, judge, store_sha256="x", template_sha256=s.template_sha256)
    assert res["status"] == "complete" and res["claims"] == 4 and res["exact_agreement"] == 0.75
    assert res["confusion_matrix"]["matrix"]["unsupported"]["supported"] == 1
    assert sum(sum(r.values()) for r in res["confusion_matrix"]["matrix"].values()) == 4
    pc = res["per_class"]
    assert (pc["unsupported"]["precision"], pc["unsupported"]["recall"]) == (1.0, 0.5)
    assert (pc["supported"]["precision"], pc["supported"]["recall"]) == (0.5, 1.0)
    fb = res["flagged_binary"]  # human flagged: 3 claims; judge flagged 2 of them, no false positive
    assert (fb["tp"], fb["fp"], fb["fn"], fb["tn"]) == (2, 0, 1, 1) and fb["specificity"] == 1.0
    assert [d["claim_id"] for d in res["disagreements"]] == ["s1:appreciated:2"]
    assert res["disagreements"][0]["human_explanation"] == "only the title"
    assert res["by_method"]["llm"]["claims"] == 3 and res["by_method"]["deterministic"]["agreement"] == 1.0
    assert "not inter-annotator reliability" in res["interpretation"]
    assert "inter_annotator" not in json.dumps({k: v for k, v in res.items() if k not in ("interpretation", "cohen_kappa_note")})


# --------------------------------------------------------------------------- status and dashboard
def _mode_copy(tmp_path: Path, labelled: int, *, verified: bool = True, comparison: bool = True) -> Path:
    """A temporary project: real judge v4 artefacts + the real 239-claim template, synthetic TEST labels for the first
    ``labelled`` claims (annotator 'Test Person'; written only to tmp, never to the project)."""
    from scripts.claim_gold import judge_map

    mode = tmp_path / "outputs" / "live"
    dst = mode / "d3" / "judge_v4" / SOURCE
    (dst / "rubric").mkdir(parents=True)
    for f in ("evaluations.jsonl", "report.json"):
        shutil.copy(V4 / f, dst / f)
    shutil.copy(V4 / "rubric" / "rubric.json", dst / "rubric" / "rubric.json")
    ver = mode / "d3" / "completion" / SOURCE
    ver.mkdir(parents=True)
    (ver / "verification.json").write_text(json.dumps({"verify_d3_final_passed": verified, "tests_failed": 0, "tests_errors": 0,
                                                       "tests_passed": 1, "tests_skipped": 0, "recorded_at": "t"}), encoding="utf-8")
    from evaluation.d3_status import paths
    p = paths(mode, SOURCE)
    p["template"].parent.mkdir(parents=True)
    shutil.copy(TEMPLATE, p["template"])
    if labelled:
        s = ClaimValidationStore(p["store"], p["template"])
        for c in s.order[:labelled]:
            s.add(c, LABELS[hash(c) % len(LABELS)], "", "Test Person", interface="test")
        if comparison:
            p["single_dir"].mkdir(parents=True)
            res = compare(s.data(), s.template, judge_map(p["evaluations"]), store_sha256=sha256_file(p["store"]),
                          template_sha256=s.template_sha256)
            p["comparison"].write_text(json.dumps(res), encoding="utf-8")
    return mode


@pytest.mark.skipif(not HAVE_V4 or not HAVE_TEMPLATE or not (V4 / "rubric" / "rubric.json").exists(), reason="artefacts not built")
def test_status_is_green_only_when_every_criterion_passes(tmp_path):
    st = compute(_mode_copy(tmp_path / "a", 0), SOURCE, n_summaries=15)
    assert not st["complete"] and [c["id"] for c in st["criteria"] if c["status"] != "PASS"] == ["C5"]
    assert "pending: human claim validation (0/239 claims labelled)" in badge_text(st)
    st = compute(_mode_copy(tmp_path / "b", 238), SOURCE, n_summaries=15)
    assert not st["complete"] and st["human_validation"]["labelled"] == 238
    st = compute(_mode_copy(tmp_path / "c", 239, comparison=False), SOURCE, n_summaries=15)
    assert not st["complete"] and "comparison not generated" in badge_text(st)  # labels alone are not enough
    st = compute(_mode_copy(tmp_path / "d", 239, verified=False), SOURCE, n_summaries=15)
    assert not st["complete"] and [c["id"] for c in st["criteria"] if c["status"] != "PASS"] == ["C9"]
    mode = _mode_copy(tmp_path / "e", 239)
    st = compute(mode, SOURCE, n_summaries=15)
    assert st["complete"] and badge_text(st).startswith("D3 complete")
    assert not compute(mode, SOURCE, n_summaries=14)["complete"]
    # changing a label after the comparison makes the comparison stale -> not complete
    from evaluation.d3_status import paths
    p = paths(mode, SOURCE)
    s = ClaimValidationStore(p["store"], p["template"])
    first = s.order[0]
    s.add(first, next(x for x in LABELS if x != s.labels()[first]["label"]), "changed", "Test Person", revise=True)
    st = compute(mode, SOURCE, n_summaries=15)
    assert not st["complete"] and "stale" in next(c for c in st["criteria"] if c["id"] == "C5")["evidence"]


@pytest.mark.skipif(not HAVE_V4 or not HAVE_TEMPLATE, reason="artefacts not built")
def test_status_rejects_a_tampered_evaluation(tmp_path):
    mode = _mode_copy(tmp_path, 239)
    p = mode / "d3" / "judge_v4" / SOURCE / "evaluations.jsonl"
    lines = p.read_text(encoding="utf-8").splitlines()
    row = json.loads(lines[0])
    row["faithfulness_score"] = 100.0
    lines[0] = json.dumps(row)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    st = compute(mode, SOURCE, n_summaries=15)
    assert not st["complete"]
    assert next(c for c in st["criteria"] if c["id"] == "C3")["status"] != "PASS"


@pytest.mark.skipif(not HAVE_TEMPLATE, reason="claim template not exported")
def test_real_template_matches_judge_v4_and_project_has_no_fabricated_labels():
    from config import load_settings
    from evaluation.d3_status import paths
    from scripts.claim_gold import judge_map

    p = paths(load_settings().paths.mode_dir(False), SOURCE)
    tmpl = load_template(TEMPLATE)
    assert len(tmpl) == 239 and len({r["claim_id"] for r in tmpl}) == 239
    assert {r["claim_id"] for r in tmpl} == set(judge_map(p["evaluations"]))
    data = json.loads(p["store"].read_text(encoding="utf-8")) if p["store"].exists() else None
    if data:  # whatever exists must be valid single-annotator labels made through the interface or CLI
        assert validate_store(data, tmpl, sha256_file(TEMPLATE), set(judge_map(p["evaluations"])))["valid"]
        assert all(r["interface"] in ("streamlit", "cli") for r in data["labels"])


@pytest.mark.skipif(not HAVE_V4, reason="judge v4 not built")
def test_overview_dashboard_shows_v4_counts_and_honest_d3_badge(monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("REVIEWIQ_DEMO_READONLY", "true")
    monkeypatch.chdir(ROOT)
    at = AppTest.from_file(str(ROOT / "ui" / "home.py"), default_timeout=180)
    at.run()
    assert not at.exception
    html = " ".join(m.value for m in at.markdown)
    assert "D3 · Summaries + judge v4" in html and '<div class="riq-kpi-value">15 / 15</div>' in html
    from config import load_settings
    st = compute(load_settings().paths.mode_dir(False), SOURCE, n_summaries=15)
    assert badge_text(st).split(" · ")[0] in html
    assert "D1 implemented" in html and "D2 complete" in html
    assert ("D3 complete" in html) == st["complete"]


@pytest.mark.skipif(not HAVE_V4 or not (V4 / "rubric" / "rubric.json").exists(), reason="judge v4 / rubric not built")
def test_summary_evaluation_page_shows_rubric_and_completion_views(monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("REVIEWIQ_DEMO_READONLY", "true")
    monkeypatch.chdir(ROOT)
    at = AppTest.from_file(str(ROOT / "ui" / "summary_evaluation.py"), default_timeout=180)
    at.run()
    assert not at.exception
    if any(sc.key == "se_page_mode" for sc in at.segmented_control):
        at.segmented_control(key="se_page_mode").set_value("Evaluation results").run()
    at.segmented_control(key="v4_view").set_value("Rubric (1-5)").run()
    assert not at.exception
    dims = {"faithfulness", "coverage", "consistency", "hallucination"}
    df = next(d.value for d in at.dataframe if dims | {"missed themes"} <= set(d.value.columns))
    assert len(df) >= 15 and set(df["object"]) == {"original", "corrected"}
    at.segmented_control(key="v4_view").set_value("Completion status").run()
    assert not at.exception
    if (LIVE / "d3" / "completion" / SOURCE / "status.json").exists():
        crit = next(x.value for x in at.dataframe if "criterion" in x.value.columns)
        assert list(crit["#"]) == [f"C{i}" for i in range(1, 10)]


# --------------------------------------------------------------------------- Streamlit annotation interface
@pytest.fixture
def ui_project(monkeypatch, tmp_path):
    """Summary Evaluation page against a temporary project: real dataset + real template, temporary label store."""
    from dataclasses import replace

    import ui.common as common
    from config import Paths, load_settings

    if not HAVE_TEMPLATE or not HAVE_V4:
        pytest.skip("artefacts not built")
    real = load_settings()
    paths = Paths(root=real.paths.root, data_raw=real.paths.data_raw, data_processed=real.paths.data_processed,
                  annotations=tmp_path / "data" / "annotations", outputs=tmp_path / "outputs", reports=tmp_path / "reports")
    settings = replace(real, paths=paths)
    gold = settings.paths.mode_dir(False) / "d3" / "claim_gold" / SOURCE
    gold.mkdir(parents=True)
    shutil.copy(TEMPLATE, gold / "claim_gold_template.csv")
    common.get_settings.clear()
    monkeypatch.setattr(common, "get_settings", lambda: settings)
    monkeypatch.delenv("REVIEWIQ_DEMO_READONLY", raising=False)
    monkeypatch.chdir(ROOT)
    return settings


def _page():
    from streamlit.testing.v1 import AppTest
    return AppTest.from_file(str(ROOT / "ui" / "summary_evaluation.py"), default_timeout=180)


def _all_text(at) -> str:
    parts = [m.value for m in at.markdown] + [c.value for c in at.caption] + [c.value for c in at.code]
    parts += [x.value for x in at.info] + [x.value for x in at.success] + [x.value for x in at.warning] + [x.value for x in at.error]
    return "\n".join(str(p) for p in parts)


def _store(settings):
    from evaluation.d3_status import paths
    p = paths(settings.paths.mode_dir(False), SOURCE)
    return ClaimValidationStore(p["store"], p["template"]), p


def test_ui_opens_blind_and_hides_judge_output(ui_project):
    at = _page().run()
    assert not at.exception
    assert at.segmented_control(key="se_page_mode").value == "Human claim validation (blind)"
    text = _all_text(at)
    assert "Claim #1 of 239" in text and "Human claim validation" in text
    # nothing from the judge: no v4 section, no verdict words, no scores
    assert not any("judge v4" in h.value.lower() for h in at.subheader)
    store, _ = _store(ui_project)
    first = store.by_id[store.order[0]]
    assert first["claim_text"] in text
    for word in ("SUPPORTED", "CONTRADICTED", "UNSUPPORTED", "UNVERIFIABLE", "CONTRADICTED_CLAIMS", "/100", "Faithfulness score"):
        assert word not in text
    assert [m.label for m in at.metric][:3] == ["Completed", "Remaining", "Total claims"]


def test_ui_save_label_persists_and_resumes(ui_project):
    store, p = _store(ui_project)
    at = _page().run()
    at.text_input(key="hcv_annotator").input("Asha Rao")
    first = store.order[0]
    at.radio(key=f"hcv_label_{first}_False").set_value("partially_supported")
    at.text_area(key=f"hcv_expl_{first}_False").input("one review only")
    at.button(key=f"FormSubmitter:hcv_form_{first}_new-Save label and go to next").click().run()
    assert not at.exception
    saved = ClaimValidationStore(p["store"], p["template"]).labels()
    assert saved[first]["label"] == "partially_supported" and saved[first]["explanation"] == "one review only"
    assert saved[first]["annotator"] == "Asha Rao" and saved[first]["interface"] == "streamlit"
    # a fresh session (Streamlit closed and reopened) resumes at the first unreviewed claim with the name fixed
    at2 = _page().run()
    text = _all_text(at2)
    assert "Claim #2 of 239" in text
    assert at2.text_input(key="hcv_annotator_fixed").value == "Asha Rao"
    assert [m.value for m in at2.metric][:2] == ["1", "238"] or [m.value for m in at2.metric][:2] == [1, 238]


def test_ui_prevents_duplicates_and_requires_revise(ui_project):
    store, p = _store(ui_project)
    first = store.order[0]
    store.add(first, "supported", "", "Asha Rao")
    at = _page().run()
    at.toggle(key="hcv_only_open").set_value(False).run()
    at.session_state["hcv_cid"] = first
    at.run()
    text = _all_text(at)
    assert "Saved label: **supported**" in text or "Saved label: supported" in text
    assert at.radio(key=f"hcv_label_{first}_False").disabled  # locked until 'revise' is ticked
    at.checkbox(key=f"hcv_revise_{first}").check().run()
    at.radio(key=f"hcv_label_{first}_True").set_value("unsupported")
    at.button(key=f"FormSubmitter:hcv_form_{first}_rev-Save label and go to next").click().run()
    assert not at.exception
    d = ClaimValidationStore(p["store"], p["template"]).data()
    assert [r["label"] for r in d["labels"]] == ["unsupported"] and len(d["revisions"]) == 1


def test_ui_requires_a_label_and_a_name(ui_project):
    store, p = _store(ui_project)
    first = store.order[0]
    at = _page().run()
    at.button(key=f"FormSubmitter:hcv_form_{first}_new-Save label and go to next").click().run()
    assert any("Choose a label" in e.value for e in at.error)
    at.radio(key=f"hcv_label_{first}_False").set_value("supported")
    at.button(key=f"FormSubmitter:hcv_form_{first}_new-Save label and go to next").click().run()
    assert any("enter your name" in e.value for e in at.error)
    assert not p["store"].exists()


def test_ui_results_view_while_incomplete_warns_and_logs(ui_project):
    store, p = _store(ui_project)
    store.add(store.order[0], "supported", "", "Asha Rao")
    at = _page().run()
    at.segmented_control(key="se_page_mode").set_value("Evaluation results").run()
    assert not at.exception
    assert any("Judge results visible" in str(w) or "biases the validation" in w.value for w in at.warning)
    ev = ClaimValidationStore(p["store"], p["template"]).data()["events"]
    assert ev and ev[0]["kind"] == "judge_results_opened_before_complete" and ev[0]["labels_at_time"] == 1


def test_ui_readonly_mode_cannot_save(ui_project, monkeypatch):
    monkeypatch.setenv("REVIEWIQ_DEMO_READONLY", "true")
    store, p = _store(ui_project)
    at = _page().run()
    assert not at.exception
    assert at.radio(key=f"hcv_label_{store.order[0]}_False").disabled
    assert not p["store"].exists()
