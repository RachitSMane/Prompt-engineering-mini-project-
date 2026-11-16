"""Primary D2 evaluation: the three prompting strategies against the 300 blind human-created labels (no LLM calls).

Decision (project owner, 2026-11-16)
------------------------------------
* Reference = the 300 labels in ``data/annotations/human_annotations.json``: created blind (no AI prediction or
  AI-assisted label shown) by four annotators on the Human Annotation page, 2026-11-13..15. Actual allocation and
  identities are kept as recorded (they were not assigned 75 each). Labels are human-created and NOT verified by a
  second person (second-person verification is not part of the project workflow since 2026-11-15).
* This is a RE-EVALUATION OF STORED PREDICTIONS, not fresh inference: the predictions of run
  ``20261111-030912_qwen2.5-3b`` (299 reviews per strategy) plus the stored supplementary predictions for the one review
  that was not in the original run (``<run>/supplementary/``). No model is called.
* This evaluation is the PRIMARY D2 result. The original evaluation of the same run against 299 mostly AI-assisted
  reference labels (``<run>/metrics.json``) and the 300-review approved-reference sensitivity analysis
  (``<run>/approved_reference/``) are kept unchanged as historical experiments.

Nothing here modifies a label. ``validate_labels`` only reports problems; ``evaluate`` refuses to compute metrics if
the label set or the prediction coverage is not exactly right (it never "fixes" data silently).
Outputs: ``<run>/human_labelled_300/{metrics.json, report.md, reference_labels.csv, manifest.json}``.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from annotation.annotation_manager import AI_ASSISTED_MARKER
from annotation.human_annotation import HumanLabel
from evaluation.classification_metrics import INVALID, OVERALL_LABELS, evaluate_predictions
from evaluation.human_evaluation import _compare, independent_sentiment_metrics
from evaluation.prompt_comparison import load_run_predictions
from schemas.evaluation_schema import AnnotatedAspect, Annotation
from utils.readonly import ensure_writable
from utils.storage import read_json, write_json_atomic

OUT_DIR = "human_labelled_300"
EXPECTED_LABELS = 300
PRIMARY_RUN_ID = "20261111-030912_qwen2.5-3b"
EVALUATION_TYPE = ("re-evaluation of stored predictions (no fresh inference): run predictions for 299 reviews plus stored "
                   "supplementary predictions for 1 review, scored against 300 blind human-created labels")
REFERENCE_TEXT = ("300 blind human-created labels (data/annotations/human_annotations.json), 4 annotators, one label per review; "
                  "human-created, not verified by a second person")
LABEL_FIELDS = ("overall_sentiment", "aspects", "complaint_categories")
# Aspects and complaint categories were OPTIONAL fields of the blind form, so an empty field cannot be told apart from
# "not annotated". Multi-label (aspect/complaint) metrics are reported only if the human labelling actually covered
# them: at least this share of labels must carry aspects (rule written after seeing 8/300; most reviews mention an
# aspect — the historical reference labels have aspects on 290/300). Otherwise they are computed but flagged not valid.
MULTILABEL_MIN_ASPECT_SHARE = 0.5


class D2ValidationError(ValueError):
    """The human label set or the prediction coverage is not exactly right; no metrics are computed."""

    def __init__(self, problems: list[str]):
        super().__init__("; ".join(problems[:10]) + (f" (+{len(problems) - 10} more)" if len(problems) > 10 else ""))
        self.problems = problems


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------------------------------------- labels
def validate_labels(store_path: Path, sample_path: Path, expected: int = EXPECTED_LABELS) -> dict:
    """Check the human label store against the annotation sample. Read-only; returns problems instead of fixing."""
    problems: list[str] = []
    raw = read_json(store_path, default=None)
    sample = pd.read_csv(sample_path, dtype=str)
    sample_ids = sample["review_id"].tolist()
    if len(sample_ids) != expected:
        problems.append(f"annotation sample has {len(sample_ids)} reviews, expected {expected}")
    dup_sample = sorted(k for k, c in Counter(sample_ids).items() if c > 1)
    if dup_sample:
        problems.append(f"duplicate review ids in the annotation sample: {dup_sample[:5]}")
    if not raw or not isinstance(raw.get("labels"), dict):
        return {"valid": False, "problems": problems + ["no human label store"], "n_labels": 0}
    labels: dict = raw["labels"]
    product_of = dict(zip(sample["review_id"], sample["product_id"]))
    parsed: dict[str, HumanLabel] = {}
    for key, rec in labels.items():
        try:
            lab = HumanLabel.model_validate(rec)
        except Exception as exc:  # pydantic error text names the field
            problems.append(f"{key}: invalid record ({str(exc).splitlines()[0]})")
            continue
        if lab.review_id != key:
            problems.append(f"{key}: stored under a different review id ({lab.review_id})")
        if lab.status != "completed":
            problems.append(f"{key}: status {lab.status!r}, expected 'completed'")
        if lab.provenance != "human_created" or not lab.blind:
            problems.append(f"{key}: provenance {lab.provenance!r} / blind={lab.blind}, expected blind human_created")
        if lab.overall_sentiment is None or lab.overall_sentiment.value not in OVERALL_LABELS:
            problems.append(f"{key}: missing or invalid overall sentiment")
        if lab.notes.strip().startswith(AI_ASSISTED_MARKER):
            problems.append(f"{key}: notes carry the AI-assisted marker")
        if not lab.annotator.strip():
            problems.append(f"{key}: no annotator")
        if not lab.history:
            problems.append(f"{key}: no history")
        elif any((h.get("by") or "") != lab.annotator for h in lab.history):
            problems.append(f"{key}: history contains an action by someone other than the annotator")
        if key in product_of and product_of[key] != lab.product_id:
            problems.append(f"{key}: product id {lab.product_id} differs from the sample ({product_of[key]})")
        parsed[key] = lab
    missing = [r for r in sample_ids if r not in labels]
    extra = sorted(set(labels) - set(sample_ids))
    if missing:
        problems.append(f"{len(missing)} sample review(s) without a human label: {missing[:5]}")
    if extra:
        problems.append(f"{len(extra)} label(s) for reviews outside the sample: {extra[:5]}")
    if len(labels) != expected:
        problems.append(f"{len(labels)} labels, expected exactly {expected}")
    if raw.get("reserve_review_ids"):
        problems.append(f"reserve reviews present ({len(raw['reserve_review_ids'])}); the 300-review set is not the original sample")
    if raw.get("meta", {}).get("count") not in (None, len(labels)):
        problems.append(f"store meta count {raw['meta']['count']} != {len(labels)} labels")
    labs = list(parsed.values())
    return {
        "valid": not problems,
        "problems": problems,
        "n_labels": len(labels),
        "n_sample": len(sample_ids),
        "annotators": dict(Counter(x.annotator for x in labs).most_common()),
        "provenance": dict(Counter(x.provenance for x in labs)),
        "blind": sum(x.blind for x in labs),
        "status": dict(Counter(x.status for x in labs)),
        "overall_sentiment": {c: sum(1 for x in labs if x.overall_sentiment and x.overall_sentiment.value == c) for c in OVERALL_LABELS},
        "with_aspects": sum(1 for x in labs if x.aspects),
        "with_complaints": sum(1 for x in labs if x.complaint_categories),
        "relabelled_by_own_annotator": sorted(x.review_id for x in labs if any(h.get("action") == "relabelled" for h in x.history)),
        "labelled_between": [min((x.created_at for x in labs), default=None), max((x.updated_at for x in labs), default=None)],
        "store_sha256": sha256_file(store_path),
        "sample_sha256": sha256_file(sample_path),
    }


def human_gold(store_path: Path, sample_path: Path) -> list[Annotation]:
    """The 300 human labels as metric-ready ``Annotation`` objects, in sample order. Content is copied unchanged."""
    labels = read_json(store_path)["labels"]
    order = pd.read_csv(sample_path, dtype=str)["review_id"].tolist()
    out = []
    for rid in order:
        lab = HumanLabel.model_validate(labels[rid])
        out.append(Annotation(
            review_id=lab.review_id, product_id=lab.product_id, overall_sentiment=lab.overall_sentiment,
            aspects=[AnnotatedAspect(aspect_category=a.aspect_category, sentiment=a.sentiment) for a in lab.aspects],
            complaint_categories=lab.complaint_categories, annotator=lab.annotator,
            notes="human_created (blind); not second-person verified", status="complete", annotated_at=lab.updated_at))
    return out


# ---------------------------------------------------------------------------------------------------------- predictions
def _prediction_files(run_dir: Path, strategy: str) -> list[Path]:
    files = [run_dir / f"predictions_{strategy}.jsonl"]
    sup = run_dir / "supplementary" / f"predictions_{strategy}.jsonl"
    return files + ([sup] if sup.exists() else [])


def _raw_ids(path: Path) -> list[str]:
    ids = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rec = json.loads(line)
                ids.append(rec["record"]["review_id"])
    return ids


def prediction_coverage(run_dir: Path, review_ids: list[str]) -> dict:
    """Per strategy: which stored predictions (primary + supplementary) exist for the reference reviews."""
    config = read_json(run_dir / "config.json")
    want = set(review_ids)
    out: dict = {"strategies": {}, "problems": []}
    sets = {}
    for name in config["strategies"]:
        files = _prediction_files(run_dir, name)
        per_file = {str(f.relative_to(run_dir)).replace("\\", "/"): _raw_ids(f) for f in files}
        all_ids = [i for ids in per_file.values() for i in ids]
        dups = sorted(k for k, c in Counter(all_ids).items() if c > 1)
        recs, fails = load_run_predictions(run_dir, name)
        if len(files) > 1:
            r2, f2 = load_run_predictions(files[1].parent, name)
            recs, fails = recs + r2, fails + f2
        loaded = {o.review_id for o in recs} | {f.review_id for f in fails}
        missing = sorted(want - loaded)
        extra = sorted(set(all_ids) - want)
        if dups:
            out["problems"].append(f"{name}: duplicate prediction rows for {dups[:5]}")
        if missing:
            out["problems"].append(f"{name}: no usable prediction for {len(missing)} reference review(s): {missing[:5]}")
        if extra:
            out["problems"].append(f"{name}: predictions for {len(extra)} review(s) outside the reference set: {extra[:5]}")
        sets[name] = loaded & want
        out["strategies"][name] = {
            "files": {k: {"rows": len(v), "sha256": sha256_file(run_dir / k)} for k, v in per_file.items()},
            "valid": len({o.review_id for o in recs} & want), "failed": len({f.review_id for f in fails} & want),
            "missing": missing, "duplicates": dups,
        }
    same = len({frozenset(v) for v in sets.values()}) == 1 and all(v == want for v in sets.values())
    if not same:
        out["problems"].append("the strategies do not cover exactly the same reference reviews")
    out["same_review_set_for_all_strategies"] = same
    out["config_sha256"] = sha256_file(run_dir / "config.json")
    return out


def _merged_predictions(run_dir: Path, strategy: str, ids: set[str]) -> dict:
    files = _prediction_files(run_dir, strategy)
    recs, fails = load_run_predictions(run_dir, strategy)
    if len(files) > 1:
        r2, f2 = load_run_predictions(files[1].parent, strategy)
        recs, fails = recs + r2, fails + f2
    preds = {r.review_id: r for r in recs} | {f.review_id: None for f in fails}
    return {rid: p for rid, p in preds.items() if rid in ids}


# ---------------------------------------------------------------------------------------------------------- evaluation
def _content_sha(res: dict) -> str:
    body = {k: v for k, v in res.items() if k not in ("evaluated_at", "content_sha256")}
    return hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _historical_agreement(gold: list[Annotation], legacy_path: Path) -> dict | None:
    """How the human labels relate to the historical (mostly AI-assisted) reference labels. Informational only."""
    raw = read_json(legacy_path, default=None)
    if not raw or "annotations" not in raw:
        return None
    leg = raw["annotations"]
    pairs = [(g.overall_sentiment.value, leg[g.review_id]["overall_sentiment"]) for g in gold if g.review_id in leg]
    return {
        "n": len(pairs), "same_overall_sentiment": sum(a == b for a, b in pairs),
        "agreement": round(sum(a == b for a, b in pairs) / len(pairs), 4) if pairs else None,
        "labels": OVERALL_LABELS,
        "matrix_rows_human_cols_historical": [[sum(a == r and b == c for a, b in pairs) for c in OVERALL_LABELS] for r in OVERALL_LABELS],
        "historical_distribution": {c: sum(b == c for _, b in pairs) for c in OVERALL_LABELS},
    }


def label_quality_signals(gold: list[Annotation], sample_path: Path, few_shot: dict | None) -> dict:
    """Read-only signals that some labels may be wrong (labels are never changed). Star ratings were not shown as a
    labelling rule; a label that contradicts the rating is not necessarily wrong, but many such cases deserve a review."""
    rating = {r: float(x) for r, x in pd.read_csv(sample_path, dtype=str).set_index("review_id")["rating"].items()}
    lab = {g.review_id: g.overall_sentiment.value for g in gold}
    pos_low = sorted(r for r, v in lab.items() if v == "positive" and rating[r] <= 2)
    neg_high = sorted(r for r, v in lab.items() if v == "negative" and rating[r] >= 4)
    out = {"positive_label_on_1_2_star": len(pos_low), "negative_label_on_4_5_star": len(neg_high),
           "label_by_rating": {c: {str(int(k)): sum(1 for r, v in lab.items() if v == c and rating[r] == k) for k in (1, 2, 3, 4, 5)}
                               for c in OVERALL_LABELS},
           "review_ids_positive_on_1_2_star": pos_low,
           "note": "signals only; no label was changed. A human review of these reviews is recommended before relying on the result."}
    if few_shot is not None:
        odd = sorted(r for r, v in lab.items() if v == "positive" and few_shot.get(r) == "negative")
        out["human_positive_few_shot_negative"] = len(odd)
        out["human_positive_few_shot_negative_on_1_2_star"] = sum(rating[r] <= 2 for r in odd)
    return out


def evaluate(run_dir: Path, store_path: Path, sample_path: Path, *, legacy_path: Path | None = None, write: bool = True) -> dict:
    """Validate, then score every strategy on exactly the same 300 reviews. Raises ``D2ValidationError`` on any problem."""
    val = validate_labels(store_path, sample_path)
    if not val["valid"]:
        raise D2ValidationError(val["problems"])
    gold = human_gold(store_path, sample_path)
    ids = [g.review_id for g in gold]
    cov = prediction_coverage(run_dir, ids)
    if cov["problems"]:
        raise D2ValidationError(cov["problems"])
    config = read_json(run_dir / "config.json")
    in_run = set(config["review_ids"])
    res: dict = {
        "deliverable": "D2", "role": "primary",
        "run_id": config["run_id"], "model_name": config["model_name"], "provider": config["provider"],
        "prompt_version": config["prompt_version"], "is_mock": config["is_mock"],
        "evaluated_at": _now(),
        "evaluation_type": EVALUATION_TYPE, "fresh_inference": False,
        "reference": REFERENCE_TEXT,
        "reference_validation": {k: v for k, v in val.items() if k != "problems"},
        "n_reference": len(gold), "n_evaluated": len(ids),
        "evaluated_review_ids": ids,
        "reviews_from_supplementary_predictions": sorted(set(ids) - in_run),
        "same_review_set_for_all_strategies": cov["same_review_set_for_all_strategies"],
        "prediction_coverage": {n: {k: v for k, v in e.items() if k != "missing"} for n, e in cov["strategies"].items()},
        "config_sha256": cov["config_sha256"],
        "failure_policy": "controlled extraction failures count as wrong predictions (label 'invalid')",
        "strategies": {}, "independent_check": {}, "by_annotator": {},
    }
    problems_all: list[str] = []
    idset = set(ids)
    y_true = [g.overall_sentiment.value for g in gold]
    for name in config["strategies"]:
        preds = _merged_predictions(run_dir, name, idset)
        cl = evaluate_predictions(gold, preds)
        valid = sum(1 for p in preds.values() if p is not None)
        res["strategies"][name] = {"classification": cl, "valid_predictions": valid, "invalid_predictions": len(preds) - valid,
                                   "failure_rate": round((len(preds) - valid) / len(preds), 4)}
        y_pred = [preds[g.review_id].overall_sentiment.value if preds[g.review_id] else INVALID for g in gold]
        ind = independent_sentiment_metrics(y_true, y_pred)
        probs = _compare(cl["overall_sentiment"], ind)
        res["independent_check"][name] = {"macro_f1": round(ind["macro_f1"], 4), "accuracy": round(ind["accuracy"], 4),
                                          "matches": not probs, "problems": probs}
        problems_all += [f"{name}: {p}" for p in probs]
        per_ann = {}
        for ann in val["annotators"]:
            idx = [i for i, g in enumerate(gold) if g.annotator == ann]
            m = independent_sentiment_metrics([y_true[i] for i in idx], [y_pred[i] for i in idx])
            per_ann[ann] = {"n": len(idx), "accuracy": round(m["accuracy"], 4), "macro_f1": round(m["macro_f1"], 4)}
        res["by_annotator"][name] = per_ann
    res["independent_check_passed"] = not problems_all
    share = val["with_aspects"] / val["n_labels"]
    res["multilabel_validity"] = {
        "valid": share >= MULTILABEL_MIN_ASPECT_SHARE,
        "labels_with_aspects": val["with_aspects"], "labels_with_complaints": val["with_complaints"], "n_labels": val["n_labels"],
        "rule": f"aspect/complaint metrics are valid only if >= {MULTILABEL_MIN_ASPECT_SHARE:.0%} of the human labels carry aspects",
        "reason": None if share >= MULTILABEL_MIN_ASPECT_SHARE else (
            f"only {val['with_aspects']}/{val['n_labels']} human labels carry aspects and {val['with_complaints']}/{val['n_labels']} carry "
            "complaint categories; these fields were optional in the blind form, so empty sets mostly mean 'not annotated', not 'none'. "
            "The multi-label numbers in 'classification' are kept for transparency but are NOT valid results."),
    }
    fs = _merged_predictions(run_dir, "few_shot", idset) if "few_shot" in config["strategies"] else None
    res["label_quality_signals"] = label_quality_signals(
        gold, sample_path, {r: p.overall_sentiment.value for r, p in fs.items() if p} if fs is not None else None)
    res["historical_label_agreement"] = _historical_agreement(gold, legacy_path) if legacy_path else None
    res["status"] = "complete" if res["independent_check_passed"] else "cross_check_failed"
    res["content_sha256"] = _content_sha(res)
    if write:
        write_outputs(run_dir, res, store_path, sample_path)
    return res


def reference_rows(store_path: Path, sample_path: Path) -> list[dict]:
    labels = read_json(store_path)["labels"]
    rows = []
    for rid in pd.read_csv(sample_path, dtype=str)["review_id"]:
        lab = HumanLabel.model_validate(labels[rid])
        rows.append({"review_id": rid, "product_id": lab.product_id, "overall_sentiment": lab.overall_sentiment.value,
                     "aspects": "; ".join(f"{a.aspect_category.value}:{a.sentiment.value}" for a in lab.aspects),
                     "complaint_categories": "; ".join(c.value for c in lab.complaint_categories),
                     "annotator": lab.annotator, "provenance": lab.provenance, "blind": lab.blind, "status": lab.status,
                     "second_person_verified": False, "created_at": lab.created_at, "updated_at": lab.updated_at,
                     "history_actions": " > ".join(h.get("action", "") for h in lab.history)})
    return rows


def write_outputs(run_dir: Path, res: dict, store_path: Path, sample_path: Path) -> Path:
    out = run_dir / OUT_DIR
    ensure_writable(out / "metrics.json")
    out.mkdir(parents=True, exist_ok=True)
    write_json_atomic(out / "metrics.json", res)
    (out / "report.md").write_text(render_report(res), encoding="utf-8")
    buf = io.StringIO()
    rows = reference_rows(store_path, sample_path)
    w = csv.DictWriter(buf, fieldnames=list(rows[0]), lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    (out / "reference_labels.csv").write_text(buf.getvalue(), encoding="utf-8")
    write_json_atomic(out / "manifest.json", {
        "created_at": res["evaluated_at"], "evaluation_type": EVALUATION_TYPE, "fresh_inference": False,
        "inputs": {"human_annotations.json": res["reference_validation"]["store_sha256"],
                   "annotation_sample.csv": res["reference_validation"]["sample_sha256"],
                   "config.json": res["config_sha256"],
                   **{f: d["sha256"] for e in res["prediction_coverage"].values() for f, d in e["files"].items()}},
        "outputs": {"metrics.json content_sha256": res["content_sha256"],
                    "reference_labels.csv": sha256_file(out / "reference_labels.csv")},
        "unchanged_historical": ["metrics.json", "report.md", "approved_reference/", "human_reference/", "predictions_*.jsonl",
                                 "supplementary/", "data/annotations/annotations.json", "reports/D2_FINAL_REPORT.md"],
    })
    return out


def stored_inputs_match(run_dir: Path, store_path: Path, sample_path: Path) -> dict:
    """Is the saved evaluation still based on exactly the current label store, sample and prediction files?"""
    man = read_json(run_dir / OUT_DIR / "manifest.json", default=None)
    if not man:
        return {"present": False, "current": False, "changed": []}
    current = {"human_annotations.json": store_path, "annotation_sample.csv": sample_path, "config.json": run_dir / "config.json"}
    changed = [k for k, p in current.items() if not p.exists() or sha256_file(p) != man["inputs"].get(k)]
    changed += [k for k, v in man["inputs"].items() if k not in current and (not (run_dir / k).exists() or sha256_file(run_dir / k) != v)]
    return {"present": True, "current": not changed, "changed": changed}


# ---------------------------------------------------------------------------------------------------------- report
def _f(v) -> str:
    return "–" if v is None else f"{v:.4f}" if isinstance(v, float) else str(v)


def render_report(m: dict) -> str:
    v = m["reference_validation"]
    L = [f"# D2 primary evaluation — 300 blind human-created labels — run `{m['run_id']}`", "",
         f"**Status:** {'complete' if m['status'] == 'complete' else m['status']} · evaluated {m['evaluated_at']} · "
         f"model `{m['model_name']}` ({m['provider']}), prompt version `{m['prompt_version']}`.", "",
         f"**Evaluation type:** {m['evaluation_type']}. No model was called for this evaluation.", "",
         f"**Reference:** {m['reference']}. Annotators (labels): "
         + ", ".join(f"{a} ({n})" for a, n in v["annotators"].items())
         + f". Provenance: {v['provenance']}; blind {v['blind']}/{v['n_labels']}. Labelled {v['labelled_between'][0]} – {v['labelled_between'][1]}.", "",
         f"**Same reviews for every strategy:** {m['same_review_set_for_all_strategies']} ({m['n_evaluated']} reviews; "
         f"from supplementary predictions: {', '.join(m['reviews_from_supplementary_predictions']) or 'none'}). {m['failure_policy'].capitalize()}.", "",
         "## Overall sentiment (4 classes) — primary metric: macro F1", "",
         "| strategy | n | valid | failed | accuracy | macro P | macro R | macro F1 | weighted F1 | independent check |",
         "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for n, e in m["strategies"].items():
        s = e["classification"]["overall_sentiment"]
        L.append(f"| {n} | {s['n']} | {e['valid_predictions']} | {e['invalid_predictions']} | {_f(s['accuracy'])} | {_f(s['macro_precision'])} | "
                 f"{_f(s['macro_recall'])} | {_f(s['macro_f1'])} | {_f(s['weighted_f1'])} | {'match' if m['independent_check'][n]['matches'] else 'MISMATCH'} |")
    L += ["", f"Reference class distribution: {v['overall_sentiment']}.", ""]
    for n, e in m["strategies"].items():
        s = e["classification"]["overall_sentiment"]
        L += [f"### {n}", "", "| class | precision | recall | F1 | support |", "| --- | --- | --- | --- | --- |"]
        L += [f"| {c} | {_f(x['precision'])} | {_f(x['recall'])} | {_f(x['f1-score'])} | {x['support']} |" for c, x in s["per_class"].items()]
        L += ["", f"Confusion matrix (rows = human label, columns = prediction, order {s['labels']}; failed extractions appear in no column):", "",
              "| human \\ predicted | " + " | ".join(s["labels"]) + " |", "| --- |" + " --- |" * len(s["labels"])]
        L += [f"| {lab} | " + " | ".join(str(x) for x in row) + " |" for lab, row in zip(s["labels"], s["confusion_matrix"])]
        L.append("")
    mv = m["multilabel_validity"]
    L += ["## Complaints (multi-label) and aspects", "",
          f"Human labels with at least one aspect: {v['with_aspects']}/{v['n_labels']}; with at least one complaint category: "
          f"{v['with_complaints']}/{v['n_labels']}. Rule: {mv['rule']}.", ""]
    if not mv["valid"]:
        L += [f"**Not reported — not valid for this reference.** {mv['reason']} Multi-label results exist only for the historical "
              "evaluation against the AI-assisted reference labels (run metrics.json).", ""]
    else:
        L += ["| strategy | complaint micro F1 | complaint macro F1 (present labels) | complaint Hamming loss | aspect detection micro F1 | aspect sentiment macro F1 | aspect joint micro F1 |",
          "| --- | --- | --- | --- | --- | --- | --- |"]
    for n, e in (m["strategies"].items() if mv["valid"] else []):
        c = e["classification"]
        L.append(f"| {n} | {_f(c['complaints'].get('micro_f1'))} | {_f(c['complaints'].get('macro_f1_present_labels'))} | "
                 f"{_f(c['complaints'].get('hamming_loss'))} | {_f(c['aspect_detection'].get('micro_f1'))} | "
                 f"{_f(c['aspect_sentiment'].get('macro_f1'))} | {_f(c['aspect_joint'].get('micro_f1'))} |")
    L += ["", "## Per annotator (informational; small subsets, not a reliability measure)", "",
          "| strategy | " + " | ".join(f"{a} (n={x['n']}) acc / macro F1" for a, x in next(iter(m["by_annotator"].values())).items()) + " |",
          "| --- |" + " --- |" * len(next(iter(m["by_annotator"].values())))]
    for n, per in m["by_annotator"].items():
        L.append(f"| {n} | " + " | ".join(f"{_f(x['accuracy'])} / {_f(x['macro_f1'])}" for x in per.values()) + " |")
    q = m.get("label_quality_signals")
    if q:
        L += ["", "## Label quality signals (no label was changed)", "",
              f"* {q['positive_label_on_1_2_star']} reviews rated 1–2 stars carry the human label *positive*; "
              f"{q['negative_label_on_4_5_star']} reviews rated 4–5 stars carry *negative*.",
              f"* {q.get('human_positive_few_shot_negative', '–')} reviews labelled *positive* by a human were predicted *negative* by few-shot; "
              f"{q.get('human_positive_few_shot_negative_on_1_2_star', '–')} of them are rated 1–2 stars.",
              "* Several of these read as clearly negative (e.g. a 2-star review that only says coffee grounds end up in the coffee). "
              "The scores above therefore partly measure label noise. A review of these labels by the annotators is recommended "
              "before the result is treated as final; any change must be made by a human on the Human Annotation page and the "
              "evaluation rerun (`python scripts/d2_human_eval.py evaluate`).", "",
              "Human label by star rating (1–5):", "", "| label | 1 | 2 | 3 | 4 | 5 |", "| --- | --- | --- | --- | --- | --- |"]
        L += [f"| {c} | " + " | ".join(str(x) for x in row.values()) + " |" for c, row in q["label_by_rating"].items()]
    h = m.get("historical_label_agreement")
    if h:
        L += ["", "## Relation to the historical (mostly AI-assisted) reference labels", "",
              f"The human label equals the historical reference label for {h['same_overall_sentiment']} of {h['n']} reviews "
              f"({_f(h['agreement'])}). Historical distribution: {h['historical_distribution']}; human distribution: {v['overall_sentiment']}. "
              "Because the reference changed this much, the scores above are not comparable with the historical 299-review "
              "evaluation (macro F1 0.4380 / 0.6025 / 0.4232), which is kept unchanged in the run's metrics.json."]
    L += ["", "## Limitations", "",
          "* One label per review by one of four annotators; no second-person verification and no inter-annotator agreement was measured.",
          "* Annotators labelled different subsets of different sizes (" + "/".join(str(n) for n in v["annotators"].values()) + "); the allocation was not balanced.",
          "* Stored predictions were re-scored; the predictions for the supplementary review were produced later (2026-11-14) with the same model, prompt version and settings.",
          "* Sentiment and star rating disagree for some reviews (e.g. positive labels on 1–2-star reviews); labels were not changed.",
          f"* Content hash of this result: `{m['content_sha256']}`."]
    return "\n".join(L) + "\n"
