"""D2 evaluation of all three strategies against the APPROVED reference annotations (no LLM calls).

Reference = the existing annotations accepted on the Human Annotation page (``annotation/reference_adjudication.py``):
mostly AI-assisted labels with recorded approval, NOT independent human annotation. The provenance counts are written
into every result. Results go to ``<run>/approved_reference/{metrics.json,report.md}``; the run's original
``metrics.json`` and the human-label evaluation (``<run>/human_reference/``) are never touched.

Predictions are read from the run's ``predictions_<strategy>.jsonl``. Reviews that were not in the original run (for
example a formerly flagged review accepted later) can be predicted separately into
``<run>/supplementary/predictions_<strategy>.jsonl``; those rows are merged and counted, and the original prediction
files stay byte-identical. Every strategy is scored on exactly the same review ids; failures count as wrong.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from evaluation.classification_metrics import INVALID, evaluate_predictions
from evaluation.human_evaluation import _compare, independent_sentiment_metrics
from evaluation.prompt_comparison import load_run_predictions
from schemas.evaluation_schema import Annotation
from utils.readonly import ensure_writable
from utils.storage import read_json, write_json_atomic

OUT_DIR = "approved_reference"
STATUS_TEXT = {
    "awaiting_approval": "Awaiting approval — no approved reference annotation yet",
    "incomplete_reference": "Reference dataset not ready — not every reference record has been reviewed and accepted",
    "needs_inference": "Some accepted references have no prediction from every strategy",
    "complete": "Complete — all three strategies evaluated on the same accepted reference set",
}


def _predictions(run_dir: Path, strategy: str):
    recs, fails = load_run_predictions(run_dir, strategy)
    sup = run_dir / "supplementary"
    if (sup / f"predictions_{strategy}.jsonl").exists():
        r2, f2 = load_run_predictions(sup, strategy)
        have = {o.review_id for o in recs + fails}
        recs = recs + [r for r in r2 if r.review_id not in have]
        fails = fails + [f for f in f2 if f.review_id not in have]
    return recs, fails


def evaluate_reference(run_dir: Path, gold: list[Annotation], *, progress: dict, provenance: dict, write: bool = True) -> dict:
    config = read_json(run_dir / "config.json")
    strategies = config["strategies"]
    loaded = {n: _predictions(run_dir, n) for n in strategies}
    predicted = {n: {r.review_id for r in recs} | {f.review_id for f in fails} for n, (recs, fails) in loaded.items()}
    gold_ids = [g.review_id for g in gold]
    common = [rid for rid in gold_ids if all(rid in predicted[n] for n in strategies)]
    needs = {n: sorted(set(gold_ids) - predicted[n]) for n in strategies}
    gold_common = [g for g in gold if g.review_id in set(common)]
    if not gold:
        status = "awaiting_approval"
    elif not progress.get("ready"):
        status = "incomplete_reference"
    elif any(needs.values()):
        status = "needs_inference"
    else:
        status = "complete"
    res: dict = {
        "run_id": config["run_id"], "model_name": config["model_name"], "is_mock": config["is_mock"],
        "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "reference": "approved existing reference annotations (data/annotations/annotations.json + reference_adjudication.json); "
                     "mostly AI-assisted labels with a recorded approval decision — NOT independent human ground truth",
        "reference_provenance": provenance, "reference_progress": {k: progress[k] for k in
                                                                    ("available", "accepted", "approved", "modified", "excluded", "awaiting_review", "required", "ready", "reviewers")},
        "n_reference": len(gold), "n_evaluated_common": len(gold_common), "evaluated_review_ids": common,
        "same_review_set_for_all_strategies": True, "needs_inference": needs, "status": status,
        "strategies": {}, "independent_check": {},
    }
    problems_all = []
    for n in strategies:
        recs, fails = loaded[n]
        preds = {r.review_id: r for r in recs} | {f.review_id: None for f in fails}
        preds = {rid: p for rid, p in preds.items() if rid in set(common)}
        cl = evaluate_predictions(gold_common, preds) if gold_common else {"n_evaluated": 0}
        valid = sum(1 for p in preds.values() if p is not None)
        res["strategies"][n] = {"classification": cl, "valid_predictions": valid, "invalid_predictions": len(preds) - valid,
                                "failure_rate": round((len(preds) - valid) / len(preds), 4) if preds else None}
        if gold_common:
            y_true = [g.overall_sentiment.value for g in gold_common]
            y_pred = [preds[g.review_id].overall_sentiment.value if preds[g.review_id] else INVALID for g in gold_common]
            ind = independent_sentiment_metrics(y_true, y_pred)
            probs = _compare(cl["overall_sentiment"], ind)
            res["independent_check"][n] = {"macro_f1": round(ind["macro_f1"], 4), "accuracy": round(ind["accuracy"], 4),
                                           "matches": not probs, "problems": probs}
            problems_all += [f"{n}: {p}" for p in probs]
    res["independent_check_passed"] = bool(gold_common) and not problems_all
    if status == "complete" and not res["independent_check_passed"]:
        res["status"] = "needs_inference"  # never report complete when the cross-check fails
    if write:
        out = run_dir / OUT_DIR
        ensure_writable(out)
        out.mkdir(parents=True, exist_ok=True)
        write_json_atomic(out / "metrics.json", res)
        (out / "report.md").write_text(render_reference_report(res), encoding="utf-8")
    return res


def render_reference_report(m: dict) -> str:
    p = m["reference_progress"]
    lines = [f"# D2 evaluation against the approved reference annotations — run `{m['run_id']}`", "",
             f"**Status:** {STATUS_TEXT[m['status']]}.", "",
             f"**Reference:** {m['reference']}.", "",
             f"Accepted {p['accepted']} of {p['available']} (approved {p['approved']}, modified {p['modified']}, excluded {p['excluded']}, "
             f"awaiting review {p['awaiting_review']}); reviewer(s): {', '.join(p['reviewers']) or '–'}. "
             f"Original provenance of the accepted labels: `{json.dumps(m['reference_provenance'])}`.", "",
             "All strategies are scored on exactly the same reviews; controlled failures count as wrong predictions.", ""]
    if any(m["needs_inference"].values()):
        lines += ["**Accepted references without a prediction from every strategy:** " +
                  "; ".join(f"{n}: {', '.join(ids)}" for n, ids in m["needs_inference"].items() if ids), ""]
    if m["n_evaluated_common"]:
        lines += ["| strategy | n | valid | invalid | failure rate | accuracy | macro P | macro R | macro F1 | weighted F1 | independent check |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for n, e in m["strategies"].items():
            s = e["classification"]["overall_sentiment"]
            ic = m["independent_check"].get(n, {})
            lines.append(f"| {n} | {s['n']} | {e['valid_predictions']} | {e['invalid_predictions']} | {e['failure_rate']} | {s['accuracy']} | "
                         f"{s['macro_precision']} | {s['macro_recall']} | {s['macro_f1']} | {s['weighted_f1']} | {'match' if ic.get('matches') else 'MISMATCH'} |")
        for n, e in m["strategies"].items():
            s = e["classification"]["overall_sentiment"]
            lines += ["", f"### {n}", "", "| class | precision | recall | F1 | support |", "| --- | --- | --- | --- | --- |"]
            lines += [f"| {c} | {v['precision']} | {v['recall']} | {v['f1-score']} | {v['support']} |" for c, v in s["per_class"].items()]
            lines += ["", f"Confusion matrix (rows = reference, columns = prediction, order {s['labels']}; failures in no column): `{s['confusion_matrix']}`"]
    else:
        lines.append("_No metrics yet: no approved reference annotation overlaps the run._")
    return "\n".join(lines) + "\n"
