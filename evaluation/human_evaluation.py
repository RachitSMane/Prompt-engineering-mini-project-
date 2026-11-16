"""D2 evaluation against the genuine human reference set (``annotation.human_annotation``).

* All three strategies are scored on the SAME review set: verified human labels whose review has a stored prediction
  (valid or controlled failure) from EVERY strategy of the run. Labelled reviews that lack a prediction from any
  strategy are listed as ``needs_inference`` and excluded from all strategies alike.
* Controlled failures count as wrong predictions (documented policy, same as the original evaluation).
* Results are written to ``<run_dir>/human_reference/`` (``metrics.json``, ``report.md``); the original
  ``metrics.json`` computed against the AI-assisted reference labels is never touched.
* Every overall-sentiment metric is recomputed independently with the standard library (``independent_sentiment_metrics``)
  and compared with the scikit-learn based numbers; any disagreement is reported.
* Status is ``complete`` only when >= 300 verified human labels are evaluated and no prediction is missing.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from annotation.human_annotation import REQUIRED_HUMAN_LABELS
from evaluation.classification_metrics import INVALID, OVERALL_LABELS, evaluate_predictions
from evaluation.prompt_comparison import load_run_predictions
from schemas.evaluation_schema import Annotation
from utils.readonly import ensure_writable
from utils.storage import read_json, write_json_atomic

STATUS_TEXT = {
    "awaiting_human_labels": "Awaiting human labels — no verified human label overlaps this run yet",
    "provisional": f"Provisional — fewer than {REQUIRED_HUMAN_LABELS} verified human labels; NOT the D2 result",
    "incomplete": "Incomplete — some human-labelled reviews have no prediction from every strategy; run inference for them first",
    "complete": f"Complete — all strategies evaluated on the same >= {REQUIRED_HUMAN_LABELS} verified human labels",
}


def independent_sentiment_metrics(y_true: list[str], y_pred: list[str], classes: list[str] = OVERALL_LABELS) -> dict:
    """Accuracy, per-class P/R/F1, macro P/R/F1 and confusion matrix with plain Python (no scikit-learn)."""
    n = len(y_true)
    per = {}
    for c in classes:
        tp = sum(t == c and p == c for t, p in zip(y_true, y_pred))
        fp = sum(t != c and p == c for t, p in zip(y_true, y_pred))
        fn = sum(t == c and p != c for t, p in zip(y_true, y_pred))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        per[c] = {"precision": prec, "recall": rec, "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0, "support": tp + fn}
    present = [c for c in classes if c in set(y_true) | set(y_pred)]
    avg = lambda k: sum(per[c][k] for c in present) / len(present) if present else 0.0  # noqa: E731
    return {
        "n": n,
        "accuracy": sum(t == p for t, p in zip(y_true, y_pred)) / n if n else 0.0,
        "macro_precision": avg("precision"), "macro_recall": avg("recall"), "macro_f1": avg("f1"),
        "per_class": per,
        "confusion_matrix": [[sum(t == a and p == b for t, p in zip(y_true, y_pred)) for b in classes] for a in classes],
        "invalid_predictions": sum(p == INVALID for p in y_pred),
    }


def _compare(sk: dict, ind: dict, tol: float = 1e-4) -> list[str]:
    problems = []
    if not sk.get("n"):
        return problems
    for k in ("accuracy", "macro_precision", "macro_recall", "macro_f1"):
        if abs(round(ind[k], 4) - sk[k]) > tol:
            problems.append(f"{k}: independent {ind[k]:.4f} vs {sk[k]}")
    if ind["confusion_matrix"] != sk["confusion_matrix"]:
        problems.append("confusion matrix differs")
    if ind["invalid_predictions"] != sk["invalid_predictions"]:
        problems.append("invalid prediction count differs")
    for c, v in ind["per_class"].items():
        for k, sk_k in (("precision", "precision"), ("recall", "recall"), ("f1", "f1-score")):
            if abs(round(v[k], 4) - sk["per_class"][c][sk_k]) > tol:
                problems.append(f"{c}.{k}")
    return problems


def evaluate_against_human(run_dir: Path, gold: list[Annotation], *, label_provenance: dict | None = None,
                           required: int = REQUIRED_HUMAN_LABELS, write: bool = True, unverified_included: bool = False) -> dict:
    config = read_json(run_dir / "config.json")
    strategies = config["strategies"]
    loaded = {name: load_run_predictions(run_dir, name) for name in strategies}
    predicted = {name: {r.review_id for r in recs} | {f.review_id for f in fails} for name, (recs, fails) in loaded.items()}
    gold_ids = [g.review_id for g in gold]
    common = [rid for rid in gold_ids if all(rid in predicted[n] for n in strategies)]
    needs_inference = {n: sorted(set(gold_ids) - predicted[n]) for n in strategies}
    gold_common = [g for g in gold if g.review_id in set(common)]
    missing = any(needs_inference.values())
    if not gold_common:
        status = "awaiting_human_labels"
    elif missing:
        status = "incomplete"
    elif len(gold_common) >= required and not unverified_included:
        status = "complete"
    else:
        status = "provisional"

    result: dict = {
        "run_id": config["run_id"], "model_name": config["model_name"], "is_mock": config["is_mock"],
        "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "reference": "verified human labels (data/annotations/human_annotations.json)" + (" + UNVERIFIED completed labels (provisional analysis only)" if unverified_included else ""),
        "reference_provenance": label_provenance or {},
        "n_human_labels": len(gold), "n_evaluated_common": len(gold_common), "required": required,
        "same_review_set_for_all_strategies": True, "evaluated_review_ids": common,
        "needs_inference": needs_inference, "status": status, "strategies": {}, "independent_check": {},
    }
    all_problems = []
    for name in strategies:
        recs, fails = loaded[name]
        preds = {r.review_id: r for r in recs} | {f.review_id: None for f in fails}
        preds = {rid: p for rid, p in preds.items() if rid in set(common)}
        cl = evaluate_predictions(gold_common, preds) if gold_common else {"n_evaluated": 0}
        result["strategies"][name] = {"classification": cl}
        if gold_common:
            y_true = [g.overall_sentiment.value for g in gold_common]
            y_pred = [preds[g.review_id].overall_sentiment.value if preds[g.review_id] else INVALID for g in gold_common]
            ind = independent_sentiment_metrics(y_true, y_pred)
            probs = _compare(cl["overall_sentiment"], ind)
            result["independent_check"][name] = {"macro_f1": round(ind["macro_f1"], 4), "accuracy": round(ind["accuracy"], 4), "matches": not probs, "problems": probs}
            all_problems += [f"{name}: {p}" for p in probs]
    result["independent_check_passed"] = not all_problems
    if write:
        out = run_dir / "human_reference"
        ensure_writable(out)
        out.mkdir(parents=True, exist_ok=True)
        write_json_atomic(out / "metrics.json", result)
        (out / "report.md").write_text(render_human_report(result), encoding="utf-8")
    return result


def render_human_report(m: dict) -> str:
    lines = [f"# D2 evaluation against verified HUMAN labels — run `{m['run_id']}`", "",
             f"**Status:** {STATUS_TEXT[m['status']]} ({m['n_evaluated_common']} / {m['required']} verified human labels evaluated).", "",
             f"**Reference:** {m['reference']}. Provenance: `{json.dumps(m['reference_provenance'])}`.", "",
             "All strategies are scored on exactly the same reviews. Controlled failures count as wrong predictions.", ""]
    if any(m["needs_inference"].values()):
        lines += ["**Human-labelled reviews without a prediction (run inference for them, then re-evaluate):**", ""]
        lines += [f"- {n}: {len(ids)} ({', '.join(ids[:10])}{' …' if len(ids) > 10 else ''})" for n, ids in m["needs_inference"].items() if ids]
        lines.append("")
    if m["n_evaluated_common"]:
        lines += ["| strategy | n | accuracy | macro P | macro R | macro F1 | weighted F1 | failed (scored wrong) | independent check |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for name, e in m["strategies"].items():
            s = e["classification"]["overall_sentiment"]
            ic = m["independent_check"].get(name, {})
            lines.append(f"| {name} | {s['n']} | {s['accuracy']} | {s['macro_precision']} | {s['macro_recall']} | {s['macro_f1']} | {s['weighted_f1']} | "
                         f"{s['invalid_predictions']} | {'match' if ic.get('matches') else 'MISMATCH'} |")
        for name, e in m["strategies"].items():
            s = e["classification"]["overall_sentiment"]
            lines += ["", f"### {name}", "", "| class | precision | recall | F1 | support |", "| --- | --- | --- | --- | --- |"]
            lines += [f"| {c} | {v['precision']} | {v['recall']} | {v['f1-score']} | {v['support']} |" for c, v in s["per_class"].items()]
            lines += ["", f"Confusion matrix (rows = human label, columns = prediction, order {s['labels']}; failures in no column): `{s['confusion_matrix']}`"]
    else:
        lines += ["_No metrics: there are no verified human labels for reviews in this run yet._"]
    return "\n".join(lines) + "\n"
