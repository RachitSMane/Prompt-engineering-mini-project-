"""Classification metrics for comparing predictions with human annotations.

Three evaluation tasks
----------------------
1. Overall sentiment (single-label, 4 classes): accuracy, macro/weighted P/R/F1,
   per-class report, confusion matrix.
2. Complaint categories (multi-label): micro/macro/weighted/samples P/R/F1,
   Hamming loss, subset (exact-match) accuracy, per-label report.
3. Aspects — matched at the *aspect-category* level (see docs/experiment_methodology.md):
   a. Detection: which categories are discussed (multi-label P/R/F1).
   b. Sentiment: for categories found in BOTH gold and prediction, compare the
      sentiment (single-label, 3 classes, macro-F1).
   c. Joint: (category, sentiment) pairs as a set — micro P/R/F1 (strict).

A failed extraction (no valid output after repair) is kept in the evaluation:
its overall sentiment counts as wrong and its aspect/complaint sets are empty.
This penalises unreliable strategies instead of silently dropping hard cases.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    hamming_loss,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import MultiLabelBinarizer

from schemas.evaluation_schema import Annotation
from schemas.review_schema import ReviewRecord
from schemas.taxonomy import AspectCategory, AspectSentiment, ComplaintCategory, OverallSentiment, values

INVALID = "__invalid__"
OVERALL_LABELS = values(OverallSentiment)
ASPECT_SENT_LABELS = values(AspectSentiment)
ASPECT_CAT_LABELS = values(AspectCategory)
COMPLAINT_LABELS = values(ComplaintCategory)


def _r(x: float) -> float:
    return round(float(x), 4)


def present_labels(labels: list[str], *label_collections) -> list[str]:
    """Taxonomy labels that occur in gold or predictions (sklearn's default label set).

    Macro averages use only these: a class absent from BOTH gold and predictions has
    undefined precision/recall, and counting it as F1 = 0 would wrongly lower macro-F1.
    """
    seen: set[str] = set()
    for coll in label_collections:
        for item in coll:
            seen.update(item if isinstance(item, (set, frozenset, list, tuple)) else [item])
    return [lab for lab in labels if lab in seen]


def single_label_metrics(y_true: list[str], y_pred: list[str], labels: list[str]) -> dict:
    if not y_true:
        return {"n": 0}
    macro_labels = present_labels(labels, y_true, y_pred)
    p_mac, r_mac, f_mac, _ = precision_recall_fscore_support(y_true, y_pred, labels=macro_labels, average="macro", zero_division=0)
    p_w, r_w, f_w, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    return {
        "n": len(y_true),
        "accuracy": _r(accuracy_score(y_true, y_pred)),
        "macro_precision": _r(p_mac),
        "macro_recall": _r(r_mac),
        "macro_f1": _r(f_mac),
        "weighted_precision": _r(p_w),
        "weighted_recall": _r(r_w),
        "weighted_f1": _r(f_w),
        "per_class": {lab: {k: _r(v) if k != "support" else int(v) for k, v in report[lab].items()} for lab in labels},
        "labels": labels,
        "macro_labels": macro_labels,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "invalid_predictions": int(sum(p == INVALID for p in y_pred)),
    }


def multilabel_metrics(true_sets: list[set[str]], pred_sets: list[set[str]], labels: list[str]) -> dict:
    if not true_sets:
        return {"n": 0}
    mlb = MultiLabelBinarizer(classes=labels)
    yt = mlb.fit_transform([sorted(s & set(labels)) for s in true_sets])
    yp = mlb.transform([sorted(s & set(labels)) for s in pred_sets])
    out: dict = {"n": len(true_sets)}
    for avg in ("micro", "macro", "weighted", "samples"):
        p, r, f, _ = precision_recall_fscore_support(yt, yp, average=avg, zero_division=0)
        out[f"{avg}_precision"], out[f"{avg}_recall"], out[f"{avg}_f1"] = _r(p), _r(r), _r(f)
    out["hamming_loss"] = _r(hamming_loss(yt, yp))
    out["subset_accuracy"] = _r(accuracy_score(yt, yp))
    p, r, f, s = precision_recall_fscore_support(yt, yp, average=None, zero_division=0)
    out["per_label"] = {lab: {"precision": _r(p[i]), "recall": _r(r[i]), "f1": _r(f[i]), "support": int(s[i])} for i, lab in enumerate(labels)}
    # macro over labels occurring in gold OR predictions (macro_f1 above averages all taxonomy labels,
    # scoring labels absent from both as 0). Predicted-only labels are kept, so false positives still count.
    occurring = set(present_labels(labels, true_sets, pred_sets))
    present = [i for i, lab in enumerate(labels) if lab in occurring]
    out["macro_f1_present_labels"] = _r(np.mean([f[i] for i in present])) if present else 0.0
    out["macro_labels"] = [labels[i] for i in present]
    return out


def collapse_predicted_aspects(record: ReviewRecord | None) -> dict[str, str]:
    """One sentiment per aspect category: majority vote, ties broken by summed confidence."""
    if record is None:
        return {}
    votes: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for a in record.aspects:
        counts[a.aspect_category.value][a.sentiment.value] += 1
        votes[a.aspect_category.value][a.sentiment.value] += a.confidence
    return {cat: max(c, key=lambda s: (c[s], votes[cat][s])) for cat, c in counts.items()}


def evaluate_predictions(gold: list[Annotation], predictions: dict[str, ReviewRecord | None]) -> dict:
    """Compare predictions (keyed by review_id; None = failed) with gold annotations."""
    gold = [g for g in gold if g.review_id in predictions]
    y_true, y_pred = [], []
    c_true, c_pred = [], []
    a_true, a_pred = [], []
    pair_true, pair_pred = [], []
    s_true, s_pred = [], []
    for g in gold:
        rec = predictions[g.review_id]
        y_true.append(g.overall_sentiment.value)
        y_pred.append(rec.overall_sentiment.value if rec else INVALID)
        c_true.append({c.value for c in g.complaint_categories})
        c_pred.append({c.value for c in rec.complaint_categories} if rec else set())
        gold_aspects = {a.aspect_category.value: a.sentiment.value for a in g.aspects}
        pred_aspects = collapse_predicted_aspects(rec)
        a_true.append(set(gold_aspects))
        a_pred.append(set(pred_aspects))
        pair_true.append({f"{k}:{v}" for k, v in gold_aspects.items()})
        pair_pred.append({f"{k}:{v}" for k, v in pred_aspects.items()})
        for cat in set(gold_aspects) & set(pred_aspects):
            s_true.append(gold_aspects[cat])
            s_pred.append(pred_aspects[cat])

    pair_labels = sorted({p for s in pair_true + pair_pred for p in s})
    joint = multilabel_metrics(pair_true, pair_pred, pair_labels) if pair_labels else {"n": len(gold)}
    return {
        "n_evaluated": len(gold),
        "n_failed_extractions": int(sum(predictions[g.review_id] is None for g in gold)),
        "overall_sentiment": single_label_metrics(y_true, y_pred, OVERALL_LABELS),
        "complaints": multilabel_metrics(c_true, c_pred, COMPLAINT_LABELS),
        "aspect_detection": multilabel_metrics(a_true, a_pred, ASPECT_CAT_LABELS),
        "aspect_sentiment": single_label_metrics(s_true, s_pred, ASPECT_SENT_LABELS),
        "aspect_joint": {k: joint.get(k) for k in ("n", "micro_precision", "micro_recall", "micro_f1")},
    }
