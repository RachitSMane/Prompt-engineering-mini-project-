"""Independent re-computation of the D2 overall-sentiment metrics.

Deliberately uses NO project code and NO scikit-learn: it reads the raw prediction JSONL files and
data/annotations/annotations.json with the standard library, recomputes accuracy, per-class precision /
recall / F1, macro-F1 and the confusion matrix by hand, and compares them with the run's stored metrics.json
(produced by evaluation/classification_metrics.py). Any disagreement is reported and makes the script exit 1.

Rules reproduced from the documented methodology:
  * reference = annotations with status "complete" whose review id is in the run (the flagged review is excluded);
  * a review with an "ok" row anywhere in the file is a valid prediction, otherwise its last "failed" row is a
    failure (provider-outage failures are not predictions and count as missing);
  * a failed prediction is scored as wrong ("__invalid__"), never dropped;
  * macro averages use the 4 sentiment classes that occur in the reference or the predictions.

Usage:  python scripts/verify_metrics_independent.py --run 20261111-030912_qwen2.5-3b [--out reports/verification_independent.md]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["positive", "negative", "neutral", "mixed"]
INVALID = "__invalid__"


def load_predictions(path: Path) -> tuple[dict[str, str], int, dict[str, int]]:
    """(prediction per review id, provider-failure rows, number of rows per review id)."""
    ok, failed, provider, rows = {}, {}, 0, {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rec = row["record"]
        rows[rec["review_id"]] = rows.get(rec["review_id"], 0) + 1
        if row["status"] == "ok":
            ok[rec["review_id"]] = rec["overall_sentiment"]
            failed.pop(rec["review_id"], None)
        elif rec["review_id"] not in ok:
            if rec.get("error", "").startswith("LLM provider error"):
                provider += 1
                continue
            failed[rec["review_id"]] = INVALID
    return {**failed, **ok}, provider, rows


def score(y_true: list[str], y_pred: list[str]) -> dict:
    n = len(y_true)
    cm = [[sum(1 for t, p in zip(y_true, y_pred) if t == a and p == b) for b in CLASSES] for a in CLASSES]
    per = {}
    for c in CLASSES:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per[c] = {"precision": prec, "recall": rec, "f1": f1, "support": tp + fn}
    present = [c for c in CLASSES if c in set(y_true) | set(y_pred)]
    return {
        "n": n,
        "accuracy": sum(t == p for t, p in zip(y_true, y_pred)) / n if n else 0.0,
        "macro_f1": sum(per[c]["f1"] for c in present) / len(present) if present else 0.0,
        "macro_precision": sum(per[c]["precision"] for c in present) / len(present) if present else 0.0,
        "macro_recall": sum(per[c]["recall"] for c in present) / len(present) if present else 0.0,
        "per_class": per,
        "confusion_matrix": cm,
        "failed": sum(p == INVALID for p in y_pred),
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    run_dir = ROOT / "outputs" / ("mock" if args.run.startswith("MOCK-") else "live") / "experiments" / args.run
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    stored = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    ann = json.loads((ROOT / "data" / "annotations" / "annotations.json").read_text(encoding="utf-8"))["annotations"]
    run_ids = set(cfg["review_ids"])
    gold = {rid: a["overall_sentiment"] for rid, a in ann.items() if a.get("status", "complete") == "complete" and rid in run_ids}

    lines = [f"# Independent metric verification — `{args.run}`", "",
             "Recomputed with the Python standard library only (no project code, no scikit-learn) from the raw prediction "
             "files and `data/annotations/annotations.json`, then compared with the stored `metrics.json` (tolerance 0.0001).", "",
             f"- Reviews in run: {len(run_ids)} · complete reference labels in run: {len(gold)}", "",
             f"- Reference class counts (all four classes checked): {dict((c, sum(v == c for v in gold.values())) for c in CLASSES)}", "",
             "| strategy | ids accounted | extra ids | duplicate rows | valid | failed (scored wrong) | missing | coverage | accuracy | macro P | macro R | macro F1 | matches metrics.json |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    problems, details = [], []
    for name in cfg["strategies"]:
        path = run_dir / f"predictions_{name}.jsonl"
        preds, _provider, rows = load_predictions(path) if path.exists() else ({}, 0, {})
        ids = [rid for rid in sorted(gold) if rid in preds]
        missing = len(run_ids - set(preds))
        extra = len(set(preds) - run_ids)
        dups = sum(1 for n in rows.values() if n > 1)
        coverage = round(len(set(preds) & run_ids) / len(run_ids), 4) if run_ids else 0.0
        r = score([gold[i] for i in ids], [preds[i] for i in ids])
        entry = stored["strategies"].get(name, {})
        st = (entry.get("classification") or {}).get("overall_sentiment", {})
        mism = []
        if abs(coverage - entry.get("coverage", -1)) > 1e-4:
            mism.append(f"coverage {coverage} vs {entry.get('coverage')}")
        if entry.get("missing_predictions") != missing:
            mism.append(f"missing {missing} vs {entry.get('missing_predictions')}")
        if extra:
            mism.append(f"{extra} predicted ids outside the run")
        if dups:
            mism.append(f"{dups} review ids have more than one stored row (one attempt per review expected)")
        if r["n"]:
            for k in ("accuracy", "macro_f1", "macro_precision", "macro_recall"):
                if abs(round(r[k], 4) - st.get(k, -1)) > 1e-4:
                    mism.append(f"{k} {r[k]:.4f} vs {st.get(k)}")
            if r["confusion_matrix"] != st.get("confusion_matrix"):
                mism.append("confusion matrix differs")
            if r["failed"] != st.get("invalid_predictions"):
                mism.append(f"invalid predictions {r['failed']} vs {st.get('invalid_predictions')}")
            if st.get("labels") != CLASSES:
                mism.append(f"stored label order {st.get('labels')}")
            for c in CLASSES:
                for k, sk in (("precision", "precision"), ("recall", "recall"), ("f1", "f1-score")):
                    if abs(round(r["per_class"][c][k], 4) - st["per_class"][c][sk]) > 1e-4:
                        mism.append(f"{c}.{k}")
        problems += [f"{name}: {m}" for m in mism]
        lines.append(f"| {name} | {len(set(preds) & run_ids)}/{len(run_ids)} | {extra} | {dups} | {len(preds) - r['failed']} | {r['failed']} | {missing} | {coverage:.4f} | {r['accuracy']:.4f} | "
                     f"{r['macro_precision']:.4f} | {r['macro_recall']:.4f} | {r['macro_f1']:.4f} | {'yes' if not mism else 'NO'} |")
        if r["n"]:
            details += [f"### {name}", "", "| class | precision | recall | F1 | support |", "| --- | --- | --- | --- | --- |"]
            details += [f"| {c} | {v['precision']:.4f} | {v['recall']:.4f} | {v['f1']:.4f} | {v['support']} |" for c, v in r["per_class"].items()]
            details += ["", f"Confusion matrix (rows = reference, columns = prediction, order {CLASSES}; failed predictions appear in no column): `{r['confusion_matrix']}`", ""]
    lines += ["", *details, "**Result:** " + ("all recomputed values match the stored metrics (per-class P/R/F1 for all four classes, confusion "
              "matrices, macro P/R/F1, accuracy, invalid predictions, coverage, missing predictions; no duplicate or out-of-run ids)."
              if not problems else "MISMATCHES: " + "; ".join(problems)), ""]
    text = "\n".join(lines)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
