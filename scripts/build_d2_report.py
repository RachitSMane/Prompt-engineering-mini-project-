"""Build the final D2 report (Markdown + HTML + PDF + SVG charts) from the saved run, after verifying every number.

  python scripts/build_d2_report.py --run 20261111-030912_qwen2.5-3b

No LLM call and no write to any experiment or annotation file. Every figure is recomputed here with the Python
standard library directly from the raw ``predictions_<strategy>.jsonl`` files and ``data/annotations/annotations.json``
(no project metric code), and compared with the run's stored ``metrics.json`` (tolerance 1e-4). The 300-review
sensitivity analysis is recomputed the same way (original predictions + ``supplementary/`` rows, approved references)
and compared with ``approved_reference/metrics.json``. If any value disagrees the script stops without writing a report. Values transcribed from dashboard screenshots are compared too
and every difference is listed in the report.

Outputs
  reports/D2_FINAL_REPORT.md
  reports/d2/D2_REPORT_DATA.json          (all verified numbers used by the report)
  reports/d2/figures/*.svg                (charts)
  reports/final/D2_Final_Report.html / .pdf   (PDF via local Chrome/Edge, as for the project's final report)
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_final_report import CSS, TEAM, to_pdf  # noqa: E402  (shared report style + PDF printing)

CLASSES = ["positive", "negative", "neutral", "mixed"]
STRATS = ["zero_shot", "few_shot", "cot"]
LABEL = {"zero_shot": "Zero-shot", "few_shot": "Few-shot", "cot": "Chain-of-Thought",
         "primary": "Primary (299 reviews)", "sensitivity": "Sensitivity (300 reviews)"}
COLORS = {"zero_shot": "#64748b", "few_shot": "#4f46e5", "cot": "#0d9488", "primary": "#4f46e5", "sensitivity": "#f59e0b"}
PROVIDER_ERROR_PREFIX = "LLM provider error"  # same prefix as pipeline/extraction.py
TOL = 1e-4

# Values transcribed by the team from the Prompt Experiments screenshots (checked, never used as results).
TRANSCRIBED = {
    "zero_shot": {"accuracy": 0.6127, "macro_precision": 0.6611, "macro_recall": 0.4659, "macro_f1": 0.4380, "weighted_f1": 0.6797,
                  "f1_positive": 0.6821, "f1_negative": 0.8520, "f1_neutral": 0.0476, "f1_mixed": 0.1702, "mean_latency_s": 41.19,
                  "mean_input_tokens": 2057.4, "mean_output_tokens": 214.5, "validity_pct": 94.3, "first_attempt_pct": 91.3, "llm_repairs": 26},
    "few_shot": {"accuracy": 0.8127, "macro_precision": 0.6917, "macro_recall": 0.5823, "macro_f1": 0.6025, "weighted_f1": 0.8170,
                 "f1_positive": 0.9231, "f1_negative": 0.8639, "f1_neutral": 0.1667, "f1_mixed": 0.4561, "mean_latency_s": 53.77,
                 "mean_input_tokens": 2884.5, "mean_output_tokens": 227.9, "validity_pct": 95.3, "first_attempt_pct": 94.3, "llm_repairs": 17},
    "cot": {"accuracy": 0.5753, "macro_precision": 0.7201, "macro_recall": 0.5831, "macro_f1": 0.4232, "weighted_f1": 0.6461,
            "f1_positive": 0.6182, "f1_negative": 0.8401, "f1_neutral": 0.1010, "f1_mixed": 0.1333, "mean_latency_s": 47.53,
            "mean_input_tokens": 2297.6, "mean_output_tokens": 223.1, "validity_pct": 93.3, "first_attempt_pct": 91.6, "llm_repairs": 25},
}
TRANSCRIBED_RUN = {"git_commit": "045f16b", "temperature": 0.0, "prompt_version": "v1", "reviews": 299, "seed": 42}


# ============================================================================ independent recomputation
def read_jsonl(p: Path) -> list[dict]:
    with p.open(encoding="utf-8") as fh:
        return [json.loads(x) for x in fh if x.strip()]


def final_rows(path: Path) -> dict[str, dict]:
    """Final outcome per review: an 'ok' row wins; otherwise the last genuine (non-provider) failure."""
    out: dict[str, dict] = {}
    for r in read_jsonl(path):
        rec = r["record"]
        rid = rec["review_id"]
        if r["status"] == "ok":
            out[rid] = r
        elif not rec.get("error", "").startswith(PROVIDER_ERROR_PREFIX) and out.get(rid, {}).get("status") != "ok":
            out[rid] = r
    return out


def failure_category(err: str) -> str:
    """Category of the FIRST validation error in a stored failure message."""
    body = err.split("attempt(s): ", 1)[1] if "attempt(s): " in err else err
    e = body.split("; ")[0].lower()
    if "invalid json" in e:
        return "invalid JSON"
    if "not a verbatim quote" in e:
        return "evidence not verbatim"
    if "evidence" in e and "at most 300" in e:
        return "evidence longer than 300 characters"
    if "rationale" in e and "at most" in e:
        return "rationale too long"
    if "aspect_category" in e or "complaint_categories" in e or "input should be" in e:
        return "label outside the taxonomy"
    if "field required" in e or "extra inputs" in e:
        return "schema (missing / extra field)"
    return "other validation error"


def metrics_for(y_true: list[str], y_pred: list[str]) -> dict:
    n = len(y_true)
    cm = [[sum(1 for t, p in zip(y_true, y_pred) if t == a and p == b) for b in CLASSES] for a in CLASSES]
    per = {}
    for i, c in enumerate(CLASSES):
        tp = cm[i][i]
        predicted = sum(cm[j][i] for j in range(4))
        support = sum(1 for t in y_true if t == c)
        p = tp / predicted if predicted else 0.0
        r = tp / support if support else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        per[c] = {"precision": p, "recall": r, "f1": f, "support": support, "predicted": predicted,
                  "invalid": sum(1 for t, q in zip(y_true, y_pred) if t == c and q not in CLASSES)}
    present = [c for c in CLASSES if per[c]["support"] or per[c]["predicted"]]
    mean = lambda k: sum(per[c][k] for c in present) / len(present)
    return {"n": n, "correct": sum(cm[i][i] for i in range(4)), "accuracy": sum(cm[i][i] for i in range(4)) / n,
            "macro_precision": mean("precision"), "macro_recall": mean("recall"), "macro_f1": mean("f1"),
            "weighted_f1": sum(per[c]["f1"] * per[c]["support"] for c in CLASSES) / n,
            "per_class": per, "confusion_matrix": cm, "invalid": sum(1 for q in y_pred if q not in CLASSES), "macro_classes": present}


def recompute(run_dir: Path) -> dict:
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    ann = json.loads((ROOT / "data" / "annotations" / "annotations.json").read_text(encoding="utf-8"))
    recs = ann["annotations"]
    recs = list(recs.values()) if isinstance(recs, dict) else recs
    method = ann["meta"].get("annotation_method", {})
    human = set(method.get("human_labels", []))
    with (ROOT / "data" / "annotations" / "annotation_sample.csv").open(encoding="utf-8") as fh:
        sample = list(csv.DictReader(fh))
    sample_ids = [r["review_id"] for r in sample]
    run_ids = cfg["review_ids"]
    gold = {r["review_id"]: r["overall_sentiment"] for r in recs if r.get("status", "complete") == "complete" and r["review_id"] in set(run_ids)}
    excluded = [r for r in recs if r.get("status", "complete") != "complete"]
    rating_by_id = {r["review_id"]: r["rating"] for r in sample}
    data = {
        "run": {k: cfg[k] for k in ("run_id", "created_at", "provider", "model_name", "temperature", "prompt_version", "strategies",
                                     "max_repair_attempts", "strict_evidence_grounding", "git_commit")},
        "settings": {k: cfg["settings"].get(k) for k in ("llm_max_output_tokens", "ollama_num_ctx", "llm_timeout_seconds", "random_seed",
                                                          "annotation_sample_size", "dataset_category", "dataset_num_products")},
        "sample": {"size": len(sample_ids), "unique": len(set(sample_ids)),
                   "rating_distribution": dict(sorted(Counter(str(rating_by_id[i]) for i in sample_ids).items())),
                   "products": len({r["product_id"] for r in sample})},
        "run_ids": {"n": len(run_ids), "unique": len(set(run_ids)), "in_sample": sum(1 for i in run_ids if i in set(sample_ids)),
                    "sample_not_in_run": [i for i in sample_ids if i not in set(run_ids)]},
        "reference": {"evaluable": len(gold), "human": len(set(gold) & human), "ai_assisted": len(gold) - len(set(gold) & human),
                      "excluded": [{"review_id": r["review_id"], "status": r["status"], "note": r.get("notes", "")[:160]} for r in excluded],
                      "class_counts": {c: sum(1 for v in gold.values() if v == c) for c in CLASSES},
                      "debatable": sum(1 for r in recs if "DEBATABLE" in (r.get("notes") or "") and r["review_id"] in gold),
                      "approved_by": method.get("approved_by"), "ai_model": method.get("ai_model")},
        "strategies": {},
    }
    id_sets = {}
    for s in STRATS:
        rows_all = read_jsonl(run_dir / f"predictions_{s}.jsonl")
        fin = final_rows(run_dir / f"predictions_{s}.jsonl")
        id_sets[s] = set(fin)
        ids = [i for i in run_ids if i in gold]
        y_true = [gold[i] for i in ids]
        y_pred = [fin[i]["record"]["overall_sentiment"] if fin.get(i, {}).get("status") == "ok" else "__invalid__" for i in ids]
        m = metrics_for(y_true, y_pred)
        metas = [fin[i]["record"]["extraction_metadata"] for i in ids if i in fin]
        ok_m = [fin[i]["record"]["extraction_metadata"] for i in ids if fin.get(i, {}).get("status") == "ok"]
        bad_m = [fin[i]["record"]["extraction_metadata"] for i in ids if i in fin and fin[i]["status"] != "ok"]
        fails = [fin[i]["record"] for i in ids if i in fin and fin[i]["status"] != "ok"]
        n = len(metas)
        op = {
            "rows_in_file": len(rows_all), "final_outcomes": n, "missing": len([i for i in ids if i not in fin]),
            "valid": len(ok_m), "failed": len(bad_m), "validity_rate": len(ok_m) / n,
            "first_attempt_valid_rate": sum(1 for x in metas if x["validation_success"] and x["repair_attempts"] == 0) / n,
            "needed_llm_repair": sum(1 for x in metas if x["repair_attempts"] > 0),
            "repaired_successfully": sum(1 for x in metas if x["repair_attempts"] > 0 and x["validation_success"]),
            "local_fixes": sum(x["local_repairs"] for x in metas),
            "records_with_local_fix": sum(1 for x in metas if x["local_repairs"] > 0),
            "mean_repair_attempts": sum(x["repair_attempts"] for x in metas) / n,
            "mean_latency_s": sum(x["total_latency_s"] for x in metas) / n,
            "median_latency_s": sorted(x["total_latency_s"] for x in metas)[n // 2],
            "mean_latency_valid_s": sum(x["total_latency_s"] for x in ok_m) / len(ok_m),
            "mean_latency_failed_s": sum(x["total_latency_s"] for x in bad_m) / len(bad_m) if bad_m else None,
            "total_latency_h": sum(x["total_latency_s"] for x in metas) / 3600,
            "mean_input_tokens": sum(x["input_tokens"] for x in metas) / n, "mean_output_tokens": sum(x["output_tokens"] for x in metas) / n,
            "total_input_tokens": sum(x["input_tokens"] for x in metas), "total_output_tokens": sum(x["output_tokens"] for x in metas),
            "model_names": sorted({x["model_name"] for x in metas}), "prompt_versions": sorted({x["prompt_version"] for x in metas}),
            "failure_categories": dict(Counter(failure_category(f["error"]) for f in fails).most_common()),
            "failures_by_reference_class": {c: sum(1 for f in fails if gold.get(f["review_id"]) == c) for c in CLASSES},
        }
        data["strategies"][s] = {"classification": m, "operational": op}
    data["same_review_set"] = id_sets["zero_shot"] == id_sets["few_shot"] == id_sets["cot"] == set(run_ids)
    return data


def verify(data: dict, stored: dict) -> list[dict]:
    """Recomputed vs stored metrics.json; every compared value is listed."""
    rows = []

    def add(s, name, mine, theirs):
        ok = theirs is not None and abs(float(mine) - float(theirs)) <= TOL + 1e-9 * abs(float(theirs))
        rows.append({"strategy": s, "value": name, "recomputed": mine, "metrics_json": theirs, "match": ok})

    for s in STRATS:
        m, op = data["strategies"][s]["classification"], data["strategies"][s]["operational"]
        st = stored["strategies"][s]["classification"]["overall_sentiment"]
        so = stored["strategies"][s]["operational"]
        for k in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"):
            add(s, k, round(m[k], 4), st[k])
        for c in CLASSES:
            for k, sk in (("precision", "precision"), ("recall", "recall"), ("f1", "f1-score")):
                add(s, f"{c} {k}", round(m["per_class"][c][k], 4), st["per_class"][c][sk])
            add(s, f"{c} support", m["per_class"][c]["support"], st["per_class"][c]["support"])
        rows.append({"strategy": s, "value": "confusion matrix", "recomputed": m["confusion_matrix"], "metrics_json": st["confusion_matrix"],
                     "match": m["confusion_matrix"] == st["confusion_matrix"]})
        add(s, "invalid predictions", m["invalid"], st["invalid_predictions"])
        for k in ("valid", "failed", "validity_rate", "first_attempt_valid_rate", "needed_llm_repair", "local_fixes"):
            add(s, k, round(op[k], 4) if isinstance(op[k], float) else op[k], so[k])
        add(s, "mean_latency_s", round(op["mean_latency_s"], 4), so["mean_latency_s"])
        add(s, "mean_input_tokens", round(op["mean_input_tokens"], 1), so["mean_input_tokens"])
        add(s, "mean_output_tokens", round(op["mean_output_tokens"], 1), so["mean_output_tokens"])
    return rows


def recompute_sensitivity(run_dir: Path) -> dict:
    """The 300-review approved-reference evaluation, recomputed from raw files (no project metric code)."""
    stored = json.loads((run_dir / "approved_reference" / "metrics.json").read_text(encoding="utf-8"))
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    ann = json.loads((ROOT / "data" / "annotations" / "annotations.json").read_text(encoding="utf-8"))["annotations"]
    adj = json.loads((ROOT / "data" / "annotations" / "reference_adjudication.json").read_text(encoding="utf-8"))
    dec = adj["decisions"]
    ids = stored["evaluated_review_ids"]
    # label = original annotation unless the reviewer recorded a correction
    gold = {i: (dec[i].get("corrected_label") or {}).get("overall_sentiment", ann[i]["overall_sentiment"]) for i in ids}
    extra = [i for i in ids if i not in set(cfg["review_ids"])]
    out = {"stored": {k: stored[k] for k in ("evaluated_at", "status", "n_reference", "n_evaluated_common", "reference_provenance",
                                              "reference_progress", "independent_check_passed")},
           "n": len(ids), "unique": len(set(ids)), "extra_review_ids": extra,
           "extra_reviews": [{"review_id": i, "label": gold[i], "original_status": ann[i].get("status"), "note": ann[i].get("notes", "")[:200],
                              "decision": dec[i]["decision"], "method": dec[i]["method"], "reviewer": dec[i]["reviewer"],
                              "decided_at": dec[i]["decided_at"], "original_provenance": dec[i]["original_provenance"]} for i in extra],
           "corrections_applied": sum(1 for i in ids if dec[i].get("corrected_label")),
           "decisions": dict(Counter(dec[i]["decision"] for i in ids)), "strategies": {}, "verification": []}
    id_sets = []
    for s in STRATS:
        fin = final_rows(run_dir / f"predictions_{s}.jsonl")
        sup = run_dir / "supplementary" / f"predictions_{s}.jsonl"
        supf = final_rows(sup) if sup.exists() else {}
        allp = {**supf, **fin}  # original outcome wins; supplementary only fills reviews absent from the run
        id_sets.append({i for i in ids if i in allp})
        y_pred = [allp[i]["record"]["overall_sentiment"] if allp.get(i, {}).get("status") == "ok" else "__invalid__" for i in ids]
        m = metrics_for([gold[i] for i in ids], y_pred)
        extra_rows = {i: {"prediction": allp[i]["record"].get("overall_sentiment") if allp[i]["status"] == "ok" else "__invalid__",
                          "model": allp[i]["record"]["extraction_metadata"]["model_name"],
                          "prompt_version": allp[i]["record"]["extraction_metadata"]["prompt_version"],
                          "extracted_at": allp[i]["record"]["extraction_metadata"].get("extracted_at")} for i in extra if i in allp}
        valid = sum(1 for q in y_pred if q in CLASSES)
        out["strategies"][s] = {"classification": m, "valid": valid, "invalid": len(ids) - valid, "extra_predictions": extra_rows}
        st = stored["strategies"][s]
        so = st["classification"]["overall_sentiment"]
        chk = [(k, round(m[k], 4), so[k]) for k in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1")]
        for c in CLASSES:
            chk += [(f"{c} precision", round(m["per_class"][c]["precision"], 4), so["per_class"][c]["precision"]),
                    (f"{c} recall", round(m["per_class"][c]["recall"], 4), so["per_class"][c]["recall"]),
                    (f"{c} f1", round(m["per_class"][c]["f1"], 4), so["per_class"][c]["f1-score"]),
                    (f"{c} support", m["per_class"][c]["support"], so["per_class"][c]["support"])]
        chk += [("valid predictions", valid, st["valid_predictions"]), ("invalid predictions", len(ids) - valid, st["invalid_predictions"])]
        for name, mine, theirs in chk:
            out["verification"].append({"strategy": s, "value": name, "recomputed": mine, "approved_reference_json": theirs,
                                        "match": abs(float(mine) - float(theirs)) <= TOL})
        out["verification"].append({"strategy": s, "value": "confusion matrix", "recomputed": m["confusion_matrix"],
                                    "approved_reference_json": so["confusion_matrix"], "match": m["confusion_matrix"] == so["confusion_matrix"]})
    out["same_review_set"] = all(x == set(ids) for x in id_sets) and len(set(ids)) == len(ids)
    return out


def chart_check(charts: dict[str, str], expected: dict[str, list[str]]) -> list[dict]:
    """Every value label expected for a chart must appear as a text label in its SVG."""
    res = []
    for k, labels in expected.items():
        missing = [x for x in labels if f">{x}<" not in charts[k]]
        res.append({"chart": k, "labels_checked": len(labels), "missing": missing, "match": not missing})
    return res


def check_transcription(data: dict) -> list[dict]:
    out = []
    for s in STRATS:
        m, op = data["strategies"][s]["classification"], data["strategies"][s]["operational"]
        actual = {"accuracy": m["accuracy"], "macro_precision": m["macro_precision"], "macro_recall": m["macro_recall"],
                  "macro_f1": m["macro_f1"], "weighted_f1": m["weighted_f1"],
                  **{f"f1_{c}": m["per_class"][c]["f1"] for c in CLASSES},
                  "mean_latency_s": op["mean_latency_s"], "mean_input_tokens": op["mean_input_tokens"], "mean_output_tokens": op["mean_output_tokens"],
                  "validity_pct": 100 * op["validity_rate"], "first_attempt_pct": 100 * op["first_attempt_valid_rate"], "llm_repairs": op["needed_llm_repair"]}
        for k, v in TRANSCRIBED[s].items():
            dec = len(str(v).split(".")[1]) if "." in str(v) else 0
            out.append({"strategy": s, "value": k, "transcribed": v, "verified": round(actual[k], dec), "match": round(actual[k], dec) == v})
    r = data["run"]
    for k, v in (("git_commit", r["git_commit"]), ("temperature", r["temperature"]), ("prompt_version", r["prompt_version"]),
                 ("reviews", data["run_ids"]["n"]), ("seed", data["settings"]["random_seed"])):
        out.append({"strategy": "run", "value": k, "transcribed": TRANSCRIBED_RUN[k], "verified": v, "match": TRANSCRIBED_RUN[k] == v})
    return out


# ============================================================================ charts (inline SVG, no plotting library)
def _svg(w, h, body):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" font-family="Segoe UI, Arial, sans-serif">'
            f'<rect width="{w}" height="{h}" fill="#ffffff"/>{body}</svg>')


def grouped_bars(title: str, groups: list[str], series: dict[str, list[float]], vmax: float, fmt: str = "{:.3f}", w=760, h=320) -> str:
    left, right, top, bottom = 50, 20, 40, 60
    pw, ph = w - left - right, h - top - bottom
    gw = pw / len(groups)
    bw = gw * 0.75 / len(series)
    b = [f'<text x="{w/2}" y="22" text-anchor="middle" font-size="15" font-weight="600" fill="#111a33">{html.escape(title)}</text>']
    for k in range(6):
        y = top + ph - ph * k / 5
        b.append(f'<line x1="{left}" y1="{y:.1f}" x2="{w-right}" y2="{y:.1f}" stroke="#e5e7eb"/>'
                 f'<text x="{left-6}" y="{y+4:.1f}" text-anchor="end" font-size="10" fill="#6b7280">{fmt.format(vmax*k/5)}</text>')
    for gi, g in enumerate(groups):
        x0 = left + gi * gw + gw * 0.125
        for si, (name, vals) in enumerate(series.items()):
            v = vals[gi]
            bh = ph * (v / vmax if vmax else 0)
            x = x0 + si * bw
            b.append(f'<rect x="{x:.1f}" y="{top+ph-bh:.1f}" width="{bw-3:.1f}" height="{bh:.1f}" fill="{COLORS.get(name, "#94a3b8")}"/>'
                     f'<text x="{x+(bw-3)/2:.1f}" y="{top+ph-bh-4:.1f}" text-anchor="middle" font-size="9.5" fill="#111827">{fmt.format(v)}</text>')
        b.append(f'<text x="{left+gi*gw+gw/2:.1f}" y="{top+ph+18}" text-anchor="middle" font-size="11.5" fill="#1f2937">{html.escape(g)}</text>')
    lx = left
    for name in series:
        b.append(f'<rect x="{lx}" y="{h-22}" width="12" height="12" fill="{COLORS.get(name, "#94a3b8")}"/>'
                 f'<text x="{lx+16}" y="{h-12}" font-size="11.5" fill="#1f2937">{LABEL.get(name, name)}</text>')
        lx += 150
    return _svg(w, h, "".join(b))


def heatmap(title: str, cm: list[list[int]], invalid_per_row: list[int]) -> str:
    cell, left, top = 58, 92, 58
    w, h = left + cell * 5 + 20, top + cell * 4 + 40
    vmax = max(max(r) for r in cm) or 1
    b = [f'<text x="{w/2}" y="20" text-anchor="middle" font-size="14" font-weight="600" fill="#111a33">{html.escape(title)}</text>',
         f'<text x="{left+cell*2.5}" y="40" text-anchor="middle" font-size="10.5" fill="#6b7280">predicted →  (last column: invalid output)</text>']
    for j, c in enumerate(CLASSES + ["invalid"]):
        b.append(f'<text x="{left+j*cell+cell/2}" y="{top-4}" text-anchor="middle" font-size="10.5" fill="#374151">{c}</text>')
    for i, c in enumerate(CLASSES):
        b.append(f'<text x="{left-6}" y="{top+i*cell+cell/2+4}" text-anchor="end" font-size="10.5" fill="#374151">{c}</text>')
        for j in range(5):
            v = cm[i][j] if j < 4 else invalid_per_row[i]
            a = (v / vmax) if j < 4 else min(1.0, v / vmax)
            fill = f"rgba(79,70,229,{0.08+0.85*a:.2f})" if j < 4 else f"rgba(220,38,38,{0.08+0.6*a:.2f})"
            col = "#ffffff" if a > 0.55 else "#111827"
            b.append(f'<rect x="{left+j*cell}" y="{top+i*cell}" width="{cell-2}" height="{cell-2}" fill="{fill}"/>'
                     f'<text x="{left+j*cell+cell/2-1}" y="{top+i*cell+cell/2+5}" text-anchor="middle" font-size="14" font-weight="{600 if i == j else 400}" fill="{col}">{v}</text>')
    b.append(f'<text x="14" y="{top+cell*2}" font-size="10.5" fill="#6b7280" transform="rotate(-90 14 {top+cell*2})" text-anchor="middle">reference ↓</text>')
    return _svg(w, h, "".join(b))


# ============================================================================ report text
def f4(x):
    return f"{x:.4f}"


def pct(x, d=1):
    return f"{100*x:.{d}f} %"


def table(head: list[str], rows: list[list]) -> str:
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def build_markdown(d: dict, ver: list[dict], tr: list[dict], figs: dict[str, str], sens: dict, charts_ok: list[dict]) -> str:
    S = d["strategies"]
    C = {s: S[s]["classification"] for s in STRATS}
    O = {s: S[s]["operational"] for s in STRATS}
    ref, run, smp = d["reference"], d["run"], d["sample"]
    best = max(STRATS, key=lambda s: C[s]["macro_f1"])
    cc = ref["class_counts"]
    nver = sum(1 for v in ver if v["match"])
    tr_bad = [t for t in tr if not t["match"]]
    excl = ref["excluded"][0] if ref["excluded"] else None
    pos_neu = {s: C[s]["confusion_matrix"][0][2] for s in STRATS}
    pos_valid = {s: sum(C[s]["confusion_matrix"][0]) for s in STRATS}
    mixed_neg = {s: C[s]["confusion_matrix"][3][1] for s in STRATS}
    neu_prec = ", ".join(f"{C[s]['per_class']['neutral']['precision']:.4f}" for s in STRATS)
    SC = {s: sens["strategies"][s]["classification"] for s in STRATS}
    sver_ok = sum(1 for v in sens["verification"] if v["match"])
    xr = sens["extra_reviews"][0] if sens["extra_reviews"] else None
    dmac = {s: round(SC[s]["macro_f1"], 4) - round(C[s]["macro_f1"], 4) for s in STRATS}  # difference of the reported 4-decimal values
    valid_wrong = {s: O[s]["valid"] - C[s]["correct"] for s in STRATS}
    L = []
    a = L.append
    a(f"# ReviewIQ — Deliverable 2: Prompt Engineering Experiments (final report)\n")
    a(f"Run `{run['run_id']}` · generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} by `scripts/build_d2_report.py` · "
      f"team: {', '.join(n for n, _ in TEAM)}\n")
    a(f"> **Every number in this report was recomputed with the Python standard library from the raw prediction files and the "
      f"reference annotations, and matched against the stored results: primary evaluation vs `metrics.json` {nver} / {len(ver)} values "
      f"identical; sensitivity analysis vs `approved_reference/metrics.json` {sver_ok} / {len(sens['verification'])} identical (tolerance 1e-4).** "
      f"The reference labels are **{ref['ai_assisted']} AI-assisted + {ref['human']} human**: the metrics measure agreement with "
      f"mentor-approved AI-assisted reference labels, **not** accuracy against independent human ground truth.\n")

    a("## 1. Abstract\n")
    a(f"We compared three prompting strategies — zero-shot, few-shot and Chain-of-Thought (CoT) — for structured review "
      f"extraction with a local 3-billion-parameter model (`{run['model_name']}` on Ollama, temperature {run['temperature']}). Each strategy "
      f"processed the same {d['run_ids']['n']} Amazon appliance reviews, and every output was validated against a Pydantic schema with "
      f"automatic repair. On overall sentiment (4 classes), **few-shot was best on every headline metric**: macro-F1 "
      f"{f4(C['few_shot']['macro_f1'])} versus {f4(C['zero_shot']['macro_f1'])} (zero-shot) and {f4(C['cot']['macro_f1'])} (CoT), accuracy "
      f"{f4(C['few_shot']['accuracy'])}, and the highest output validity ({pct(O['few_shot']['validity_rate'])}). Zero-shot and CoT "
      f"label many mildly positive reviews as neutral; all strategies struggle with the rare neutral class ({cc['neutral']} examples) and "
      f"with mixed reviews. Few-shot costs about {100*(O['few_shot']['mean_input_tokens']/O['zero_shot']['mean_input_tokens']-1):.0f} % more "
      f"input tokens and {100*(O['few_shot']['mean_latency_s']/O['zero_shot']['mean_latency_s']-1):.0f} % more latency than zero-shot. "
      f"Because {ref['ai_assisted']} of the {ref['evaluable']} reference labels are AI-assisted, the results are agreement with an "
      f"AI-assisted reference and need confirmation on independent human labels. A sensitivity analysis on the approved 300-review "
      f"reference set, which adds one formerly excluded review, gives macro-F1 {f4(SC['zero_shot']['macro_f1'])} / {f4(SC['few_shot']['macro_f1'])} / "
      f"{f4(SC['cot']['macro_f1'])} with the same ranking; the 299-review evaluation remains the primary result.\n")

    a("## 2. Introduction\n")
    a("Star ratings say *how much* customers like a product, not *why*. ReviewIQ turns free-text reviews into validated structured "
      "records (overall sentiment, aspects with sentiment and verbatim evidence, complaint categories). Deliverable 1 built and verified "
      "the extraction pipeline; Deliverable 2, reported here, asks which prompting strategy produces the most accurate and reliable "
      "records with a small local model, and at what cost.\n")

    a("## 3. Objectives\n")
    a("1. Compare zero-shot, few-shot and CoT prompting on the same reviews, model and settings.\n"
      "2. Measure accuracy, macro and weighted precision / recall / F1, per-class performance and confusion matrices.\n"
      "3. Measure reliability (schema validity, repairs, controlled failures) and efficiency (latency, tokens).\n"
      "4. Verify every metric independently and state the provenance of the reference labels honestly.\n")

    a("## 4. Dataset and reference annotations\n")
    a(table(["Item", "Value"], [
        ["Source", "Amazon Reviews 2023, category Appliances; 1,500 reviews from 15 products (`data/processed/reviews.csv`)"],
        ["Annotation sample", f"**{smp['size']} reviews** ({smp['unique']} unique ids), stratified by star rating, seed {d['settings']['random_seed']}, "
                              f"covering {smp['products']} products; rating distribution {smp['rating_distribution']}"],
        ["Excluded from evaluation", f"{len(ref['excluded'])} review: `{excl['review_id']}` (status *{excl['status']}*: text garbled / unreadable)" if excl else "none"],
        ["**Evaluated set**", f"**{d['run_ids']['n']} reviews** ({d['run_ids']['unique']} unique ids, all in the sample) = 300 − 1 excluded"],
        ["Reference labels on the evaluated set", f"{ref['evaluable']}: **{ref['human']} human** (annotator Akash) + **{ref['ai_assisted']} AI-assisted**"],
        ["AI-assisted labels", f"{ref['ai_model']} (star rating not used); approval: {ref['approved_by']}; "
                               f"{ref['debatable']} marked `DEBATABLE`"],
        ["Reference class distribution", f"positive {cc['positive']} · negative {cc['negative']} · neutral {cc['neutral']} · mixed {cc['mixed']}"],
    ]) + "\n")
    a(f"**300 vs 299.** The annotation sample has 300 reviews. One (`{excl['review_id'] if excl else '–'}`) was flagged as unreadable before the "
      f"experiment, so the run evaluated {d['run_ids']['n']} (the **primary evaluation**, §9–§13). That review was not predicted in this run. "
      "It was later approved as a reference label and predicted separately (2026-11-14, `supplementary/`); it enters **only** the "
      "sensitivity analysis in §14.\n")
    a(f"**Reference approval (2026-11-14).** All {sens['stored']['reference_progress']['accepted']} / {sens['stored']['reference_progress']['available']} "
      f"reference annotations were accepted by a named reviewer ({', '.join(sens['stored']['reference_progress']['reviewers'])}; decisions in "
      "`data/annotations/reference_adjudication.json`, no label corrected). Approval records acceptance only: the original provenance is "
      f"unchanged (accepted labels: {sens['stored']['reference_provenance'].get('human_created', 0)} human-created, "
      f"{sens['stored']['reference_provenance'].get('ai_assisted', 0)} AI-assisted). Independent human annotation (blind labelling plus "
      "verification by a second person) is a separate process and is not used in any result here.\n")
    a("**Provenance.** The labels are stored in `data/annotations/annotations.json` with the human / AI-assisted split in "
      "`meta.annotation_method` and an `AI-assisted label` note on every AI-assisted record. They were not changed for this report. "
      "No output of the evaluated model was ever written into the label file.\n")

    a("## 5. Experimental environment\n")
    a(table(["Setting", "Value"], [
        ["Run id / created", f"`{run['run_id']}` / {run['created_at']}"],
        ["Provider / model", f"{run['provider']} / `{run['model_name']}` (local; no cloud API)"],
        ["Temperature", run["temperature"]], ["Prompt version", run["prompt_version"]],
        ["Max output tokens / context window", f"{d['settings']['llm_max_output_tokens']} / {d['settings']['ollama_num_ctx']}"],
        ["Timeout per call", f"{d['settings']['llm_timeout_seconds']} s"],
        ["Max LLM repair attempts", run["max_repair_attempts"]], ["Strict evidence grounding", run["strict_evidence_grounding"]],
        ["Git commit at run start", f"`{run['git_commit']}`"],
        ["Hardware", "student laptop with a 4 GB GPU; GPU thermal watchdog active during inference"],
    ]) + "\n")

    a("## 6. Prompting strategies\n")
    a(table(["Strategy", "Prompt content (prompts/)", "Mean input tokens"], [
        ["Zero-shot", "system role, extraction rules, taxonomy with definitions, JSON output format", f"{O['zero_shot']['mean_input_tokens']:.1f}"],
        ["Few-shot", "zero-shot content + 5 synthetic labelled demonstrations written by the team (not from the dataset): overall labels positive, negative, mixed, negative (compatibility issue), neutral (no product aspects)", f"{O['few_shot']['mean_input_tokens']:.1f}"],
        ["Chain-of-Thought", "zero-shot content + a 7-step analysis procedure (split statements, map aspects, decide sentiments, map complaints, weigh positive vs negative for the overall label, verify evidence) carried out *silently*; output is only the JSON with a 1–2 sentence `rationale` (no reasoning trace is stored)", f"{O['cot']['mean_input_tokens']:.1f}"],
    ]) + "\n")
    a("All strategies share the same schema, taxonomy, extraction rules, repair loop and model settings; only the prompt differs. "
      "Prompt version `v1` (`prompts/zero_shot.py`, `prompts/few_shot.py`, `prompts/cot.py`).\n")

    a("## 7. Experimental methodology\n")
    a(f"1. Each strategy processed all {d['run_ids']['n']} reviews once (one attempt per review per strategy; stored outcomes are final and "
      "were never re-sampled to improve scores).\n"
      "2. Every output was parsed and validated against the Pydantic schema (labels from the closed taxonomy, verbatim evidence ≤ 300 "
      "characters). Deterministic local fixes were applied first; if still invalid, up to 2 LLM repair attempts followed.\n"
      "3. An output still invalid after repair is a **controlled failure**: stored with every raw attempt and error, never dropped and "
      "never given invented values.\n"
      f"4. **Same review set:** all three prediction files contain final outcomes for exactly the same {d['run_ids']['n']} ids "
      f"(verified: {'yes' if d['same_review_set'] else 'NO'}); no missing and no duplicate outcomes.\n"
      f"5. **Denominators and invalid outputs:** every metric uses all {d['run_ids']['n']} reviews. A controlled failure is scored as a wrong "
      "prediction (label `__invalid__`): it lowers accuracy and the recall of its true class, it appears in no predicted-class column of the "
      "confusion matrix (so a matrix row sums to the class support minus that class's failures), and it does not change any class's "
      "precision. Macro averages are the unweighted mean over the 4 sentiment classes; weighted F1 weights each class by its support.\n")

    a("## 8. Evaluation metrics\n")
    a("- **Accuracy** = correct / 299.\n- **Precision (class c)** = correct c predictions / all c predictions; **recall** = correct c predictions / "
      "reference c reviews; **F1** = harmonic mean.\n- **Macro P / R / F1** = unweighted mean over positive, negative, neutral, mixed — every "
      "class counts equally, so rare classes matter (headline metric because the sample is stratified by rating, not by sentiment).\n"
      "- **Weighted F1** = support-weighted mean of class F1.\n- **Validity** = valid outputs / 299; **first-attempt validity** = valid with no LLM "
      "repair / 299; **LLM repairs** = reviews that needed at least one LLM repair call.\n- **Latency / tokens** = per review, including repair "
      "calls, averaged over all 299 outcomes.\n")

    a("## 9. Primary results — 299 reviews\n")
    a(table(["Strategy", "Accuracy", "Macro P", "Macro R", "**Macro F1**", "Weighted F1", "Correct / 299", "Invalid"],
            [[("**" + LABEL[s] + "**") if s == best else LABEL[s], f4(C[s]["accuracy"]), f4(C[s]["macro_precision"]), f4(C[s]["macro_recall"]),
              ("**" + f4(C[s]["macro_f1"]) + "**") if s == best else f4(C[s]["macro_f1"]), f4(C[s]["weighted_f1"]), C[s]["correct"], C[s]["invalid"]] for s in STRATS]) + "\n")
    a(f"![Overall metrics]({figs['overall']})\n")
    a(f"Few-shot leads on accuracy (+{C['few_shot']['accuracy']-C['zero_shot']['accuracy']:.4f} over zero-shot), macro-F1 "
      f"(+{C['few_shot']['macro_f1']-C['zero_shot']['macro_f1']:.4f}) and weighted F1. CoT has the highest macro precision "
      f"({f4(C['cot']['macro_precision'])}) but the lowest accuracy: it predicts mixed and positive rarely but correctly, and pushes many "
      "reviews into neutral.\n")

    a("## 10. Confusion matrix analysis\n")
    a("Rows = reference label, columns = prediction (order positive, negative, neutral, mixed); the extra column counts invalid outputs "
      "for that reference class.\n")
    for s in STRATS:
        cm = C[s]["confusion_matrix"]
        inv = [C[s]["per_class"][c]["invalid"] for c in CLASSES]
        a(f"**{LABEL[s]}**\n")
        a(table(["reference \\ predicted"] + CLASSES + ["invalid", "support"],
                [[c] + cm[i] + [inv[i], C[s]["per_class"][c]["support"]] for i, c in enumerate(CLASSES)]) + "\n")
    a(f"![Confusion matrices]({figs['cm']})\n")
    a(f"- **Positive → neutral** is the dominant error for zero-shot ({pos_neu['zero_shot']} of {pos_valid['zero_shot']} valid positive reviews) "
      f"and CoT ({pos_neu['cot']} of {pos_valid['cot']}); few-shot makes it only {pos_neu['few_shot']} times.\n"
      f"- **Mixed → negative** is the dominant error for all strategies ({mixed_neg['zero_shot']} / {mixed_neg['few_shot']} / {mixed_neg['cot']}): "
      "reviews with both praise and complaints are read as negative.\n"
      "- **Negative** is recognised well by all three strategies (largest, most clear-cut class).\n")

    a("## 11. Per-class analysis\n")
    a(table(["Class (support)"] + [f"{LABEL[s]} P / R / F1" for s in STRATS],
            [[f"{c} ({cc[c]})"] + [f"{C[s]['per_class'][c]['precision']:.4f} / {C[s]['per_class'][c]['recall']:.4f} / **{C[s]['per_class'][c]['f1']:.4f}**" for s in STRATS]
             for c in CLASSES]) + "\n")
    a(f"![Per-class F1]({figs['perclass']})\n")
    a(f"**Neutral is the weakest class for every strategy** (F1 {', '.join(f4(C[s]['per_class']['neutral']['f1']) for s in STRATS)}). "
      f"Only {cc['neutral']} of {ref['evaluable']} reference labels ({100*cc['neutral']/ref['evaluable']:.1f} %) are neutral, while the models "
      f"predict neutral {', '.join(str(C[s]['per_class']['neutral']['predicted']) for s in STRATS)} times (zero-shot / few-shot / CoT). "
      f"With so few true neutrals, almost every neutral prediction is a false positive (precision "
      f"{neu_prec}), "
      f"and a single correct or missed neutral moves its recall by 20 points. Because macro-F1 weights neutral equally with positive "
      f"({cc['positive']}) and negative ({cc['negative']}), this one rare class pulls every macro average down. **Mixed** "
      f"({cc['mixed']} reviews) is the second weakness: recall {', '.join(f4(C[s]['per_class']['mixed']['recall']) for s in STRATS)}.\n")

    a("## 12. Efficiency analysis\n")
    a(table(["Metric (per review, all 299 outcomes)"] + [LABEL[s] for s in STRATS], [
        ["Mean latency"] + [f"{O[s]['mean_latency_s']:.2f} s" for s in STRATS],
        ["Median latency"] + [f"{O[s]['median_latency_s']:.2f} s" for s in STRATS],
        ["Mean latency, valid / failed outputs"] + [f"{O[s]['mean_latency_valid_s']:.1f} s / {O[s]['mean_latency_failed_s']:.1f} s" for s in STRATS],
        ["Total inference time"] + [f"{O[s]['total_latency_h']:.2f} h" for s in STRATS],
        ["Mean input tokens"] + [f"{O[s]['mean_input_tokens']:.1f}" for s in STRATS],
        ["Mean output tokens"] + [f"{O[s]['mean_output_tokens']:.1f}" for s in STRATS],
        ["Total input / output tokens"] + [f"{O[s]['total_input_tokens']:,} / {O[s]['total_output_tokens']:,}" for s in STRATS],
        ["Macro-F1 per 1,000 input tokens"] + [f"{1000*C[s]['macro_f1']/O[s]['mean_input_tokens']:.3f}" for s in STRATS],
    ]) + "\n")
    a(f"![Efficiency]({figs['efficiency']})\n")
    a(f"Few-shot's demonstrations make its prompt the longest ({100*(O['few_shot']['mean_input_tokens']/O['zero_shot']['mean_input_tokens']-1):.0f} % more input "
      f"tokens than zero-shot) and slowest ({100*(O['few_shot']['mean_latency_s']/O['zero_shot']['mean_latency_s']-1):.0f} % more latency). Per 1,000 input tokens "
      "zero-shot and few-shot deliver almost the same macro-F1 (zero-shot marginally more) and CoT the least, so few-shot buys its large quality "
      "gain at a roughly proportional token cost. Failed outputs cost several times more time than valid ones because they include the repair "
      "calls. Output length is similar for all strategies because CoT's procedure is performed silently.\n")

    a("## 13. Invalid output and repair analysis\n")
    a(table(["Metric"] + [LABEL[s] for s in STRATS], [
        ["Valid / controlled failures"] + [f"{O[s]['valid']} / {O[s]['failed']}" for s in STRATS],
        ["Validity rate"] + [pct(O[s]["validity_rate"]) for s in STRATS],
        ["First-attempt validity (no LLM repair)"] + [pct(O[s]["first_attempt_valid_rate"]) for s in STRATS],
        ["Reviews needing ≥ 1 LLM repair"] + [O[s]["needed_llm_repair"] for s in STRATS],
        ["… of which repaired successfully"] + [O[s]["repaired_successfully"] for s in STRATS],
        ["Deterministic local fixes (reviews / fixes)"] + [f"{O[s]['records_with_local_fix']} / {O[s]['local_fixes']}" for s in STRATS],
        ["Failures by reference class (pos / neg / neu / mix)"] + [" / ".join(str(O[s]["failures_by_reference_class"][c]) for c in CLASSES) for s in STRATS],
    ]) + "\n")
    cats = sorted({k for s in STRATS for k in O[s]["failure_categories"]})
    a(table(["Failure cause (first validation error)"] + [LABEL[s] for s in STRATS], [[k] + [O[s]["failure_categories"].get(k, 0) for s in STRATS] for k in cats]) + "\n")
    a(f"![Validity]({figs['validity']})\n")
    a("Most failures are grounding errors (evidence that is not an exact quote or is too long) and labels outside the closed taxonomy — "
      "the strict schema rejects them instead of accepting unverifiable output. Few-shot has the most deterministic local fixes but the "
      "fewest LLM repairs and failures: its examples show the exact JSON shape, so its errors are small and mechanically fixable. CoT "
      "adds its own failure mode (rationale over the length limit).\n")
    a("**Validation success is not classification accuracy.** Validity only says that the output matched the schema (allowed labels, "
      "verbatim evidence); it says nothing about whether the label is right. "
      + "; ".join(f"{LABEL[s]}: {O[s]['valid']} valid outputs, {C[s]['correct']} correct labels, so {valid_wrong[s]} valid but wrong" for s in STRATS)
      + ". CoT's validity (" + pct(O['cot']['validity_rate']) + ") is close to zero-shot's (" + pct(O['zero_shot']['validity_rate'])
      + "), yet its accuracy is lower, because its valid outputs more often carry the wrong (neutral) label.\n")

    a("## 14. Sensitivity analysis — 300 reviews\n")
    a(f"After the reference annotations were approved, all three strategies were also scored on the full approved set of {sens['n']} reviews "
      f"(`approved_reference/metrics.json`, evaluated {sens['stored']['evaluated_at']}). The only difference from the primary evaluation is "
      f"{len(sens['extra_review_ids'])} review:\n")
    if xr:
        preds = ", ".join(f"{LABEL[s]} → {sens['strategies'][s]['extra_predictions'][xr['review_id']]['prediction']}" for s in STRATS)
        a(table(["Item", "Value"], [
            ["Review", f"`{xr['review_id']}`"], ["Original status", f"{xr['original_status']}: excluded from the original run (text garbled / unreadable)"],
            ["Reference label", f"{xr['label']} ({xr['original_provenance']}; note: \"{xr['note']}\")"],
            ["Approval", f"{xr['decision']} individually by {xr['reviewer']} at {xr['decided_at']}"],
            ["Predictions", f"{preds}; produced separately on 2026-11-14 with the same model (`{run['model_name']}`), prompt version and temperature (`supplementary/`)"],
        ]) + "\n")
    a(table(["Strategy", "n", "Accuracy", "Macro P", "Macro R", "Macro F1", "Weighted F1", "Valid / invalid", "Macro F1 change vs primary"],
            [[LABEL[s], sens["n"], f4(SC[s]["accuracy"]), f4(SC[s]["macro_precision"]), f4(SC[s]["macro_recall"]), f4(SC[s]["macro_f1"]),
              f4(SC[s]["weighted_f1"]), f"{sens['strategies'][s]['valid']} / {sens['strategies'][s]['invalid']}", f"{dmac[s]:+.4f}"] for s in STRATS]) + "\n")
    a(table(["Class F1 (support 299 → 300)"] + [f"{LABEL[s]} 299 → 300" for s in STRATS],
            [[f"{c} ({C['few_shot']['per_class'][c]['support']} → {SC['few_shot']['per_class'][c]['support']})"]
             + [f"{C[s]['per_class'][c]['f1']:.4f} → {SC[s]['per_class'][c]['f1']:.4f}" for s in STRATS] for c in CLASSES]) + "\n")
    a(f"![Sensitivity analysis]({figs['sensitivity']})\n")
    a(f"**Effect.** The added review is labelled neutral and all three strategies predicted neutral for it, so it adds one correct "
      f"neutral prediction and brings the neutral class to {SC['few_shot']['per_class']['neutral']['support']} examples. Positive, negative and mixed "
      f"results are unchanged. Because neutral is so small, one extra correct example raises its F1 noticeably and, since macro-F1 weights "
      f"the four classes equally, moves macro-F1 by {', '.join(f'{dmac[s]:+.4f}' for s in STRATS)} (zero-shot / few-shot / CoT), while accuracy "
      f"moves by about 0.001. The ranking (few-shot > zero-shot > CoT on macro-F1) is the same in both evaluations. Both evaluations use "
      f"identical review ids for all three strategies (primary {d['run_ids']['n']}, sensitivity {sens['n']}).\n")
    a("**Why the 299-review evaluation stays primary.** (1) The review was excluded as unreadable *before* the experiment, so the 299-review "
      "set is the pre-specified evaluation; (2) all 299 × 3 predictions come from one run, while the added review was predicted three days "
      "later in a separate session; (3) the added label is AI-assisted with low confidence and marked debatable; (4) the sensitivity result "
      "shows how strongly a single rare-class example moves macro-F1, which argues for reporting it separately rather than as the headline.\n")

    a("## 15. Discussion\n")
    pos_gain = C['few_shot']['confusion_matrix'][0][0] - C['zero_shot']['confusion_matrix'][0][0]
    acc_gain = C['few_shot']['correct'] - C['zero_shot']['correct']
    a(f"**Why few-shot performed best.** The measured difference is concentrated in one place: few-shot labels {C['few_shot']['confusion_matrix'][0][0]} "
      f"positive reviews correctly versus {C['zero_shot']['confusion_matrix'][0][0]} for zero-shot (+{pos_gain}), which is {100*pos_gain/acc_gain:.0f} % of its "
      f"{acc_gain} additional correct predictions; positive recall rises from {f4(C['zero_shot']['per_class']['positive']['recall'])} to "
      f"{f4(C['few_shot']['per_class']['positive']['recall'])}, and mixed recall from {f4(C['zero_shot']['per_class']['mixed']['recall'])} to "
      f"{f4(C['few_shot']['per_class']['mixed']['recall'])}. Our interpretation: the five demonstrations show concretely what a positive, mixed, negative "
      "and neutral review looks like, so the model no longer treats an ordinary positive review as opinion-free (neutral); and because they "
      "show the exact JSON shape, few-shot also has the best first-attempt validity and the fewest failures. (This is an interpretation of the "
      "error pattern; the run does not isolate which demonstration causes which effect.)\n")
    a(f"**Why CoT did not help.** CoT predicts neutral {C['cot']['per_class']['neutral']['predicted']} times for {cc['neutral']} true neutrals (recall "
      f"{f4(C['cot']['per_class']['neutral']['recall'])}, precision {C['cot']['per_class']['neutral']['precision']:.4f}), mostly for positive reviews "
      f"({C['cot']['confusion_matrix'][0][2]}) and negative reviews ({C['cot']['confusion_matrix'][1][2]}). The procedure's step 6 asks the model to weigh "
      "positive against negative aspects; our interpretation is that the 3B model, performing this silently, often concludes 'no clear "
      "opinion' and falls back to neutral. The procedure did not improve aspect handling either (aspect-detection micro-F1 is lower than "
      "zero-shot in `metrics.json`) and it adds one failure mode (rationale over the length limit). Structured reasoning prompts are generally "
      "expected to help larger models more than a 3B model; this run does not test that.\n")
    a("**Class imbalance.** The sample is stratified by star rating, not by sentiment, so neutral reviews are rare. Macro-F1 is reported "
      "as the headline metric because accuracy would hide the minority classes; the price is that a 5-example class has a large, noisy "
      "influence on every macro average.\n")

    a("## 16. Limitations\n")
    a(f"- **AI-assisted reference labels.** {ref['ai_assisted']} of {ref['evaluable']} reference labels were generated by an AI model "
      f"({ref['ai_model']}) and approved by the mentor ({ref['approved_by']}); only {ref['human']} is human. The metrics measure **agreement "
      "with these labels**. If the evaluated model and the labelling model share biases, agreement can overstate accuracy; the ranking "
      "should be confirmed on independent human labels (a blind two-person workflow exists in the app; 0 of 300 human labels are "
      f"verified). {ref['debatable']} AI labels are marked `DEBATABLE`. All 300 reference labels were later approved by a named reviewer; "
      "approval records acceptance and does not turn them into independent human annotations.\n"
      "- **Sensitivity to single examples:** with 5–6 neutral reviews, one review changes few-shot macro-F1 by about 0.03 (§14).\n"
      f"- **Small rare classes:** neutral n = {cc['neutral']}, mixed n = {cc['mixed']}; their per-class scores have wide uncertainty.\n"
      "- **One model, one run:** `qwen2.5:3b` at temperature 0, one attempt per review; results may differ for larger models, other prompts "
      "or other demonstrations, and no confidence intervals or significance tests were computed.\n"
      "- **Overall sentiment only** is analysed here in depth; aspect and complaint metrics are in `metrics.json` and the D1/D2 reports.\n"
      "- The sample (300 of 1,500 reviews, stratified by rating) does not reflect the natural rating distribution.\n")

    a("## 17. Conclusion\n")
    a(f"Few-shot prompting is the best strategy for this task and model: macro-F1 {f4(C['few_shot']['macro_f1'])}, accuracy "
      f"{f4(C['few_shot']['accuracy'])}, validity {pct(O['few_shot']['validity_rate'])}, at a moderate cost in tokens and latency. "
      "Zero-shot and CoT blur the positive/neutral boundary, and all strategies under-detect mixed and neutral reviews. Every number was "
      "independently recomputed and matches the stored results. The evaluation is against mentor-approved, mostly AI-assisted reference "
      "labels; confirming the ranking on independent human labels is the main open step. The 300-review sensitivity analysis "
      f"(macro-F1 {f4(SC['zero_shot']['macro_f1'])} / {f4(SC['few_shot']['macro_f1'])} / {f4(SC['cot']['macro_f1'])}) confirms the ranking and shows how "
      "sensitive macro-F1 is to the rare neutral class.\n")

    a("## Appendix A. Verification\n")
    a(f"- Recomputed (stdlib, raw files) vs stored `metrics.json`: **{nver} / {len(ver)} values identical** (tolerance 1e-4), including all "
      "confusion matrices.\n"
      f"- Same review set for all strategies: **{'yes' if d['same_review_set'] else 'NO'}** ({d['run_ids']['n']} ids; missing outcomes "
      f"{', '.join(str(O[s]['missing']) for s in STRATS)}).\n"
      "- Also verified separately: `scripts/verify_metrics_independent.py` (all match), `scripts/verify_d2_records.py` (11 / 11 PASS).\n"
      f"- Sensitivity analysis recomputed from the raw files (original predictions + `supplementary/`, approved labels) vs "
      f"`approved_reference/metrics.json`: **{sver_ok} / {len(sens['verification'])} values identical**, including the confusion matrices; "
      f"same {sens['n']} ids for all strategies: **{'yes' if sens['same_review_set'] else 'NO'}**; label corrections applied: {sens['corrections_applied']}.\n"
      f"- Charts: every value label checked against the data ({sum(c['labels_checked'] for c in charts_ok)} labels in {len(charts_ok)} charts, "
      f"{'all present' if all(c['match'] for c in charts_ok) else 'MISSING LABELS'}).\n")
    a("**Values transcribed from the dashboard screenshots vs verified data**\n")
    if tr_bad:
        a(table(["Strategy", "Value", "Transcribed", "Verified (run data)", "Explanation"],
                [[t["strategy"], t["value"], t["transcribed"], f"{t['verified']:.4f}" if isinstance(t["verified"], float) else t["verified"],
                  "183 correct / 299 = 0.61204" if (t["strategy"], t["value"]) == ("zero_shot", "accuracy") else "–"] for t in tr_bad]) + "\n")
        a(f"{len(tr) - len(tr_bad)} of {len(tr)} transcribed values match the run data exactly; the value(s) above differ and the **verified** "
          "value is used throughout this report.\n")
    else:
        a(f"All {len(tr)} transcribed values match the run data.\n")
    a("Data file with every number used: `reports/d2/D2_REPORT_DATA.json`.\n")
    return "\n".join(L)


# ============================================================================ minimal Markdown -> HTML (headings, tables, lists, images, emphasis)
def inline(t: str) -> str:
    t = html.escape(t, quote=False)
    t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t)
    t = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"(?<![\w*])\*([^*]+)\*(?!\w)", r"<i>\1</i>", t)
    return t


def md_to_html(md: str, svgs: dict[str, str]) -> str:
    out, lines, i = [], md.split("\n"), 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("!["):
            src = re.search(r"\(([^)]+)\)", ln).group(1)
            cap = re.search(r"!\[([^\]]*)\]", ln).group(1)
            out.append(f"<figure>{svgs.get(src, '')}<figcaption>{html.escape(cap)}</figcaption></figure>")
        elif ln.startswith("# "):
            out.append(f"<h1>{inline(ln[2:])}</h1>")
        elif ln.startswith("## "):
            out.append(f"<h2>{inline(ln[3:])}</h2>")
        elif ln.startswith("> "):
            out.append(f"<div class='ok'>{inline(ln[2:])}</div>")
        elif ln.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append(lines[i])
                i += 1
            cells = lambda r: [c.strip() for c in r.strip().strip("|").split("|")]
            h = "".join(f"<th>{inline(c)}</th>" for c in cells(rows[0]))
            body = "".join("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in cells(r)) + "</tr>" for r in rows[2:])
            out.append(f"<table><tr>{h}</tr>{body}</table>")
            continue
        elif re.match(r"^(\d+\.|-) ", ln):
            tag = "ol" if ln[0].isdigit() else "ul"
            items = []
            while i < len(lines) and re.match(r"^(\d+\.|-) ", lines[i]):
                items.append(re.sub(r"^(\d+\.|-) ", "", lines[i]))
                i += 1
            out.append(f"<{tag}>" + "".join(f"<li>{inline(x)}</li>" for x in items) + f"</{tag}>")
            continue
        elif ln.strip():
            out.append(f"<p>{inline(ln)}</p>")
        i += 1
    return "".join(out)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="20261111-030912_qwen2.5-3b")
    p.add_argument("--no-pdf", action="store_true")
    args = p.parse_args()
    run_dir = ROOT / "outputs" / "live" / "experiments" / args.run
    stored = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    data = recompute(run_dir)
    ver = verify(data, stored)
    bad = [v for v in ver if not v["match"]]
    if bad or not data["same_review_set"]:
        print(json.dumps(bad[:20], indent=1, default=str))
        sys.exit("VERIFICATION FAILED: recomputed values differ from metrics.json (or review sets differ); no report written")
    tr = check_transcription(data)
    sens = recompute_sensitivity(run_dir)
    sbad = [v for v in sens["verification"] if not v["match"]]
    if sbad or not sens["same_review_set"]:
        print(json.dumps(sbad[:20], indent=1, default=str))
        sys.exit("VERIFICATION FAILED: 300-review sensitivity values differ from approved_reference/metrics.json; no report written")

    C = {s: data["strategies"][s]["classification"] for s in STRATS}
    O = {s: data["strategies"][s]["operational"] for s in STRATS}
    figdir = ROOT / "reports" / "d2" / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    charts = {
        "overall": grouped_bars("Overall sentiment — agreement with the reference labels (299 reviews)", ["Accuracy", "Macro precision", "Macro recall", "Macro F1", "Weighted F1"],
                                {s: [C[s][k] for k in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1")] for s in STRATS}, 1.0),
        "perclass": grouped_bars("Per-class F1", [f"{c} (n={C['few_shot']['per_class'][c]['support']})" for c in CLASSES],
                                 {s: [C[s]["per_class"][c]["f1"] for c in CLASSES] for s in STRATS}, 1.0),
        "efficiency": grouped_bars("Per review: mean latency (s), mean input tokens ÷ 100, mean output tokens ÷ 10", ["Latency (s)", "Input tokens ÷ 100", "Output tokens ÷ 10"],
                                   {s: [O[s]["mean_latency_s"], O[s]["mean_input_tokens"] / 100, O[s]["mean_output_tokens"] / 10] for s in STRATS}, 60, fmt="{:.1f}"),
        "validity": grouped_bars("Output reliability (% of 299 reviews)", ["Valid", "Valid at first attempt", "Needed LLM repair", "Controlled failure"],
                                 {s: [100 * O[s]["validity_rate"], 100 * O[s]["first_attempt_valid_rate"], 100 * O[s]["needed_llm_repair"] / 299,
                                      100 * O[s]["failed"] / 299] for s in STRATS}, 100, fmt="{:.1f}"),
    }
    SC = {s: sens["strategies"][s]["classification"] for s in STRATS}
    charts["sensitivity"] = grouped_bars("Macro-F1: primary (299 reviews) vs sensitivity (300 reviews)", [LABEL[s] for s in STRATS],
                                         {"primary": [C[s]["macro_f1"] for s in STRATS], "sensitivity": [SC[s]["macro_f1"] for s in STRATS]}, 1.0)
    f3 = lambda v: f"{v:.3f}"
    f1 = lambda v: f"{v:.1f}"
    expected = {
        "overall": [f3(C[s][k]) for s in STRATS for k in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1")],
        "perclass": [f3(C[s]["per_class"][c]["f1"]) for s in STRATS for c in CLASSES],
        "efficiency": [f1(O[s]["mean_latency_s"]) for s in STRATS] + [f1(O[s]["mean_input_tokens"] / 100) for s in STRATS] + [f1(O[s]["mean_output_tokens"] / 10) for s in STRATS],
        "validity": [f1(100 * O[s]["validity_rate"]) for s in STRATS] + [f1(100 * O[s]["first_attempt_valid_rate"]) for s in STRATS],
        "sensitivity": [f3(C[s]["macro_f1"]) for s in STRATS] + [f3(SC[s]["macro_f1"]) for s in STRATS],
    }
    cm_svgs = [heatmap(LABEL[s], C[s]["confusion_matrix"], [C[s]["per_class"][c]["invalid"] for c in CLASSES]) for s in STRATS]
    charts["cm"] = "<div style='display:flex;gap:6px;flex-wrap:wrap'>" + "".join(cm_svgs) + "</div>"
    figs = {}
    for k, svg in charts.items():
        if k == "cm":
            for s, one in zip(STRATS, cm_svgs):
                (figdir / f"confusion_{s}.svg").write_text(one, encoding="utf-8")
            figs[k] = "d2/figures/confusion_few_shot.svg"
        else:
            (figdir / f"{k}.svg").write_text(svg, encoding="utf-8")
            figs[k] = f"d2/figures/{k}.svg"
    charts_ok = chart_check(charts, expected)
    for s, one in zip(STRATS, cm_svgs):
        cells = [str(v) for row in C[s]["confusion_matrix"] for v in row]
        charts_ok.append({"chart": f"confusion_{s}", "labels_checked": len(cells),
                          "missing": [x for x in cells if f">{x}<" not in one], "match": all(f">{x}<" in one for x in cells)})
    if not all(c["match"] for c in charts_ok):
        sys.exit(f"CHART CHECK FAILED: {[c for c in charts_ok if not c['match']]}")
    md = build_markdown(data, ver, tr, figs, sens, charts_ok)
    md_cm = md.replace(f"![Confusion matrices]({figs['cm']})",
                       " ".join(f"![{LABEL[s]} confusion matrix](d2/figures/confusion_{s}.svg)" for s in STRATS))
    (ROOT / "reports" / "D2_FINAL_REPORT.md").write_text(md_cm, encoding="utf-8")
    out_data = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **data,
                "verification": {"compared": len(ver), "identical": sum(v["match"] for v in ver), "rows": ver},
                "transcription_check": tr, "sensitivity_300": sens, "chart_check": charts_ok}
    (ROOT / "reports" / "d2" / "D2_REPORT_DATA.json").write_text(json.dumps(out_data, indent=1, default=str), encoding="utf-8")
    svg_by_src = {figs[k]: charts[k] for k in charts}
    body = md_to_html(md, svg_by_src)
    doc = (f"<!doctype html><html><head><meta charset='utf-8'><title>ReviewIQ — D2 Final Report</title><style>{CSS}"
           "figure svg{max-width:100%;height:auto} figure{text-align:center}</style></head><body>"
           "<div class='muted'>PES University · Prompt Engineering Mini Project · Deliverable 2</div>"
           f"{body}</body></html>")
    out_html = ROOT / "reports" / "final" / "D2_Final_Report.html"
    out_html.write_text(doc, encoding="utf-8")
    pdf = None if args.no_pdf else to_pdf(out_html)
    print(json.dumps({"verified_values": f"{sum(v['match'] for v in ver)}/{len(ver)}", "same_review_set": data["same_review_set"],
                      "sensitivity_verified": f"{sum(v['match'] for v in sens['verification'])}/{len(sens['verification'])}",
                      "sensitivity_same_review_set": sens["same_review_set"], "charts": f"{sum(c['match'] for c in charts_ok)}/{len(charts_ok)}",
                      "transcription_mismatches": [t for t in tr if not t["match"]],
                      "markdown": "reports/D2_FINAL_REPORT.md", "html": str(out_html.relative_to(ROOT)), "pdf": str(pdf.relative_to(ROOT)) if pdf else None,
                      "figures": sorted(str(x.relative_to(ROOT)) for x in figdir.glob("*.svg"))}, indent=1, default=str))


if __name__ == "__main__":
    main()
