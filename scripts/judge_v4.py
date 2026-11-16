"""Judge v4 (claim-level faithfulness evaluation) and negative control v3: build, run, report.

  python scripts/judge_v4.py build-nc --source dataset__few_shot__qwen2.5-3b__v1     # freeze nc-v3 (no LLM)
  python scripts/judge_v4.py run      --source ... --objects original nc corrected  # sequential local LLM calls, resumable
  python scripts/judge_v4.py repro    --source ...                                  # repeat a fixed subset (separate store)
  python scripts/judge_v4.py report   --source ...                                  # evaluation files + metrics (no LLM)

Stores (append-only; earlier v1/v2/v3 results are never touched):
  outputs/<mode>/judgements/v4/{claim_verdicts,failures}.jsonl, run_manifest.json   LLM claim verdicts (raw responses kept)
  outputs/<mode>/judgements/v4_repro/...                                            repeat of a fixed subset
  outputs/<mode>/d3/negative_control_v3/<source>/{cases.jsonl,manifest.json}         pre-registered nc-v3 dataset
  outputs/<mode>/d3/judge_v4/<source>/                                               evaluations, CSV exports, metrics
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import load_settings  # noqa: E402
from evaluation.claim_checks import Stats  # noqa: E402
from evaluation.judge_v4 import (  # noqa: E402
    Item, V4Store, evaluate_items, get_judge_v4_client, item_from_summary, load_review_index, prompt_sha256, run_config, run_llm,
)
from evaluation.negative_control_v3 import (  # noqa: E402
    DETECTION_RULE, ERROR_TYPES, NC_VERSION, SEED, confusion, dataset_sha256, flagged, generate, to_claim, wilson,
)
from llm.base import LLMError, LLMUnavailableError  # noqa: E402
from schemas.claim_schema import WEIGHTS, ClaimStatus, SummaryEvaluationV4, derive_scores  # noqa: E402
from schemas.evaluation_schema import ProductSummary  # noqa: E402
from scripts.run_judge import load_inputs  # noqa: E402
from utils.gpu_guard import GpuGuard, GpuTooHot  # noqa: E402
from utils.storage import read_json, read_jsonl, write_json_atomic  # noqa: E402

NOW = lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")  # noqa: E731


def nc_dir(mode: Path, source: str) -> Path:
    return mode / "d3" / "negative_control_v3" / source


def out_dir(mode: Path, source: str) -> Path:
    return mode / "d3" / "judge_v4" / source


def corrected_items(mode: Path, source: str, records) -> list[Item]:
    items = []
    for fname, tag in (("corrected_summaries", "corr1"), ("corrected_summaries_v2", "corr2")):
        p = mode / "d3" / fname / source / "corrected_summaries.jsonl"
        for r in read_jsonl(p):
            s = ProductSummary.model_validate(r["corrected_summary"])
            items.append(item_from_summary("corrected", s, records, original_summary_id=r["original_summary_id"],
                                           extra={"correction_set": tag}))
    return items


def load_nc(mode: Path, source: str) -> list[dict]:
    d = nc_dir(mode, source)
    cases = read_jsonl(d / "cases.jsonl")
    man = read_json(d / "manifest.json", default=None)
    if not cases or not man:
        raise SystemExit("nc-v3 dataset not built: run `build-nc` first")
    if dataset_sha256(cases) != man["dataset_sha256"]:
        raise SystemExit("nc-v3 cases.jsonl does not match the SHA-256 in its manifest: refusing to use a modified dataset")
    return cases


def nc_items(cases: list[dict], summaries: dict[str, ProductSummary], records) -> list[Item]:
    items = []
    for c in cases:
        summ = summaries[c["product_id"]]
        st = Stats.from_aggregate(summ.aggregate, records)
        for which, kind in (("corrupted_claim", "nc_corrupted"), ("clean_claim", "nc_clean")):
            cl = to_claim(c[which])
            items.append(Item(kind=kind, summary_id=cl.summary_id, product_id=c["product_id"], claims=[cl], stats=st, frequent_list=[],
                              original_summary_id=c["original_summary_id"], extra={"case_id": c["case_id"], "error_type": c["error_type"]}))
    return items


def repro_subset(originals: list[Item], ncs: list[Item]) -> list[Item]:
    """Fixed rule (decided before the repeat): the first 2 point claims of every original summary, plus the claims of the
    first 6 review-based nc-v3 cases (corrupted and clean)."""
    out = []
    for it in originals:
        pts = [c for c in it.claims if c.kind == "point"][:2]
        if pts:
            out.append(Item(it.kind, it.summary_id, it.product_id, pts, it.stats, it.frequent_list, it.original_summary_id))
    review_nc = [it for it in ncs if it.claims[0].kind == "point"]
    case_ids = []
    for it in review_nc:
        if it.extra["case_id"] not in case_ids:
            case_ids.append(it.extra["case_id"])
    keep = set(case_ids[:6])
    out += [it for it in review_nc if it.extra["case_id"] in keep]
    return out


# --------------------------------------------------------------------------- commands
def cmd_build_nc(s, mode: Path, source: str, summaries, records, reviews) -> None:
    human = read_json(s.paths.annotations / "human_annotations.json", default={}).get("labels", {})
    cases = generate(summaries, human, reviews, records, seed=SEED)
    d = nc_dir(mode, source)
    sha = dataset_sha256(cases)
    man_path = d / "manifest.json"
    old = read_json(man_path, default=None)
    if old and old["dataset_sha256"] != sha:
        raise SystemExit(f"a different nc-v3 dataset is already frozen ({old['dataset_sha256'][:12]}); not overwriting it")
    if old:
        print(f"nc-v3 already frozen and identical ({sha[:12]}), {len(cases)} cases")
        return
    d.mkdir(parents=True, exist_ok=True)
    (d / "cases.jsonl").write_text("".join(json.dumps(c, sort_keys=True, ensure_ascii=False) + "\n" for c in cases), encoding="utf-8")
    by_type = {t: sum(1 for c in cases if c["error_type"] == t) for t in ERROR_TYPES}
    write_json_atomic(man_path, {
        "nc_version": NC_VERSION, "frozen_at": NOW(), "seed": SEED, "source": source, "dataset_sha256": sha, "cases": len(cases),
        "claims_to_judge": 2 * len(cases), "cases_by_type": by_type, "detection_rule": DETECTION_RULE,
        "positive_class": "corrupted claim; each case also has a matched clean claim of the same form (negative class)",
        "frozen_before_judging": True, "judge_v4_prompt_sha256_at_freeze": prompt_sha256(),
        "construction": "evaluation/negative_control_v3.py (seeded, no LLM). Review-based cases were read against their review "
                        "texts by the AI coding assistant before freezing; that QA is not a human verification.",
        "previous_control": "nc-v2 (30 copies, outputs/<mode>/d3/judge_negative_control_v2) is unchanged and reported separately",
    })
    print(f"nc-v3 frozen: {len(cases)} cases, sha256 {sha}")
    print(json.dumps(by_type))


def _guarded_run(s, items, reviews, store: V4Store, *, is_mock: bool, label: str, limit=None) -> dict:
    try:
        client = get_judge_v4_client(s, use_mock=is_mock)
    except LLMUnavailableError as exc:
        sys.exit(f"Judge unavailable: {exc}")
    cfg = run_config(client, s)
    store.folder.mkdir(parents=True, exist_ok=True)
    man_path = store.folder / "run_manifest.json"
    man = read_json(man_path, default=None) or {"created_at": NOW(), "config": cfg, "sessions": []}
    if man["config"]["prompt_sha256"] != cfg["prompt_sha256"] or man["config"]["judge_model"] != cfg["judge_model"]:
        sys.exit("this v4 store was produced with a different prompt or model; use a new store instead of mixing runs")
    run_id = datetime.now(timezone.utc).strftime(f"{label}-%Y%m%dT%H%M%SZ")
    man["sessions"].append({"run_id": run_id, "started_at": NOW(), "objects": sorted({it.kind for it in items})})
    write_json_atomic(man_path, man)
    guard = GpuGuard(s.paths.root / "logs" / "gpu_guard.csv")
    n_llm = sum(1 for it in items for c in it.claims if c.kind == "point" or c.kind == "opinion_sentence")
    print(f"Judge v4 {cfg['judge_provider']}/{cfg['judge_model']} constrained={cfg['constrained_decoding']} · {len(items)} objects, "
          f"<= {n_llm} candidate LLM claims (stored ones are skipped)", flush=True)
    if is_mock:
        print("*** MOCK judge: placeholder verdicts, NOT an evaluation result ***")

    def progress(it, c, ok, rec):
        tag = f"{rec.get('status', 'FAIL'):19s} q={'Y' if rec.get('quote_verified') else 'n'}" if ok else "VALIDATION FAILURE"
        print(f"  {it.kind:12s} {c.claim_id:42s} {tag} {rec['latency_s']:.0f}s", flush=True)

    try:
        stats = run_llm(client, items, reviews, store, config=cfg, run_id=run_id, before_call=guard.wait_until_safe, progress=progress, limit=limit)
    except (LLMError, GpuTooHot) as exc:
        man["sessions"][-1].update({"stopped_at": NOW(), "stop_reason": str(exc)[:300], "max_gpu_temp_c": guard.max_seen})
        write_json_atomic(man_path, man)
        sys.exit(f"STOPPED: {exc}\nEverything judged so far is stored. Re-run the same command to resume.")
    man["sessions"][-1].update({"finished_at": NOW(), **stats, "max_gpu_temp_c": guard.max_seen})
    write_json_atomic(man_path, man)
    print(json.dumps(stats))
    return stats


# --------------------------------------------------------------------------- report helpers
def claim_rows(evals: list[SummaryEvaluationV4]) -> list[dict]:
    rows = []
    for ev in evals:
        for e in ev.claim_evaluations:
            c = e.claim
            rows.append({"evaluated_object": ev.evaluated_object, "summary_id": ev.summary_id, "product_id": ev.product_id,
                         "claim_id": c.claim_id, "source_field": c.source_field, "claim_kind": c.kind, "category": c.category,
                         "polarity": c.polarity, "is_numeric": c.is_numeric, "cited_review_ids": " ".join(c.cited_review_ids),
                         "claim_text": c.text, "status": e.status.value, "method": e.method,
                         "deterministic_status": e.deterministic_status.value if e.deterministic_status else "",
                         "llm_status": e.llm_status.value if e.llm_status else "",
                         "llm_status_effective": e.llm_status_effective.value if e.llm_status_effective else "",
                         "llm_quote": e.llm_quote or "", "llm_quote_verified": "" if e.llm_quote_verified is None else e.llm_quote_verified,
                         "numeric_checks": len(e.numeric_checks), "numeric_failed": sum(1 for n in e.numeric_checks if not n.passed),
                         "numeric_detail": " | ".join(f"{n.mention}: {'PASS' if n.passed else n.severity.upper()} ({n.calculation})" for n in e.numeric_checks),
                         "explanation": e.explanation})
    return rows


def summary_rows(evals: list[SummaryEvaluationV4]) -> list[dict]:
    rows = []
    for ev in evals:
        rows.append({"evaluated_object": ev.evaluated_object, "summary_id": ev.summary_id, "product_id": ev.product_id,
                     "original_summary_id": ev.original_summary_id or "", "total_claims": ev.total_claims, **{k: v for k, v in ev.counts.items()},
                     "verifiable_claims": ev.verifiable_claims, "evidence_coverage": ev.evidence_coverage, "numeric_claims": ev.numeric_claims,
                     "numeric_passed": ev.numeric_claims_passed, "numeric_failed": ev.numeric_claims_failed,
                     "faithfulness_score": ev.faithfulness_score, "faithfulness_band_1_5": ev.faithfulness_band_1_5,
                     "validity_status": ev.validity_status, "unsupported_claims": " ".join(ev.unsupported_claims),
                     "contradicted_claims": " ".join(ev.contradicted_claims)})
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def kendall_tau(a: list[float], b: list[float]) -> float | None:
    n = len(a)
    conc = disc = 0
    for i in range(n):
        for j in range(i + 1, n):
            x, y = (a[i] - a[j]), (b[i] - b[j])
            if x * y > 0:
                conc += 1
            elif x * y < 0:
                disc += 1
    return round((conc - disc) / (conc + disc), 3) if conc + disc else None


def sensitivity(evals: list[SummaryEvaluationV4]) -> dict:
    schemes = {
        "primary (S1, P0.5, U0, C-1, caps)": (WEIGHTS, True),
        "P=0.25": ({**WEIGHTS, ClaimStatus.PARTIALLY_SUPPORTED: 0.25}, True),
        "P=0.75": ({**WEIGHTS, ClaimStatus.PARTIALLY_SUPPORTED: 0.75}, True),
        "C=0 (contradicted = unsupported)": ({**WEIGHTS, ClaimStatus.CONTRADICTED: 0.0}, True),
        "C=-2": ({**WEIGHTS, ClaimStatus.CONTRADICTED: -2.0}, True),
        "strict (only SUPPORTED counts)": ({ClaimStatus.SUPPORTED: 1, ClaimStatus.PARTIALLY_SUPPORTED: 0, ClaimStatus.UNSUPPORTED: 0,
                                             ClaimStatus.CONTRADICTED: 0}, True),
        "primary weights, no caps": (WEIGHTS, False),
    }
    base = None
    out = {}
    for name, (w, caps) in schemes.items():
        sc = [derive_scores([e.status for e in ev.claim_evaluations], w, caps=caps)["faithfulness_score"] or 0.0 for ev in evals]
        if base is None:
            base = sc
        out[name] = {"scores": dict(zip([ev.product_id for ev in evals], sc)), "kendall_tau_vs_primary": kendall_tau(base, sc),
                     "mean": round(sum(sc) / len(sc), 1) if sc else None}
    return out


def human_sentiment_check(s, summaries: list[ProductSummary]) -> dict:
    """Separate analysis: product sentiment distribution from the 300 independent human annotations (overall sentiment
    only; not second-person verified) vs the extraction-based aggregate used by the summaries."""
    labels = read_json(s.paths.annotations / "human_annotations.json", default={}).get("labels", {})
    rows = []
    for summ in summaries:
        ids = set(summ.aggregate.review_ids)
        hs = [labels[r]["overall_sentiment"] for r in ids if r in labels and labels[r].get("status") == "completed"]
        n = len(hs)
        hp = {k: round(100 * sum(1 for x in hs if x == k) / n, 1) if n else None for k in ("positive", "negative", "neutral", "mixed")}
        rows.append({"product_id": summ.product_id, "reviews_in_aggregate": len(ids), "human_labelled": n,
                     "extraction_pct": summ.aggregate.sentiment_percentages, "human_pct": hp,
                     "abs_diff_positive_pts": round(abs(hp["positive"] - summ.aggregate.sentiment_percentages.get("positive", 0)), 1) if n else None,
                     "abs_diff_negative_pts": round(abs(hp["negative"] - summ.aggregate.sentiment_percentages.get("negative", 0)), 1) if n else None,
                     "same_majority_direction": (max(("positive", "negative"), key=lambda k: hp[k]) ==
                                                 max(("positive", "negative"), key=lambda k: summ.aggregate.sentiment_percentages.get(k, 0))) if n else None})
    import pandas as pd

    rv = pd.read_csv(s.paths.reviews_csv, dtype={"review_id": str}, keep_default_na=False).set_index("review_id")["rating"].to_dict()
    xtab: dict[str, dict[str, int]] = {}
    for rid, v in labels.items():
        if rid in rv and v.get("status") == "completed":
            row = xtab.setdefault(f"{float(rv[rid]):.0f} star", {})
            row[v["overall_sentiment"]] = row.get(v["overall_sentiment"], 0) + 1
    low = [r for k, r in xtab.items() if k in ("1 star", "2 star")]
    low_pos, low_n = sum(r.get("positive", 0) for r in low), sum(sum(r.values()) for r in low)
    aspects = sum(1 for v in labels.values() if v.get("aspects"))
    complaints = sum(1 for v in labels.values() if v.get("complaint_categories"))
    return {"note": "Human labels are independent, human-created and NOT second-person verified; they cover overall sentiment. "
                    f"Only {aspects} of {len(labels)} carry aspect labels and {complaints} carry complaint labels, so aspect/complaint "
                    "statistics cannot be checked against them. This is a separate analysis, not claim-level ground truth.",
            "rows": rows, "rating_by_human_label": dict(sorted(xtab.items())),
            "low_star_labelled_positive": low_pos, "low_star_reviews": low_n,
            "interpretation": (f"{low_pos} of {low_n} one- and two-star reviews carry the human label 'positive', so the human labels "
                               "disagree with the star ratings far more than the extraction does; this check is therefore inconclusive "
                               "about extraction quality and the human labels themselves need review."),
            "mean_abs_diff_positive_pts": round(sum(r["abs_diff_positive_pts"] for r in rows if r["abs_diff_positive_pts"] is not None) / len(rows), 1),
            "mean_abs_diff_negative_pts": round(sum(r["abs_diff_negative_pts"] for r in rows if r["abs_diff_negative_pts"] is not None) / len(rows), 1),
            "products_same_majority_direction": sum(1 for r in rows if r["same_majority_direction"]),
            "products": len(rows)}


def nc_analysis(cases: list[dict], nc_evals: list[SummaryEvaluationV4], orig_by_summary: dict[str, SummaryEvaluationV4]) -> dict:
    by_sid = {ev.summary_id: ev for ev in nc_evals}
    rows = []
    for c in cases:
        k = by_sid.get(c["clean_claim"]["summary_id"])
        x = by_sid.get(c["corrupted_claim"]["summary_id"])
        if not k or not x:
            continue
        ek, ex = k.claim_evaluations[0], x.claim_evaluations[0]

        def v(e, f):
            val = getattr(e, f)
            return val.value if val is not None else None

        # copy-level score: the original summary's claim statuses plus the planted claim (v4 judges each claim on its own
        # evidence, so the original's other claims and their verdicts are identical in the copy)
        o = orig_by_summary.get(c["original_summary_id"])
        base = [e.status for e in o.claim_evaluations] if o else []
        row = {"case_id": c["case_id"], "product_id": c["product_id"], "error_type": c["error_type"], "variant": c["variant"],
               "planted_change": c["planted_change"], "expected_status_corrupted": c["expected_status_corrupted"],
               "corrupted_text": c["corrupted_claim"]["text"], "clean_text": c["clean_claim"]["text"],
               "corrupted_status": v(ex, "status"), "clean_status": v(ek, "status"),
               "corrupted_deterministic": v(ex, "deterministic_status"), "clean_deterministic": v(ek, "deterministic_status"),
               "corrupted_llm": v(ex, "llm_status_effective"), "clean_llm": v(ek, "llm_status_effective"),
               "corrupted_llm_raw": v(ex, "llm_status"), "clean_llm_raw": v(ek, "llm_status"),
               "corrupted_quote_verified": ex.llm_quote_verified, "clean_quote_verified": ek.llm_quote_verified,
               "corrupted_explanation": ex.explanation, "clean_explanation": ek.explanation,
               "copy_score_original": derive_scores(base)["faithfulness_score"] if o else None,
               "copy_score_with_corrupted": derive_scores(base + [ex.status])["faithfulness_score"] if o else None,
               "copy_score_with_clean": derive_scores(base + [ek.status])["faithfulness_score"] if o else None}
        for rule in ("primary", "strict", "lenient"):
            row[f"corrupted_flagged_{rule}"] = flagged(row["corrupted_status"], c["error_type"], rule)
            row[f"clean_flagged_{rule}"] = flagged(row["clean_status"], c["error_type"], rule)
        rows.append(row)

    def layer_rows(layer):
        # deterministic layer: no deterministic verdict = not flagged; LLM layer: only claims the LLM judged
        out = []
        for r in rows:
            if layer == "deterministic":
                out.append({"error_type": r["error_type"], "corrupted_s": r["corrupted_deterministic"] or "SUPPORTED",
                            "clean_s": r["clean_deterministic"] or "SUPPORTED"})
            elif layer == "llm":
                if r["corrupted_llm_raw"] is None and r["clean_llm_raw"] is None:
                    continue
                out.append({"error_type": r["error_type"], "corrupted_s": r["corrupted_llm"], "clean_s": r["clean_llm"]})
            else:
                out.append({"error_type": r["error_type"], "corrupted_s": r["corrupted_status"], "clean_s": r["clean_status"]})
        return out

    metrics = {rule: {layer: confusion(layer_rows(layer), "s", rule) for layer in ("combined", "deterministic", "llm")}
               for rule in ("primary", "strict", "lenient")}
    by_type = {}
    for t in ERROR_TYPES:
        rs = [r for r in rows if r["error_type"] == t]
        if not rs:
            continue
        det = sum(1 for r in rs if r["corrupted_flagged_primary"])
        fp = sum(1 for r in rs if r["clean_flagged_primary"])
        llm_rs = [r for r in rs if r["corrupted_llm_raw"] is not None]
        by_type[t] = {"cases": len(rs), "detected": det, "detection_rate": round(det / len(rs), 3), "detection_wilson95": wilson(det, len(rs)),
                      "clean_flagged": fp, "specificity": round(1 - fp / len(rs), 3), "specificity_wilson95": wilson(len(rs) - fp, len(rs)),
                      "expected_status": rs[0]["expected_status_corrupted"],
                      "status_distribution_corrupted": {s.value: sum(1 for r in rs if r["corrupted_status"] == s.value) for s in ClaimStatus},
                      "status_distribution_clean": {s.value: sum(1 for r in rs if r["clean_status"] == s.value) for s in ClaimStatus},
                      "llm_judged": len(llm_rs),
                      "llm_only_detected": sum(1 for r in llm_rs if flagged(r["corrupted_llm"], t, "primary")),
                      "deterministic_only_detected": sum(1 for r in rs if flagged(r["corrupted_deterministic"] or "SUPPORTED", t, "primary"))}
    paired = [r for r in rows if r["copy_score_with_corrupted"] is not None]
    copy = {"pairs": len(paired),
            "corrupted_copy_lower": sum(1 for r in paired if r["copy_score_with_corrupted"] < r["copy_score_with_clean"]),
            "equal": sum(1 for r in paired if r["copy_score_with_corrupted"] == r["copy_score_with_clean"]),
            "corrupted_copy_higher": sum(1 for r in paired if r["copy_score_with_corrupted"] > r["copy_score_with_clean"]),
            "note": "copy-level scores are recomputed from the original summary's claim verdicts plus the planted claim's verdict"}
    stat_types = [t for t in ERROR_TYPES if t.startswith("incorrect_") and t != "incorrect_aspect_attribution"]
    review_rows = [r for r in rows if r["error_type"] not in stat_types]
    review_layer = {layer: confusion([x for x in [
        {"error_type": r["error_type"], "corrupted_s": (r["corrupted_deterministic"] or "SUPPORTED") if layer == "deterministic" else
         r["corrupted_llm"] if layer == "llm" else r["corrupted_status"],
         "clean_s": (r["clean_deterministic"] or "SUPPORTED") if layer == "deterministic" else r["clean_llm"] if layer == "llm" else r["clean_status"]}
        for r in review_rows]], "s", "primary") for layer in ("combined", "deterministic", "llm")}
    return {"rows": rows, "metrics": metrics, "by_type": by_type, "copy_level_scores": copy,
            "review_based_types_primary": review_layer, "statistic_types": stat_types,
            "detection_rule": DETECTION_RULE}


def prior_judges(mode: Path, source: str) -> dict:
    """Headline numbers of judges v1-v3 and nc-v2, read from their stored artifacts (never recomputed with new rules)."""
    v1 = read_jsonl(mode / "judgements" / "judgements.jsonl")
    v1f = read_jsonl(mode / "judgements" / "judge_failures.jsonl")
    v1 = [j for j in v1 if j.get("prompt_version", "v1") == "v1"]
    v2 = read_jsonl(mode / "judgements" / "v2" / "judgements.jsonl")
    v3rep = read_json(mode / "d3" / "judge_v3_report" / source / "report.json", default={})
    nc2 = read_json(mode / "d3" / "judge_negative_control_v2" / source / "summary.json", default={}) or {}
    v3n = v3rep.get("negative_control_v3", {}).get("overall", {})
    v3o = v3rep.get("originals", {})
    js3 = read_jsonl(mode / "judgements" / "v3" / "judgements.jsonl")
    inconsistent = sum(1 for j in js3 if j.get("hallucination_reconciliation", "consistent") != "consistent")
    return {
        "v1": {"valid": len({j["summary_id"] for j in v1}), "failures": len({f["summary_id"] for f in v1f if f.get("prompt_version", "v1") == "v1"}),
               "note": "nested schema; 4 of 5 valid verdicts score hallucination 1-2 with zero claims"},
        "v2": {"valid": len(v2), "hallucination_scores": sorted({j["verdict"]["hallucination"]["score"] for j in v2}) if v2 else [],
               "nc_v2_detected": nc2.get("overall", {}).get("planted_claim_detected"),
               "note": "validator forced H<5 <=> claims; H = 4 for every summary"},
        "v3": {"valid_total": len(js3), "inconsistent_score_claim_verdicts": inconsistent, "originals_valid": v3o.get("valid"),
               "nc_v2_set_detected": v3n.get("detected"), "nc_v2_set_judged": v3n.get("corrupted_judged"),
               "nc_v2_faithfulness_lower": v3n.get("faithfulness_lower"), "nc_v2_pairs": v3n.get("pairs_with_v3_original"),
               "note": "scores generated independently of the claim list"},
    }


def repro_analysis(main_store: V4Store, repro_store: V4Store) -> dict:
    a = main_store.index()
    pairs = []
    for key, r in repro_store.index().items():
        m = a.get(key)
        if m is None:
            continue
        pairs.append({"key": "|".join(key), "first": m.get("status"), "repeat": r.get("status"),
                      "first_valid": m.get("valid"), "repeat_valid": r.get("valid"),
                      "raw_identical": m.get("raw_response") == r.get("raw_response"),
                      "same_input": m.get("input_sha256") == r.get("input_sha256")})
    n = len(pairs)
    same = sum(1 for p in pairs if p["first"] == p["repeat"])
    return {"pairs": n, "same_status": same, "status_agreement": round(same / n, 3) if n else None,
            "status_agreement_wilson95": wilson(same, n), "raw_identical": sum(1 for p in pairs if p["raw_identical"]),
            "same_input_prompt": sum(1 for p in pairs if p["same_input"]), "disagreements": [p for p in pairs if p["first"] != p["repeat"]]}


def cmd_report(s, mode: Path, source: str, summaries, records, reviews) -> dict:
    store = V4Store(mode / "judgements" / "v4")
    man = read_json(store.folder / "run_manifest.json", default={})
    cfg = man.get("config", {})
    run_id = "v4-report"
    originals = [item_from_summary("original", x, records) for x in summaries]
    corrected = corrected_items(mode, source, records)
    evals_o = evaluate_items(originals, reviews, store, run_id=run_id, config=cfg)
    evals_c = evaluate_items(corrected, reviews, store, run_id=run_id, config=cfg)
    for ev, it in zip(evals_c, corrected):
        ev.config = {**cfg, "correction_set": it.extra.get("correction_set")}
    cases = load_nc(mode, source)
    by_pid = {x.product_id: x for x in summaries}
    ncs = nc_items(cases, by_pid, records)
    evals_nc = evaluate_items(ncs, reviews, store, run_id=run_id, config=cfg)
    d = out_dir(mode, source)
    d.mkdir(parents=True, exist_ok=True)
    (d / "evaluations.jsonl").write_text("".join(ev.model_dump_json() + "\n" for ev in evals_o + evals_c), encoding="utf-8")
    (d / "nc_evaluations.jsonl").write_text("".join(ev.model_dump_json() + "\n" for ev in evals_nc), encoding="utf-8")
    write_csv(d / "claims.csv", claim_rows(evals_o + evals_c))
    write_csv(d / "summaries.csv", summary_rows(evals_o + evals_c))
    orig_by = {ev.summary_id: ev for ev in evals_o}
    nca = nc_analysis(cases, evals_nc, orig_by)
    write_csv(d / "nc_results.csv", [{k: v for k, v in r.items()} for r in nca["rows"]])
    repro = repro_analysis(store, V4Store(mode / "judgements" / "v4_repro"))
    fails = store.failures()
    recs = store.records()
    statuses = [e.status.value for ev in evals_o for e in ev.claim_evaluations]
    rep = {
        "generated_at": NOW(), "source": source, "judge_version": "v4", "config": cfg, "prompt_sha256": prompt_sha256(),
        "llm_calls": {"valid": len(recs), "validation_failures": len(fails), "kinds": {k: sum(1 for r in recs if r["evaluated_object"] == k)
                                                                                      for k in sorted({r["evaluated_object"] for r in recs})},
                      "quote_verified_rate": round(sum(1 for r in recs if r.get("quote_verified")) / len(recs), 3) if recs else None,
                      "mean_latency_s": round(sum(r["latency_s"] for r in recs) / len(recs), 1) if recs else None,
                      "sessions": man.get("sessions", [])},
        "originals": {"summaries": len(evals_o), "claims": len(statuses), "status_counts": {sv.value: statuses.count(sv.value) for sv in ClaimStatus},
                      "by_method": {m: sum(1 for ev in evals_o for e in ev.claim_evaluations if e.method == m)
                                    for m in ("deterministic", "llm", "deterministic+llm", "none")},
                      "numeric_claims": sum(ev.numeric_claims for ev in evals_o), "numeric_failed": sum(ev.numeric_claims_failed for ev in evals_o),
                      "validity": {v: sum(1 for ev in evals_o if ev.validity_status == v) for v in sorted({ev.validity_status for ev in evals_o})},
                      "mean_faithfulness": round(sum(ev.faithfulness_score or 0 for ev in evals_o) / len(evals_o), 1),
                      "per_product": summary_rows(evals_o)},
        "corrected": {"per_summary": summary_rows(evals_c),
                      "comparison": [{"product_id": ev.product_id, "corrected_summary_id": ev.summary_id, "correction_set": ev.config.get("correction_set"),
                                      "original_score": orig_by[ev.original_summary_id].faithfulness_score if ev.original_summary_id in orig_by else None,
                                      "corrected_score": ev.faithfulness_score,
                                      "original_validity": orig_by[ev.original_summary_id].validity_status if ev.original_summary_id in orig_by else None,
                                      "corrected_validity": ev.validity_status,
                                      "original_counts": orig_by[ev.original_summary_id].counts if ev.original_summary_id in orig_by else None,
                                      "corrected_counts": ev.counts} for ev in evals_c]},
        "negative_control_v3": {k: v for k, v in nca.items() if k != "rows"} | {"cases": len(nca["rows"]),
                                                                               "manifest": read_json(nc_dir(mode, source) / "manifest.json")},
        "sensitivity": sensitivity(evals_o),
        "reproducibility": repro,
        "human_sentiment_check": human_sentiment_check(s, summaries),
        "prior_judges": prior_judges(mode, source),
        "consistency_check": {"evaluations_validated": len(evals_o) + len(evals_c) + len(evals_nc),
                              "rule": "every SummaryEvaluationV4 re-derives counts, lists and score from its claims on validation"},
    }
    write_json_atomic(d / "report.json", rep)
    print(f"written: {d}")
    return rep


def main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description="Judge v4 claim-level evaluation")
    p.add_argument("cmd", choices=["build-nc", "run", "repro", "report"])
    p.add_argument("--source", required=True)
    p.add_argument("--objects", nargs="*", default=["original", "nc", "corrected"], choices=["original", "nc", "corrected"])
    p.add_argument("--limit", type=int, default=None, help="at most N new LLM calls (pilot)")
    p.add_argument("--mock", action="store_true", help="OFFLINE mock judge (pipeline check, not a result)")
    args = p.parse_args(argv)
    s = load_settings()
    is_mock = args.mock or s.is_mock
    mode = s.paths.mode_dir(is_mock)
    summaries, records = load_inputs(s, is_mock, args.source)
    reviews = load_review_index(s.paths.reviews_csv)
    if args.cmd == "build-nc":
        cmd_build_nc(s, mode, args.source, summaries, records, reviews)
        return
    if args.cmd == "report":
        rep = cmd_report(s, mode, args.source, summaries, records, reviews)
        print(json.dumps({k: rep[k] for k in ("llm_calls", "originals")}, indent=1, default=str)[:3000])
        print(json.dumps(rep["negative_control_v3"]["metrics"]["primary"], indent=1))
        return
    originals = [item_from_summary("original", x, records) for x in summaries]
    by_pid = {x.product_id: x for x in summaries}
    if args.cmd == "repro":
        ncs = nc_items(load_nc(mode, args.source), by_pid, records)
        _guarded_run(s, repro_subset(originals, ncs), reviews, V4Store(mode / "judgements" / "v4_repro"), is_mock=is_mock, label="v4repro")
        return
    items: list[Item] = []
    if "original" in args.objects:
        items += originals
    if "nc" in args.objects:
        items += nc_items(load_nc(mode, args.source), by_pid, records)
    if "corrected" in args.objects:
        items += corrected_items(mode, args.source, records)
    _guarded_run(s, items, reviews, V4Store(mode / "judgements" / "v4"), is_mock=is_mock, label="v4", limit=args.limit)


if __name__ == "__main__":
    main()
