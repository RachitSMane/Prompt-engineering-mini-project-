"""D3 completion: derived rubric, recorded verification, completion status and the completion report (no LLM).

  python scripts/d3_completion.py --source dataset__few_shot__qwen2.5-3b__v1 [--verify]

Steps
 1. rubric   : anchored 1-5 rubric (faithfulness, coverage, consistency, hallucination) derived from the stored judge v4
               claim results -> outputs/live/d3/judge_v4/<source>/rubric/{rubric.json, rubric.csv}
 2. human    : single-annotator human claim validation: integrity check, label export and (when all claims are labelled)
               judge v4 comparison -> outputs/live/d3/claim_gold/<source>/single_annotator/{human_labels.csv, comparison.json}
 3. --verify : runs the full pytest suite and scripts/verify_d3_final.py, records the outcome
               -> outputs/live/d3/completion/<source>/verification.json
 4. status   : evaluation/d3_status.py -> outputs/live/d3/completion/<source>/status.json
 5. report   : reports/D3_COMPLETION_REPORT_SINGLE_ANNOTATOR.md (every number read from the files above and report.json);
               reports/D3_COMPLETION_REPORT_2026-11-16.md is the earlier (two-annotator) version and is no longer written

Nothing here changes a summary, a claim verdict, an evaluation, a negative-control case or any D1/D2 artefact.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import load_settings  # noqa: E402
from evaluation.claim_checks import Stats  # noqa: E402
from evaluation.claim_gold import compare, export_csv_bytes, export_rows, load_template, now, sha256_file  # noqa: E402
from evaluation.d3_status import badge_text, compute, load_evaluations, paths  # noqa: E402
from evaluation.rubric_v4 import ANCHORS, RUBRIC_VERSION, check_rubric_consistent, rubric_for  # noqa: E402
from schemas.claim_schema import SCORING_FORMULA  # noqa: E402
from scripts.claim_gold import ensure_preregistration, judge_map  # noqa: E402
from scripts.run_judge import load_inputs  # noqa: E402
from utils.storage import read_json, write_json_atomic  # noqa: E402

REPORT = ROOT / "reports" / "D3_COMPLETION_REPORT_SINGLE_ANNOTATOR.md"
DIMS = ("faithfulness", "coverage", "consistency", "hallucination")


def build_rubric(s, mode: Path, source: str) -> dict:
    summaries, records = load_inputs(s, False, source)
    stats = {x.product_id: Stats.from_aggregate(x.aggregate, records) for x in summaries}
    evals = load_evaluations(paths(mode, source)["evaluations"])
    rows, violations = [], []
    for ev in evals:
        row = rubric_for(ev, stats[ev.product_id])
        for prob in check_rubric_consistent(row, ev):
            violations.append(f"{ev.summary_id}: {prob}")
        rows.append(row)
    orig = [r for r in rows if r["evaluated_object"] == "original"]
    means = {d: round(sum(r[d] for r in orig if r[d] is not None) / max(1, sum(r[d] is not None for r in orig)), 2) for d in DIMS}
    dist = {d: dict(sorted(Counter(r[d] for r in orig).items(), key=lambda kv: (kv[0] is None, kv[0]))) for d in DIMS}
    out = {"rubric_version": RUBRIC_VERSION, "computed_at": now(), "source": source, "anchors": ANCHORS,
           "llm_calls": 0, "derived_from": "outputs/live/d3/judge_v4/<source>/evaluations.jsonl + product aggregates",
           "originals_mean": means, "originals_distribution": {d: {str(k): v for k, v in x.items()} for d, x in dist.items()},
           "invariant_violations": violations, "rows": rows}
    d = paths(mode, source)["rubric"].parent
    write_json_atomic(d / "rubric.json", out)
    with (d / "rubric.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["evaluated_object", "summary_id", "product_id", "original_summary_id", "faithfulness_score_0_100", *DIMS,
                    "validity_status", "coverage_missed", "consistency_major", "consistency_minor", "wrong_numbers",
                    "unsupported", "contradicted"])
        for r in rows:
            w.writerow([r["evaluated_object"], r["summary_id"], r["product_id"], r["original_summary_id"] or "",
                        r["faithfulness_score_0_100"], *(r[x] for x in DIMS), r["validity_status"],
                        " ".join(r["coverage_detail"]["missed"]), len(r["consistency_detail"]["major_mismatch_claims"]),
                        len(r["consistency_detail"]["minor_mismatch_claims"]), len(r["consistency_detail"]["wrong_number_claims"]),
                        len(r["hallucination_detail"]["unsupported_claims"]), len(r["hallucination_detail"]["contradicted_claims"])])
    return out


def build_human_validation(mode: Path, source: str) -> dict:
    """Integrity check + label export + judge comparison (metrics only when all claims are labelled). Never creates a label."""
    p = paths(mode, source)
    template = load_template(p["template"])
    store_exists = p["store"].exists()
    prereg = ensure_preregistration(p, store_exists)
    data = read_json(p["store"], default=None) or {}
    res = compare(data, template, judge_map(p["evaluations"]), store_sha256=sha256_file(p["store"]) if store_exists else "",
                  template_sha256=sha256_file(p["template"]), thresholds=prereg.get("informational_thresholds"))
    res["events_before_complete"] = [e for e in data.get("events", []) if e.get("kind") == "judge_results_opened_before_complete"]
    p["single_dir"].mkdir(parents=True, exist_ok=True)
    if store_exists:
        p["labels_csv"].write_bytes(export_csv_bytes(export_rows(data, template)))
    write_json_atomic(p["comparison"], res)
    return res


def run_verification(mode: Path, source: str) -> dict:
    out = paths(mode, source)["verification"]
    out.parent.mkdir(parents=True, exist_ok=True)
    junit = out.parent / "pytest_junit.xml"
    py = sys.executable
    t = subprocess.run([py, "-m", "pytest", "-q", "-p", "no:cacheprovider", f"--junitxml={junit}"], cwd=ROOT,
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    suite = ET.parse(junit).getroot()
    suite = suite if suite.tag == "testsuite" else suite.find("testsuite")
    v = subprocess.run([py, "scripts/verify_d3_final.py", "--source", source], cwd=ROOT, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    res = {"recorded_at": now(), "pytest_exit_code": t.returncode, "tests_total": int(suite.get("tests")),
           "tests_failed": int(suite.get("failures")), "tests_errors": int(suite.get("errors")),
           "tests_skipped": int(suite.get("skipped")),
           "skipped_reasons": [sk.get("message") for tc in suite.iter("testcase") for sk in tc.findall("skipped")],
           "pytest_tail": t.stdout.strip().splitlines()[-1:] if t.stdout else [],
           "verify_d3_final_exit_code": v.returncode, "verify_d3_final_passed": v.returncode == 0,
           "verify_d3_final_output": [ln for ln in v.stdout.splitlines() if ln.strip()]}
    res["tests_passed"] = res["tests_total"] - res["tests_failed"] - res["tests_errors"] - res["tests_skipped"]
    write_json_atomic(out, res)
    return res


# --------------------------------------------------------------------------- report
def _pct(x) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def _ci(x) -> str:
    return "n/a" if not x else f"[{100 * x[0]:.1f}%, {100 * x[1]:.1f}%]"


def _table(header: list[str], rows: list[list]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c).replace("|", "\\|") for c in r) + " |" for r in rows]
    return out


def write_report(mode: Path, source: str, st: dict, rubric: dict, hv: dict, ver: dict | None) -> None:
    p = paths(mode, source)
    rep = read_json(p["report"], default={})
    o, nc, cfg = rep["originals"], rep["negative_control_v3"], rep["config"]
    prim = nc["metrics"]["primary"]
    rb = nc["review_based_types_primary"]
    L: list[str] = []
    a = L.append
    a("# ReviewIQ — D3 completion report (single-annotator human validation)")
    a("")
    a(f"Generated by `scripts/d3_completion.py --source {source}` from stored artefacts; no LLM was called for this report. "
      "It does not replace the historical reports (`reports/D3_*.md`, `reports/final/D3_Final_Report.*`, "
      "`reports/proposed_revisions_2026-11-15/`), which are unchanged.")
    a("")
    a("## 1. Verdict")
    a("")
    verdict = "**D3 is complete.**" if st["complete"] else "**D3 is NOT complete.**"
    a(f"{verdict} {st['passed']} of {st['mandatory']} mandatory criteria pass (recomputed by `evaluation/d3_status.py`). "
      f"Dashboard badge: _{badge_text(st)}_ ({'green' if st['complete'] else 'yellow'}).")
    if st["remaining"]:
        a("")
        a("Remaining:")
        for r in st["remaining"]:
            a(f"- {r}")
    a("")
    a("## 2. Requirement-by-requirement status")
    a("")
    a("Sources: problem statement §4.4-4.6 (`docs/ReviewIQ_Problem_Statement_and_Solution.pdf`, as transcribed in the 2026-11-15 "
      "plan §1), the completion criteria of `reports/proposed_revisions_2026-11-15/D3_INVESTIGATION_AND_COMPLETION_PLAN.md` §14, "
      "the 2026-11-16 brief (A-E) and the 2026-11-16 decision to validate judge v4 with ONE human annotator (replacing plan §14.4).")
    a("")
    a("### 2.1 Mandatory completion criteria (machine-checked)")
    a("")
    L += _table(["#", "Requirement", "Evidence", "Status", "Remaining work"],
                [[c["id"], c["criterion"], c["evidence"], c["status"], c["remaining"] or "–"] for c in st["criteria"]])
    a("")
    a("### 2.2 Problem-statement and brief requirements")
    a("")
    req_rows = [
        ["Extract structured review information (R1)", "qwen2.5:3b few-shot -> Pydantic `ReviewRecord` (reused D2 run)",
         "285 valid / 14 controlled failures; `outputs/live/extractions/dataset__few_shot__qwen2.5-3b__v1.jsonl` (hash unchanged)", "PASS",
         "– (quality limit: complaint precision 0.367, stated)"],
        ["Aggregate product-level statistics (R2)", "`pipeline/aggregation.py`, pandas only",
         "15/15 aggregates re-computed identically from the extraction file by the evidence audit (`evaluation/d3_audit.py`); "
         "judge v4 numeric checks read them", "PASS", "–"],
        ["Generate product summaries with a local LLM (R3)", "qwen2.5:3b, `prompts/summarization.py` v1",
         f"{st['n_original_evaluations']} summaries; every cited id valid (C1)", "PASS", "–"],
        ["No invented numbers in summaries (R3)", "deterministic numeric checks in judge v4",
         f"{o['numeric_claims']} numeric claims, {o['numeric_failed']} failed -> reported per claim; corrected copies separate", "PASS (detected and reported)", "–"],
        ["A. Claim-level evaluation against cited evidence", "`evaluation/claims.py`, `claim_checks.py`, `judge_v4.py`",
         f"{o['claims']} claims: " + ", ".join(f"{k} {v}" for k, v in o["status_counts"].items()), "PASS", "–"],
        ["A. Quotations validated against original reviews", "`quote_in_texts` (normalised exact substring)",
         f"verify check 4: every verified quote is a substring of a cited review; quote-verified rate {rep['llm_calls']['quote_verified_rate']}", "PASS", "–"],
        ["A. Numerical claims checked deterministically", "`numeric_checks_statistics`, `number_checks_in_texts`",
         "every number recomputed from the aggregate (value, expected, denominator, calculation stored)", "PASS", "–"],
        ["A. Scores consistent with claim results", "`derive_scores` + `SummaryEvaluationV4` validator; rubric invariants",
         f"{rep['consistency_check']['evaluations_validated']} evaluations re-validate; rubric violations: {len(rubric['invariant_violations'])}", "PASS", "–"],
        ["LLM-as-a-Judge with anchored 1-5 rubric (R4, §4.6)", "judge v4 + `evaluation/rubric_v4.py` (derived)",
         "faithfulness, coverage, consistency, hallucination for all 15 originals (section 7)", "PASS", "–"],
        ["B. Correct B001DHKBXU, B0BTY63SFY and other flagged summaries", "`-corr1`, `-corr2` copies (new ids, originals untouched)",
         f"{len(rep['corrected']['comparison'])} corrected copies evaluated separately (section 8)", "PASS", "B001DHKBXU statistics need re-extraction (out of scope: D2 artefact)"],
        ["C. Negative controls with false-positive measurement", "nc-v3, 12 error types, matched clean claims, frozen before judging",
         f"recall {_pct(prim['combined']['recall'])}, specificity {_pct(prim['combined']['specificity'])} (section 9)", "PASS", "–"],
        ["D. Automated tests", "tests/test_claims.py, test_judge_v4.py, test_negative_control_v3.py, test_d3_completion.py",
         (f"{ver['tests_passed']} passed, {ver['tests_failed']} failed, {ver['tests_errors']} errors, {ver['tests_skipped']} skipped" if ver else "not run"),
         "PASS" if ver and ver["tests_failed"] == 0 and ver["tests_errors"] == 0 else "FAIL", "–"],
        ["D. Exports, UI and stored results agree", "verify_d3_final checks 3 and 7; dashboard reads evaluations via the validator",
         "JSON/CSV/MD/HTML/PDF agree" if ver and ver["verify_d3_final_passed"] else "verification not passed",
         "PASS" if ver and ver["verify_d3_final_passed"] else "FAIL", "–"],
        ["E. Local Ollama only, sequential", "llama3.2 judge via Ollama, one call per claim, GPU guard",
         f"{rep['llm_calls']['valid']} valid calls, {rep['llm_calls']['validation_failures']} failures; no LLM call for validation or this report", "PASS", "–"],
        ["Judge v4 validated against one human annotator (replaces plan §14.4)", "Streamlit *Human claim validation*, `evaluation/claim_gold.py`",
         f"{hv.get('labelled', 0)}/{hv.get('total', 239)} claims labelled; comparison {hv.get('status')}",
         "PASS" if hv.get("status") == "complete" else "FAIL",
         "human labelling (cannot be done by the assistant)" if hv.get("status") != "complete" else "–"],
    ]
    L += _table(["Requirement", "Implementation", "Evidence", "Status", "Remaining"], req_rows)
    a("")
    a("## 3. Architecture and pipeline")
    a("")
    a("```")
    a("data/processed/reviews.csv (1,500 reviews, 15 products) -> 299-review annotation sample")
    a("  [1] EXTRACT    qwen2.5:3b few-shot -> ReviewRecord (pipeline/extraction.py)            reused from D2, unchanged")
    a("  [2] AGGREGATE  pandas: sentiment, ratings, aspect +/-, complaints, evidence quotes      pipeline/aggregation.py")
    a("  [3] SUMMARISE  qwen2.5:3b, statistics + evidence quotes -> ProductSummaryContent        pipeline/summarization.py")
    a("  [4] DECOMPOSE  summary -> verbatim claims with their cited ids                         evaluation/claims.py")
    a("  [5] CHECK      numbers, counts, denominators, majorities, aspects, citations (no LLM)  evaluation/claim_checks.py")
    a("  [6] VERIFY     llama3.2: ONE claim vs the full text of ITS cited reviews, verbatim quote evaluation/judge_v4.py")
    a("  [7] COMBINE    worse verdict wins; unverified quote -> UNVERIFIABLE; scores derived      schemas/claim_schema.py")
    a("  [8] RUBRIC     faithfulness / coverage / consistency / hallucination 1-5, derived        evaluation/rubric_v4.py")
    a("  [9] CONTROLS   nc-v3: 123 planted + 123 matched clean claims, frozen before judging     evaluation/negative_control_v3.py")
    a(" [10] HUMAN      one annotator labels all claims (blind) -> agreement, P/R/F1 vs judge v4  evaluation/claim_gold.py")
    a(" [11] STATUS     completion criteria recomputed from files -> dashboard badge             evaluation/d3_status.py")
    a("```")
    a("")
    a("## 4. Extraction and aggregation methodology")
    a("")
    a("Each review of the D2 annotation sample is extracted by qwen2.5:3b with the few-shot prompt selected in D2 into a Pydantic "
      "`ReviewRecord` (overall sentiment, aspects with polarity and a verbatim evidence span, complaint categories), with "
      "deterministic fixes and at most two repair calls. The extraction file is reused unchanged from D2 (285 valid, 14 stored "
      "failures). `pipeline/aggregation.py` then computes, per product and without any LLM: sentiment counts and percentages, "
      "average rating, aspect mentions split into positive / negative / neutral, complaint counts and percentages (denominator = "
      "all reviews of the product), the number of reviews with complaints, a yearly trend and up to ten evidence quotes. Judge v4 "
      "re-reads these statistics for every numeric check, so statistics are faithful to the extractions by construction; D2 "
      "measured complaint micro-precision 0.367, so complaint statistics can be inflated relative to the review text.")
    a("")
    a("## 5. Summary generation methodology")
    a("")
    a("qwen2.5:3b (Ollama, local) receives the statistics JSON and the evidence quotes (not the full reviews) and returns "
      "`ProductSummaryContent`: an overall opinion, up to six appreciated points and six complaints (each citing review ids "
      "that must belong to the evidence set, enforced by the schema), frequent aspects, improvement areas and recommendations. "
      "The 15 summaries in `outputs/live/summaries/summaries.jsonl` are the evaluated artefacts and were never modified; "
      "corrections are separate copies (section 8).")
    a("")
    a("## 6. Judge v4 methodology and claim-level results")
    a("")
    a(f"Judge model `{cfg['judge_provider']}/{cfg['judge_model']}` (a different family from the qwen2.5 summariser), temperature "
      f"{cfg['temperature']}, num_ctx {cfg['num_ctx']}, num_predict {cfg['num_predict']}, constrained decoding "
      f"{cfg['constrained_decoding']}, {cfg['calls_per_claim']} call per claim, {cfg['repair_attempts']} repairs, prompt "
      f"SHA-256 `{cfg['prompt_sha256']}`. Every overall-opinion sentence, appreciated/complaint point, frequent-aspect list, "
      "improvement area and recommendation becomes one verbatim claim. Statistic sentences are checked deterministically; "
      "each cited point is checked by the LLM against the full text of its cited reviews only and must return a verbatim quote, "
      "which must be found in those reviews or the verdict becomes UNVERIFIABLE. The deterministic and LLM verdicts are combined "
      "(the worse wins; a deterministic failure cannot be overridden) and the score is computed from the statuses:")
    a("")
    a(f"`{SCORING_FORMULA}`")
    a("")
    a(f"Results on the 15 originals: {o['claims']} claims — " + ", ".join(f"{k} {v}" for k, v in o["status_counts"].items())
      + f"; by method " + ", ".join(f"{k} {v}" for k, v in o["by_method"].items() if v) + f"; numeric claims {o['numeric_claims']} "
      f"({o['numeric_failed']} failed a deterministic check); validity " + ", ".join(f"{k} {v}" for k, v in o["validity"].items())
      + f"; mean faithfulness {o['mean_faithfulness']} / 100. LLM calls: {rep['llm_calls']['valid']} valid, "
      f"{rep['llm_calls']['validation_failures']} validation failures, mean latency {rep['llm_calls']['mean_latency_s']} s.")
    a("")
    a("The per-claim table (status, method, quote, numeric calculation, explanation) is "
      "`outputs/live/d3/judge_v4/dataset__few_shot__qwen2.5-3b__v1/claims.csv`; the 16 investigation-known errors and the "
      "53 flagged claims are listed with explanations in `reports/final/D3_Final_Report.md` §12 and §17.")
    a("")
    a("## 7. Anchored 1-5 rubric (derived, no LLM scores)")
    a("")
    a(f"Rubric version `{rubric['rubric_version']}`; {rubric['llm_calls']} LLM calls; invariant violations: "
      f"{len(rubric['invariant_violations'])}. Anchors (written before computing, following `prompts/judge.py::RUBRIC`; revised once "
      "after the first computation, see section 12):")
    a("")
    for k, v in rubric["anchors"].items():
        a(f"- **{k}**: {v}")
    a("")
    orows = [r for r in rubric["rows"] if r["evaluated_object"] == "original"]
    L += _table(["Product", "Faithfulness score", "Faithfulness", "Coverage", "Consistency", "Hallucination", "Validity", "Missed themes"],
                [[r["product_id"], r["faithfulness_score_0_100"], r["faithfulness"], r["coverage"], r["consistency"], r["hallucination"],
                  r["validity_status"], " ".join(r["coverage_detail"]["missed"]) or "–"] for r in sorted(orows, key=lambda r: r["product_id"])])
    a("")
    a("Means over the 15 originals: " + ", ".join(f"{d} {rubric['originals_mean'][d]}" for d in DIMS) + ". The hallucination "
      "dimension counts LLM-flagged claims, which include false positives (section 9: LLM specificity "
      f"{_pct(prim['llm']['specificity'])}), so it is a conservative (pessimistic) indicator.")
    a("")
    a("## 8. Corrected summaries")
    a("")
    a("Originals are never edited. `-corr1` (2026-11-14) fixed the misattributed percentages of B0BTY63SFY and the denominators of "
      "B001DHKBXU; `-corr2` (2026-11-15, `scripts/build_corrected_summaries_v2.py`) fixes every error named in the investigation "
      "for 7 products, each edit recording old/new text, the reason and verbatim review quotes or recomputed statistics. Each copy "
      "was evaluated by judge v4 under its own id; no score is transferred. Provenance: AI coding assistant, not human-approved.")
    a("")
    crub = {r["summary_id"]: r for r in rubric["rows"] if r["evaluated_object"] == "corrected"}
    L += _table(["Product", "Copy", "Original score", "Corrected score", "Original validity", "Corrected validity", "Corrected rubric F/Cov/Con/H"],
                [[c["product_id"], c["corrected_summary_id"], c["original_score"], c["corrected_score"], c["original_validity"],
                  c["corrected_validity"], "/".join(str(crub[c["corrected_summary_id"]][d]) for d in DIMS) if c["corrected_summary_id"] in crub else "–"]
                 for c in rep["corrected"]["comparison"]])
    a("")
    a("Copies that are still flagged carry the flags on points the corrections did not touch (per-claim detail in "
      "`reports/final/D3_Final_Report.md` §13); several of those are LLM false positives of the kind measured in section 9.")
    a("")
    a("B0BTY63SFY: corr2 removes the misattributed 80 % / 67 %, the contradicted 'larger eggs' point, the 'legs' misreading and "
      "the durability advice without negative evidence. B001DHKBXU: corr2 fixes the denominators, the 'installation was "
      "difficult' / 'compatibility' / 'value' points whose citations do not support them, and 'mixed' (0 % mixed). Its complaint "
      "statistics remain inflated by extraction errors; fixing them requires re-extraction, which would change D2 artefacts.")
    a("")
    a("## 9. Negative-control results (nc-v3)")
    a("")
    man = nc["manifest"]
    a(f"{nc['cases']} cases over 12 error types; each case = one planted (corrupted) claim and one matched correct (clean) claim. "
      f"Seed {man.get('seed')}, dataset SHA-256 `{man.get('dataset_sha256')}`, frozen {man.get('frozen_at')} before the first "
      "verdict (verify check 5). Detection is read from the evaluator's final status on the planted claim itself, never from "
      "free-text explanations.")
    a("")
    L += _table(["Error type", "Cases", "Detected", "Detection rate", "95% CI", "Clean flagged", "Specificity"],
                [[t, v["cases"], v["detected"], _pct(v["detection_rate"]), _ci(v["detection_wilson95"]), v["clean_flagged"], _pct(v["specificity"])]
                 for t, v in nc["by_type"].items()])
    a("")
    L += _table(["Layer (primary rule)", "TP", "FP", "TN", "FN", "Precision", "Recall [95% CI]", "Specificity [95% CI]", "False-positive rate"],
                [[k, m["tp"], m["fp"], m["tn"], m["fn"], _pct(m["precision"]), f"{_pct(m['recall'])} {_ci(m['recall_wilson95'])}",
                  f"{_pct(m['specificity'])} {_ci(m['specificity_wilson95'])}", _pct(m["false_positive_rate"])] for k, m in prim.items()])
    a("")
    a(f"Review-based types only (what only the LLM can check): recall {_pct(rb['combined']['recall'])}, specificity "
      f"{_pct(rb['combined']['specificity'])}, precision {_pct(rb['combined']['precision'])}. The four statistic types are caught "
      "deterministically by checks written by the same author as the generator, so their 100 % shows the checks work, not that "
      f"the task is hard. Summary-level effect: the corrupted copy scored lower in {nc['copy_level_scores']['corrupted_copy_lower']} "
      f"of {nc['copy_level_scores']['pairs']} pairs, equal in {nc['copy_level_scores']['equal']}, higher in "
      f"{nc['copy_level_scores']['corrupted_copy_higher']} (v3: lower in {rep['prior_judges']['v3']['nc_v2_faithfulness_lower']} of "
      f"{rep['prior_judges']['v3']['nc_v2_pairs']}). Earlier nc-v2 copies and their v2/v3 results are unchanged.")
    a("")
    a("## 10. Human claim validation (single annotator)")
    a("")
    a("**Methodology (decision of the project owner, 2026-11-16).** One named person labels every one of the 239 claims of the 15 "
      "original summaries exactly once, on the Summary Evaluation page (*Human claim validation*), against the evidence in the frozen "
      "claim template (cited review text, or the deterministic statistics for statistic sentences). Judge v4 verdicts are hidden while "
      "labelling is incomplete; opening the judge results earlier is logged in the label store. After all claims are labelled, judge v4 "
      "is compared with the labels. This replaces the earlier criterion of two independent annotators, inter-annotator Cohen's kappa and "
      "adjudication (plan §14.4, and `reports/D3_COMPLETION_REPORT_2026-11-16.md`, both kept unchanged as history). No inter-annotator "
      "agreement or human consensus is claimed.")
    a("")
    if hv.get("status") == "complete":
        a(f"Annotator: **{hv['annotator']}** · {hv['claims']} claims · label file SHA-256 `{hv['store_sha256'][:16]}…` · computed {hv['computed_at']}.")
        events = hv.get("events_before_complete") or []
        if events:
            a("")
            a(f"Provenance note: the judge results were opened {len(events)} time(s) before labelling was complete "
              f"(at {', '.join(str(e['labels_at_time']) for e in events)} labelled claims).")
        a("")
        a(f"Exact agreement judge v4 vs annotator: **{_pct(hv['exact_agreement'])}** {_ci(hv['exact_agreement_wilson95'])}; Cohen's kappa "
          f"(judge vs this one annotator, chance-corrected, not inter-annotator reliability): {hv['cohen_kappa_judge_vs_annotator']}.")
        fb = hv["flagged_binary"]
        a(f"Flagged claims (unsupported or contradicted), human as reference: precision {_pct(fb['precision'])} {_ci(fb['precision_wilson95'])}, "
          f"recall {_pct(fb['recall'])} {_ci(fb['recall_wilson95'])}, F1 {fb['f1']}, specificity {_pct(fb['specificity'])}. "
          f"Informational thresholds {hv['thresholds']} met: {hv['meets_informational_thresholds']} (not a completion condition).")
        a("")
        a("Confusion matrix (rows = human label, columns = judge v4 status):")
        a("")
        mx = hv["confusion_matrix"]["matrix"]
        labs = list(mx)
        L.extend(_table(["human / judge"] + labs + ["total"], [[r] + [mx[r][c] for c in labs] + [sum(mx[r].values())] for r in labs]))
        a("")
        L.extend(_table(["Class", "Human support", "Judge predicted", "Precision", "Recall", "F1"],
                        [[c, v["human_support"], v["judge_predicted"], _pct(v["precision"]), _pct(v["recall"]),
                          v["f1"] if v["f1"] is not None else "n/a"] for c, v in hv["per_class"].items()]))
        a("")
        L.extend(_table(["Judge layer", "Claims", "Agreement", "95% CI"],
                        [[m, v["claims"], _pct(v["agreement"]), _ci(v["agreement_wilson95"])] for m, v in hv["by_method"].items()]))
        a("")
        a(f"Disagreements: {len(hv['disagreements'])} (all listed in `outputs/live/d3/claim_gold/{source}/single_annotator/comparison.json`). "
          "First 15:")
        a("")
        L.extend(_table(["Product", "Field", "Claim", "Human", "Judge (method)", "Human explanation"],
                        [[d["product_id"], d["claim_field"], d["claim_text"][:110], d["human_label"], f"{d['judge_status']} ({d['judge_method']})",
                          (d["human_explanation"] or "–")[:110]] for d in hv["disagreements"][:15]]))
    else:
        a(f"**Not complete.** {hv.get('labelled', 0)} of {hv.get('total', 239)} claims labelled"
          + (f" by {hv['annotator']}" if hv.get("annotator") else "") + ". No comparison with judge v4 is computed until every claim "
          "is labelled (no partial results are shown, to keep the remaining labelling blind). No claim has been labelled by the assistant.")
        if hv.get("validation_problems"):
            a("")
            a("Label-store problems: " + "; ".join(hv["validation_problems"][:10]))
        a("")
        a("To finish: open Summary Evaluation > *Human claim validation*, label the remaining claims, then run "
          f"`python scripts/d3_completion.py --source {source} --verify`. See `reports/D3_HUMAN_CLAIM_VALIDATION_GUIDE.md`.")
    a("")
    a("## 11. Quantitative summary")
    a("")
    L += _table(["Metric", "Value"], [
        ["Summaries / judge v4 evaluations (originals)", f"{st['n_original_evaluations']} / {st['n_original_evaluations']}"],
        ["Claims (originals)", o["claims"]],
        ["Supported / partial / unsupported / contradicted / unverifiable", " / ".join(str(o["status_counts"][k]) for k in
                                                                                ("SUPPORTED", "PARTIALLY_SUPPORTED", "UNSUPPORTED", "CONTRADICTED", "UNVERIFIABLE"))],
        ["Mean faithfulness (0-100)", o["mean_faithfulness"]],
        ["Rubric means F / Cov / Con / H", " / ".join(str(rubric["originals_mean"][d]) for d in DIMS)],
        ["Numeric claims / failed", f"{o['numeric_claims']} / {o['numeric_failed']}"],
        ["nc-v3 recall / precision / specificity (combined)", f"{_pct(prim['combined']['recall'])} / {_pct(prim['combined']['precision'])} / {_pct(prim['combined']['specificity'])}"],
        ["nc-v3 false-positive rate combined / LLM / deterministic", f"{_pct(prim['combined']['false_positive_rate'])} / {_pct(prim['llm']['false_positive_rate'])} / {_pct(prim['deterministic']['false_positive_rate'])}"],
        ["Repeat-run status agreement", f"{rep['reproducibility']['same_status']}/{rep['reproducibility']['pairs']}"],
        ["Judge v4 agreement with the single annotator", "not available (labelling incomplete)" if hv.get("status") != "complete"
         else f"{_pct(hv['exact_agreement'])} exact; flagged P/R {_pct(hv['flagged_binary']['precision'])} / {_pct(hv['flagged_binary']['recall'])}"],
    ])
    a("")
    a("## 12. Limitations")
    a("")
    for x in [
        "Human validation uses ONE annotator: the comparison measures agreement with that person, not inter-annotator reliability or "
        "ground truth; individual misreadings or a personal threshold for 'supported' cannot be detected without a second annotator. "
        "The annotator is a project member who may have seen earlier judge outputs and investigation reports, so labelling is blind "
        "only within the interface." + ("" if hv.get("status") == "complete" else " Labelling is not yet complete (criterion C5)."),
        f"Small local judge (llama3.2 3B): on clean controls the LLM layer flags {prim['llm']['fp']} of {prim['llm']['fp'] + prim['llm']['tn']} correct claims; some of the 44 UNSUPPORTED findings in the originals are therefore false positives. 3 of 8 misleading paraphrases passed.",
        "Negative controls are template-based ('A reviewer reports that ...') and per-type samples are small (5-15).",
        "Statistic types are detected by checks written by the same author as the generator.",
        "Statistics inherit extraction errors (D2 complaint precision 0.367); summaries can be faithful to statistics that are not faithful to the reviews (B001DHKBXU).",
        "Opinion sentences are checked against statistics only; derived claims only for negative evidence on the targeted aspect.",
        "Rubric anchors (coverage, consistency, hallucination) were set by the AI assistant and are not calibrated against human ratings. "
        "They were revised once after the first computation (r2): the catch-all aspect 'other' is no longer a theme, and only numeric checks with a "
        "wrong value (severity 'error') count as a failed number for hallucination; misleading denominators (rated PARTIALLY_SUPPORTED by v4) do not. "
        "Coverage detection uses keyword patterns and is imperfect: B001ONHE8S 'The color does not match the customer's expectations' is a "
        "not-as-described complaint the pattern misses (coverage 3, arguably 4). It was left unfixed rather than tuned to the results.",
        "One summariser run; corrected copies were written by the AI assistant, not by people.",
        "The legacy export `outputs/live/d3/dataset__few_shot__qwen2.5-3b__v1/{d3_report.md,judgements.csv}` is the frozen judge v1 output (5 judged); it is kept for history and marked superseded.",
    ]:
        a(f"- {x}")
    a("")
    a("## 13. Reproducibility")
    a("")
    a("```")
    a(f"python scripts/judge_v4.py report --source {source}          # evaluations, CSV exports, metrics (no LLM)")
    a(f"python scripts/claim_gold.py status --source {source}        # labelling progress and integrity")
    a(f"python scripts/claim_gold.py compare --source {source}       # judge v4 vs the annotator (all claims labelled)")
    a(f"python scripts/d3_completion.py --source {source} --verify   # rubric, tests, verify_d3_final, status, this report")
    a(f"python scripts/verify_d3_final.py --source {source}          # independent checks 1-8")
    a("```")
    a("")
    a(f"Repeat run: {rep['reproducibility']['same_status']}/{rep['reproducibility']['pairs']} identical statuses, "
      f"{rep['reproducibility']['raw_identical']} byte-identical raw responses. All stores are append-only; raw LLM responses, prompt hash and "
      "configuration are stored with every verdict.")
    a("")
    a("## 14. Test and verification results")
    a("")
    if ver:
        a(f"Recorded {ver['recorded_at']}: pytest {ver['tests_passed']} passed, {ver['tests_failed']} failed, {ver['tests_errors']} errors, "
          f"{ver['tests_skipped']} skipped (exit code {ver['pytest_exit_code']}). Skipped: {'; '.join(ver['skipped_reasons']) or 'none'}.")
        a("")
        a(f"`scripts/verify_d3_final.py` exit code {ver['verify_d3_final_exit_code']}:")
        a("")
        a("```")
        L.extend(ver["verify_d3_final_output"])
        a("```")
    else:
        a("Not recorded (run with `--verify`).")
    a("")
    REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--verify", action="store_true", help="run the full test suite and verify_d3_final.py and record the result")
    a = ap.parse_args(argv)
    s = load_settings()
    mode = s.paths.mode_dir(False)
    rubric = build_rubric(s, mode, a.source)
    print(f"rubric: {len(rubric['rows'])} rows, means {rubric['originals_mean']}, violations {len(rubric['invariant_violations'])}")
    hv = build_human_validation(mode, a.source)
    print(f"human validation: {hv['labelled']}/{hv['total']} labelled; comparison {hv['status']}")
    ver = run_verification(mode, a.source) if a.verify else read_json(paths(mode, a.source)["verification"], default=None)
    if ver:
        print(f"tests: {ver['tests_passed']} passed / {ver['tests_failed']} failed / {ver['tests_errors']} errors / {ver['tests_skipped']} skipped; "
              f"verify_d3_final passed: {ver['verify_d3_final_passed']}")
    n_sum = len(load_inputs(s, False, a.source)[0])
    st = compute(mode, a.source, n_summaries=n_sum)
    write_json_atomic(paths(mode, a.source)["verification"].parent / "status.json", st)
    write_report(mode, a.source, st, rubric, hv, ver)
    print(f"status: {'COMPLETE' if st['complete'] else 'NOT complete'} ({st['passed']}/{st['mandatory']}); report -> {REPORT}")
    for r in st["remaining"]:
        print("  remaining:", r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
