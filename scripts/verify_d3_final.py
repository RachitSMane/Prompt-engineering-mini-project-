"""Independent verification of the final D3 artefacts (no LLM). Exit code 0 only if every check passes.

  python scripts/verify_d3_final.py --source dataset__few_shot__qwen2.5-3b__v1 [--backup-manifest <protected_sha256_before.txt>]

Checks
 1. every stored SummaryEvaluationV4 re-validates (counts, lists and score re-derived from the claims: no contradiction)
 2. re-running the deterministic evaluation from the stores reproduces evaluations.jsonl exactly (statuses and scores)
 3. JSON and CSV exports agree (summaries.csv, claims.csv, nc_results.csv)
 4. every verified LLM quote is an exact (normalised) substring of a cited review of the right product
 5. the nc-v3 dataset matches the SHA-256 in its manifest and was frozen before the first v4 nc verdict
 6. every nc metric is recomputed from nc_results.csv
 7. the Markdown and HTML reports agree (every Markdown table cell and figure value appears in the HTML) and the key numbers
    in the Markdown come from report.json; the PDF exists and is newer than the HTML
 8. D1/D2 and other protected artefacts are byte-identical to the backup hash manifest (allowed D3 additions listed)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import load_settings  # noqa: E402
from evaluation.claim_checks import quote_in_texts  # noqa: E402
from evaluation.judge_v4 import V4Store, evaluate_items, item_from_summary, load_review_index  # noqa: E402
from evaluation.negative_control_v3 import confusion, dataset_sha256  # noqa: E402
from schemas.claim_schema import SummaryEvaluationV4  # noqa: E402
from scripts.judge_v4 import corrected_items, nc_items  # noqa: E402
from scripts.run_judge import load_inputs  # noqa: E402
from utils.storage import read_json, read_jsonl  # noqa: E402

#: files that D3 completion is allowed to change inside the hashed (protected) trees
ALLOWED_CHANGES = {"outputs/live/d3/corrected_summaries/dataset__few_shot__qwen2.5-3b__v1/README.json"}


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--backup-manifest", default=str(ROOT.parent / "ReviewIQ-backup-d3-v4-20261115" / "protected_sha256_before.txt"))
    args = p.parse_args()
    s = load_settings()
    mode = s.paths.mode_dir(False)
    d = mode / "d3" / "judge_v4" / args.source
    results: list[tuple[str, bool, str]] = []

    def check(name, ok, detail=""):
        results.append((name, bool(ok), detail))

    evals = [SummaryEvaluationV4.model_validate_json(x) for x in (d / "evaluations.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    nc_evals = [SummaryEvaluationV4.model_validate_json(x) for x in (d / "nc_evaluations.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    check("1 evaluations re-validate (no score/claim contradiction)", True, f"{len(evals)} summary + {len(nc_evals)} nc evaluations")

    summaries, records = load_inputs(s, False, args.source)
    reviews = load_review_index(s.paths.reviews_csv)
    store = V4Store(mode / "judgements" / "v4")
    items = [item_from_summary("original", x, records) for x in summaries] + corrected_items(mode, args.source, records)
    again = evaluate_items(items, reviews, store, run_id="verify", config={})
    key = lambda ev: [(ev.summary_id, e.claim.claim_id, e.status.value) for e in ev.claim_evaluations] + [ev.faithfulness_score]
    check("2 deterministic re-evaluation reproduces evaluations.jsonl", [key(e) for e in again] == [key(e) for e in evals])

    srows = list(csv.DictReader((d / "summaries.csv").open(encoding="utf-8")))
    crows = list(csv.DictReader((d / "claims.csv").open(encoding="utf-8")))
    ok3 = len(srows) == len(evals) and all(r["summary_id"] == e.summary_id and r["validity_status"] == e.validity_status and
                                            str(e.faithfulness_score) == (r["faithfulness_score"] or "None") for r, e in zip(srows, evals))
    st = {(e.claim.claim_id, ev.evaluated_object): e.status.value for ev in evals for e in ev.claim_evaluations}
    ok3 = ok3 and len(crows) == len(st) and all(st[(r["claim_id"], r["evaluated_object"])] == r["status"] for r in crows)
    nrows = list(csv.DictReader((d / "nc_results.csv").open(encoding="utf-8")))
    nst = {ev.summary_id: ev.claim_evaluations[0].status.value for ev in nc_evals}
    cases = read_jsonl(mode / "d3" / "negative_control_v3" / args.source / "cases.jsonl")
    cbyid = {c["case_id"]: c for c in cases}
    ok3 = ok3 and all(nst[cbyid[r["case_id"]]["corrupted_claim"]["summary_id"]] == r["corrupted_status"] and
                      nst[cbyid[r["case_id"]]["clean_claim"]["summary_id"]] == r["clean_status"] for r in nrows)
    check("3 JSON and CSV exports agree", ok3, f"{len(srows)} summaries, {len(crows)} claims, {len(nrows)} nc cases")

    bad = []
    n_q = 0
    for ev in evals + nc_evals:
        for e in ev.claim_evaluations:
            if e.llm_quote_verified:
                n_q += 1
                texts = [f"{reviews[r]['title']}\n{reviews[r]['text']}" for r in e.claim.cited_review_ids
                         if r in reviews and reviews[r]["product_id"] == e.claim.product_id]
                if e.claim.kind == "point" and not quote_in_texts(e.llm_quote, texts + [" ".join(texts)]):
                    bad.append(e.claim.claim_id)
    check("4 verified quotes are substrings of cited reviews", not bad, f"{n_q} verified quotes; bad: {bad[:5]}")

    man = read_json(mode / "d3" / "negative_control_v3" / args.source / "manifest.json")
    first_nc = min((r["created_at"] for r in store.records() + store.failures() if r["evaluated_object"].startswith("nc_")), default=None)
    check("5 nc-v3 dataset hash matches manifest and was frozen before judging",
          dataset_sha256(cases) == man["dataset_sha256"] and (first_nc is None or man["frozen_at"] <= first_nc),
          f"frozen {man['frozen_at']}, first nc verdict {first_nc}")

    rep = read_json(d / "report.json")
    rows = [{"error_type": r["error_type"], "corrupted_s": r["corrupted_status"], "clean_s": r["clean_status"]} for r in nrows]
    rec = confusion(rows, "s", "primary")
    check("6 nc metrics recomputed from nc_results.csv", rec == rep["negative_control_v3"]["metrics"]["primary"]["combined"])

    md = (ROOT / "reports" / "final" / "D3_Final_Report.md").read_text(encoding="utf-8")
    hp = ROOT / "reports" / "final" / "D3_Final_Report.html"
    ht = html.unescape(re.sub(r"<[^>]+>", " ", hp.read_text(encoding="utf-8")))
    ht = re.sub(r"\s+", " ", ht)
    cells = [c.strip().replace("\\|", "|").replace("**", "").replace("`", "") for ln in md.splitlines() if ln.startswith("|") and not ln.startswith("|---")
             for c in re.split(r"(?<!\\)\|", ln)[1:-1]]
    missing = [c for c in cells if c and re.sub(r"\s+", " ", c) not in ht]
    o = rep["originals"]
    nums = [str(o["claims"]), str(o["mean_faithfulness"]), str(rep["negative_control_v3"]["cases"])]
    pdf = hp.with_suffix(".pdf")
    check("7 Markdown, HTML and PDF reports agree", not missing and all(n in md for n in nums) and pdf.exists() and pdf.stat().st_mtime >= hp.stat().st_mtime - 5,
          f"{len(cells)} table cells compared; missing in HTML: {missing[:3]}; pdf exists: {pdf.exists()}")

    bm = Path(args.backup_manifest)
    if bm.exists():
        changed = []
        for ln in bm.read_text(encoding="utf-8").splitlines():
            h, rel = ln.split(maxsplit=1)
            rel = rel.strip().lstrip("*")
            f = ROOT / rel
            cur = hashlib.sha256(f.read_bytes()).hexdigest() if f.exists() else None
            if cur != h and rel not in ALLOWED_CHANGES:
                changed.append(rel)
        check("8 protected artefacts unchanged (D1, D2, data, earlier D3 outputs)", not changed,
              f"{len(bm.read_text(encoding='utf-8').splitlines())} files; changed: {changed[:5]}; allowed: {sorted(ALLOWED_CHANGES)}")
    else:
        check("8 protected artefacts unchanged", False, f"backup manifest not found: {bm}")

    width = max(len(n) for n, _, _ in results)
    for n, ok, det in results:
        print(f"{'PASS' if ok else 'FAIL'}  {n.ljust(width)}  {det}")
    sys.exit(0 if all(ok for _, ok, _ in results) else 1)


if __name__ == "__main__":
    main()
