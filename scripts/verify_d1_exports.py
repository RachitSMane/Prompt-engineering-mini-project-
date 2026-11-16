"""Independent consistency check of the D1 JSON / CSV exports against the run's saved predictions.

For each strategy folder under ``outputs/<mode>/d1/<run_id>/`` this verifies, without any LLM call:

1. ``records.json`` parses and every record re-validates with the Pydantic schema (evidence grounded in its own text);
2. it contains exactly the valid records of ``predictions_<strategy>.jsonl`` (same ids, same labels);
3. ``records.csv`` has one row per JSON record with identical sentiment, complaint categories and aspect list;
4. ``records_aspects.csv`` has exactly one row per extracted aspect (category, sentiment, evidence identical);
5. ``failures.csv`` lists exactly the stored controlled failures, and no review is both a record and a failure;
6. records + failures cover every review in the run (no review silently dropped).

Usage:  python scripts/verify_d1_exports.py --run 20261111-030912_qwen2.5-3b [--out reports/d1/export_verification.md]
Exit code 1 if any check fails.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import load_settings  # noqa: E402
from evaluation.prompt_comparison import experiments_dir  # noqa: E402
from pipeline.extraction import ExtractionStore  # noqa: E402
from schemas.review_schema import ReviewRecord  # noqa: E402
from utils.storage import read_json  # noqa: E402


def _split(cell) -> list[str]:
    return [x.strip() for x in str(cell or "").split(";") if x.strip()] if not (isinstance(cell, float) and pd.isna(cell)) else []


def verify_strategy(run_dir: Path, d1_dir: Path, strategy: str) -> dict:
    checks: dict[str, bool] = {}
    problems: list[str] = []
    cfg = read_json(run_dir / "config.json")
    run_ids = set(cfg["review_ids"])
    store = ExtractionStore(run_dir / f"predictions_{strategy}.jsonl")
    stored = {r.review_id: r for r in store.records}
    stored_fail = {f.review_id for f in store.failures}
    out = d1_dir / strategy

    data = json.loads((out / "records.json").read_text(encoding="utf-8"))
    invalid = []
    for d in data:
        try:
            ReviewRecord.model_validate(d, context={"review_text": d["review_text"]})
        except Exception as exc:  # noqa: BLE001
            invalid.append(f"{d.get('review_id')}: {str(exc)[:120]}")
    checks["json_parses_and_revalidates"] = not invalid
    problems += invalid[:5]

    json_ids = [d["review_id"] for d in data]
    checks["json_ids_unique"] = len(json_ids) == len(set(json_ids))
    checks["json_matches_stored_records"] = set(json_ids) == set(stored) and all(
        d["overall_sentiment"] == stored[d["review_id"]].overall_sentiment.value
        and [c for c in d["complaint_categories"]] == [c.value for c in stored[d["review_id"]].complaint_categories]
        and len(d["aspects"]) == len(stored[d["review_id"]].aspects)
        for d in data if d["review_id"] in stored
    )

    csv = pd.read_csv(out / "records.csv", dtype={"review_id": str, "product_id": str}, keep_default_na=False)
    by_id = {d["review_id"]: d for d in data}
    checks["csv_row_per_record"] = len(csv) == len(data) and set(csv["review_id"]) == set(json_ids)
    mism = []
    for row in csv.itertuples(index=False):
        d = by_id.get(row.review_id)
        if d is None:
            mism.append(f"{row.review_id}: in CSV but not JSON")
            continue
        if row.overall_sentiment != d["overall_sentiment"]:
            mism.append(f"{row.review_id}: sentiment {row.overall_sentiment} != {d['overall_sentiment']}")
        if _split(row.complaint_categories) != d["complaint_categories"]:
            mism.append(f"{row.review_id}: complaints differ")
        if int(row.num_aspects) != len(d["aspects"]) or json.loads(row.aspects_json) != d["aspects"]:
            mism.append(f"{row.review_id}: aspects differ")
    checks["csv_values_match_json"] = not mism
    problems += mism[:5]

    asp = pd.read_csv(out / "records_aspects.csv", dtype={"review_id": str}, keep_default_na=False)
    expected = sorted((d["review_id"], a["aspect_category"], a["sentiment"], a["evidence"]) for d in data for a in d["aspects"])
    got = sorted(zip(asp["review_id"], asp["aspect_category"], asp["sentiment"], asp["evidence"]))
    checks["aspects_csv_row_per_aspect"] = expected == got

    fcsv = pd.read_csv(out / "failures.csv", dtype={"review_id": str}, keep_default_na=False)
    fail_ids = set(fcsv["review_id"])
    checks["failures_csv_matches_stored_failures"] = fail_ids == stored_fail
    checks["no_review_both_record_and_failure"] = not (fail_ids & set(json_ids))
    covered = set(json_ids) | fail_ids
    checks["every_run_review_accounted_for"] = run_ids <= covered
    if not checks["every_run_review_accounted_for"]:
        problems.append(f"missing from exports: {sorted(run_ids - covered)[:5]}")

    return {
        "strategy": strategy,
        "records": len(data),
        "aspect_rows": len(asp),
        "failures": len(fcsv),
        "reviews_in_run": len(run_ids),
        "checks": checks,
        "passed": all(checks.values()),
        "problems": problems,
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run", required=True)
    p.add_argument("--out", default=None, help="write a Markdown report here")
    args = p.parse_args()
    s = load_settings()
    is_mock = args.run.startswith("MOCK-")
    run_dir = experiments_dir(s, is_mock) / args.run
    d1_dir = s.paths.mode_dir(is_mock) / "d1" / args.run
    strategies = [d.name for d in sorted(d1_dir.iterdir()) if d.is_dir()]
    results = [verify_strategy(run_dir, d1_dir, st) for st in strategies]

    lines = [f"# D1 export consistency check · run `{args.run}`", "",
             "Generated by `scripts/verify_d1_exports.py` (no LLM calls). Source of truth: the run's `predictions_<strategy>.jsonl`.", "",
             "| strategy | records (JSON) | aspect rows | failures | reviews in run | all checks |", "| --- | --- | --- | --- | --- | --- |"]
    for r in results:
        lines.append(f"| {r['strategy']} | {r['records']} | {r['aspect_rows']} | {r['failures']} | {r['reviews_in_run']} | {'PASS' if r['passed'] else 'FAIL'} |")
    for r in results:
        lines += ["", f"## {r['strategy']}", ""] + [f"- {'✔' if ok else '✘'} {name}" for name, ok in r["checks"].items()]
        lines += [f"  - problem: {x}" for x in r["problems"]]
    report = "\n".join(lines) + "\n"
    print(report)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(report, encoding="utf-8")
    sys.exit(0 if all(r["passed"] for r in results) else 1)


if __name__ == "__main__":
    main()
