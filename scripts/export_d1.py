"""Deliverable 1 export: validated extraction records of an experiment run as JSON + CSV, with reports.

For each strategy of a saved run this writes, under ``outputs/<mode>/d1/<run_id>/<strategy>/``:

  records.json            valid structured records (re-validated after reload, evidence re-grounded)
  records.csv             one row per review (fixed columns)
  records_aspects.csv     one row per extracted aspect (with its evidence quote)
  failures.csv            reviews with no valid output after repair (never silently dropped)
  validation_report.json  validity, repair and failure statistics, runtime and token totals

plus ``summary.md`` (execution summary for all strategies). No LLM calls are made; everything is
derived from the run's saved predictions.

Usage:  python scripts/export_d1.py --run <run_id> [--strategies zero_shot few_shot cot]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import load_settings  # noqa: E402
from evaluation.prompt_comparison import experiments_dir  # noqa: E402
from pipeline.extraction import ExtractionStore, is_provider_failure  # noqa: E402
from schemas.review_schema import ReviewRecord  # noqa: E402
from utils.export import export_records, failures_to_dataframe  # noqa: E402
from utils.storage import read_json, write_json_atomic  # noqa: E402


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    v = sorted(values)
    return round(v[min(len(v) - 1, int(q * (len(v) - 1) + 0.5))], 2)


def revalidate(json_path: Path) -> dict:
    """Reload the exported JSON and validate every record again, with evidence grounding against its own text."""
    data = json.loads(json_path.read_text(encoding="utf-8"))
    invalid = []
    for d in data:
        try:
            ReviewRecord.model_validate(d, context={"review_text": d["review_text"]})
        except Exception as exc:  # noqa: BLE001 - report every problem
            invalid.append({"review_id": d.get("review_id"), "error": str(exc)[:300]})
    return {"reloaded_records": len(data), "revalidated_ok": len(data) - len(invalid), "revalidation_errors": invalid}


def export_strategy(run_dir: Path, strategy: str, review_ids: list[str], out_root: Path) -> dict:
    store = ExtractionStore(run_dir / f"predictions_{strategy}.jsonl")
    records, failures = store.records, store.failures
    provider_failures = [f for f in failures if is_provider_failure(f)]
    output_failures = [f for f in failures if not is_provider_failure(f)]
    out = out_root / strategy
    paths = export_records(records, out, "records")
    failures_to_dataframe(failures).to_csv(out / "failures.csv", index=False, encoding="utf-8")
    metas = [r.extraction_metadata for r in records] + [f.extraction_metadata for f in failures]
    lat = [m.total_latency_s for m in metas]
    attempt_errors = Counter(e.split(":")[0][:80] for m in metas for a in m.attempts for e in a.errors)
    predicted = ({r.review_id for r in records} | {f.review_id for f in failures}) & set(review_ids)
    report = {
        "run_id": run_dir.name,
        "strategy": strategy,
        "model": metas[0].model_name if metas else None,
        "provider": metas[0].provider if metas else None,
        "reviews_in_run": len(review_ids),
        "predictions_stored": len(predicted),
        "missing_predictions": len(set(review_ids) - predicted),
        "coverage": round(len(predicted) / len(review_ids), 4) if review_ids else 0.0,
        "valid_records": len(records),
        "failed_output": len(output_failures),
        "failed_provider": len(provider_failures),
        "validity_rate": round(len(records) / len(metas), 4) if metas else None,
        "first_attempt_valid": sum(m.validation_success and m.repair_attempts == 0 for m in metas),
        "needed_llm_repair": sum(m.repair_attempts > 0 for m in metas),
        "repaired_successfully": sum(m.validation_success and m.repair_attempts > 0 for m in metas),
        "repair_attempts_distribution": {str(k): v for k, v in sorted(Counter(m.repair_attempts for m in metas).items())},
        "records_with_local_fixes": sum(m.local_repairs > 0 for m in metas),
        "top_attempt_errors": dict(attempt_errors.most_common(10)),
        "failure_errors": [{"review_id": f.review_id, "error": f.error[:300]} for f in failures],
        "runtime_s": {
            "total": round(sum(lat), 1),
            "mean": round(statistics.mean(lat), 2) if lat else None,
            "median": _quantile(lat, 0.5),
            "p95": _quantile(lat, 0.95),
            "max": round(max(lat), 2) if lat else None,
        },
        "tokens": {"input_total": sum(m.input_tokens or 0 for m in metas), "output_total": sum(m.output_tokens or 0 for m in metas)},
        "files": {k: str(v) for k, v in paths.items()} | {"failures_csv": str(out / "failures.csv")},
    }
    report["revalidation"] = revalidate(paths["json"])
    write_json_atomic(out / "validation_report.json", report)
    return report


def summary_md(reports: list[dict], run_id: str) -> str:
    lines = [
        f"# D1 — Structured extraction export · run `{run_id}`",
        "",
        "Derived from the run's saved predictions (no new LLM calls). Every exported record was reloaded and "
        "re-validated with the Pydantic schema, including evidence grounding against its own review text.",
        "",
        "| strategy | coverage | valid | failed (output) | failed (provider) | first-attempt valid | needed repair | repaired OK | re-validated | mean s | p95 s | input tokens | output tokens |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in reports:
        rv = r["revalidation"]
        lines.append(
            f"| {r['strategy']} | {r['predictions_stored']}/{r['reviews_in_run']} | {r['valid_records']} | {r['failed_output']} | {r['failed_provider']} | "
            f"{r['first_attempt_valid']} | {r['needed_llm_repair']} | {r['repaired_successfully']} | {rv['revalidated_ok']}/{rv['reloaded_records']} | "
            f"{r['runtime_s']['mean']} | {r['runtime_s']['p95']} | {r['tokens']['input_total']:,} | {r['tokens']['output_total']:,} |"
        )
    for r in reports:
        lines += [
            "",
            f"## {r['strategy']}",
            "",
            f"- Model: `{r['provider']}/{r['model']}`",
            f"- Repair attempts distribution (attempts: reviews): `{r['repair_attempts_distribution']}`",
            f"- Records with deterministic (local) fixes: {r['records_with_local_fixes']}",
            f"- Most frequent validation errors before repair: `{r['top_attempt_errors'] or 'none'}`",
            f"- Failures: {len(r['failure_errors'])}" + "".join(f"\n  - `{f['review_id']}`: {f['error'][:160]}" for f in r["failure_errors"]),
        ]
    return "\n".join(lines) + "\n"


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles default to cp1252
    p = argparse.ArgumentParser(description="Export D1 structured extraction results of a saved run.")
    p.add_argument("--run", required=True, help="experiment run id (outputs/<mode>/experiments/<run_id>)")
    p.add_argument("--strategies", nargs="*", default=None)
    args = p.parse_args()
    s = load_settings()
    is_mock = args.run.startswith("MOCK-")
    run_dir = experiments_dir(s, is_mock) / args.run
    config = read_json(run_dir / "config.json")
    if not config:
        sys.exit(f"No run found at {run_dir}")
    out_root = s.paths.mode_dir(is_mock) / "d1" / args.run
    names = [n for n in (args.strategies or config["strategies"]) if (run_dir / f"predictions_{n}.jsonl").exists()]
    reports = [export_strategy(run_dir, name, config["review_ids"], out_root) for name in names]
    (out_root / "summary.md").write_text(summary_md(reports, args.run), encoding="utf-8")
    print((out_root / "summary.md").read_text(encoding="utf-8"))
    print(f"Wrote {out_root}")


if __name__ == "__main__":
    main()
