"""Negative control for the LLM-as-a-Judge: does it score a deliberately corrupted summary lower?

For every product summary of one extraction source, a corrupted copy is built with
``evaluation.summary_evaluation.corrupt_summary`` (inverted overall opinion + fabricated percentage,
an invented feature, all complaints removed) and judged with the same judge model and prompt.
The original summary's judgement is reused from ``judgements.jsonl`` (judged by scripts/run_chain.py --judge);
originals without a judgement are judged here, unless --only-judged-originals is given: then such products are skipped
(and listed in summary.json), so an original whose stored judgement attempt failed is never sent to the judge again.

Corrupted summaries and their judgements are written ONLY to ``outputs/<mode>/d3/judge_sanity/`` and never to the
summary/judgement stores the app displays.

Usage:  python scripts/judge_sanity_check.py --source dataset__few_shot__qwen2.5-3b__v1
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import load_settings  # noqa: E402
from evaluation.summary_evaluation import CORRUPTION_FABRICATED_FEATURE, JudgementStore, corrupt_summary, judge_summary  # noqa: E402
from llm.factory import get_llm_client  # noqa: E402
from pipeline.extraction import ExtractionStore  # noqa: E402
from pipeline.summarization import SummaryStore  # noqa: E402
from utils.logging_config import setup_logging  # noqa: E402
from utils.storage import append_jsonl, read_jsonl, write_json_atomic  # noqa: E402

DIMS = ("faithfulness", "coverage", "consistency", "hallucination")


def scores(verdict) -> dict:
    return {d: getattr(verdict, d).score for d in DIMS} | {"mean": round(statistics.mean(getattr(verdict, d).score for d in DIMS), 2)}


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles default to cp1252
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, help="extraction source name, e.g. dataset__few_shot__qwen2.5-3b__v1")
    p.add_argument("--mock", action="store_true")
    p.add_argument("--only-judged-originals", action="store_true", help="skip products whose original summary has no stored judgement (no re-sampling of failed judgements)")
    args = p.parse_args()
    setup_logging()
    s = load_settings()
    is_mock = args.mock or s.is_mock
    judge = get_llm_client(s, use_mock=is_mock, role="judge")
    mode = s.paths.mode_dir(is_mock)
    summaries = {pid: x for (pid, src), x in SummaryStore.for_mode(s, is_mock).latest().items() if src == args.source}
    if not summaries:
        sys.exit(f"No summaries for source {args.source}")
    records = ExtractionStore(mode / "extractions" / f"{args.source}.jsonl").records
    real = JudgementStore.for_mode(s, is_mock).latest_by_summary()
    out_dir = mode / "d3" / "judge_sanity" / args.source
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "pairs.jsonl"
    done = {r["product_id"]: r for r in read_jsonl(out_path) if r.get("original_summary_id") in {x.summary_id for x in summaries.values()}}
    print(f"Judge {judge.provider}/{judge.model_name}; {len(summaries)} summaries; {len(done)} pairs already judged")

    skipped: list[str] = []
    for pid, summ in summaries.items():
        if pid in done:
            continue
        orig = real.get(summ.summary_id)
        if orig is None and args.only_judged_originals:
            skipped.append(pid)
            print(f"  {pid}: skipped (original summary has no stored judgement)")
            continue
        if orig is None:
            orig, res = judge_summary(judge, summ, records, s)
            if orig is None:
                print(f"  {pid}: original could not be judged: {res.error}")
                continue
        bad = corrupt_summary(summ)
        bj, res = judge_summary(judge, bad, records, s)
        row = {"product_id": pid, "original_summary_id": summ.summary_id, "judge_model": judge.model_name,
               "original_scores": scores(orig.verdict), "original_unsupported_claims": orig.verdict.unsupported_claims}
        if bj is None:
            row |= {"corrupted_judged": False, "error": res.error}
        else:
            claims = " ".join(bj.verdict.unsupported_claims).lower()
            row |= {"corrupted_judged": True, "corrupted_scores": scores(bj.verdict), "corrupted_unsupported_claims": bj.verdict.unsupported_claims,
                    "flagged_fabricated_feature": "wi-fi" in claims or "voice" in claims,
                    "flagged_fabricated_percentage": "92%" in claims or "98%" in claims,
                    "corrupted_judgement": bj.model_dump(mode="json"), "corrupted_summary": bad.content.model_dump(mode="json")}
        append_jsonl(out_path, [row])
        done[pid] = row
        print(f"  {pid}: original {row['original_scores']['mean']} -> corrupted {row.get('corrupted_scores', {}).get('mean', 'n/a')}", flush=True)

    pairs = [r for r in done.values() if r.get("corrupted_judged")]
    n = len(pairs)
    summary = {
        "source": args.source, "judge_model": judge.model_name, "products": len(summaries), "pairs_judged": n,
        "failed_corrupted_judgements": len(done) - n,
        "skipped_original_not_judged": sorted(skipped),
        "corruption": ["overall opinion inverted with a fabricated percentage", f"invented feature: {CORRUPTION_FABRICATED_FEATURE!r}", "all complaints removed"],
    }
    if n:
        summary |= {
            "mean_original": {d: round(statistics.mean(r["original_scores"][d] for r in pairs), 2) for d in (*DIMS, "mean")},
            "mean_corrupted": {d: round(statistics.mean(r["corrupted_scores"][d] for r in pairs), 2) for d in (*DIMS, "mean")},
            "corrupted_scored_lower_mean": sum(r["corrupted_scores"]["mean"] < r["original_scores"]["mean"] for r in pairs),
            "corrupted_scored_equal_mean": sum(r["corrupted_scores"]["mean"] == r["original_scores"]["mean"] for r in pairs),
            "corrupted_scored_higher_mean": sum(r["corrupted_scores"]["mean"] > r["original_scores"]["mean"] for r in pairs),
            "corrupted_faithfulness_lower": sum(r["corrupted_scores"]["faithfulness"] < r["original_scores"]["faithfulness"] for r in pairs),
            "flagged_fabricated_feature": sum(r["flagged_fabricated_feature"] for r in pairs),
            "flagged_fabricated_percentage": sum(r["flagged_fabricated_percentage"] for r in pairs),
        }
    write_json_atomic(out_dir / "summary.json", summary)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
