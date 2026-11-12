"""Run the zero-shot / few-shot / CoT comparison on the annotation sample.

Examples
  python scripts/run_experiment.py                       # live, all strategies, all 300 sample reviews
  python scripts/run_experiment.py --limit 20            # quick pilot on 20 reviews
  python scripts/run_experiment.py --resume <run_id>     # continue an interrupted run
  python scripts/run_experiment.py --mock --limit 50     # OFFLINE demo (not a result)
  python scripts/run_experiment.py --reevaluate <run_id> # recompute metrics after more annotations
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from annotation.annotation_manager import AnnotationStore, label_provenance, load_sample, provenance_text  # noqa: E402
from config import load_settings  # noqa: E402
from evaluation.prompt_comparison import ExperimentStopped, comparison_table, evaluate_run, experiments_dir, new_run_id, run_experiment  # noqa: E402
from llm.base import LLMUnavailableError  # noqa: E402
from llm.factory import get_llm_client  # noqa: E402
from prompts.strategies import STRATEGIES  # noqa: E402
from utils.data_loader import dataframe_to_inputs  # noqa: E402
from utils.logging_config import setup_logging  # noqa: E402


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles default to cp1252
    p = argparse.ArgumentParser()
    p.add_argument("--strategies", nargs="+", default=list(STRATEGIES), choices=list(STRATEGIES))
    p.add_argument("--limit", type=int, default=None, help="only the first N sample reviews")
    p.add_argument("--only-annotated", action="store_true", help="only reviews with a complete reference label (flagged reviews excluded)")
    p.add_argument("--resume", default=None, help="run_id to resume")
    p.add_argument("--reevaluate", default=None, help="run_id to re-evaluate without LLM calls")
    p.add_argument("--mock", action="store_true", help="use the OFFLINE mock client (development only)")
    args = p.parse_args()

    setup_logging()
    s = load_settings()
    store = AnnotationStore(s.paths.annotations_json)
    sample = load_sample(s.paths.annotation_sample_csv)
    if sample.empty:
        sys.exit("No annotation sample. Run scripts/build_annotation_sample.py first.")
    gold = store.completed(sample["review_id"].tolist())
    required = store.progress(sample["review_id"].tolist())["evaluable"]  # flagged reviews are excluded

    if args.reevaluate:
        run_dir = experiments_dir(s, args.reevaluate.startswith("MOCK-")) / args.reevaluate
        metrics = evaluate_run(run_dir, gold, required)
        print(comparison_table(metrics).to_string(index=False))
        return

    try:
        client = get_llm_client(s, use_mock=args.mock or s.is_mock)
    except LLMUnavailableError as exc:
        sys.exit(f"Live LLM unavailable: {exc}\nUse --mock for an offline pipeline demo (not an experimental result).")

    if args.only_annotated:
        sample = sample[sample["review_id"].isin({g.review_id for g in gold})]
    if args.limit:
        sample = sample.head(args.limit)
    reviews = dataframe_to_inputs(sample)
    if client.is_mock:
        print("*** MOCK MODE: deterministic offline fixtures. Output is NOT an experimental result. ***")
    print(f"Running {args.strategies} on {len(reviews)} reviews with {client.provider}/{client.model_name}; {provenance_text(label_provenance(gold))}.")

    def progress(strategy: str, i: int, n: int) -> None:
        if i == n or i % 10 == 0:
            print(f"  {strategy}: {i}/{n}", flush=True)

    run_id = args.resume or new_run_id(client.model_name, client.is_mock)
    resume_cmd = f"python scripts/run_experiment.py --resume {run_id}" + (" --only-annotated" if args.only_annotated else "") + (f" --limit {args.limit}" if args.limit else "")
    print(f"Run id: {run_id}  (resume with: {resume_cmd})", flush=True)
    try:
        run_dir = run_experiment(s, client, args.strategies, reviews, gold, run_id=run_id, progress=progress, required_gold=required)
    except ExperimentStopped as exc:
        print(f"\nSTOPPED: {exc}\nMetrics are marked 'incomplete'. After fixing the provider, resume with:\n  {resume_cmd}", flush=True)
        sys.exit(2)
    print(f"\nRun directory: {run_dir}\nReport: {run_dir / 'report.md'}")


if __name__ == "__main__":
    main()
