"""Run extract -> aggregate -> summarise (+ optional LLM-as-a-Judge) on the dataset.

Examples
  python scripts/run_chain.py --per-product 30 --judge     # live
  python scripts/run_chain.py --mock --judge                # OFFLINE demo (not a result)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import load_settings  # noqa: E402
from evaluation.summary_evaluation import JudgementStore, export_inspection_sample, judge_summary  # noqa: E402
from llm.base import LLMUnavailableError  # noqa: E402
from llm.factory import get_llm_client  # noqa: E402
from pipeline.summarization import run_chain  # noqa: E402
from prompts.strategies import STRATEGIES  # noqa: E402
from utils.data_loader import dataframe_to_inputs, load_reviews  # noqa: E402
from utils.logging_config import setup_logging  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--strategy", default="few_shot", choices=list(STRATEGIES))
    p.add_argument("--per-product", type=int, default=None, help="max reviews per product (cost control)")
    p.add_argument("--products", nargs="*", default=None, help="restrict to these product ids")
    p.add_argument("--judge", action="store_true", help="also run LLM-as-a-Judge on each summary")
    p.add_argument("--mock", action="store_true", help="OFFLINE mock client (development only)")
    args = p.parse_args()

    setup_logging()
    s = load_settings()
    use_mock = args.mock or s.is_mock
    try:
        client = get_llm_client(s, use_mock=use_mock)
        judge = get_llm_client(s, use_mock=use_mock, role="judge") if args.judge else None
    except LLMUnavailableError as exc:
        sys.exit(f"Live LLM unavailable: {exc}\nUse --mock for an offline pipeline demo (not a result).")

    df = load_reviews(s.paths.reviews_csv)
    if args.products:
        df = df[df["product_id"].isin(args.products)]
    if args.per_product:
        df = df.groupby("product_id", group_keys=False).head(args.per_product)
    if use_mock:
        print("*** MOCK MODE: deterministic offline fixtures. Output is NOT an experimental result. ***")
    print(f"Chain on {len(df)} reviews / {df['product_id'].nunique()} products with {client.provider}/{client.model_name}")

    def progress(stage: str, i: int, n: int) -> None:
        if i == n or i % 25 == 0:
            print(f"  {stage}: {i}/{n}", flush=True)

    result = run_chain(client, dataframe_to_inputs(df), s, strategy=args.strategy, source="dataset", progress=progress)
    print(f"Extracted {len(result.extraction.records)} ok / {len(result.extraction.failures)} failed; {len(result.summaries)} summaries")
    for pid, err in result.summary_errors.items():
        print(f"  summary failed for {pid}: {err}")

    if judge:
        jstore = JudgementStore.for_mode(s, use_mock)
        judgements = []
        for pid, summary in result.summaries.items():
            j, res = judge_summary(judge, summary, result.extraction.records, s)
            if j:
                jstore.add(j)
                judgements.append(j)
                v = j.verdict
                print(f"  judged {pid}: F={v.faithfulness.score} C={v.coverage.score} K={v.consistency.score} H={v.hallucination.score}")
            else:
                print(f"  judge failed for {pid}: {res.error}")
        out = s.paths.mode_dir(use_mock) / "judgements" / "inspection_sample.md"
        out.write_text(export_inspection_sample(judgements, {x.summary_id: x for x in result.summaries.values()}), encoding="utf-8")
        print(f"Inspection sample: {out}")


if __name__ == "__main__":
    main()
