"""Run extract -> aggregate -> summarise (+ optional LLM-as-a-Judge) on the dataset.

Examples
  python scripts/run_chain.py --per-product 30 --judge     # live
  python scripts/run_chain.py --mock --judge                # OFFLINE demo (not a result)
  python scripts/run_chain.py --per-product 20 --extract-only   # stage 1 only (cached; resumable)
  python scripts/run_chain.py --sample --strategy few_shot --seed-from-run <run_id> --no-retry-failed --judge
      # chain on the reference-labelled sample, reusing the D2 run's predictions (no repeated inference)
  add --resume to continue an interrupted chain: stored summaries/judgements are reused when inputs, model and
  prompt version are identical, so no completed LLM call is repeated and no duplicate rows are written
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from annotation.annotation_manager import AnnotationStore, load_sample  # noqa: E402
from config import load_settings  # noqa: E402
from evaluation.prompt_comparison import experiments_dir  # noqa: E402
from evaluation.summary_evaluation import JudgementStore, export_inspection_sample, judge_summary  # noqa: E402
from llm.base import LLMUnavailableError  # noqa: E402
from llm.factory import get_llm_client  # noqa: E402
from pipeline.extraction import ExtractionStopped, ExtractionStore  # noqa: E402
from pipeline.summarization import run_chain  # noqa: E402
from prompts.common import PROMPT_VERSION  # noqa: E402
from prompts.judge import JUDGE_PROMPT_VERSION  # noqa: E402
from prompts.strategies import STRATEGIES  # noqa: E402
from utils.data_loader import dataframe_to_inputs, load_reviews  # noqa: E402
from utils.logging_config import setup_logging  # noqa: E402
from utils.storage import append_jsonl, read_json, read_jsonl  # noqa: E402


def seed_from_run(s, client, strategy: str, run_id: str, review_ids: set[str]) -> dict:
    """Copy a prompt-comparison run's predictions into the chain's extraction cache, after checking that
    they were produced with the same provider, model, prompt version and validation settings."""
    run_dir = experiments_dir(s, client.is_mock) / run_id
    cfg = read_json(run_dir / "config.json")
    if not cfg:
        sys.exit(f"--seed-from-run: no run at {run_dir}")
    expected = {"provider": client.provider, "model_name": client.model_name, "is_mock": client.is_mock, "prompt_version": PROMPT_VERSION,
                "max_repair_attempts": s.max_repair_attempts, "strict_evidence_grounding": s.strict_evidence_grounding}
    mismatch = {k: (cfg.get(k), v) for k, v in expected.items() if cfg.get(k) != v}
    if mismatch:
        sys.exit(f"--seed-from-run: run {run_id} is not compatible (run value, current value): {mismatch}")
    src = run_dir / f"predictions_{strategy}.jsonl"
    if not src.exists():
        sys.exit(f"--seed-from-run: run {run_id} has no predictions for strategy {strategy}")
    store = ExtractionStore.for_run(s, client.is_mock, "dataset", strategy, client.model_name)
    return store.seed_from(src, review_ids)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles default to cp1252
    p = argparse.ArgumentParser()
    p.add_argument("--strategy", default="few_shot", choices=list(STRATEGIES))
    p.add_argument("--per-product", type=int, default=None, help="max reviews per product (cost control)")
    p.add_argument("--products", nargs="*", default=None, help="restrict to these product ids")
    p.add_argument("--sample", action="store_true", help="only the reference-labelled annotation sample (flagged reviews excluded)")
    p.add_argument("--seed-from-run", default=None, help="reuse this prompt-comparison run's predictions for --strategy (same model/prompt/settings)")
    p.add_argument("--no-retry-failed", action="store_true", help="keep cached output failures instead of calling the model again")
    p.add_argument("--judge", action="store_true", help="also run LLM-as-a-Judge on each summary")
    p.add_argument("--extract-only", action="store_true", help="only run (and cache) the extraction stage; no summaries or judging")
    p.add_argument("--mock", action="store_true", help="OFFLINE mock client (development only)")
    p.add_argument("--resume", action="store_true", help="reuse stored summaries/judgements produced from identical inputs with the same model and prompt version")
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
    if args.sample:
        sample_ids = load_sample(s.paths.annotation_sample_csv)["review_id"].tolist()
        evaluable = {a.review_id for a in AnnotationStore(s.paths.annotations_json).completed(sample_ids)}
        df = df[df["review_id"].isin(evaluable)]
    if args.products:
        df = df[df["product_id"].isin(args.products)]
    if args.per_product:
        df = df.groupby("product_id", group_keys=False).head(args.per_product)
    if use_mock:
        print("*** MOCK MODE: deterministic offline fixtures. Output is NOT an experimental result. ***")
    if args.seed_from_run:
        seeded = seed_from_run(s, client, args.strategy, args.seed_from_run, set(df["review_id"]))
        print(f"Seeded extraction cache from run {args.seed_from_run}: {seeded}")
    print(f"Chain on {len(df)} reviews / {df['product_id'].nunique()} products with {client.provider}/{client.model_name}")

    def progress(stage: str, i: int, n: int) -> None:
        if i == n or i % 25 == 0:
            print(f"  {stage}: {i}/{n}", flush=True)

    try:
        result = run_chain(client, dataframe_to_inputs(df), s, strategy=args.strategy, source="dataset", progress=progress, summarize=not args.extract_only, retry_failed=not args.no_retry_failed, reuse_summaries=args.resume)
    except ExtractionStopped as exc:
        sys.exit(f"STOPPED: {exc}\nSuccessful extractions are cached; re-run the same command to continue.")
    ex = result.extraction
    print(f"Extracted {len(ex.records)} ok / {len(ex.failures)} failed ({ex.from_cache} from cache, {ex.new_calls} new LLM calls); {len(result.summaries)} summaries ({result.reused_summaries} reused from storage)")
    for pid, err in result.summary_errors.items():
        print(f"  summary failed for {pid}: {err}")

    if judge and not args.extract_only:
        jstore = JudgementStore.for_mode(s, use_mock)
        judgements = []
        stored = jstore.latest_by_summary() if args.resume else {}
        fail_path = jstore.path.parent / "judge_failures.jsonl"
        stored_failures = {r["summary_id"]: r for r in read_jsonl(fail_path)
                           if (r.get("judge_provider"), r.get("judge_model"), r.get("is_mock"), r.get("prompt_version")) == (judge.provider, judge.model_name, judge.is_mock, JUDGE_PROMPT_VERSION)} if args.resume else {}
        for pid, summary in result.summaries.items():
            prev = stored.get(summary.summary_id)
            if prev is not None and (prev.judge_provider, prev.judge_model, prev.is_mock, prev.prompt_version) == (judge.provider, judge.model_name, judge.is_mock, JUDGE_PROMPT_VERSION):
                judgements.append(prev)
                print(f"  judged {pid}: reused stored judgement {prev.judgement_id}")
                continue
            if summary.summary_id in stored_failures:  # one judge attempt per summary: stored failures are final
                print(f"  judge failed for {pid}: stored failure kept (not re-sampled)")
                continue
            j, res = judge_summary(judge, summary, result.extraction.records, s)
            if j:
                jstore.add(j)
                judgements.append(j)
                v = j.verdict
                print(f"  judged {pid}: F={v.faithfulness.score} C={v.coverage.score} K={v.consistency.score} H={v.hallucination.score}")
            else:
                append_jsonl(fail_path, [{"summary_id": summary.summary_id, "product_id": pid, "judge_provider": judge.provider, "judge_model": judge.model_name,
                                          "is_mock": judge.is_mock, "prompt_version": JUDGE_PROMPT_VERSION, "error": res.error, "source": "run_chain"}])
                print(f"  judge failed for {pid}: {res.error}")
        out = s.paths.mode_dir(use_mock) / "judgements" / "inspection_sample.md"
        out.write_text(export_inspection_sample(judgements, {x.summary_id: x for x in result.summaries.values()}), encoding="utf-8")
        print(f"Inspection sample: {out}")


if __name__ == "__main__":
    main()
