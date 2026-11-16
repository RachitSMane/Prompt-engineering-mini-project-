"""Resumable LLM-as-a-Judge run over the stored product summaries of one extraction source.

  python scripts/run_judge.py --source dataset__few_shot__qwen2.5-3b__v1                 # judge v2, all summaries
  python scripts/run_judge.py --source dataset__few_shot__qwen2.5-3b__v1 --limit 2       # pilot on 2 summaries
  python scripts/run_judge.py --source dataset__few_shot__qwen2.5-3b__v1 --products B09WWM1WLS
  python scripts/run_judge.py --source ... --mock                                         # OFFLINE pipeline check only
  python scripts/run_judge.py --version v3 --source ...                                   # judge v3 (see scripts/judge_v3.py)

Behaviour
* Results go to ``outputs/<mode>/judgements/<version>/`` (v1 results in ``outputs/<mode>/judgements/`` are never touched).
* Resumable: a summary with a stored judgement from the same judge model + version is skipped (never re-judged);
  a stored invalid-output failure is final unless ``--retry-failed`` is given (temperature 0 would reproduce it).
* Bounded: at most 1 + MAX_REPAIR_ATTEMPTS judge calls per summary; provider errors retried at most twice, then the
  run stops WITHOUT recording a failure, so the same command resumes it.
* Thermal guard before every call (pause >= 88 °C until <= 75 °C, abort >= 94 °C).
* Failures are stored with every attempt's raw output and validation errors.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import load_settings  # noqa: E402
from evaluation.summary_evaluation import JudgementStore, JudgeProviderStopped, get_judge_client, judge_summaries  # noqa: E402
from llm.base import LLMUnavailableError  # noqa: E402
from pipeline.extraction import ExtractionStore  # noqa: E402
from pipeline.summarization import SummaryStore  # noqa: E402
from prompts.judge import CURRENT_JUDGE_VERSION, JUDGE_PROMPT_VERSION, JUDGE_PROMPT_VERSION_V2  # noqa: E402
from utils.gpu_guard import GpuGuard, GpuTooHot  # noqa: E402
from utils.logging_config import setup_logging  # noqa: E402


def load_inputs(s, is_mock: bool, source: str):
    mode = s.paths.mode_dir(is_mock)
    summaries = [x for (pid, src), x in sorted(SummaryStore.for_mode(s, is_mock).latest().items()) if src == source]
    records = ExtractionStore(mode / "extractions" / f"{source}.jsonl").records
    return summaries, records


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    argv = sys.argv[1:]
    if "v3" in argv and argv[argv.index("v3") - 1] == "--version":
        # Judge v3 (single call, raw response kept, no repair): separate runner, same process name for the GPU watchdog.
        setup_logging()
        from scripts.judge_v3 import main as main_v3

        main_v3(argv)
        return
    p = argparse.ArgumentParser(description="Resumable LLM-as-a-Judge run")
    p.add_argument("--source", required=True, help="extraction source, e.g. dataset__few_shot__qwen2.5-3b__v1")
    p.add_argument("--version", default=CURRENT_JUDGE_VERSION, choices=[JUDGE_PROMPT_VERSION, JUDGE_PROMPT_VERSION_V2])
    p.add_argument("--products", nargs="*", default=None)
    p.add_argument("--limit", type=int, default=None, help="only the first N summaries (pilot)")
    p.add_argument("--retry-failed", action="store_true", help="re-judge summaries whose stored judgement for this version FAILED")
    p.add_argument("--mock", action="store_true", help="OFFLINE mock judge (pipeline check, not a result)")
    args = p.parse_args()

    setup_logging()
    s = load_settings()
    is_mock = args.mock or s.is_mock
    summaries, records = load_inputs(s, is_mock, args.source)
    if args.products:
        summaries = [x for x in summaries if x.product_id in set(args.products)]
    if args.limit:
        summaries = summaries[: args.limit]
    if not summaries:
        sys.exit(f"No summaries for source {args.source}")
    try:
        client = get_judge_client(s, use_mock=is_mock, version=args.version)
    except LLMUnavailableError as exc:
        sys.exit(f"Judge unavailable: {exc}")
    store = JudgementStore.for_mode(s, is_mock, args.version)
    guard = GpuGuard(s.paths.root / "logs" / "gpu_guard.csv")
    constrained = getattr(client, "constrained_decoding", False)
    print(f"Judge {client.provider}/{client.model_name} · version {args.version} · constrained decoding {constrained} · "
          f"{len(summaries)} summaries · store {store.path}", flush=True)
    if is_mock:
        print("*** MOCK judge: placeholder verdicts, NOT an evaluation result ***")

    def progress(event: str, summ, j, err: str) -> None:
        name = f"{summ.product_id} ({(summ.product_title or '')[:40]})"
        if event == "judged":
            v = j.verdict
            print(f"  judged   {name}: F={v.faithfulness.score} C={v.coverage.score} K={v.consistency.score} H={v.hallucination.score} "
                  f"claims={len(v.unsupported_claims)} repairs={j.repair_attempts} {j.latency_s:.0f}s", flush=True)
        elif event == "failed":
            print(f"  FAILED   {name}: {err[:300]}", flush=True)
        elif event == "reused":
            print(f"  reused   {name}: stored judgement {j.judgement_id}", flush=True)
        else:
            print(f"  skipped  {name}: stored failure for this version is final (use --retry-failed to re-run)", flush=True)

    try:
        res = judge_summaries(client, summaries, records, s, store, version=args.version, retry_failed=args.retry_failed,
                              before_call=lambda _s: guard.wait_until_safe(), progress=progress, source="run_judge")
    except (JudgeProviderStopped, GpuTooHot) as exc:
        sys.exit(f"STOPPED: {exc}\nEverything judged so far is stored. Re-run the same command to resume.")
    summary = {"version": args.version, "judge_model": client.model_name, "summaries": len(summaries), "judged_ok": len(res.judged),
               "reused": len(res.reused), "new_failures": len(res.failed), "skipped_stored_failures": len(res.skipped_stored_failure),
               "new_judge_runs": res.new_calls, "max_gpu_temp_c": guard.max_seen}
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
