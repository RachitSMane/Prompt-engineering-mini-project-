"""Run the deterministic D3 audits (evidence traceability + judge quality) over stored results. No LLM calls.

  python scripts/audit_d3.py --source dataset__few_shot__qwen2.5-3b__v1

Writes outputs/<mode>/d3/audit/<source>/{evidence_audit.json, judge_quality.json} and prints a summary.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import load_settings  # noqa: E402
from evaluation.d3_audit import audit_judgement, audit_summary  # noqa: E402
from evaluation.summary_evaluation import JudgementStore  # noqa: E402
from scripts.run_judge import load_inputs  # noqa: E402
from utils.storage import write_json_atomic  # noqa: E402


def run(s, is_mock: bool, source: str, write: bool = True) -> dict:
    summaries, records = load_inputs(s, is_mock, source)
    by_sid = {x.summary_id: x for x in summaries}
    evidence = [audit_summary(x, records) for x in summaries]
    quality = {}
    for v in ("v1", "v2"):
        store = JudgementStore.for_mode(s, is_mock, v)
        quality[v] = [audit_judgement(j, by_sid[j.summary_id]) for sid, j in sorted(store.latest_by_summary().items()) if sid in by_sid]
    if write:
        out = s.paths.mode_dir(is_mock) / "d3" / "audit" / source
        out.mkdir(parents=True, exist_ok=True)
        write_json_atomic(out / "evidence_audit.json", evidence)
        write_json_atomic(out / "judge_quality.json", quality)
    return {"evidence": evidence, "quality": quality}


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--mock", action="store_true")
    args = p.parse_args()
    s = load_settings()
    res = run(s, args.mock or s.is_mock, args.source)
    ev = res["evidence"]
    print(f"summaries: {len(ev)} · ids+statistics verified: {sum(e['ids_and_statistics_verified'] for e in ev)} · "
          f"points traceable/partial/weak: {sum(e['points_traceable'] for e in ev)}/{sum(e['points_partial'] for e in ev)}/{sum(e['points_weak'] for e in ev)}")
    for e in ev:
        if e["issues"]:
            print(f"  {e['product_id']}: {'; '.join(e['issues'])}")
    for v, rows in res["quality"].items():
        print(f"judge {v}: {len(rows)} judgements · scores valid {sum(r['scores_valid'] for r in rows)} · justifications {sum(r['justifications_present'] for r in rows)} · "
              f"hallucination consistent {sum(r['hallucination_consistent_with_claims'] for r in rows)} · claims {sum(r['claims_flagged'] for r in rows)} "
              f"(likely false positives {sum(r['claims_likely_false_positive'] for r in rows)})")
    print(json.dumps({"written": str(s.paths.mode_dir(args.mock or s.is_mock) / 'd3' / 'audit' / args.source)}))


if __name__ == "__main__":
    main()
