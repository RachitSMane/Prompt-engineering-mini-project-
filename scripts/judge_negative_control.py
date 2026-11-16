"""Judge v2 negative control: one planted unsupported claim per corrupted summary (see evaluation/negative_control.py).

  python scripts/judge_negative_control.py --source dataset__few_shot__qwen2.5-3b__v1            # all products
  python scripts/judge_negative_control.py --source ... --limit 3                                # pilot
  python scripts/judge_negative_control.py --source ... --report-only                            # summarise stored pairs, no LLM

* Originals are the v2 judgements in ``outputs/<mode>/judgements/v2/`` (produced by scripts/run_judge.py). A product
  whose original has no successful v2 judgement is skipped (listed), so failed originals are not re-sampled here.
* Each product gets two corrupted copies (invented feature, fabricated statistic), judged with the same judge/version.
* Results: ``outputs/<mode>/d3/judge_negative_control_v2/<source>/pairs.jsonl`` (append-only, resumable) and
  ``summary.json``. Corrupted summaries never enter the summary or judgement stores shown in the app.
* Same safety as run_judge.py: thermal guard, bounded repairs, provider errors stop the run without recording a result.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import load_settings  # noqa: E402
from evaluation.negative_control import build_planted, planted_claim_detected, planted_keyword_in_reason_only, scores  # noqa: E402
from schemas.evaluation_schema import SummaryJudgement  # noqa: E402
from evaluation.summary_evaluation import JudgementStore, failure_record, get_judge_client, judge_summary  # noqa: E402
from llm.base import LLMError  # noqa: E402
from prompts.judge import CURRENT_JUDGE_VERSION  # noqa: E402
from scripts.run_judge import load_inputs  # noqa: E402
from utils.gpu_guard import GpuGuard, GpuTooHot  # noqa: E402
from utils.logging_config import setup_logging  # noqa: E402
from utils.storage import append_jsonl, read_json, read_jsonl, write_json_atomic  # noqa: E402


def wilson_ci(k: int, n: int, z: float = 1.96) -> list[float] | None:
    if not n:
        return None
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [round((c - h) / d, 3), round((c + h) / d, 3)]


def _redetect(r: dict, originals: dict | None = None) -> None:
    """Recompute detection from the STORED judgements with the current rule (claim text only); pairs.jsonl is never edited.

    The stored ``detected`` field was written with the earlier rule (claims + reasons) and is kept as ``detected_stored``.
    """
    kw = tuple(r["keywords"])
    r["detected_stored"] = r.get("detected")
    if r.get("corrupted_judged"):
        j = SummaryJudgement.model_validate(r["corrupted_judgement"])
        r["detected"] = planted_claim_detected(j, kw)
        r["keyword_in_reason_only"] = planted_keyword_in_reason_only(j, kw)
    if originals and r.get("original_judgement_id") in originals:
        r["original_flagged_planted_keywords"] = planted_claim_detected(originals[r["original_judgement_id"]], kw)


def summarise(rows: list[dict], skipped: list[str], judge_model: str, originals_by_id: dict | None = None) -> dict:
    rows = [dict(r) for r in rows]
    for r in rows:
        _redetect(r, originals_by_id)
    out = {"judge_model": judge_model, "version": CURRENT_JUDGE_VERSION, "skipped_original_not_judged": sorted(skipped), "by_kind": {},
           "detection_rule": "claim_text_only",
           "note": "Detection = one of the judge's unsupported CLAIMS contains the planted claim's keyword(s) (the 'reason' field is not "
                   "counted: a judge that cites the planted number as fact has not detected it). Keyword control = the ORIGINAL "
                   "summary's judgement already names that keyword. Every original judgement is reused for both defect kinds."}
    controls = {}
    for r in rows:  # one keyword check per (product, kind) against the original's judgement
        controls[(r["product_id"], r["kind"])] = r.get("original_flagged_planted_keywords", False)
    out["original_false_positive"] = {"n": len(controls), "flagged": sum(controls.values()),
                                      "rate": round(sum(controls.values()) / len(controls), 3) if controls else None,
                                      "distinct_original_judgements": len({r["original_judgement_id"] for r in rows})}
    if originals_by_id:
        used = [originals_by_id[i] for i in {r["original_judgement_id"] for r in rows} if i in originals_by_id]
        out["originals_flagging_any_claim"] = {"n": len(used), "flagged": sum(bool(j.verdict.unsupported_claims) for j in used),
                                               "note": "Original (uncorrupted) summaries for which the judge listed at least one unsupported claim. "
                                                       "This is the general false-positive exposure; see D3_JUDGE_EVALUATION.md for which claims are legitimate."}

    def block(rs: list[dict]) -> dict:
        ok = [r for r in rs if r["corrupted_judged"]]
        n = len(ok)
        det = sum(r["detected"] for r in ok)
        cmp = lambda d, op: sum(op(r["corrupted_scores"][d], r["original_scores"][d]) for r in ok)
        lt, gt = (lambda a, b: a < b), (lambda a, b: a > b)
        return {
            "pairs_attempted": len(rs), "corrupted_judged": n, "corrupted_judge_failed": len(rs) - n,
            "planted_claim_detected": det, "missed": n - det,
            "detection_rate": round(det / n, 3) if n else None, "detection_wilson95": wilson_ci(det, n),
            "false_negative_rate": round((n - det) / n, 3) if n else None,
            "detected_with_previous_rule": sum(bool(r.get("detected_stored")) for r in ok),
            "keyword_in_reason_only": sum(bool(r.get("keyword_in_reason_only")) for r in ok),
            "faithfulness_lower": cmp("faithfulness", lt), "faithfulness_higher": cmp("faithfulness", gt),
            "hallucination_lower": cmp("hallucination", lt), "hallucination_higher": cmp("hallucination", gt),
            "mean_score_lower": cmp("mean", lt),
        }

    for kind in sorted({r["kind"] for r in rows}):
        out["by_kind"][kind] = block([r for r in rows if r["kind"] == kind])
    out["overall"] = block(rows)
    out["missed"] = [{"product_id": r["product_id"], "kind": r["kind"], "planted_text": r["planted_text"]}
                     for r in rows if r["corrupted_judged"] and not r["detected"]]
    return out


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--mock", action="store_true")
    p.add_argument("--report-only", action="store_true")
    args = p.parse_args()
    setup_logging()
    s = load_settings()
    is_mock = args.mock or s.is_mock
    summaries, records = load_inputs(s, is_mock, args.source)
    texts = {r.review_id: r.review_text for r in records}
    originals = JudgementStore.for_mode(s, is_mock, CURRENT_JUDGE_VERSION).latest_by_summary()
    out_dir = s.paths.mode_dir(is_mock) / "d3" / "judge_negative_control_v2" / args.source
    pairs_path = out_dir / "pairs.jsonl"
    rows = read_jsonl(pairs_path) if pairs_path.exists() else []
    done = {(r["product_id"], r["kind"]) for r in rows}
    skipped = [x.product_id for x in summaries if x.summary_id not in originals]
    eligible = [x for x in summaries if x.summary_id in originals][: args.limit]
    judge_model = rows[0]["judge_model"] if rows else "?"
    if not args.report_only:
        client = get_judge_client(s, use_mock=is_mock, version=CURRENT_JUDGE_VERSION)
        judge_model = client.model_name
        guard = GpuGuard(s.paths.root / "logs" / "gpu_guard.csv")
        print(f"Judge {client.provider}/{client.model_name} {CURRENT_JUDGE_VERSION}; {len(eligible)} products with a judged original; {len(skipped)} skipped; {len(done)} pairs stored", flush=True)
        try:
            for idx, summ in enumerate(eligible):
                for planted in build_planted(summ, texts, idx):
                    if (summ.product_id, planted.kind) in done:
                        continue
                    guard.wait_until_safe()
                    try:
                        j, res = judge_summary(client, planted.summary, records, s, version=CURRENT_JUDGE_VERSION, raise_provider_errors=True)
                    except LLMError as exc:
                        sys.exit(f"STOPPED (provider error, nothing recorded for this pair): {exc}\nRe-run to resume.")
                    orig = originals[summ.summary_id]
                    row = {"product_id": summ.product_id, "kind": planted.kind, "planted_text": planted.planted_text, "keywords": list(planted.keywords),
                           "judge_model": client.model_name, "original_judgement_id": orig.judgement_id, "original_scores": scores(orig),
                           "original_flagged_planted_keywords": planted_claim_detected(orig, planted.keywords), "corrupted_judged": j is not None}
                    if j is None:
                        row["failure"] = failure_record(planted.summary, client, CURRENT_JUDGE_VERSION, res, "negative_control")
                    else:
                        row |= {"corrupted_scores": scores(j), "detected": planted_claim_detected(j, planted.keywords),
                                "corrupted_unsupported_claims": j.verdict.unsupported_claims, "corrupted_judgement": j.model_dump(mode="json"),
                                "corrupted_summary": planted.summary.content.model_dump(mode="json")}
                    append_jsonl(pairs_path, [row])
                    rows.append(row)
                    print(f"  {summ.product_id} {planted.kind}: " + (f"detected={row['detected']} F {row['original_scores']['faithfulness']}->{row['corrupted_scores']['faithfulness']} "
                          f"H {row['original_scores']['hallucination']}->{row['corrupted_scores']['hallucination']}" if j else "judge FAILED"), flush=True)
        except GpuTooHot as exc:
            print(f"STOPPED: {exc}. Stored pairs are kept; re-run to resume.")
    summary = summarise(rows, skipped, judge_model, {j.judgement_id: j for j in originals.values()})
    out_dir.mkdir(parents=True, exist_ok=True)
    prev = read_json(out_dir / "summary.json", default=None) if (out_dir / "summary.json").exists() else None
    if prev and prev.get("detection_rule") != "claim_text_only" and not (out_dir / "summary_previous_rule_claims_and_reasons.json").exists():
        shutil.copy2(out_dir / "summary.json", out_dir / "summary_previous_rule_claims_and_reasons.json")  # keep the earlier result
    write_json_atomic(out_dir / "summary.json", summary)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
