"""Judge v3 run + report (see evaluation/judge_v3.py). Started through ``run_judge.py --version v3`` so that the GPU
watchdog (which watches run_judge.py) guards it.

  python scripts/run_judge.py --version v3 --source dataset__few_shot__qwen2.5-3b__v1                      # originals, corrected, NC
  python scripts/run_judge.py --version v3 --source ... --objects original corrected                       # subset
  python scripts/run_judge.py --version v3 --source ... --report-only                                      # analysis only, no LLM

Outputs (append-only, resumable, nothing earlier is touched):
  outputs/<mode>/judgements/v3/judgements.jsonl   valid verdicts (evaluated_object = original | corrected | nc_corrupted)
  outputs/<mode>/judgements/v3/failures.jsonl     validation failures with the raw response
  outputs/<mode>/judgements/v3/run_manifest.json  model, prompt hash, configuration, dates
  outputs/<mode>/d3/judge_v3_report/<source>/report.json   analysis (no LLM)

The negative-control copies are the SAME 30 corrupted summaries as the v2 control: their content is taken from the
v2 ``pairs.jsonl`` (read-only) and checked to be identical to what ``build_planted`` produces.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import load_settings  # noqa: E402
from evaluation.judge_v3 import (  # noqa: E402
    V3Store, get_judge_v3_client, judge_items, run_config, summarise_v3,
)
from evaluation.negative_control import _norm, build_planted  # noqa: E402
from llm.base import LLMError, LLMUnavailableError  # noqa: E402
from schemas.evaluation_schema import ProductSummary  # noqa: E402
from scripts.judge_negative_control import wilson_ci  # noqa: E402
from scripts.run_judge import load_inputs  # noqa: E402
from utils.gpu_guard import GpuGuard, GpuTooHot  # noqa: E402
from utils.storage import read_json, read_jsonl, write_json_atomic  # noqa: E402

KIND_SUFFIX = {"invented_feature": "nc-feature", "fabricated_statistic": "nc-statistic"}


def corrected_summaries(mode_dir: Path, source: str) -> list[ProductSummary]:
    p = mode_dir / "d3" / "corrected_summaries" / source / "corrected_summaries.jsonl"
    return [ProductSummary.model_validate(r["corrected_summary"]) for r in read_jsonl(p)]


def nc_items(mode_dir: Path, source: str, summaries: list[ProductSummary], texts: dict[str, str]) -> list[tuple]:
    """The v2 negative-control corrupted copies, rebuilt and verified against the stored v2 pairs (read-only)."""
    pairs = read_jsonl(mode_dir / "d3" / "judge_negative_control_v2" / source / "pairs.jsonl")
    by_key = {(r["product_id"], r["kind"]): r for r in pairs}
    items = []
    for idx, summ in enumerate(summaries):
        for planted in build_planted(summ, texts, idx):
            stored = by_key.get((summ.product_id, planted.kind))
            if stored is None:
                continue
            rebuilt = planted.summary.content.model_dump(mode="json")
            if stored.get("corrupted_summary") and stored["corrupted_summary"] != rebuilt:
                raise SystemExit(f"corrupted copy for {summ.product_id}/{planted.kind} differs from the stored v2 pair; refusing to run")
            if stored["planted_text"] != planted.planted_text:
                raise SystemExit(f"planted text for {summ.product_id}/{planted.kind} differs from the v2 pair; refusing to run")
            items.append(("nc_corrupted", planted.summary,
                          {"nc_kind": planted.kind, "planted_text": planted.planted_text, "keywords": list(planted.keywords),
                           "original_summary_id": summ.summary_id}))
    return items


def claim_detects(rec: dict, keywords) -> bool:
    blob = _norm(" ".join(c["claim"] for c in rec["unsupported_claims"]))
    return any(_norm(k) in blob for k in keywords)


def reason_only(rec: dict, keywords) -> bool:
    reasons = _norm(" ".join(c.get("reason", "") for c in rec["unsupported_claims"]) + " " +
                    " ".join(rec.get("justifications", {}).values()))
    return not claim_detects(rec, keywords) and any(_norm(k) in reasons for k in keywords)


def nc_report(judgements: list[dict], failures: list[dict]) -> dict:
    orig = {j["summary_id"]: j for j in judgements if j["evaluated_object"] == "original"}
    rows = []
    for j in (x for x in judgements if x["evaluated_object"] == "nc_corrupted"):
        o = orig.get(j["original_summary_id"])
        rows.append({
            "product_id": j["product_id"], "kind": j["nc_kind"], "planted_text": j["planted_text"],
            "detected": claim_detects(j, j["keywords"]),
            "keyword_only_in_reason_or_justification": reason_only(j, j["keywords"]),
            "original_judged": o is not None,
            "original_names_keyword": claim_detects(o, j["keywords"]) if o else None,
            "corrupted_scores": j["scores"], "original_scores": o["scores"] if o else None,
        })
    nc_fail = [f for f in failures if f["evaluated_object"] == "nc_corrupted"]

    def block(rs):
        n = len(rs)
        det = sum(r["detected"] for r in rs)
        paired = [r for r in rs if r["original_scores"]]
        def cmp(d, op):  # pairs where either score was excluded (inconsistent verdict) are not compared
            return sum(op(r["corrupted_scores"][d], r["original_scores"][d]) for r in paired
                       if r["corrupted_scores"][d] is not None and r["original_scores"][d] is not None)
        return {"corrupted_judged": n, "detected": det, "missed": n - det,
                "detection_rate": round(det / n, 3) if n else None, "detection_wilson95": wilson_ci(det, n),
                "keyword_only_in_reason_or_justification": sum(r["keyword_only_in_reason_or_justification"] for r in rs),
                "pairs_with_v3_original": len(paired),
                "pairs_comparable_on_hallucination": sum(1 for r in paired if r["corrupted_scores"]["hallucination"] is not None
                                                         and r["original_scores"]["hallucination"] is not None),
                "faithfulness_lower": cmp("faithfulness", lambda a, b: a < b),
                "faithfulness_higher": cmp("faithfulness", lambda a, b: a > b),
                "hallucination_lower": cmp("hallucination", lambda a, b: a < b),
                "hallucination_higher": cmp("hallucination", lambda a, b: a > b),
                "mean_lower": cmp("mean", lambda a, b: a < b), "mean_higher": cmp("mean", lambda a, b: a > b)}

    return {
        "detection_rule": "claim_text_only (same keyword rule as the v2 control)",
        "validation_failures": len(nc_fail),
        "by_kind": {k: block([r for r in rows if r["kind"] == k]) for k in sorted({r["kind"] for r in rows})},
        "overall": block(rows),
        "keyword_control_on_v3_originals": {"checked": sum(r["original_judged"] for r in rows),
                                            "named": sum(bool(r["original_names_keyword"]) for r in rows)},
        "missed": [{"product_id": r["product_id"], "kind": r["kind"]} for r in rows if not r["detected"]],
        "rows": rows,
    }


def original_vs_corrected(judgements: list[dict]) -> list[dict]:
    orig = {j["product_id"]: j for j in judgements if j["evaluated_object"] == "original"}
    out = []
    for c in (j for j in judgements if j["evaluated_object"] == "corrected"):
        o = orig.get(c["product_id"])
        out.append({"product_id": c["product_id"], "corrected_summary_id": c["summary_id"],
                    "original_summary_id": o["summary_id"] if o else None,
                    "original_scores_v3": o["scores"] if o else None, "corrected_scores_v3": c["scores"],
                    "original_claims": [x["claim"] for x in o["unsupported_claims"]] if o else None,
                    "corrected_claims": [x["claim"] for x in c["unsupported_claims"]]})
    return out


def report(mode_dir: Path, source: str, store: V3Store) -> dict:
    js, fs = store.judgements(), store.failures()
    rep = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": source,
        "note": "Judge v3 is a separately labelled run. v1/v2 results are unchanged. Scores of corrected copies belong to the "
                "corrected summary ids only; no score is transferred between summaries. Scores are model opinions.",
        "originals": summarise_v3(js, fs, "original"),
        "corrected": summarise_v3(js, fs, "corrected"),
        "nc_corrupted": summarise_v3(js, fs, "nc_corrupted"),
        "original_vs_corrected": original_vs_corrected(js),
        "negative_control_v3": nc_report(js, fs),
    }
    out = mode_dir / "d3" / "judge_v3_report" / source
    out.mkdir(parents=True, exist_ok=True)
    write_json_atomic(out / "report.json", rep)
    return rep


def main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description="Judge v3 (single call, raw response kept, no repair)")
    p.add_argument("--version", default="v3")
    p.add_argument("--source", required=True)
    p.add_argument("--objects", nargs="*", default=["original", "corrected", "nc_corrupted"],
                   choices=["original", "corrected", "nc_corrupted"])
    p.add_argument("--limit", type=int, default=None, help="at most N new judge calls (pilot)")
    p.add_argument("--report-only", action="store_true")
    p.add_argument("--mock", action="store_true", help="OFFLINE mock judge (pipeline check, not a result)")
    args = p.parse_args(argv)

    s = load_settings()
    is_mock = args.mock or s.is_mock
    mode_dir = s.paths.mode_dir(is_mock)
    summaries, records = load_inputs(s, is_mock, args.source)
    store = V3Store(mode_dir / "judgements" / "v3")
    if not args.report_only:
        texts = {r.review_id: r.review_text for r in records}
        items = []
        if "original" in args.objects:
            items += [("original", x, None) for x in summaries]
        if "corrected" in args.objects:
            items += [("corrected", x, {"corrects_summary_id": x.summary_id.removesuffix("-corr1")})
                      for x in corrected_summaries(mode_dir, args.source)]
        if "nc_corrupted" in args.objects:
            items += nc_items(mode_dir, args.source, summaries, texts)
        done = store.done()
        todo = [it for it in items if (it[0], it[1].summary_id) not in done]
        n_stored = len(items) - len(todo)
        if args.limit is not None:
            todo = todo[: args.limit]
        try:
            client = get_judge_v3_client(s, use_mock=is_mock)
        except LLMUnavailableError as exc:
            sys.exit(f"Judge unavailable: {exc}")
        cfg = run_config(client, s)
        store.folder.mkdir(parents=True, exist_ok=True)
        manifest_path = store.folder / "run_manifest.json"
        manifest = read_json(manifest_path, default=None) or {"created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                                               "source": args.source, "config": cfg, "sessions": []}
        if manifest["config"]["prompt_sha256"] != cfg["prompt_sha256"] or manifest["config"]["judge_model"] != cfg["judge_model"]:
            sys.exit("v3 store was produced with a different prompt or model; use a new version label instead of mixing runs")
        run_id = datetime.now(timezone.utc).strftime("v3-%Y%m%dT%H%M%SZ")
        manifest["sessions"].append({"run_id": run_id, "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                     "planned_new_calls": len(todo)})
        write_json_atomic(manifest_path, manifest)
        guard = GpuGuard(s.paths.root / "logs" / "gpu_guard.csv")
        print(f"Judge v3 {cfg['judge_provider']}/{cfg['judge_model']} constrained={cfg['constrained_decoding']} · "
              f"{len(items)} items, {n_stored} already stored, {len(todo)} to judge in this session", flush=True)
        if is_mock:
            print("*** MOCK judge: placeholder verdicts, NOT an evaluation result ***")

        def progress(kind, summ, ok, rec):
            if ok:
                sc = rec["scores"]
                print(f"  {kind:12s} {summ.summary_id}: F={sc['faithfulness']} C={sc['coverage']} K={sc['consistency']} "
                      f"H={sc['hallucination']} (raw {rec['hallucination_score_raw']}) claims={len(rec['unsupported_claims'])} "
                      f"{rec['latency_s']:.0f}s", flush=True)
            else:
                print(f"  {kind:12s} {summ.summary_id}: VALIDATION FAILURE {rec['validation_error'][:200]}", flush=True)

        try:
            stats = judge_items(client, todo, records, store, config=cfg, run_id=run_id,
                                before_call=lambda _s: guard.wait_until_safe(), progress=progress)
        except (LLMError, GpuTooHot) as exc:
            sys.exit(f"STOPPED: {exc}\nEverything judged so far is stored. Re-run the same command to resume.")
        manifest["sessions"][-1].update({"finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **stats,
                                         "max_gpu_temp_c": guard.max_seen})
        write_json_atomic(manifest_path, manifest)
        print(json.dumps(stats))
    rep = report(mode_dir, args.source, store)
    print(json.dumps({k: rep[k] for k in ("originals", "corrected")}, indent=1))
    print(json.dumps({k: v for k, v in rep["negative_control_v3"].items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    main()
