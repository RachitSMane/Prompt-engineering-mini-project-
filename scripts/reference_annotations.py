"""Approved reference annotations for D2 (see annotation/reference_adjudication.py).

  python scripts/reference_annotations.py validate                         # 300 ids, fields, sample match (read-only)
  python scripts/reference_annotations.py status                           # approval progress (read-only)
  python scripts/reference_annotations.py approve --reviewer "Name" --confirm    # bulk approval (idempotent)
  python scripts/reference_annotations.py decide R5cfb12ef75b4 approved --reviewer "Name"
  python scripts/reference_annotations.py decide R5cfb12ef75b4 excluded --reviewer "Name" --reason "unreadable"
  python scripts/reference_annotations.py export                           # -> data/annotations/approved_reference/
  python scripts/reference_annotations.py evaluate --run 20261111-030912_qwen2.5-3b    # no LLM
  python scripts/reference_annotations.py predict-missing --run 20261111-030912_qwen2.5-3b --i-understand-this-calls-the-llm

``approve`` and ``decide`` record the name you give as the reviewer: run them yourself, as the person who reviewed the
labels. Approval never changes the original provenance (AI-assisted stays AI-assisted).
``predict-missing`` runs the three strategies (local Ollama, same model as the run) ONLY for sample reference records that
have no prediction yet, into ``<run>/supplementary/`` — the original prediction files are not modified.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from annotation.reference_adjudication import AdjudicationError, ReferenceAdjudicationStore, load_reference  # noqa: E402
from config import load_settings  # noqa: E402
from evaluation.prompt_comparison import experiments_dir  # noqa: E402
from evaluation.reference_evaluation import evaluate_reference, render_reference_report  # noqa: E402


def run_evaluation(s, store: ReferenceAdjudicationStore, run_id: str, write: bool = True) -> dict:
    prog = store.progress()
    run_dir = experiments_dir(s, run_id.startswith("MOCK-")) / run_id
    return evaluate_reference(run_dir, store.accepted_gold(), progress=prog, provenance=prog["accepted_by_original_provenance"], write=write)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("validate")
    sub.add_parser("status")
    sub.add_parser("export")
    a = sub.add_parser("approve")
    a.add_argument("--reviewer", required=True)
    a.add_argument("--confirm", action="store_true", help="I have reviewed these reference labels and accept them as the D2 reference")
    d = sub.add_parser("decide")
    d.add_argument("review_id")
    d.add_argument("decision", choices=["approved", "excluded"])
    d.add_argument("--reviewer", required=True)
    d.add_argument("--reason", default="")
    e = sub.add_parser("evaluate")
    e.add_argument("--run", required=True)
    pm = sub.add_parser("predict-missing")
    pm.add_argument("--run", required=True)
    pm.add_argument("--i-understand-this-calls-the-llm", action="store_true")
    args = p.parse_args()

    s = load_settings()
    store = ReferenceAdjudicationStore.default(s)
    try:
        if args.cmd == "validate":
            ref = load_reference(s.paths.annotations_json, s.paths.annotation_sample_csv)
            prov = {k: sum(1 for v in ref["provenance"].values() if v == k) for k in ("human_created", "ai_assisted")}
            print(json.dumps({"valid": ref["valid"], "records": len(ref["records"]), "unique_ids": len(set(ref["records"])),
                              "sample_size": len(ref["sample_ids"]), "missing": ref["missing"], "invalid": ref["invalid"],
                              "original_provenance": prov, "flagged_in_source": [r for r, x in ref["records"].items() if x.get("status") != "complete"],
                              "problems": ref["problems"]}, indent=1))
            sys.exit(0 if ref["valid"] else 1)
        if args.cmd == "export":
            from annotation.reference_adjudication import export_reference

            res = export_reference(store, s.paths.annotations / "approved_reference")
            print(json.dumps({k: str(v) for k, v in res.items()}, indent=1))
            return
        if args.cmd == "approve":
            print(json.dumps(store.bulk_approve(args.reviewer, confirmed=args.confirm), indent=1))
        elif args.cmd == "decide":
            print(json.dumps(store.decide(args.review_id, args.decision, args.reviewer, reason=args.reason), indent=1))
        elif args.cmd == "evaluate":
            res = run_evaluation(s, store, args.run)
            print(render_reference_report(res))
            sys.exit(0 if res["status"] == "complete" else 2)
        elif args.cmd == "predict-missing":
            if not args.i_understand_this_calls_the_llm:
                sys.exit("predict-missing calls the local LLM; add --i-understand-this-calls-the-llm")
            predict_missing(s, store, args.run)
            return
    except AdjudicationError as exc:
        sys.exit(f"REFUSED: {exc}")
    print(json.dumps(store.progress(), indent=1))


def predict_missing(s, store: ReferenceAdjudicationStore, run_id: str) -> None:
    from evaluation.prompt_comparison import load_run_predictions
    from llm.factory import get_llm_client
    from pipeline.extraction import ExtractionStore, ReviewExtractor
    from prompts.strategies import get_strategy
    from schemas.review_schema import ReviewInput
    from utils.data_loader import load_reviews
    from utils.gpu_guard import GpuGuard
    from utils.storage import read_json

    run_dir = experiments_dir(s, False) / run_id
    cfg = read_json(run_dir / "config.json")
    # Every valid reference record of the sample (predictions do not depend on approval decisions).
    gold_ids = list(load_reference(store.annotations_json, store.sample_csv)["records"])
    todo = {}
    for name in cfg["strategies"]:
        recs, fails = load_run_predictions(run_dir, name)
        have = {o.review_id for o in recs + fails}
        sup = run_dir / "supplementary" / f"predictions_{name}.jsonl"
        if sup.exists():
            r2, f2 = load_run_predictions(run_dir / "supplementary", name)
            have |= {o.review_id for o in r2 + f2}
        todo[name] = [rid for rid in gold_ids if rid not in have]
    if not any(todo.values()):
        print("nothing to predict: every accepted reference has a prediction from every strategy")
        return
    client = get_llm_client(s, role="extraction")
    if client.model_name != cfg["model_name"]:
        sys.exit(f"run {run_id} used {cfg['model_name']}; configured model is {client.model_name}; refusing to mix models")
    df = load_reviews(s.paths.reviews_csv).set_index("review_id")
    guard = GpuGuard(s.paths.root / "logs" / "gpu_guard.csv")
    for name, ids in todo.items():
        if not ids:
            continue
        reviews = [ReviewInput(review_id=rid, product_id=df.loc[rid, "product_id"], review_text=df.loc[rid, "review_text"]) for rid in ids]
        store_s = ExtractionStore(run_dir / "supplementary" / f"predictions_{name}.jsonl")
        guard.wait_until_safe()
        ReviewExtractor(client, get_strategy(name), s).extract_many(reviews, store=store_s, retry_failed=False)
        print(f"{name}: predicted {ids}")


if __name__ == "__main__":
    main()
