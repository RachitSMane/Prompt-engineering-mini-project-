"""Human annotation workflow: status, migration, export, reserve reviews, and D2 evaluation against human labels.

  python scripts/human_annotation.py status                       # annotation completion (pending / completed / flagged)
  python scripts/human_annotation.py migrate                      # copy the documented HUMAN label(s) from annotations.json (idempotent)
  python scripts/human_annotation.py export                       # final human-verified reference -> data/annotations/human_reference/
  python scripts/human_annotation.py reserve --n 1                # add replacement reviews for flagged ones
  python scripts/human_annotation.py evaluate --run 20261111-030912_qwen2.5-3b [--include-unverified]
  python scripts/human_annotation.py finalise --run 20261111-030912_qwen2.5-3b    # export + evaluate, ONLY once 300 verified labels exist
  python scripts/human_annotation.py progress-report [--out reports/d2/annotation_progress.md]

``evaluate`` makes no LLM calls: it scores the run's saved predictions against verified human labels and writes
``<run>/human_reference/{metrics.json,report.md}``. Labelling itself happens on the Human Annotation page.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from annotation.annotation_manager import load_sample  # noqa: E402
from annotation.human_annotation import HumanAnnotationStore, migrate_legacy_human_labels  # noqa: E402
from config import load_settings  # noqa: E402
from evaluation.human_evaluation import evaluate_against_human, render_human_report  # noqa: E402
from evaluation.prompt_comparison import experiments_dir  # noqa: E402
from utils.data_loader import load_reviews  # noqa: E402


def progress_markdown(store: HumanAnnotationStore, queue: list[str]) -> str:
    """Annotation completion, computed from the human store only (AI-assisted labels never count).

    Completion = every queued review has a human label or a flag (pending == 0). Second-person verification is not
    part of the project's required workflow; completed labels are human-created but not independently verified."""
    from datetime import datetime, timezone

    prog = store.progress(queue)
    lines = [f"# D2 human annotation progress ({datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')})", "",
             "Computed from `data/annotations/human_annotations.json` (`scripts/human_annotation.py progress-report`). "
             "The AI-assisted labels in `annotations.json` are not counted and never will be.", "",
             "| | Count |", "|---|---|",
             f"| Queue (stratified sample + reserves) | {prog['queue']} |",
             f"| Human-created annotations | {prog['human_created']} |",
             f"| **Completed annotations** | **{prog['labelled']} / {prog['queue']}** |",
             f"| Pending annotations | {prog['pending']} |",
             f"| Flagged annotations | {prog['flagged']} |",
             f"| Annotators | {', '.join(prog['annotators']) or '–'} |", ""]
    pending = [rid for rid in queue if store.status_of(rid) == "pending"]
    if pending:
        lines += ["Reviews without a human label: " + ", ".join(f"`{r}`" for r in pending[:50]) + (" …" if len(pending) > 50 else ""), ""]
    status = (f"complete: all {prog['labelled']} independent human annotations are done" if prog["annotation_complete"]
              else f"in progress: {prog['pending']} review(s) still need a human label")
    lines += [f"**Annotation status:** {status}.", "",
              "Second-person verification is not part of this project's required workflow. The labels are human-created "
              "(blind) but have **not** been independently verified by a second annotator, and they are kept separate from "
              "the AI-assisted reference annotations.", ""]
    return "\n".join(lines)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    sub.add_parser("migrate")
    sub.add_parser("export")
    r = sub.add_parser("reserve")
    r.add_argument("--n", type=int, default=1)
    e = sub.add_parser("evaluate")
    e.add_argument("--run", required=True)
    e.add_argument("--include-unverified", action="store_true", help="provisional analysis only: also use completed, unverified labels")
    f = sub.add_parser("finalise", help="export + evaluate once the required number of verified human labels exists; otherwise change nothing")
    f.add_argument("--run", required=True)
    pr = sub.add_parser("progress-report", help="write a Markdown progress report (read-only on the store)")
    pr.add_argument("--out", default=None)
    args = p.parse_args()

    s = load_settings()
    store = HumanAnnotationStore.default(s)
    sample_ids = load_sample(s.paths.annotation_sample_csv)["review_id"].tolist()
    queue = store.queue(sample_ids)

    if args.cmd == "migrate":
        moved = migrate_legacy_human_labels(store, s.paths.annotations_json)
        print(f"migrated {len(moved)} documented human label(s): {moved}")
    elif args.cmd == "export":
        paths = store.export_reference(s.paths.annotations / "human_reference")
        print(json.dumps({k: str(v) for k, v in paths.items()}, indent=1))
    elif args.cmd == "reserve":
        flagged = sum(1 for rid in queue if store.status_of(rid) == "flagged")
        picked = store.add_reserve(load_reviews(s.paths.reviews_csv), set(queue), args.n, s.random_seed)
        print(f"{flagged} flagged in queue; added reserve review(s): {picked}")
    elif args.cmd == "progress-report":
        text = progress_markdown(store, queue)
        print(text)
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(text, encoding="utf-8")
        return
    elif args.cmd == "finalise":
        prog = store.progress(queue)
        if prog["eligible"] < prog["required"]:
            print(f"NOT READY: {prog['eligible']} / {prog['required']} verified human labels "
                  f"({prog['remaining_to_required']} still needed). Nothing was exported or evaluated.")
            sys.exit(2)
        paths = store.export_reference(s.paths.annotations / "human_reference")
        print(json.dumps({k: str(v) for k, v in paths.items()}, indent=1))
        run_dir = experiments_dir(s, args.run.startswith("MOCK-")) / args.run
        res = evaluate_against_human(run_dir, store.gold(queue), label_provenance={"verified_human_created": prog["eligible_blind"],
                                     "verified_human_verified": prog["eligible"] - prog["eligible_blind"]})
        print(render_human_report(res))
        print(f"written: {run_dir / 'human_reference'}")
        sys.exit(0 if res["independent_check_passed"] else 1)
    elif args.cmd == "evaluate":
        run_dir = experiments_dir(s, args.run.startswith("MOCK-")) / args.run
        gold = store.gold(queue, include_unverified=args.include_unverified)
        prog = store.progress(queue)
        res = evaluate_against_human(run_dir, gold, label_provenance={"verified_human_created": prog["eligible_blind"],
                                     "verified_human_verified": prog["eligible"] - prog["eligible_blind"]},
                                     unverified_included=args.include_unverified)
        print(render_human_report(res))
        print(f"written: {run_dir / 'human_reference'}")
        sys.exit(0 if res["independent_check_passed"] else 1)
    print(json.dumps(store.progress(queue), indent=1))


if __name__ == "__main__":
    main()
