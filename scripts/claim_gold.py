"""Single-annotator human claim validation for judge v4 (no LLM). See evaluation/claim_gold.py.

Labelling happens on the Streamlit Summary Evaluation page ("Human claim validation"). This CLI only exports, checks and
compares; it never creates a label.

  python scripts/claim_gold.py export        --source dataset__few_shot__qwen2.5-3b__v1   # frozen 239-claim template (refuses to overwrite)
  python scripts/claim_gold.py status        --source ...                                  # progress + integrity check
  python scripts/claim_gold.py export-labels --source ...                                  # completed labels -> CSV
  python scripts/claim_gold.py compare       --source ...                                  # judge v4 vs the annotator (needs all 239)

Files: template outputs/<mode>/d3/claim_gold/<source>/claim_gold_template.csv (unchanged since 2026-11-16);
labels data/annotations/d3_claim_validation.json; outputs outputs/<mode>/d3/claim_gold/<source>/single_annotator/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import load_settings  # noqa: E402
from evaluation.claim_checks import Stats  # noqa: E402
from evaluation.claim_gold import (  # noqa: E402
    LABEL_GUIDE, LABELS, PROPOSED_THRESHOLDS, compare, export_csv_bytes, export_rows, load_template, now, sha256_file,
    template_rows, validate_store, write_template,
)
from evaluation.d3_status import load_evaluations, paths  # noqa: E402
from evaluation.judge_v4 import load_review_index  # noqa: E402
from scripts.run_judge import load_inputs  # noqa: E402
from utils.readonly import ensure_writable  # noqa: E402
from utils.storage import read_json, write_json_atomic  # noqa: E402


def judge_map(evals_path: Path) -> dict[str, dict]:
    originals = [e for e in load_evaluations(evals_path) if e.evaluated_object == "original"]
    return {ce.claim.claim_id: {"status": ce.status.value, "method": ce.method, "explanation": ce.explanation}
            for e in originals for ce in e.claim_evaluations}


def ensure_preregistration(p: dict, store_exists: bool) -> dict:
    """Written once, before labelling when possible; never rewritten."""
    path = p["single_dir"] / "preregistration_single_annotator.json"
    data = read_json(path, default=None)
    if data:
        return data
    data = {"created_at": now(), "methodology": "single_human_annotator",
            "created_before_any_label": not store_exists,
            "decision": "Project owner decision 2026-11-16: one human annotator labels all 239 claims; no second annotator, "
                        "no inter-annotator agreement, no adjudication. Supersedes plan §14.4 (two annotators, kappa).",
            "labels": list(LABELS), "scope": "all 239 claims of the 15 original summaries (claim_gold_template.csv)",
            "comparison": "judge v4 final status vs the human label: exact agreement, per-class precision/recall/F1, confusion "
                          "matrix, flagged (unsupported+contradicted) precision/recall/specificity, disagreements",
            "informational_thresholds": PROPOSED_THRESHOLDS,
            "completion_rule": "D3 human validation is complete when all 239 claims carry exactly one valid label from the one "
                               "annotator and the comparison has been generated from that exact label file; meeting the "
                               "informational thresholds is NOT required and the result is reported as is."}
    ensure_writable(path)
    write_json_atomic(path, data)
    return data


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="Single-annotator human claim validation (judge v4)")
    ap.add_argument("cmd", choices=["export", "status", "export-labels", "compare"])
    ap.add_argument("--source", required=True)
    ap.add_argument("--force", action="store_true", help="export: replace the frozen template (invalidates existing labels)")
    a = ap.parse_args(argv)
    s = load_settings()
    mode = s.paths.mode_dir(False)
    p = paths(mode, a.source)

    if a.cmd == "export":
        originals = [e for e in load_evaluations(p["evaluations"]) if e.evaluated_object == "original"]
        summaries, records = load_inputs(s, False, a.source)
        stats = {x.product_id: Stats.from_aggregate(x.aggregate, records) for x in summaries}
        rows = template_rows(originals, load_review_index(s.paths.reviews_csv), stats)
        try:
            write_template(p["template"], rows, force=a.force)
        except FileExistsError as e:
            print(f"REFUSED: {e}")
            return 1
        print(f"wrote {len(rows)} claims -> {p['template']} (no judge output included)")
        return 0

    if not p["template"].exists():
        print("no claim template; run `export` first")
        return 2
    template = load_template(p["template"])
    tsha = sha256_file(p["template"])
    judge = judge_map(p["evaluations"])
    data = read_json(p["store"], default=None)
    v = validate_store(data, template, tsha, set(judge))
    print(json.dumps({k: v[k] for k in ("valid", "complete", "labelled", "total", "annotator")} | {"problems": v["problems"][:10]}
                     | ({"note": v["note"]} if v.get("note") else {}), indent=1))
    if a.cmd == "status":
        return 0 if v["valid"] else 1
    p["single_dir"].mkdir(parents=True, exist_ok=True)
    if a.cmd == "export-labels":
        rows = export_rows(data, template)
        ensure_writable(p["labels_csv"])
        p["labels_csv"].write_bytes(export_csv_bytes(rows))
        print(f"wrote {len(rows)} labels -> {p['labels_csv']}")
        return 0
    prereg = ensure_preregistration(p, p["store"].exists())
    res = compare(data or {}, template, judge, store_sha256=sha256_file(p["store"]) if p["store"].exists() else "",
                  template_sha256=tsha, thresholds=prereg.get("informational_thresholds"))
    ensure_writable(p["comparison"])
    write_json_atomic(p["comparison"], res)
    print(f"comparison status: {res['status']} -> {p['comparison']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
