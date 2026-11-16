"""Read-only integrity check of the D2 evaluation records (standard library only, no project code, no LLM).

Checks, per strategy and across strategies:
  * every prediction row has a review id; duplicate rows per review id are counted (retries are legitimate only if the
    final row per id is unique in outcome);
  * the set of evaluated review ids is IDENTICAL for zero-shot, few-shot and CoT;
  * it equals the evaluable reference set (annotations with status "complete"), with no missing or extra ids;
  * invalid outputs (no "ok" row for a review) are counted, never dropped;
  * reference provenance: how many evaluable labels are human vs AI-assisted (from meta.annotation_method).

Usage:  python scripts/verify_d2_records.py --run 20261111-030912_qwen2.5-3b [--out reports/d2/record_integrity.md]
Exit code 1 if any check fails.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STRATEGIES = ("zero_shot", "few_shot", "cot")


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def review_id(row: dict) -> str | None:
    rec = row.get("record") or {}
    return rec.get("review_id") or row.get("review_id") or (row.get("cache_key") or "").split(":")[0] or None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    run = ROOT / "outputs" / "live" / "experiments" / args.run
    ann = json.loads((ROOT / "data" / "annotations" / "annotations.json").read_text(encoding="utf-8"))
    method = ann["meta"].get("annotation_method", {})
    human_ids = set(method.get("human_labels", []))
    records = list(ann["annotations"].values()) if isinstance(ann["annotations"], dict) else ann["annotations"]
    complete = {a["review_id"] for a in records if a.get("status", "complete") == "complete"}
    flagged = sorted(a["review_id"] for a in records if a.get("status", "complete") != "complete")

    checks: list[tuple[str, bool, str]] = []
    per = {}
    for s in STRATEGIES:
        rows = load_rows(run / f"predictions_{s}.jsonl")
        ids = [review_id(r) for r in rows]
        no_id = sum(1 for i in ids if not i)
        counts = Counter(i for i in ids if i)
        ok = {review_id(r) for r in rows if r.get("status") == "ok"}
        evaluated = set(counts)
        dup_ok = sum(1 for i, c in Counter(review_id(r) for r in rows if r.get("status") == "ok").items() if c > 1)
        per[s] = {"rows": len(rows), "unique_ids": len(evaluated), "rows_without_id": no_id,
                  "ids_with_several_rows": sum(1 for c in counts.values() if c > 1), "ids_with_several_ok_rows": dup_ok,
                  "valid": len(ok), "invalid": len(evaluated - ok), "ids": evaluated}
        checks.append((f"{s}: every row has a review id", no_id == 0, f"{no_id} rows without id"))
        checks.append((f"{s}: no review has two successful predictions", dup_ok == 0, f"{dup_ok} ids"))
        checks.append((f"{s}: evaluated ids == evaluable reference ids", evaluated == complete,
                       f"missing {len(complete - evaluated)}, extra {len(evaluated - complete)}"))
    same = per["zero_shot"]["ids"] == per["few_shot"]["ids"] == per["cot"]["ids"]
    checks.append(("all three strategies use the identical review set", same, f"{len(per['zero_shot']['ids'])} ids"))
    prov_h = len(complete & human_ids)
    checks.append(("reference provenance recorded (human vs AI-assisted)", bool(method), f"{prov_h} human, {len(complete) - prov_h} AI-assisted"))

    failed = [c for c in checks if not c[1]]
    lines = [f"# D2 record integrity — run `{args.run}`", "",
             "Read-only, standard library only, no project code and no LLM (`scripts/verify_d2_records.py`).", "",
             f"Evaluable reference labels: **{len(complete)}** ({prov_h} human, {len(complete) - prov_h} AI-assisted); "
             f"excluded (flagged): {', '.join(flagged) or 'none'}.", "",
             "| Strategy | Rows | Unique review ids | Ids with several rows (retries) | Valid | Invalid (scored as wrong) |",
             "|---|---|---|---|---|---|"]
    for s in STRATEGIES:
        d = per[s]
        lines.append(f"| {s} | {d['rows']} | {d['unique_ids']} | {d['ids_with_several_rows']} | {d['valid']} | {d['invalid']} |")
    lines += ["", "| Check | Result | Detail |", "|---|---|---|"]
    lines += [f"| {n} | {'PASS' if ok else 'FAIL'} | {d} |" for n, ok, d in checks]
    lines += ["", f"**{'ALL CHECKS PASS' if not failed else f'{len(failed)} CHECK(S) FAILED'}.** The reference labels are "
              "mostly AI-assisted, so these metrics measure agreement with AI-assisted labels, not with human ground truth."]
    text = "\n".join(lines) + "\n"
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
