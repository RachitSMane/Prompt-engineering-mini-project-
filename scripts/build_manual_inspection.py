"""Manual-inspection sheet: every summary claim next to the full text of the reviews it cites (no LLM).

  python scripts/build_manual_inspection.py --source dataset__few_shot__qwen2.5-3b__v1

Writes ``outputs/<mode>/d3/manual_inspection/<source>/``:
  claims_for_review.csv  one row per summary point: claim, cited ids, cited review texts + extracted labels, an automatic
                         sentiment-consistency hint, the v1/v2 judge status, and EMPTY columns for the human verdict
                         (human_verdict: supported / partly / unsupported, human_notes, inspector).
  inspection.md          the same material, readable.
Fill the CSV by hand, then run with ``--score <filled.csv>`` to report human verdict counts and agreement with the
judge's unsupported-claim list. Nothing is ever filled in automatically.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import load_settings  # noqa: E402
from evaluation.summary_evaluation import JudgementStore, claim_in_text  # noqa: E402
from scripts.run_judge import load_inputs  # noqa: E402

HUMAN_COLS = ["human_verdict", "human_notes", "inspector"]


def build(s, is_mock: bool, source: str) -> pd.DataFrame:
    summaries, records = load_inputs(s, is_mock, source)
    by_id = {r.review_id: r for r in records}
    v1 = JudgementStore.for_mode(s, is_mock, "v1").latest_by_summary()
    v2 = JudgementStore.for_mode(s, is_mock, "v2").latest_by_summary()
    rows = []
    for summ in summaries:
        for kind, points in (("appreciated", summ.content.appreciated), ("complaint", summ.content.complaints)):
            for pt in points:
                cited = [by_id[r] for r in pt.supporting_review_ids if r in by_id]
                want = "positive" if kind == "appreciated" else "negative"
                agree = sum(r.overall_sentiment.value in (want, "mixed") or any(a.sentiment.value == want for a in r.aspects) for r in cited)
                flagged = {ver: any(claim_in_text(c, pt.point, 0.5) or claim_in_text(pt.point, c, 0.5) for c in store[summ.summary_id].verdict.unsupported_claims)
                           if summ.summary_id in store else None for ver, store in (("v1", v1), ("v2", v2))}
                rows.append({
                    "product_id": summ.product_id, "product_title": summ.product_title or "", "summary_id": summ.summary_id,
                    "point_type": kind, "claim": pt.point, "cited_review_ids": "; ".join(pt.supporting_review_ids),
                    "cited_ids_known": f"{len(cited)}/{len(pt.supporting_review_ids)}",
                    "auto_sentiment_consistent": f"{agree}/{len(cited)}",
                    "cited_reviews": "\n---\n".join(f"[{r.review_id}] ({r.overall_sentiment.value}; "
                                                   + ", ".join(f"{a.aspect_category.value}:{a.sentiment.value}" for a in r.aspects) + f") {r.review_text}" for r in cited),
                    "v1_judge_flagged": flagged["v1"], "v2_judge_flagged": flagged["v2"],
                    **{c: "" for c in HUMAN_COLS},
                })
    return pd.DataFrame(rows)


def score(path: Path) -> dict:
    df = pd.read_csv(path, keep_default_na=False)
    done = df[df["human_verdict"].str.strip() != ""]
    out = {"points": len(df), "inspected": len(done), "verdicts": done["human_verdict"].str.lower().value_counts().to_dict()}
    for ver in ("v1", "v2"):
        sub = done[done[f"{ver}_judge_flagged"].astype(str).isin(["True", "False"])]
        if len(sub):
            human_unsupported = sub["human_verdict"].str.lower().eq("unsupported")
            judge_flag = sub[f"{ver}_judge_flagged"].astype(str).eq("True")
            out[f"{ver}_agreement"] = round(float((human_unsupported == judge_flag).mean()), 3)
            out[f"{ver}_n"] = len(sub)
    return out


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--mock", action="store_true")
    p.add_argument("--score", default=None, help="a filled claims_for_review.csv to score")
    args = p.parse_args()
    if args.score:
        print(json.dumps(score(Path(args.score)), indent=1))
        return
    s = load_settings()
    is_mock = args.mock or s.is_mock
    df = build(s, is_mock, args.source)
    out = s.paths.mode_dir(is_mock) / "d3" / "manual_inspection" / args.source
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "claims_for_review.csv", index=False, encoding="utf-8")
    lines = [f"# Manual inspection sheet · `{args.source}`", "", "Fill `human_verdict` (supported / partly / unsupported) in the CSV. "
             "Automatic columns are hints only; nothing here is a human judgement.", ""]
    for pid, g in df.groupby("product_id", sort=False):
        lines += [f"## {g.iloc[0]['product_title'][:80]} (`{pid}`)", ""]
        for r in g.itertuples():
            lines += [f"- **{r.point_type}:** {r.claim}  ", f"  cited {r.cited_review_ids} · sentiment-consistent {r.auto_sentiment_consistent} · "
                      f"judge flagged v1={r.v1_judge_flagged} v2={r.v2_judge_flagged}", ""]
    (out / "inspection.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"{len(df)} claims from {df['product_id'].nunique()} summaries -> {out}")


if __name__ == "__main__":
    main()
