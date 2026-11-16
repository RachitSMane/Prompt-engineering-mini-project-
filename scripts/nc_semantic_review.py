"""Semantic review of the v2 negative-control pairs (no LLM; reads the stored pairs.jsonl only).

The automatic metric (``planted_claim_detected``) is a keyword match in the judge's claim texts. A keyword match shows
that the judge *named* the planted claim, not that it understood the evidence, and keywords can miss paraphrases.
This script adds a per-pair reading of every stored judgement:

* pairs with a keyword match are re-checked automatically: the matching claim and its reason are recorded;
* the 12 pairs without a keyword match were read in full (claims, reasons, all four justifications) and classified
  by hand. That reading was done by the AI coding assistant (Claude Code) on 2026-11-14 — it is NOT a human review,
  and the classifications below quote the stored text so anyone can check them.

Categories
  named_in_claims            planted claim listed as unsupported, keyword match (automatic)
  named_in_claims_paraphrase planted claim listed as unsupported in other words (manual)
  justification_only         claim list empty / without it, but a justification calls the planted content unsupported
                             (internally contradictory verdict) (manual)
  accepted_as_fact           the judge used the planted claim as if it were true (manual)
  not_mentioned              the planted content appears nowhere in the judgement (manual)

Output: outputs/<mode>/d3/judge_negative_control_v2/<source>/semantic_review.json (pairs.jsonl is never modified).

  python scripts/nc_semantic_review.py --source dataset__few_shot__qwen2.5-3b__v1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import load_settings  # noqa: E402
from evaluation.negative_control import _norm, planted_claim_detected, planted_keyword_in_reason_only  # noqa: E402
from schemas.evaluation_schema import SummaryJudgement  # noqa: E402
from utils.storage import read_jsonl, write_json_atomic  # noqa: E402

REVIEWER = "Claude Code (AI coding assistant), manual reading of stored judge output, 2026-11-14 — not a human review"
ASSESSMENT_TYPE = "model-assisted assessment (Claude); NOT independent human ground truth; reported separately from the keyword detection rate"

# (product_id, kind) -> (category, quote from the stored judgement that supports the category)
MANUAL: dict[tuple[str, str], tuple[str, str]] = {
    ("B07XG8QQL5", "fabricated_statistic"): ("named_in_claims_paraphrase",
        'claim: "Users would buy this product again." reason: "Not supported by the reviews, which mention performance issues and poor durability."'),
    ("B001ONHE8S", "invented_feature"): ("justification_only",
        'unsupported_claims empty, hallucination 5; hallucination justification: "The summary introduces unsupported claims, such as the product\'s solar power feature"'),
    ("B01FZJXITU", "fabricated_statistic"): ("justification_only",
        'unsupported_claims empty, hallucination 5; hallucination justification: "The summary introduces the idea that users would buy the product again, which is not supported by the reviews."'),
    ("B0081N3YW4", "fabricated_statistic"): ("accepted_as_fact",
        'reason: "The actual statistic is 91% of reviewers saying they would buy the product again, which is a positive sentiment."'),
    ("B001DHKBXU", "invented_feature"): ("not_mentioned", "claims: value for money, ease of use; no mention of Wi-Fi/voice"),
    ("B001TH7GZU", "fabricated_statistic"): ("not_mentioned", "claims: value for money, fits the stove; no mention of 88% or buying again"),
    ("B0081N3YW4", "invented_feature"): ("not_mentioned", "claims: mixed reception, great idea, good cup of coffee; no mention of touchscreen/recipes"),
    ("B008YDTFYW", "invented_feature"): ("not_mentioned", "claims: easy to customise, not as clean; no mention of a lifetime guarantee"),
    ("B00LHRAGHW", "invented_feature"): ("not_mentioned", "claim: delivered quickly; no mention of solar power"),
    ("B0759KR8JX", "fabricated_statistic"): ("not_mentioned", "claim: ensure compatibility before purchasing; no mention of 97% or buying again"),
    ("B0BTY63SFY", "fabricated_statistic"): ("not_mentioned", "claim: ensure containers fit the eggs; no mention of 91% or buying again"),
    ("B0C5G1WGBN", "fabricated_statistic"): ("not_mentioned", "claim: significant performance/ease-of-use/durability issues; no mention of 86% or buying again"),
}
NOTICED = ("named_in_claims", "named_in_claims_paraphrase", "justification_only")


def review(rows: list[dict], manual: dict | None = None) -> dict:
    manual = MANUAL if manual is None else manual
    keys = [(r["product_id"], r["kind"]) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError(f"duplicate (product, kind) pairs in pairs.jsonl: {sorted({k for k in keys if keys.count(k) > 1})}")
    items = []
    used_manual = set()
    for r in rows:
        key = (r["product_id"], r["kind"])
        j = SummaryJudgement.model_validate(r["corrupted_judgement"])
        kw = tuple(r["keywords"])
        if planted_claim_detected(j, kw):
            if key in manual:
                raise ValueError(f"{key}: has a keyword match but also a manual entry")
            d = next(d for d in j.verdict.unsupported_claim_details if any(_norm(k) in _norm(d.claim) for k in kw))
            cat, quote = "named_in_claims", f'claim: "{d.claim}" reason: "{d.reason}"'
        else:
            if key not in manual:
                raise ValueError(f"{key}: no keyword match and no manual classification — read it and add one")
            cat, quote = manual[key]
            used_manual.add(key)
        items.append({"product_id": r["product_id"], "kind": r["kind"], "planted_text": r["planted_text"], "category": cat,
                      "keyword_detected": cat == "named_in_claims", "n_unsupported_claims": len(j.verdict.unsupported_claims),
                      "hallucination_score": j.verdict.hallucination.score,
                      "planted_keyword_in_reason_only": planted_keyword_in_reason_only(j, kw), "evidence": quote})
    unused = [k for k in manual if k not in used_manual]  # a manual entry that matches no keyword-missed pair is stale
    if unused:
        raise ValueError(f"manual classifications not used (pair absent from pairs.jsonl): {unused}")
    if len(items) != len(rows):
        raise AssertionError("every stored pair must appear exactly once in the review")
    count = lambda pred: sum(pred(i) for i in items)
    by_kind = {}
    for kind in sorted({i["kind"] for i in items}):
        ks = [i for i in items if i["kind"] == kind]
        by_kind[kind] = {c: sum(i["category"] == c for i in ks) for c in ("named_in_claims", "named_in_claims_paraphrase", "justification_only", "accepted_as_fact", "not_mentioned")}
    return {
        "reviewer": REVIEWER,
        "assessment_type": ASSESSMENT_TYPE,
        "n": len(items),
        "pairs_in_file": len(rows),
        "keyword_detected_in_claims": count(lambda i: i["category"] == "named_in_claims"),
        "named_in_claims_incl_paraphrase": count(lambda i: i["category"] in ("named_in_claims", "named_in_claims_paraphrase")),
        "noticed_anywhere": count(lambda i: i["category"] in NOTICED),
        "accepted_as_fact": count(lambda i: i["category"] == "accepted_as_fact"),
        "not_mentioned": count(lambda i: i["category"] == "not_mentioned"),
        "planted_keyword_in_reason_only": count(lambda i: i["planted_keyword_in_reason_only"]),
        "by_kind": by_kind,
        "caveat": "A planted claim named as unsupported shows the judge flagged it, not that it verified the evidence: the reasons are generic "
                  "('not supported by the reviews'), and some verdicts list many claims including true ones (e.g. 7 claims for B001DHKBXU).",
        "items": items,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--mock", action="store_true")
    args = p.parse_args()
    s = load_settings()
    d = s.paths.mode_dir(args.mock or s.is_mock) / "d3" / "judge_negative_control_v2" / args.source
    out = review(read_jsonl(d / "pairs.jsonl"))
    write_json_atomic(d / "semantic_review.json", out)
    print(json.dumps({k: v for k, v in out.items() if k != "items"}, indent=1))


if __name__ == "__main__":
    main()
