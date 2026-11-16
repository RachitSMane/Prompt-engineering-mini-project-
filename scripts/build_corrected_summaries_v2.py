"""Correction set 2 (``-corr2``): separately labelled corrections of the factual errors found in the D3 investigation
(reports/proposed_revisions_2026-11-15/D3_INVESTIGATION_AND_COMPLETION_PLAN.md, sections 8-9). No LLM.

* The original summaries (``outputs/<mode>/summaries/summaries.jsonl``) and correction set 1 (``-corr1``) are never changed.
* The list of edits was fixed from the investigation findings BEFORE judge v4 was run on any corrected copy; no edit was
  added or changed to improve a score. Only the errors named in the investigation are corrected; other weaknesses
  (for example generic recommendations) are left as they are and are reported by the evaluation.
* Every edit records the old text, the new text, the reason, and the evidence: verbatim review quotes (checked here to
  be exact substrings of the review text) or recomputed statistics.
* B0BTY63SFY and B001DHKBXU build on their ``-corr1`` copy (which already fixed the misattributed percentages and
  denominators); the other products build on the original.
* Known limitation that cannot be fixed in a summary: B001DHKBXU's complaint statistics are inflated by extraction
  errors (about 2 of 7 installation_difficulty and 1 of 6 poor_value labels match the review text); fixing them needs
  re-extraction, which would change D2-derived artefacts, so the statistics are left unchanged and the limitation is
  recorded.

Provenance: deterministic text correction by the AI coding assistant (Claude Code), 2026-11-15; not summariser output;
not written or approved by a human.

  python scripts/build_corrected_summaries_v2.py --source dataset__few_shot__qwen2.5-3b__v1
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import load_settings  # noqa: E402
from evaluation.claim_checks import quote_in_texts  # noqa: E402
from evaluation.judge_v4 import load_review_index  # noqa: E402
from pipeline.summarization import SummaryStore  # noqa: E402
from schemas.evaluation_schema import ProductSummary, SummaryPoint  # noqa: E402
from utils.storage import read_json, read_jsonl, write_json_atomic  # noqa: E402

PROVENANCE = ("deterministic text correction by the AI coding assistant (Claude Code), 2026-11-15, justified by quoted review text "
              "or recomputed statistics; not summariser output; not human-written or human-approved")

# Edit kinds: ("opinion", old, new) | ("point", field, old_point, new_point, new_ids) | ("remove", field, old_text)
#             | ("replace_item", field, old_text, new_text)
CORRECTIONS: dict[str, dict] = {
    "B0BTY63SFY": {"base": "corr1", "edits": [
        {"edit": ("point", "appreciated", "The containers are suitable for holding larger eggs and fit well in the fridge.",
                  "The containers hold small to large chicken eggs and fit in the fridge.", ["R1c492cdbe101", "R24b8b37aae15"]),
         "reason": "the only citation (R80f511ff60c4) says larger eggs do NOT fit ('larger eggs the let will not close properly')",
         "evidence": {"R1c492cdbe101": "they are better for storing small and large eggs", "R24b8b37aae15": "also to fit in the fridge"}},
        {"edit": ("point", "complaints", "The containers do not fit properly for larger eggs and are too tall for the container.",
                  "Larger eggs such as duck eggs are too tall for the container, and with larger eggs the lid may not close properly.",
                  ["R90688f008c98", "R80f511ff60c4"]),
         "reason": "garbled wording ('containers ... too tall for the container'); the eggs are too tall",
         "evidence": {"R90688f008c98": "they’re too tall for the container", "R80f511ff60c4": "larger eggs the let will not close properly"}},
        {"edit": ("point", "complaints", "The containers have small legs that do not fit properly.",
                  "The small feet on the bottom have to line up just right, or the stack can slide around.", ["Rbd2a0049f12f"]),
         "reason": "'small legs' misreads 'Only small legs fit' (R80f511ff60c4, a typo for eggs); the cited review talks about little feet",
         "evidence": {"Rbd2a0049f12f": "The little feet on the bottom have to line up just right or the stack can slide around"}},
        {"edit": ("opinion", " Customers also had concerns about the installation setup and usability.", ""),
         "reason": "sentence removed: 'installation setup' is an extraction artefact (lid snapping labelled as installation) and the "
                   "usability concern is 1 negative vs 7 positive mentions; the lid and stacking problems are already stated, with "
                   "citations, in the complaint points",
         "evidence": {"Rbd2a0049f12f": "you have to apply decent pressure to snap it on all the way around", "Rd1d0ae81cb05": "The lids do not seal at all"}},
        {"edit": ("replace_item", "improvement_areas", "Optimizing the installation setup for better fit and stability.",
                  "Making the lid easier to snap on and seal."),
         "reason": "same extraction artefact", "evidence": {"Rbd2a0049f12f": "you have to apply decent pressure to snap it on all the way around"}},
        {"edit": ("remove", "improvement_areas", "Enhancing the design and material quality for durability."),
         "reason": "no review complains about durability or material (quality_durability 3 positive / 0 negative)",
         "evidence": {"Rbd2a0049f12f": "The plastic is fairly thin but sturdy enough"}},
        {"edit": ("replace_item", "improvement_areas", "Ensuring the containers fit properly with the included legs.",
                  "Making the feet line up more easily so that stacked containers do not slide."),
         "reason": "'included legs' does not exist (misreading)", "evidence": {"Rbd2a0049f12f": "The little feet on the bottom have to line up just right"}},
        {"edit": ("replace_item", "recommendations", "Ensure that the included legs are compatible with the container design.",
                  "Redesign the feet so that stacked containers lock in place."),
         "reason": "'included legs' does not exist (misreading)", "evidence": {"Rbd2a0049f12f": "I hoped they would stack better, even snap together"}},
        {"edit": ("remove", "recommendations", "Invest in better material quality and design for increased durability."),
         "reason": "no review complains about durability or material (quality_durability 3 positive / 0 negative)",
         "evidence": {"Rbd2a0049f12f": "The plastic is fairly thin but sturdy enough"}},
        {"edit": ("remove", "recommendations", "Ensure that the included legs fit properly with the container design."),
         "reason": "duplicate of the legs recommendation, based on the same misreading", "evidence": {}},
    ]},
    "B001DHKBXU": {"base": "corr1", "edits": [
        {"edit": ("opinion", "received a mixed to negative overall sentiment", "received a predominantly negative overall sentiment"),
         "reason": "0 of 19 reviews are mixed and 15 of 19 (78.9%) are negative", "evidence": {"statistics": "negative 15/19 = 78.9%, mixed 0/19"}},
        {"edit": ("point", "complaints", "Installation was difficult and the part did not work as intended.",
                  "Some reviewers had to modify or adjust the mounting to install it, and one installed part did not work at all.",
                  ["R88206a7f14bb", "R2079a0152739", "R037714d5ebd7"]),
         "reason": "cited R326a034e9581 says 'easy to install' (it supports the opposite appreciated point)",
         "evidence": {"R88206a7f14bb": "i have to modified mounting holds", "R2079a0152739": "Had to adjust screws at lid to make it work",
                      "R037714d5ebd7": "Installed part, but did not work"}},
        {"edit": ("point", "complaints", "The product did not provide good value for money.",
                  "One reviewer felt it was not worth the money because it quit working within months.", ["Re98c674dfc8e"]),
         "reason": "cited R28878e067e18 contains no value complaint about this part (it compares Sears' price)",
         "evidence": {"Re98c674dfc8e": "because it quit working. Buyer beware you get what you pay for"}},
        {"edit": ("point", "complaints", "The product had issues with functionality and customer service.",
                  "Parts failed or arrived not working, and one reviewer advised against buying from the seller.", ["R55ca2e9dc390", "R037714d5ebd7"]),
         "reason": "the cited reviews criticise the seller, not customer service",
         "evidence": {"R55ca2e9dc390": "I would recommend not buying from SupplyEdge", "R037714d5ebd7": "no switch function, no continuity"}},
        {"edit": ("point", "complaints", "The product did not perform as expected and had compatibility issues.",
                  "Several parts stopped working within weeks or months, and one did not fit right without adjustment.",
                  ["R28878e067e18", "R326a034e9581", "R2079a0152739"]),
         "reason": "neither cited review reports incompatibility (an uncited review even says 'exact match')",
         "evidence": {"R28878e067e18": "for less than two weeks", "R326a034e9581": "worked perfectly for about 6 months and then it simply stopped working",
                      "R2079a0152739": "it would not fit right"}},
    ]},
    "B00DM8K1NS": {"base": "original", "edits": [
        {"edit": ("opinion", "has a mixed overall sentiment with a neutral to slightly positive tone, indicated by the average star rating of 3.38 and 3 positive reviews out of 13.",
                  "has a mixed overall sentiment with a slightly negative tone: the average star rating is 3.38, with 5 positive and 6 negative reviews out of 13."),
         "reason": "5 (not 3) of 13 reviews are positive and negative (6) outnumbers positive (5)",
         "evidence": {"statistics": "positive 5/13 = 38.5%, negative 6/13 = 46.2%, average rating 3.38"}},
    ]},
    "B00LHRAGHW": {"base": "original", "edits": [
        {"edit": ("opinion", "The product received mixed feedback with a slight majority of positive reviews, indicating that the product is generally well-received, though with notable issues in durability and compatibility.",
                  "The product received mixed feedback, with slightly more negative (5 of 10) than positive (4 of 10) reviews, and notable issues in durability and compatibility."),
         "reason": "positive is 40% and negative 50%: the original claims the wrong direction",
         "evidence": {"statistics": "positive 4/10 = 40.0%, negative 5/10 = 50.0%"}},
    ]},
    "B01FZJXITU": {"base": "original", "edits": [
        {"edit": ("opinion", "as indicated by the majority of reviews (7 out of 9) praising its performance and price value.",
                  "with 7 of the 9 reviews positive and praise mainly for its performance and price value."),
         "reason": "7 is the number of positive reviews; positive mentions of performance (4) and price value (2) are fewer",
         "evidence": {"statistics": "positive reviews 7/9; performance 4 positive, price_value 2 positive"}},
    ]},
    "B07XG8QQL5": {"base": "original", "edits": [
        {"edit": ("opinion", "The dryer has received mixed reviews with a majority of users experiencing performance issues and poor durability, leading to frequent complaints and negative sentiment.",
                  "The dryer has received mostly negative reviews (13 of 19), with performance issues and poor durability among the most common complaints (6 of 19 reviews each)."),
         "reason": "performance_issue and poor_durability are each mentioned in 6 of 19 reviews (31.6%), not by a majority",
         "evidence": {"statistics": "negative 13/19 = 68.4%; performance_issue 6/19, poor_durability 6/19"}},
        {"edit": ("point", "complaints", "Performance issues are the most common complaint, with users experiencing heating coil failure and sensor malfunctions.",
                  "Performance issues are among the most common complaints; one reviewer received a dead sensor.", ["R00181bfd96a7"]),
         "reason": "two of the three citations (R86c40cc45487, R30ffca1cf480) are 5-star positive reviews, and R00181bfd96a7 says the heating element works",
         "evidence": {"R00181bfd96a7": "One of the sensors was dead in arrival DOA"}},
    ]},
    "B0C5G1WGBN": {"base": "original", "edits": [
        {"edit": ("point", "appreciated", "It offers good value for money, as seen in reviews like [R4cc079d33dae].",
                  "It offers good value for money, as seen in reviews like [R1a5040c839f6] and [R83348325447b].", ["R1a5040c839f6", "R83348325447b"]),
         "reason": "cited R4cc079d33dae is a 1-star complaint that says nothing about value",
         "evidence": {"R1a5040c839f6": "Yep great price and item", "R83348325447b": "Great dryer for half the price of a full-size"}},
    ]},
}


def apply_edits(summary: ProductSummary, edits: list[dict], reviews: dict[str, dict]) -> tuple[ProductSummary, list[dict]]:
    c = summary.content.model_dump(mode="json")
    log = []
    for e in edits:
        kind = e["edit"][0]
        for rid, quote in e["evidence"].items():
            if rid == "statistics":
                continue
            r = reviews.get(rid)
            if not r or r["product_id"] != summary.product_id or not quote_in_texts(quote, [r["text"], r["title"]], min_words=2):
                raise SystemExit(f"{summary.product_id}: evidence quote for {rid} is not in that review: {quote!r}")
        if kind == "opinion":
            _, old, new = e["edit"]
            if c["overall_opinion"].count(old) != 1:
                raise SystemExit(f"{summary.product_id}: expected one occurrence of {old!r} in the overall opinion")
            c["overall_opinion"] = c["overall_opinion"].replace(old, new).strip()
            log.append({"field": "overall_opinion", "action": "replace_text" if new else "remove_sentence", "old": old.strip(), "new": new})
        elif kind == "point":
            _, field, old, new, ids = e["edit"]
            idx = [i for i, p in enumerate(c[field]) if p["point"] == old]
            if len(idx) != 1:
                raise SystemExit(f"{summary.product_id}: expected one {field} point {old!r}")
            old_ids = c[field][idx[0]]["supporting_review_ids"]
            for rid in ids:
                if reviews.get(rid, {}).get("product_id") != summary.product_id or rid not in summary.aggregate.review_ids:
                    raise SystemExit(f"{summary.product_id}: new citation {rid} is not a review of this product's aggregate")
            c[field][idx[0]] = SummaryPoint(point=new, supporting_review_ids=ids).model_dump()
            log.append({"field": field, "action": "replace_point", "old": old, "old_ids": old_ids, "new": new, "new_ids": ids})
        elif kind in ("remove", "replace_item"):
            field, old = e["edit"][1], e["edit"][2]
            if c[field].count(old) != 1:
                raise SystemExit(f"{summary.product_id}: expected one {field} item {old!r}")
            if kind == "remove":
                c[field].remove(old)
                log.append({"field": field, "action": "remove_item", "old": old})
            else:
                c[field][c[field].index(old)] = e["edit"][3]
                log.append({"field": field, "action": "replace_item", "old": old, "new": e["edit"][3]})
        log[-1].update({"reason": e["reason"], "evidence": e["evidence"]})
    from schemas.evaluation_schema import ProductSummaryContent

    content = ProductSummaryContent.model_validate(c)
    return summary.model_copy(update={"summary_id": f"{summary.summary_id.removesuffix('-corr1')}-corr2", "content": content}), log


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    args = p.parse_args()
    s = load_settings()
    mode = s.paths.mode_dir(s.is_mock)
    originals = {pid: x for (pid, src), x in SummaryStore.for_mode(s, s.is_mock).latest().items() if src == args.source}
    corr1 = {r["product_id"]: ProductSummary.model_validate(r["corrected_summary"])
             for r in read_jsonl(mode / "d3" / "corrected_summaries" / args.source / "corrected_summaries.jsonl")}
    reviews = load_review_index(s.paths.reviews_csv)
    out = []
    for pid, spec in CORRECTIONS.items():
        base = corr1[pid] if spec["base"] == "corr1" else originals[pid]
        fixed, log = apply_edits(base, spec["edits"], reviews)
        out.append({"product_id": pid, "original_summary_id": originals[pid].summary_id, "base_summary_id": base.summary_id,
                    "corrected_summary_id": fixed.summary_id, "correction_set": "corr2", "provenance": PROVENANCE, "edits": log,
                    "corrected_summary": fixed.model_dump(mode="json")})
    d = mode / "d3" / "corrected_summaries_v2" / args.source
    d.mkdir(parents=True, exist_ok=True)
    (d / "corrected_summaries.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in out), encoding="utf-8")
    write_json_atomic(d / "README.json", {
        "correction_set": "corr2", "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "provenance": PROVENANCE,
        "originals_unchanged": True, "corr1_unchanged": True, "products": [x["product_id"] for x in out],
        "scope": "only the factual errors named in the 2026-11-15 investigation (sections 8-9); fixed before judge v4 evaluated any corrected copy",
        "judge_scores": "corrected copies are evaluated by judge v4 under their own summary ids (evaluated_object = corrected); no score is "
                        "transferred to or from an original summary",
        "unfixable_in_summary": {"B001DHKBXU": "complaint statistics inflated by extraction errors (needs re-extraction)"}})
    # dated note on correction set 1 (its 'never judged' statement became outdated when judge v3 evaluated the copies)
    r1 = mode / "d3" / "corrected_summaries" / args.source / "README.json"
    meta = read_json(r1, default={})
    notes = meta.get("notes", [])
    note = ("2026-11-15: the statement 'The corrected copies have never been judged' was true when written (2026-11-14) but is outdated: "
            "judge v3 evaluated both -corr1 copies (outputs/live/judgements/v3, evaluated_object = corrected) and judge v4 evaluates them "
            "under outputs/live/d3/judge_v4. Correction set 2 (-corr2) is in ../corrected_summaries_v2/.")
    if note not in notes:
        meta["notes"] = notes + [note]
        write_json_atomic(r1, meta)
    for x in out:
        print(f"{x['product_id']}: {x['corrected_summary_id']} ({len(x['edits'])} edit(s), base {x['base_summary_id']})")


if __name__ == "__main__":
    main()
