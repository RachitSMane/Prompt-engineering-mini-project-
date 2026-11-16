"""Deterministic D3 audits (no LLM): summary evidence traceability and judge-verdict quality.

Evidence audit (per summary)
  * every cited review id exists in the source extractions and belongs to the summarised product;
  * the stored aggregate equals a fresh re-aggregation of the source records (counts, percentages, complaints, aspects);
  * every percentage written in the summary matches a computed statistic (±0.5);
  * every appreciated/complaint point is checked against its cited reviews: sentiment direction (appreciated ->
    a positive aspect or positive/mixed review; complaint -> a negative aspect or negative/mixed review) and content-word
    overlap with the cited review texts / evidence quotes. Points failing both are reported as ``weak`` — explicitly, not
    silently accepted — and listed for manual inspection;
  * ``frequent_aspects`` must name aspect categories that occur in the statistics.

Judge-quality audit (per judgement)
  * integer scores 1-5, non-empty justifications, claims at top level, hallucination score consistent with claims;
  * evidence review ids cited by the judge exist for that product;
  * each flagged "unsupported" claim is checked against the SOURCE DATA: if it names a complaint category / aspect that the
    product's statistics or evidence actually contain, it is marked ``source_supported`` — a likely false positive of the
    judge (e.g. calling "poor value" unsupported when ``poor_value`` is a top complaint).
"""

from __future__ import annotations

import re

from evaluation.summary_evaluation import _PCT_RE, claim_in_text, summary_text
from pipeline.aggregation import aggregate_product
from schemas.evaluation_schema import ProductSummary, SummaryJudgement

# "8 out of 15 reviews", "4 of 19 reviews", "3 of 15 negative reviews"
_COUNT_RE = re.compile(r"\b(\d+)\s+(?:out\s+of|of)\s+(\d+)\s+(?:\w+\s+)?reviews?\b", re.IGNORECASE)
from schemas.review_schema import ReviewRecord

_WORD = re.compile(r"[a-z0-9]+")
_STOP = set("the a an and or of to is are was were it its in on for with that this as be by at their they them has have had "
            "not no but very so too some many most more customers customer reviewers reviewer review reviews product products "
            "item one also overall which who from about can could would will".split())

# Words that identify complaint categories / aspect categories in free text (used to match judge claims to source data).
CATEGORY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "poor_value": ("value", "price", "money", "expensive", "cost", "cheap", "worth"),
    "price_value": ("value", "price", "money", "expensive", "cost", "cheap", "worth", "affordable"),
    "poor_durability": ("durab", "broke", "break", "last", "sturdy", "flimsy", "wear", "fail", "crack"),
    "quality_durability": ("durab", "quality", "sturdy", "flimsy", "last", "broke", "build"),
    "noise_issue": ("noise", "noisy", "loud", "quiet"),
    "noise": ("noise", "noisy", "loud", "quiet"),
    "performance_issue": ("perform", "work", "effective", "function"),
    "performance": ("perform", "work", "effective", "function"),
    "compatibility_fit_issue": ("fit", "compatib", "size"),
    "compatibility": ("fit", "compatib"),
    "installation_difficulty": ("install", "setup", "set up", "assembl"),
    "installation_setup": ("install", "setup", "set up", "assembl"),
    "usability_issue": ("use", "easy", "difficult", "clean"),
    "ease_of_use": ("use", "easy", "difficult", "clean"),
    "product_defect": ("defect", "broken", "damaged", "faulty", "leak"),
    "shipping_damage": ("shipping", "arrived", "damaged", "package"),
    "missing_parts": ("missing",),
    "not_as_described": ("described", "description", "advertised", "picture"),
    "customer_service_issue": ("service", "support", "refund", "return", "warranty"),
    "customer_service": ("service", "support", "refund", "return", "warranty"),
    "safety_concern": ("safety", "fire", "smoke", "burn", "hazard"),
    "design_build": ("design", "look", "material", "build"),
    "size_capacity": ("size", "capacity", "small", "large", "big"),
    "shipping_packaging": ("shipping", "packag", "deliver"),
}


def _content(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP and len(w) > 2}


def _overlap(point: str, texts: list[str]) -> float:
    words = _content(point)
    if not words:
        return 0.0
    pool = set().union(*(_content(t) for t in texts)) if texts else set()
    return round(len(words & pool) / len(words), 2)


def _matches(text: str, category: str) -> bool:
    low = text.lower()
    return any(k in low for k in CATEGORY_KEYWORDS.get(category, (category.replace("_", " "),)))


_NEG_WORDS = ("not ", "n't", "no ", "poor", "fail", "issue", "problem", "broke", "break", "less ", "worse", "difficult", "hard to",
              "complain", "negative", "defect", "leak", "loud", "noisy", "flimsy", "cheaply", "disappoint", "malfunction")


def claim_polarity(text: str) -> str:
    """Crude polarity of a claim: 'negative' if it contains a negation/complaint cue, else 'positive'."""
    low = f" {text.lower()} "
    return "negative" if any(w in low for w in _NEG_WORDS) else "positive"


def _names_valid(category: str, names: list[str], records: list[ReviewRecord]) -> bool:
    """Each stored top name occurs in the category and is not out-ranked by more names than allowed (tie-aware)."""
    from collections import Counter

    counts = Counter(a.aspect_name for r in records for a in r.aspects if a.aspect_category.value == category)
    if not counts:
        return not names
    ranked = sorted(counts.values(), reverse=True)
    cutoff = ranked[min(len(ranked), max(1, len(names))) - 1]
    return all(n in counts and counts[n] >= cutoff for n in names)


def audit_summary(summary: ProductSummary, records: list[ReviewRecord]) -> dict:
    by_id = {r.review_id: r for r in records}
    product_records = [r for r in records if r.product_id == summary.product_id]
    agg = summary.aggregate
    issues: list[str] = []

    # 1. cited ids
    points = [("appreciated", p) for p in summary.content.appreciated] + [("complaint", p) for p in summary.content.complaints]
    cited = [rid for _, p in points for rid in p.supporting_review_ids]
    unknown = sorted({rid for rid in cited if rid not in by_id})
    wrong_product = sorted({rid for rid in cited if rid in by_id and by_id[rid].product_id != summary.product_id})
    if unknown:
        issues.append(f"cited ids not in extractions: {unknown}")
    if wrong_product:
        issues.append(f"cited ids from another product: {wrong_product}")

    # 2. statistics re-computed from the source records
    fresh = aggregate_product(product_records)
    stat_checks = {
        "total_reviews": fresh.total_reviews == agg.total_reviews,
        "sentiment_counts": fresh.sentiment_counts == agg.sentiment_counts,
        "sentiment_percentages": fresh.sentiment_percentages == agg.sentiment_percentages,
        "complaint_counts": fresh.complaint_counts == agg.complaint_counts,
        # numeric aspect statistics must match exactly; top_aspect_names only up to ties (value_counts order among equal
        # counts follows record order, which differs between the cache at summary time and now)
        "aspect_stats": [a.model_dump(exclude={"top_aspect_names"}) for a in fresh.aspect_stats] == [a.model_dump(exclude={"top_aspect_names"}) for a in agg.aspect_stats]
                        and all(_names_valid(cat.aspect_category, cat.top_aspect_names, product_records) for cat in agg.aspect_stats),
        "review_ids": sorted(fresh.review_ids) == sorted(agg.review_ids),
    }
    issues += [f"aggregate mismatch: {k}" for k, ok in stat_checks.items() if not ok]

    # 3. percentages in the text
    text = summary_text(summary)
    known = list(agg.sentiment_percentages.values()) + list(agg.complaint_percentages.values())
    pcts = [float(x) for x in _PCT_RE.findall(text)]
    not_stat = [x for x in pcts if not any(abs(x - k) <= 0.5 for k in known)]
    cited_text = " ".join(by_id[r].review_text for r in cited if r in by_id)
    review_pcts = {float(x) for x in _PCT_RE.findall(cited_text)}
    quoted_pcts = [x for x in not_stat if x in review_pcts]  # e.g. "40% cheaper" copied from a cited review: sourced, not invented
    bad_pcts = [x for x in not_stat if x not in review_pcts]
    if bad_pcts:
        issues.append(f"percentages not in statistics: {bad_pcts}")

    # 3b. "N out of M reviews" count statements (added 2026-11-14; the percentage check alone missed B001DHKBXU, which
    #     reported complaint counts "out of 15 reviews" where 15 is the number of COMPLAINING reviews, not the 19 total)
    all_counts = set(agg.sentiment_counts.values()) | set(agg.complaint_counts.values()) | {a.mentions for a in agg.aspect_stats}         | {agg.reviews_with_complaints, agg.total_reviews}
    count_rows = []
    for m in _COUNT_RE.finditer(text):
        n, d = int(m.group(1)), int(m.group(2))
        if d == agg.total_reviews and n in all_counts:
            status = "ok"
        elif d == agg.reviews_with_complaints and d != agg.total_reviews and n in set(agg.complaint_counts.values()):
            status = "misleading_denominator"
        else:
            status = "not_in_statistics"
        count_rows.append({"text": m.group(0), "count": n, "denominator": d, "status": status})
    bad_counts = [c for c in count_rows if c["status"] != "ok"]
    if bad_counts:
        mis = [c["text"] for c in bad_counts if c["status"] == "misleading_denominator"]
        unk = [c["text"] for c in bad_counts if c["status"] == "not_in_statistics"]
        if mis:
            issues.append(f"{len(mis)} count(s) use the {agg.reviews_with_complaints} complaining reviews as denominator although the product has "
                          f"{agg.total_reviews} reviews (misleading): {mis}")
        if unk:
            issues.append(f"counts not in statistics: {unk}")

    # 4. each point against its cited reviews
    quotes = {q.review_id: [] for q in agg.positive_evidence + agg.negative_evidence}
    for q in agg.positive_evidence + agg.negative_evidence:
        quotes[q.review_id].append(q.quote)
    point_rows = []
    for kind, p in points:
        cited_recs = [by_id[r] for r in p.supporting_review_ids if r in by_id]
        want = "positive" if kind == "appreciated" else "negative"
        direction = [r.overall_sentiment.value in (want, "mixed") or any(a.sentiment.value == want for a in r.aspects) for r in cited_recs]
        texts = [r.review_text for r in cited_recs] + [q for r in cited_recs for q in quotes.get(r.review_id, [])] + \
                [a.aspect_category.value.replace("_", " ") for r in cited_recs for a in r.aspects]
        ov = _overlap(p.point, texts)
        dir_ok = bool(cited_recs) and all(direction)
        status = "traceable" if dir_ok and ov >= 0.3 else ("partial" if dir_ok or ov >= 0.3 else "weak")
        point_rows.append({"type": kind, "point": p.point, "cited": p.supporting_review_ids, "sentiment_direction_ok": f"{sum(direction)}/{len(cited_recs)}",
                           "word_overlap": ov, "status": status})
    weak = [r for r in point_rows if r["status"] == "weak"]
    if weak:
        issues.append(f"{len(weak)} point(s) weakly traceable to their cited reviews")

    # 5. frequent aspects must be real categories in the stats
    stat_cats = {a.aspect_category for a in agg.aspect_stats}
    stat_words = " ".join(stat_cats).replace("_", " ") + " " + " ".join(n for a in agg.aspect_stats for n in a.top_aspect_names).lower()
    fa_bad = [fa for fa in summary.content.frequent_aspects if not (_content(fa) & _content(stat_words))]
    if fa_bad:
        issues.append(f"frequent_aspects not found in statistics: {fa_bad}")

    return {
        "product_id": summary.product_id, "summary_id": summary.summary_id, "records": len(product_records),
        "cited_ids": len(cited), "unknown_ids": unknown, "wrong_product_ids": wrong_product,
        "statistics_match_source": all(stat_checks.values()), "stat_checks": stat_checks,
        "percentages_in_summary": pcts, "percentages_quoted_from_cited_reviews": quoted_pcts, "percentages_not_in_statistics": bad_pcts,
        "points": point_rows, "points_traceable": sum(r["status"] == "traceable" for r in point_rows),
        "points_partial": sum(r["status"] == "partial" for r in point_rows), "points_weak": len(weak),
        "frequent_aspects_not_in_stats": fa_bad,
        "count_statements": count_rows, "count_statements_bad": len(bad_counts),
        "ids_and_statistics_verified": not unknown and not wrong_product and all(stat_checks.values()) and not bad_pcts and not bad_counts,
        "issues": issues,
    }


def audit_judgement(j: SummaryJudgement, summary: ProductSummary) -> dict:
    v = j.verdict
    dims = ("faithfulness", "coverage", "consistency", "hallucination")
    scores = {d: getattr(v, d).score for d in dims}
    agg = summary.aggregate
    complaint_cats = [c for c, n in agg.complaint_counts.items() if n > 0]
    neg_aspects = [a.aspect_category for a in agg.aspect_stats if a.negative > 0]
    pos_aspects = [a.aspect_category for a in agg.aspect_stats if a.positive > 0]
    evidence_text = " ".join(q.quote for q in agg.positive_evidence + agg.negative_evidence)
    claims = []
    for c in v.unsupported_claims:
        polarity = claim_polarity(c)
        pool = (complaint_cats + neg_aspects) if polarity == "negative" else pos_aspects
        supported_by = sorted({cat for cat in pool if _matches(c, cat)})
        in_text = claim_in_text(c, summary_text(summary), 0.5)
        claims.append({
            "claim": c,
            "polarity": polarity,
            "in_summary_text": in_text,
            "source_categories_supporting": supported_by,
            "evidence_overlap": _overlap(c, [evidence_text]),
            # The judge called this unsupported, yet the product's own statistics contain a matching category with the
            # same polarity (e.g. "did not provide good value" while poor_value is a recorded complaint).
            "likely_false_positive": bool(supported_by) and in_text,
        })
    ev_ids = {r for d in v.unsupported_claim_details for r in d.evidence_review_ids}
    return {
        "product_id": j.product_id, "judge_version": j.prompt_version, "judge_model": j.judge_model, "scores": scores,
        "scores_valid": all(isinstance(x, int) and 1 <= x <= 5 for x in scores.values()),
        "justifications_present": all(len(getattr(v, d).justification.strip()) >= 5 for d in dims),
        "hallucination_consistent_with_claims": (scores["hallucination"] < 5) == bool(v.unsupported_claims),
        "judge_evidence_ids_valid": ev_ids <= set(agg.review_ids), "judge_evidence_ids": sorted(ev_ids),
        "claims": claims, "claims_flagged": len(claims),
        "claims_likely_false_positive": sum(c["likely_false_positive"] for c in claims),
    }
