"""Stage 2 — AGGREGATE: group validated records by product and compute statistics.

All numbers are computed here with pandas — never by the LLM. The summariser
receives these numbers and is instructed to copy, not calculate, them.
"""

from __future__ import annotations

from collections import Counter, defaultdict

import pandas as pd

from schemas.evaluation_schema import AspectStat, EvidenceQuote, ProductAggregate
from schemas.review_schema import ReviewRecord
from schemas.taxonomy import OverallSentiment

SENTIMENTS = [s.value for s in OverallSentiment]


def _pct(count: int, total: int) -> float:
    return round(100.0 * count / total, 1) if total else 0.0


def aspect_rows(records: list[ReviewRecord]) -> pd.DataFrame:
    rows = [
        {
            "review_id": r.review_id,
            "product_id": r.product_id,
            "aspect_category": a.aspect_category.value,
            "aspect_name": a.aspect_name,
            "sentiment": a.sentiment.value,
            "evidence": a.evidence,
            "confidence": a.confidence,
            "timestamp": r.timestamp,
        }
        for r in records
        for a in r.aspects
    ]
    return pd.DataFrame(rows, columns=["review_id", "product_id", "aspect_category", "aspect_name", "sentiment", "evidence", "confidence", "timestamp"])


def _select_evidence(df: pd.DataFrame, sentiment: str, max_quotes: int, per_category: int = 2) -> list[EvidenceQuote]:
    """Diverse evidence: up to ``per_category`` quotes from each category, most-mentioned categories first."""
    sub = df[df["sentiment"] == sentiment]
    if sub.empty:
        return []
    order = sub["aspect_category"].value_counts().index.tolist()
    quotes: list[EvidenceQuote] = []
    for cat in order:
        picks = sub[sub["aspect_category"] == cat].sort_values(["confidence", "review_id"], ascending=[False, True])
        picks = picks.drop_duplicates("review_id").head(per_category)
        quotes += [EvidenceQuote(review_id=p.review_id, aspect_category=cat, sentiment=sentiment, quote=p.evidence) for p in picks.itertuples()]
        if len(quotes) >= max_quotes:
            break
    return quotes[:max_quotes]


def aggregate_product(records: list[ReviewRecord], max_evidence: int = 10) -> ProductAggregate:
    if not records:
        raise ValueError("cannot aggregate an empty record list")
    pid = records[0].product_id
    if any(r.product_id != pid for r in records):
        raise ValueError("aggregate_product expects records of a single product")
    total = len(records)
    sent_counts = Counter(r.overall_sentiment.value for r in records)
    sentiment_counts = {s: int(sent_counts.get(s, 0)) for s in SENTIMENTS}

    adf = aspect_rows(records)
    stats: list[AspectStat] = []
    if not adf.empty:
        for cat, grp in adf.groupby("aspect_category"):
            # count each review at most once per (category, sentiment)
            per_review = grp.drop_duplicates(["review_id", "sentiment"])
            c = per_review["sentiment"].value_counts()
            mentions = int(grp["review_id"].nunique())
            pos, neg, neu = int(c.get("positive", 0)), int(c.get("negative", 0)), int(c.get("neutral", 0))
            stats.append(
                AspectStat(
                    aspect_category=str(cat),
                    mentions=mentions,
                    positive=pos,
                    negative=neg,
                    neutral=neu,
                    net_sentiment=round((pos - neg) / max(1, pos + neg + neu), 3),
                    top_aspect_names=grp["aspect_name"].value_counts().head(3).index.tolist(),
                )
            )
    stats.sort(key=lambda s: (-s.mentions, s.aspect_category))

    comp = Counter(c.value for r in records for c in r.complaint_categories)
    complaint_counts = dict(sorted(comp.items(), key=lambda kv: (-kv[1], kv[0])))

    trend: list[dict] = []
    if not adf.empty and adf["timestamp"].notna().any():
        t = adf.copy()
        t["period"] = pd.to_datetime(t["timestamp"], errors="coerce", utc=True).dt.year
        t = t.dropna(subset=["period"])
        if not t.empty:
            t["score"] = t["sentiment"].map({"positive": 1, "negative": -1, "neutral": 0})
            g = t.groupby(["period", "aspect_category"]).agg(mentions=("review_id", "nunique"), net_sentiment=("score", "mean")).reset_index()
            trend = [
                {"period": int(row.period), "aspect_category": row.aspect_category, "mentions": int(row.mentions), "net_sentiment": round(float(row.net_sentiment), 3)}
                for row in g.itertuples()
            ]

    ratings = [r.rating for r in records if r.rating is not None]
    half = max(1, max_evidence // 2)
    return ProductAggregate(
        product_id=pid,
        product_title=records[0].product_title,
        total_reviews=total,
        sentiment_counts=sentiment_counts,
        sentiment_percentages={s: _pct(c, total) for s, c in sentiment_counts.items()},
        average_rating=round(sum(ratings) / len(ratings), 2) if ratings else None,
        aspect_stats=stats,
        complaint_counts=complaint_counts,
        complaint_percentages={k: _pct(v, total) for k, v in complaint_counts.items()},
        reviews_with_complaints=sum(bool(r.complaint_categories) for r in records),
        aspect_trend=trend,
        positive_evidence=_select_evidence(adf, "positive", half) if not adf.empty else [],
        negative_evidence=_select_evidence(adf, "negative", half) if not adf.empty else [],
        review_ids=[r.review_id for r in records],
    )


def group_by_product(records: list[ReviewRecord]) -> dict[str, list[ReviewRecord]]:
    groups: dict[str, list[ReviewRecord]] = defaultdict(list)
    for r in records:
        groups[r.product_id].append(r)
    return dict(groups)


def aggregate_all(records: list[ReviewRecord], max_evidence: int = 10) -> dict[str, ProductAggregate]:
    return {pid: aggregate_product(recs, max_evidence) for pid, recs in sorted(group_by_product(records).items())}


def overview_stats(records: list[ReviewRecord]) -> dict:
    """Corpus-level statistics for the dashboard."""
    total = len(records)
    sent = Counter(r.overall_sentiment.value for r in records)
    adf = aspect_rows(records)
    return {
        "total": total,
        "products": len({r.product_id for r in records}),
        "sentiment_counts": {s: int(sent.get(s, 0)) for s in SENTIMENTS},
        "aspect_counts": adf.groupby(["aspect_category", "sentiment"]).size().unstack(fill_value=0).to_dict(orient="index") if not adf.empty else {},
        "complaint_counts": dict(Counter(c.value for r in records for c in r.complaint_categories).most_common()),
    }
