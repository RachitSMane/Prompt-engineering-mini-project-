"""Dataset preparation and loading.

Source: Amazon Reviews 2023 (McAuley Lab, UCSD) — https://amazon-reviews-2023.github.io/
The preparation is fully deterministic given ``RANDOM_SEED`` so anyone can
reproduce exactly the same subset and the same 300-review annotation sample.
"""

from __future__ import annotations

import gzip
import hashlib
import html
import json
import logging
import random
import re
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Iterator

import pandas as pd

from config import Settings
from schemas.review_schema import ReviewInput

logger = logging.getLogger(__name__)

BASE_URL = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw"
REVIEW_COLUMNS = ["review_id", "product_id", "product_title", "rating", "review_title", "review_text", "timestamp", "verified_purchase", "helpful_vote"]

MIN_WORDS = 5
MAX_WORDS = 400
MIN_PRODUCT_REVIEWS = 200
MAX_PRODUCT_REVIEWS = 5000

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def review_url(category: str) -> str:
    return f"{BASE_URL}/review_categories/{category}.jsonl.gz"


def meta_url(category: str) -> str:
    return f"{BASE_URL}/meta_categories/meta_{category}.jsonl.gz"


def download(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        logger.info("Using cached %s", dest.name)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    logger.info("Downloading %s", url)
    urllib.request.urlretrieve(url, tmp)  # noqa: S310 - fixed https URL
    tmp.replace(dest)
    return dest


def clean_text(text: str | None) -> str:
    if not text:
        return ""
    text = html.unescape(_TAG_RE.sub(" ", str(text)))
    return _WS_RE.sub(" ", text).strip()


def is_valid_review(text: str) -> bool:
    n = len(text.split())
    return MIN_WORDS <= n <= MAX_WORDS


def make_review_id(user_id: str, asin: str, timestamp: int | str) -> str:
    return "R" + hashlib.sha1(f"{user_id}|{asin}|{timestamp}".encode()).hexdigest()[:12]


def _iter_jsonl_gz(path: Path) -> Iterator[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def prepare_dataset(settings: Settings, download_if_missing: bool = True) -> dict:
    """Build ``data/processed/reviews.csv`` and return a dataset card (stats)."""
    paths = settings.paths
    cat = settings.dataset_category
    reviews_gz = paths.data_raw / f"{cat}.jsonl.gz"
    meta_gz = paths.data_raw / f"meta_{cat}.jsonl.gz"
    if download_if_missing:
        download(review_url(cat), reviews_gz)
        download(meta_url(cat), meta_gz)

    # Pass 1: count valid reviews per product
    counts: Counter[str] = Counter()
    total_raw = 0
    for r in _iter_jsonl_gz(reviews_gz):
        total_raw += 1
        if is_valid_review(clean_text(r.get("text"))):
            counts[r["parent_asin"]] += 1
    candidates = sorted(p for p, c in counts.items() if MIN_PRODUCT_REVIEWS <= c <= MAX_PRODUCT_REVIEWS)

    # Titles for candidates (products without a title are skipped)
    titles: dict[str, str] = {}
    cand_set = set(candidates)
    for m in _iter_jsonl_gz(meta_gz):
        pid = m.get("parent_asin")
        if pid in cand_set and m.get("title"):
            titles[pid] = clean_text(m["title"])[:200]
    candidates = [p for p in candidates if p in titles]
    rng = random.Random(settings.random_seed)
    selected = sorted(rng.sample(candidates, min(settings.dataset_num_products, len(candidates))))
    selected_set = set(selected)

    # Pass 2: collect, clean and de-duplicate reviews for selected products
    seen_user_product: set[tuple[str, str]] = set()
    seen_text: set[tuple[str, str]] = set()
    per_product: dict[str, list[dict]] = {p: [] for p in selected}
    dropped = Counter()
    for r in _iter_jsonl_gz(reviews_gz):
        pid = r.get("parent_asin")
        if pid not in selected_set:
            continue
        text = clean_text(r.get("text"))
        if not is_valid_review(text):
            dropped["invalid_length_or_empty"] += 1
            continue
        up_key = (r.get("user_id", ""), pid)
        txt_key = (pid, text.lower())
        if up_key in seen_user_product:
            dropped["duplicate_user_product"] += 1
            continue
        if txt_key in seen_text:
            dropped["duplicate_text"] += 1
            continue
        seen_user_product.add(up_key)
        seen_text.add(txt_key)
        per_product[pid].append(
            {
                "review_id": make_review_id(r.get("user_id", ""), r.get("asin", pid), r.get("timestamp", "")),
                "product_id": pid,
                "product_title": titles[pid],
                "rating": float(r.get("rating", 0) or 0),
                "review_title": clean_text(r.get("title")),
                "review_text": text,
                "timestamp": pd.to_datetime(r.get("timestamp"), unit="ms", utc=True).strftime("%Y-%m-%d"),
                "verified_purchase": bool(r.get("verified_purchase")),
                "helpful_vote": int(r.get("helpful_vote", 0) or 0),
            }
        )

    rows: list[dict] = []
    for pid in selected:
        items = sorted(per_product[pid], key=lambda d: d["review_id"])
        rng_p = random.Random(f"{settings.random_seed}-{pid}")
        rows.extend(rng_p.sample(items, min(settings.dataset_reviews_per_product, len(items))))

    df = pd.DataFrame(rows, columns=REVIEW_COLUMNS).drop_duplicates("review_id")
    paths.data_processed.mkdir(parents=True, exist_ok=True)
    df.to_csv(paths.reviews_csv, index=False, encoding="utf-8")

    card = {
        "source": "Amazon Reviews 2023 (McAuley Lab, UCSD)",
        "source_url": "https://amazon-reviews-2023.github.io/",
        "category": cat,
        "raw_review_url": review_url(cat),
        "raw_meta_url": meta_url(cat),
        "random_seed": settings.random_seed,
        "total_raw_reviews_in_category": total_raw,
        "candidate_products": len(candidates),
        "selected_products": len(selected),
        "reviews_per_product_cap": settings.dataset_reviews_per_product,
        "final_reviews": int(len(df)),
        "dropped_for_selected_products": dict(dropped),
        "filters": {"min_words": MIN_WORDS, "max_words": MAX_WORDS, "product_review_count_range": [MIN_PRODUCT_REVIEWS, MAX_PRODUCT_REVIEWS]},
        "rating_distribution": {str(k): int(v) for k, v in df["rating"].value_counts().sort_index().items()},
    }
    (paths.data_processed / "dataset_card.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
    return card


# --------------------------------------------------------------------------- loading
def load_reviews(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"review_id": str, "product_id": str}, keep_default_na=False)
    if "rating" in df.columns:
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    return df


def normalize_uploaded_reviews(df: pd.DataFrame, text_col: str, product_col: str, id_col: str | None = None, title_col: str | None = None) -> pd.DataFrame:
    """Map an arbitrary uploaded CSV to the project's review columns and clean it."""
    if text_col not in df.columns or product_col not in df.columns:
        raise ValueError(f"CSV must contain the selected columns '{text_col}' and '{product_col}'")
    out = pd.DataFrame()
    out["review_text"] = df[text_col].astype(str).map(clean_text)
    out["product_id"] = df[product_col].astype(str).str.strip()
    if id_col and id_col in df.columns:
        out["review_id"] = df[id_col].astype(str).str.strip()
    else:
        out["review_id"] = [
            "U" + hashlib.sha1(f"{p}|{t}".encode()).hexdigest()[:12] for p, t in zip(out["product_id"], out["review_text"])
        ]
    out["product_title"] = df[title_col].astype(str) if title_col and title_col in df.columns else ""
    for opt in ("rating", "timestamp"):
        out[opt] = df[opt] if opt in df.columns else None
    before = len(out)
    out = out[(out["review_text"].str.len() > 0) & (out["product_id"].str.len() > 0)]
    out = out.drop_duplicates(subset=["product_id", "review_text"]).drop_duplicates(subset=["review_id"])
    out.attrs["dropped"] = before - len(out)
    return out.reset_index(drop=True)


def dataframe_to_inputs(df: pd.DataFrame) -> list[ReviewInput]:
    items: list[ReviewInput] = []
    for row in df.to_dict(orient="records"):
        rating = row.get("rating")
        items.append(
            ReviewInput(
                review_id=str(row["review_id"]),
                product_id=str(row["product_id"]),
                review_text=str(row["review_text"]),
                product_title=(str(row.get("product_title")) or None) if row.get("product_title") else None,
                rating=float(rating) if rating not in (None, "") and pd.notna(rating) else None,
                timestamp=row.get("timestamp") or None,
            )
        )
    return items
