"""Negative control v3 (nc-v3): a pre-registered, seeded set of planted errors WITH matched clean controls.

The 30 nc-v2 copies (``evaluation/negative_control.py``) and their v2/v3 results are kept unchanged. nc-v3 adds:

* 12 realistic error types (the kinds of error actually found in the 15 summaries), each paired with a CLEAN claim of
  the same form that is correct, so false positives (specificity) can be measured;
* claim-level detection: a planted claim counts as detected from the verifier's status on THAT claim, not from
  keywords;
* a manifest with the SHA-256 of the dataset, written before any judging; the runner refuses to judge a dataset whose
  hash differs from the manifest.

Construction (no LLM). Statistic types use the product aggregate. Review-based types are anchored on reviews where
three signals agree (see ``clear_anchors``): an extraction aspect whose evidence span is verbatim in the review, the
star rating, and the independent human-created overall sentiment. The human labels are used only as one of these
agreement signals; they are not treated as verified claim-level ground truth (they are not second-person verified).
Fabricated details are planted only where their key words do not occur in the cited review (compatibility brands: in
no review of the product). Every review-based case was read against its review text before any judging; ambiguous
cases led to rule changes (POLARITY_WORDING, ATTRIBUTION_EXTRA_EXCLUDE) before the dataset was frozen, never after.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from dataclasses import dataclass

from evaluation.claim_checks import ASPECT_PATTERNS
from schemas.claim_schema import Claim, ClaimStatus

NC_VERSION = "nc-v3"
SEED = 20261115
PRODUCTS_PER_REVIEW_TYPE = 10

STAT_TYPES = ("incorrect_review_count", "incorrect_percentage", "incorrect_majority", "incorrect_complaint_statistic")
# Generation order (types with the scarcest anchors first, so they get first pick of each product's anchors).
REVIEW_TYPES = ("contradictory_citation", "fabricated_compatibility", "flipped_sentiment", "unsupported_performance",
                "unsupported_durability", "fabricated_warranty", "incorrect_aspect_attribution", "misleading_paraphrase")
ERROR_TYPES = STAT_TYPES + ("flipped_sentiment", "unsupported_performance", "unsupported_durability", "fabricated_compatibility",
                             "fabricated_warranty", "incorrect_aspect_attribution", "contradictory_citation", "misleading_paraphrase")

EXPECTED = {t: "CONTRADICTED" for t in STAT_TYPES} | {
    "flipped_sentiment": "CONTRADICTED", "unsupported_performance": "UNSUPPORTED", "unsupported_durability": "UNSUPPORTED",
    "fabricated_compatibility": "UNSUPPORTED", "fabricated_warranty": "UNSUPPORTED", "incorrect_aspect_attribution": "UNSUPPORTED",
    "contradictory_citation": "CONTRADICTED", "misleading_paraphrase": "PARTIALLY_SUPPORTED"}
DETECTION_RULE = {
    "primary": "flagged = final status in {UNSUPPORTED, CONTRADICTED}; for misleading_paraphrase (expected PARTIALLY_SUPPORTED) "
               "flagged = final status != SUPPORTED. The same per-type rule is applied to the matched clean claims.",
    "strict": "flagged = final status in {UNSUPPORTED, CONTRADICTED} for every type",
    "lenient": "flagged = final status != SUPPORTED for every type (PARTIALLY_SUPPORTED and UNVERIFIABLE count as flagged)",
}

ASPECT_PHRASES = {
    "performance": ("the product works well", "the product does not work properly"),
    "quality_durability": ("the product is well made", "the product is poorly made or did not last"),
    "compatibility": ("the part fits their appliance", "the part does not fit their appliance"),
    "installation_setup": ("installation was easy", "installation was difficult"),
    "price_value": ("the product is good value for the price", "the product is poor value for the price"),
    "ease_of_use": ("the product is easy to use", "the product is hard to use"),
    "design_build": ("they like the design of the product", "they dislike the design of the product"),
    "size_capacity": ("the size suits their needs", "the size does not suit their needs"),
    "customer_service": ("the seller's customer service was helpful", "the seller's customer service was unhelpful"),
    "shipping_packaging": ("the product was delivered promptly", "the product arrived late or damaged"),
    "noise": ("the product is quiet", "the product is noisy"),
}
#: The anchor's evidence span must contain wording for the aspect AND the polarity of the template phrase, so that the
#: clean claim "A reviewer reports that <phrase>" is actually stated by the review (an extraction label alone is not enough).
POLARITY_WORDING = {
    ("performance", "positive"): r"\bworks?\b|worked|working|perfect\w*|great|fixed|corrected|solved|does the job|as expected|good",
    ("performance", "negative"): r"(?:not|n't|never|no longer)\s+\w*\s*work|stopped working|quit|died|fail\w*|useless|doesn't do",
    ("quality_durability", "positive"): r"sturdy|well made|solid|durable|good quality|high quality|holds? up|yet to fail|heavy duty",
    ("quality_durability", "negative"): r"broke|broken|crack\w*|flims\w*|wore|fell apart|didn't last|doesn't hold up|lasted (?:only|about|less)|poor quality|cheap(?:ly)? made",
    ("compatibility", "positive"): r"\bfits?\b|fitted|exact (?:fit|match|replacement)|perfect fit|compatible|match\w*",
    ("compatibility", "negative"): r"(?:not|n't|doesn't|didn't|won't|wouldn't)\s+fit|wrong (?:size|part)|too (?:big|small|tall|short|long)|not compatible|incompatible",
    ("installation_setup", "positive"): r"easy|simple|quick\w*|snap\w*|no problem|minutes",
    ("installation_setup", "negative"): r"difficult|hard to install|tricky|struggl\w*|pain|impossible",
    ("price_value", "positive"): r"price|cheap\w*|value|worth|deal|sav(?:e|ed|ing)|less than|inexpensive|bargain",
    ("price_value", "negative"): r"overpriced|expensive|not worth|waste of money|you get what you pay|rip[- ]?off",
    ("ease_of_use", "positive"): r"easy|simple|convenient|handy",
    ("ease_of_use", "negative"): r"hard to use|difficult|awkward|confusing|messy|pain",
    ("design_build", "positive"): r"(?:like|love|nice|beautiful|great|good|attractive)\w*.{0,30}(?:design|look|colou?r)|looks? (?:great|good|nice)",
    ("design_build", "negative"): r"(?:ugly|cheap[- ]looking|poor design|bad design|flimsy design|dislike)",
    ("size_capacity", "positive"): r"(?:right|perfect|good|great) size|holds? (?:a lot|plenty|lots)|roomy|spacious|lots of",
    ("size_capacity", "negative"): r"too (?:small|big|large|tall|short|long)|not big enough|smaller than|larger than|doesn't hold",
    ("customer_service", "positive"): r"helpful|great (?:customer )?service|responsive|replaced|refund\w*|excellent service",
    ("customer_service", "negative"): r"unhelpful|rude|never (?:responded|replied|heard)|no response|terrible service|poor service|bad service",
    ("shipping_packaging", "positive"): r"quick\w*|fast|on time|promptly|well packaged|arrived (?:early|quickly|fast|in good)",
    ("shipping_packaging", "negative"): r"\blate\b|damaged|never arrived|slow (?:shipping|delivery)|arrived broken",
    ("noise", "positive"): r"quiet|silent",
    ("noise", "negative"): r"noisy|loud|rattl\w*|squeak\w*",
}
#: Extra wording that makes an attributed aspect plausibly implied (attribution cases are skipped for such reviews).
#: "performance" is never used as the wrongly attributed aspect: almost any review implies whether the product works.
ATTRIBUTION_EXTRA_EXCLUDE = {"quality_durability": r"years?\b|lasted|still working|holding up", "size_capacity": r"\bfits?\b|fit just right"}
FAKE_PERFORMANCE = [("it cut their energy bill in half", r"energy|bill|half"),
                    ("it works twice as fast as the original part", r"twice|fast"),
                    ("it runs completely silently even at full power", r"silent|full power")]
FAKE_DURABILITY = ("the product has lasted more than five years without any problems", r"year|five|\b5\b")
FAKE_COMPATIBILITY = [("the part also fits Samsung and Bosch models", r"samsung|bosch"),
                      ("the part also fits Miele and Electrolux models", r"miele|electrolux"),
                      ("the part also fits Haier and Smeg models", r"haier|smeg")]
FAKE_WARRANTY = ("the product comes with a two-year manufacturer warranty", r"warrant|guarantee|two[- ]year")
COMPLAINT_LABEL = {
    "poor_durability": "Poor durability", "installation_difficulty": "Installation difficulty", "poor_value": "Poor value",
    "product_defect": "A product defect", "customer_service_issue": "A customer service issue", "performance_issue": "A performance issue",
    "compatibility_fit_issue": "A compatibility or fit issue", "usability_issue": "A usability issue",
    "not_as_described": "An item not as described", "missing_parts": "Missing parts", "noise_issue": "A noise issue",
    "safety_concern": "A safety concern",
}


@dataclass
class Anchor:
    review_id: str
    aspect: str
    polarity: str  # positive | negative
    text: str
    rating: float
    evidence_span: str = ""


def clear_anchors(product_id: str, sample_ids: set[str], human_labels: dict, reviews: dict[str, dict], records=None) -> list[Anchor]:
    """Reviews with an unambiguous aspect sentiment. Three independent signals must agree:
    (1) an extraction aspect label whose evidence span is a verbatim substring of the review text AND contains
        wording for that aspect and polarity (``POLARITY_WORDING``),
    (2) the star rating (positive >= 4, negative <= 2) and
    (3) the independent human-created OVERALL sentiment label (the human annotations carry almost no aspect labels:
        292 of 300 have none, so they cannot anchor an aspect by themselves)."""
    from evaluation.claim_checks import quote_in_texts

    out = []
    by_id = {r.review_id: r for r in (records or [])}
    for rid in sorted(sample_ids):
        lab, rev, rec = human_labels.get(rid), reviews.get(rid), by_id.get(rid)
        if not lab or not rev or not rec or rev["product_id"] != product_id or lab.get("status") != "completed":
            continue
        if lab.get("provenance") != "human_created" or rev["rating"] is None or len(rev["text"].split()) < 5:
            continue
        human = lab.get("overall_sentiment")
        seen = set()
        for a in rec.aspects:
            cat = str(getattr(a.aspect_category, "value", a.aspect_category))
            sen = str(getattr(a.sentiment, "value", a.sentiment))
            if cat not in ASPECT_PHRASES or cat in seen or sen not in ("positive", "negative") or sen != human:
                continue
            if not (sen == "positive" and rev["rating"] >= 4 or sen == "negative" and rev["rating"] <= 2):
                continue
            if not quote_in_texts(a.evidence, [rev["text"]], min_words=2):
                continue
            if not re.search(POLARITY_WORDING[(cat, sen)], a.evidence, re.I):
                continue
            seen.add(cat)
            out.append(Anchor(rid, cat, sen, rev["title"] + "\n" + rev["text"], rev["rating"], a.evidence))
    return out


def _phrase(aspect: str, polarity: str) -> str:
    return ASPECT_PHRASES[aspect][0 if polarity == "positive" else 1]


def _absent(pattern: str, text: str) -> bool:
    return not re.search(pattern, text, re.I)


def _claim(case_id: str, which: str, product_id: str, text: str, polarity: str, cited: list[str], kind: str) -> dict:
    return {"claim_id": f"{case_id}-{which}:planted:0", "summary_id": f"{case_id}-{which}", "product_id": product_id, "text": text,
            "polarity": polarity, "cited_review_ids": cited, "kind": kind}


def to_claim(d: dict) -> Claim:
    from evaluation.claims import categorise

    kind = d["kind"]
    return Claim(claim_id=d["claim_id"], summary_id=d["summary_id"], product_id=d["product_id"], source_field="planted", index=0,
                 kind=kind, text=d["text"], polarity=d["polarity"], category=categorise(d["text"], kind),
                 is_numeric=bool(re.search(r"\d", d["text"])), cited_review_ids=d["cited_review_ids"])


def _insert(content: dict, claim: dict) -> dict:
    c = json.loads(json.dumps(content))
    if claim["kind"] == "opinion_sentence":
        c["overall_opinion"] = c["overall_opinion"].rstrip() + " " + claim["text"]
    else:
        key = "appreciated" if claim["polarity"] == "positive" else "complaints"
        c[key] = list(c.get(key, [])) + [{"point": claim["text"], "supporting_review_ids": claim["cited_review_ids"]}]
    return c


def _stat_case(t: str, agg, rng: random.Random) -> tuple[str, str, str, str] | None:
    """(clean text, corrupted text, variant, planted change) for a statistic error type, or None if not applicable."""
    n = agg.total_reviews
    sc, sp = agg.sentiment_counts, agg.sentiment_percentages
    dom = "positive" if sc.get("positive", 0) >= sc.get("negative", 0) else "negative"
    other = "negative" if dom == "positive" else "positive"
    if t == "incorrect_review_count":
        k = sc.get(dom, 0)
        clean = f"{k} of the {n} reviews are {dom}."
        if rng.random() < 0.5:
            opts = [k + d for d in (-3, -2, -1, 1, 2, 3) if 0 <= k + d <= n and k + d != k]
            k2 = rng.choice(opts)
            return clean, f"{k2} of the {n} reviews are {dom}.", "wrong_count", f"count {k} -> {k2}"
        opts = [n + d for d in (3, 4, 5, 6, 8, 10, -2, -3) if n + d >= k and n + d not in (n, agg.reviews_with_complaints) and n + d > 0]
        m = rng.choice(opts)
        return clean, f"{k} of the {m} reviews are {dom}.", "wrong_denominator", f"denominator {n} -> {m}"
    if t == "incorrect_percentage":
        s = rng.choice([x for x in ("positive", "negative") if sc.get(x, 0) > 0])
        p = sp[s]
        others = list(sp.values())
        for _ in range(50):
            d = rng.choice([8.3, 10.5, 12.7, 15.2, 19.4]) * rng.choice([-1, 1])
            p2 = round(p + d, 1)
            if 0 < p2 < 100 and all(abs(p2 - o) > 3 for o in others):
                return f"{p}% of the reviews are {s}.", f"{p2}% of the reviews are {s}.", s, f"{s} share {p}% -> {p2}%"
        return None
    if t == "incorrect_majority":
        if sc.get("positive", 0) == sc.get("negative", 0):
            return None
        if sp[dom] > 50:
            return (f"The majority of reviews are {dom}.", f"The majority of reviews are {other}.", "majority",
                    f"majority {dom} ({sp[dom]}%) -> {other} ({sp[other]}%)")
        return (f"More reviews are {dom} than {other}.", f"More reviews are {other} than {dom}.", "plurality",
                f"more {dom} ({sp[dom]}%) -> more {other} ({sp[other]}%)")
    if t == "incorrect_complaint_statistic":
        if not agg.complaint_counts:
            return None
        top = max(agg.complaint_counts.values())
        c = sorted(x for x, v in agg.complaint_counts.items() if v == top)[0]
        k = agg.complaint_counts[c]
        aspect_counts = {v for a in agg.aspect_stats for v in (a.positive, a.negative, a.mentions)}
        opts = [k + d for d in (-3, -2, 2, 3, 4, -4, 5) if 0 < k + d <= n and k + d != k and k + d not in aspect_counts]
        if not opts:
            opts = [k + d for d in (-3, -2, 2, 3, 4, -4, 5) if 0 < k + d <= n and k + d != k]
        if not opts:
            return None
        k2 = rng.choice(opts)
        lab = COMPLAINT_LABEL.get(c, c.replace("_", " ").capitalize())
        return (f"{lab} is mentioned in {k} of the {n} reviews.", f"{lab} is mentioned in {k2} of the {n} reviews.", c,
                f"{c} count {k} -> {k2}")
    raise ValueError(t)


def _review_case(t: str, pid: str, anchors: list[Anchor], reviews: dict[str, dict], product_text: str, rng: random.Random,
                 used: set) -> dict | None:
    """dict(clean_text, clean_cited, corrupt_text, corrupt_cited, polarity_clean, polarity_corrupt, variant, change, evidence_ids)."""
    def pick(cands):
        cands = [a for a in cands if (a.review_id, a.aspect, a.polarity) not in used]
        if not cands:
            return None
        a = rng.choice(cands)
        used.add((a.review_id, a.aspect, a.polarity))
        return a

    def clean_for(aspect_set=None):
        a = pick([x for x in anchors if aspect_set is None or x.aspect in aspect_set])
        return a

    if t == "flipped_sentiment":
        a = pick(anchors)
        if not a:
            return None
        flip = "negative" if a.polarity == "positive" else "positive"
        return dict(clean_text=f"A reviewer reports that {_phrase(a.aspect, a.polarity)}.", clean_cited=[a.review_id], polarity_clean=a.polarity,
                    corrupt_text=f"A reviewer reports that {_phrase(a.aspect, flip)}.", corrupt_cited=[a.review_id], polarity_corrupt=flip,
                    variant=f"{a.aspect}:{a.polarity}->{flip}", change=f"'{_phrase(a.aspect, a.polarity)}' -> '{_phrase(a.aspect, flip)}'",
                    evidence_ids=[a.review_id])
    if t in ("unsupported_performance", "unsupported_durability", "fabricated_compatibility", "fabricated_warranty"):
        if t == "unsupported_performance":
            detail, pat = rng.choice(FAKE_PERFORMANCE)
            clean_aspects = {"performance"}
        elif t == "unsupported_durability":
            detail, pat = FAKE_DURABILITY
            clean_aspects = {"quality_durability"}
        elif t == "fabricated_compatibility":
            options = [(d, p) for d, p in FAKE_COMPATIBILITY if _absent(p, product_text)]  # brands named in NO review of the product
            if not options:
                return None
            detail, pat = rng.choice(options)
            clean_aspects = {"compatibility"}
        else:
            detail, pat = FAKE_WARRANTY
            clean_aspects = {"customer_service", "shipping_packaging", "price_value", "installation_setup"}
        hosts = sorted({x.review_id for x in anchors if _absent(pat, x.text)})
        if not hosts:
            return None
        clean = clean_for(clean_aspects) or (clean_for(None) if t == "fabricated_warranty" else None)
        if not clean:
            return None
        host = rng.choice(hosts)
        return dict(clean_text=f"A reviewer reports that {_phrase(clean.aspect, clean.polarity)}.", clean_cited=[clean.review_id],
                    polarity_clean=clean.polarity, corrupt_text=f"A reviewer reports that {detail}.", corrupt_cited=[host],
                    polarity_corrupt="positive", variant=detail, change=f"planted '{detail}' citing {host} (key words absent from it)",
                    evidence_ids=[host, clean.review_id])
    if t == "incorrect_aspect_attribution":
        for _ in range(20):
            a = pick(anchors)
            if not a:
                return None
            labelled = {x.aspect for x in anchors if x.review_id == a.review_id}
            opts = [b for b in ASPECT_PHRASES if b not in labelled and b != a.aspect and b != "performance"
                    and not re.search(ATTRIBUTION_EXTRA_EXCLUDE.get(b, r"(?!x)x"), a.text, re.I)
                    and not re.search(rf"\b(?:{ASPECT_PATTERNS[b]})", a.text, re.I)
                    and not re.search(POLARITY_WORDING[(b, a.polarity)], a.text, re.I)]
            if opts:
                b = rng.choice(sorted(opts))
                return dict(clean_text=f"A reviewer reports that {_phrase(a.aspect, a.polarity)}.", clean_cited=[a.review_id],
                            polarity_clean=a.polarity, corrupt_text=f"A reviewer reports that {_phrase(b, a.polarity)}.",
                            corrupt_cited=[a.review_id], polarity_corrupt=a.polarity, variant=f"{a.aspect}->{b}",
                            change=f"aspect {a.aspect} -> {b} (not mentioned in {a.review_id})", evidence_ids=[a.review_id])
        return None
    if t == "contradictory_citation":
        pairs = sorted({x.aspect for x in anchors if x.polarity == "positive"} & {x.aspect for x in anchors if x.polarity == "negative"})
        if not pairs:
            return None
        asp = rng.choice(pairs)
        pol = rng.choice(["positive", "negative"])
        opp = "negative" if pol == "positive" else "positive"
        good = pick([x for x in anchors if x.aspect == asp and x.polarity == pol])
        bad = [x for x in anchors if x.aspect == asp and x.polarity == opp]
        if not good or not bad:
            return None
        b = rng.choice(bad)
        text = f"A reviewer reports that {_phrase(asp, pol)}."
        return dict(clean_text=text, clean_cited=[good.review_id], polarity_clean=pol, corrupt_text=text, corrupt_cited=[b.review_id],
                    polarity_corrupt=pol, variant=f"{asp}:{pol}", change=f"citation {good.review_id} -> {b.review_id} ({asp} is {opp} in the cited review)",
                    evidence_ids=[good.review_id, b.review_id])
    if t == "misleading_paraphrase":
        a = pick(anchors)
        if not a:
            return None
        ph = _phrase(a.aspect, a.polarity)
        if rng.random() < 0.5:
            corrupt, var = f"All reviewers agree that {ph}.", "overgeneralisation"
        else:
            word = "best" if a.polarity == "positive" else "worst"
            if not _absent(r"\bbest\b|\bworst\b|\bever\b", a.text):
                corrupt, var = f"All reviewers agree that {ph}.", "overgeneralisation"
            else:
                corrupt, var = f"A reviewer reports that {ph}, calling it the {word} purchase they have ever made.", "intensification"
        return dict(clean_text=f"A reviewer reports that {ph}.", clean_cited=[a.review_id], polarity_clean=a.polarity, corrupt_text=corrupt,
                    corrupt_cited=[a.review_id], polarity_corrupt=a.polarity, variant=var, change=f"'{ph}' -> '{corrupt}'",
                    evidence_ids=[a.review_id])
    raise ValueError(t)


def generate(summaries, human_labels: dict, reviews: dict[str, dict], records=None, seed: int = SEED) -> list[dict]:
    """Deterministic for a given seed and inputs. One case = one corrupted claim + one matched clean claim."""
    rng = random.Random(seed)
    summ = sorted(summaries, key=lambda s: s.product_id)
    cases: list[dict] = []

    def add(t, s, clean_text, clean_cited, pc, corrupt_text, corrupt_cited, pk, kind, variant, change, evidence):
        case_id = f"{NC_VERSION}-{len(cases) + 1:03d}"
        clean = _claim(case_id, "k", s.product_id, clean_text, pc, clean_cited, kind)
        corrupt = _claim(case_id, "c", s.product_id, corrupt_text, pk, corrupt_cited, kind)
        cases.append({"case_id": case_id, "nc_version": NC_VERSION, "seed": seed, "product_id": s.product_id,
                      "original_summary_id": s.summary_id, "error_type": t, "variant": variant, "planted_change": change,
                      "expected_status_corrupted": EXPECTED[t], "expected_status_clean": "SUPPORTED",
                      "expected_detection": "flagged" + (" (status != SUPPORTED)" if t == "misleading_paraphrase" else " (UNSUPPORTED or CONTRADICTED)"),
                      "corrupted_claim": corrupt, "clean_claim": clean, "source_evidence": evidence,
                      "original_content": s.content.model_dump(mode="json"),
                      "corrupted_content": _insert(s.content.model_dump(mode="json"), corrupt),
                      "clean_content": _insert(s.content.model_dump(mode="json"), clean)})

    for t in STAT_TYPES:
        for s in summ:
            r = _stat_case(t, s.aggregate, rng)
            if r is None:
                continue
            clean_text, corrupt_text, variant, change = r
            ev = {"statistics": {"total_reviews": s.aggregate.total_reviews, "sentiment_counts": s.aggregate.sentiment_counts,
                                 "sentiment_percentages": s.aggregate.sentiment_percentages, "complaint_counts": s.aggregate.complaint_counts}}
            add(t, s, clean_text, [], "neutral", corrupt_text, [], "neutral", "opinion_sentence", variant, change, ev)
    used_by_product: dict[str, set] = {}
    for t in REVIEW_TYPES:
        eligible = []
        prepared = {}
        for s in summ:
            anchors = clear_anchors(s.product_id, set(s.aggregate.review_ids), human_labels, reviews, records)
            if anchors:
                eligible.append(s)
                prepared[s.product_id] = anchors
        chosen = sorted(rng.sample(eligible, min(PRODUCTS_PER_REVIEW_TYPE, len(eligible))), key=lambda s: s.product_id)
        for s in chosen:
            ptext = " ".join(f"{r['title']} {r['text']}" for rid, r in reviews.items() if rid in set(s.aggregate.review_ids))
            used = used_by_product.setdefault(s.product_id, set())  # an anchor is used at most once per product
            r = _review_case(t, s.product_id, prepared[s.product_id], reviews, ptext, rng, used)
            if r is None:
                continue
            spans = {a.review_id + ":" + a.aspect: a.evidence_span for a in prepared[s.product_id] if a.review_id in r["evidence_ids"]}
            ev = {"reviews": {rid: {"rating": reviews[rid]["rating"], "text": reviews[rid]["text"]} for rid in r["evidence_ids"]},
                  "anchor_evidence_spans": spans,
                  "anchor_source": "extraction aspect with a verbatim evidence span, agreeing with the star rating and the human overall sentiment"}
            add(t, s, r["clean_text"], r["clean_cited"], r["polarity_clean"], r["corrupt_text"], r["corrupt_cited"], r["polarity_corrupt"],
                "point", r["variant"], r["change"], ev)
    return cases


def dataset_sha256(cases: list[dict]) -> str:
    return hashlib.sha256("".join(json.dumps(c, sort_keys=True, ensure_ascii=False) + "\n" for c in cases).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- metrics
def wilson(k: int, n: int, z: float = 1.96) -> list[float] | None:
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 3), round(min(1.0, c + h), 3)]


def flagged(status: str | None, error_type: str, rule: str = "primary") -> bool | None:
    if status is None:
        return None
    if rule == "lenient" or (rule == "primary" and error_type == "misleading_paraphrase"):
        return status != ClaimStatus.SUPPORTED.value
    return status in (ClaimStatus.UNSUPPORTED.value, ClaimStatus.CONTRADICTED.value)


def confusion(rows: list[dict], key: str, rule: str = "primary") -> dict:
    """rows: {error_type, corrupted_<key>, clean_<key>} statuses. Positive class = corrupted claim."""
    tp = fn = fp = tn = 0
    for r in rows:
        a, b = flagged(r.get(f"corrupted_{key}"), r["error_type"], rule), flagged(r.get(f"clean_{key}"), r["error_type"], rule)
        if a is not None:
            tp += a
            fn += not a
        if b is not None:
            fp += b
            tn += not b
    prec = tp / (tp + fp) if tp + fp else None
    rec = tp / (tp + fn) if tp + fn else None
    spec = tn / (tn + fp) if tn + fp else None
    f1 = 2 * prec * rec / (prec + rec) if prec and rec else (0.0 if prec == 0 or rec == 0 else None)
    r3 = lambda x: None if x is None else round(x, 3)
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "n_corrupted": tp + fn, "n_clean": tn + fp,
            "precision": r3(prec), "precision_wilson95": wilson(tp, tp + fp), "recall": r3(rec), "recall_wilson95": wilson(tp, tp + fn),
            "f1": r3(f1), "specificity": r3(spec), "specificity_wilson95": wilson(tn, tn + fp),
            "false_positive_rate": r3(1 - spec) if spec is not None else None}
