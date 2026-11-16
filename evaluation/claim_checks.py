"""Deterministic (no-LLM) checks for D3 claim-level evaluation (judge v4).

Everything here is pure Python over stored data: the product aggregate (``ProductAggregate``), the original review texts
(``data/processed/reviews.csv``) and, for shares among negative/positive reviews, the extraction records. Numbers,
counts, percentages, denominators, majorities and averages are always recomputed here, never by an LLM.

Every check returns ``NumericCheck`` / ``DeterministicFinding`` objects with the extracted value, the expected value,
the denominator and the calculation, so each verdict can be audited. The phrase patterns are heuristics: a sentence
whose wording no pattern recognises gets no deterministic verdict (and is sent to the LLM verifier instead); it is
never silently passed.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from schemas.claim_schema import ClaimStatus, DeterministicFinding, EvidenceExcerpt, NumericCheck

# --------------------------------------------------------------------------- vocabularies
ASPECT_PATTERNS: dict[str, str] = {
    "performance": r"performance|perform\w*|functional\w*|functionality|works? well|work as intended|resolv\w*|works? great",
    "quality_durability": r"durab\w*|quality|lasting|lasted|sturdy|broke\w*|breaks?|reliab\w*|wears? out",
    "design_build": r"design\w*|build quality|built|colou?r|hinge|materials?",
    "ease_of_use": r"ease of use|easy to use|usab\w*|user[- ]friendly|convenien\w*|hard to use",
    "installation_setup": r"install\w*|set ?up",
    "compatibility": r"compatib\w*|\bfit\w*",
    "price_value": r"\bprice\w*|\bvalue\b|money|afford\w*|cheap\w*|\bcost\w*|expensive",
    "customer_service": r"customer service|customer support|seller|warrant(?:y|ies)",
    "shipping_packaging": r"\bship\w*|deliver\w*|arriv\w*|packag\w*",
    "size_capacity": r"\bsize\w*|capacit\w*|\btall\b|longer than expected|storing",
    "noise": r"nois\w*|\bloud\w*|\bquiet\w*",
    "safety": r"\bsafe\b|\bsafety\b|unsafe|dangerous",
}
_ASPECT_RX = {a: re.compile(rf"\b(?:{p})", re.I) for a, p in ASPECT_PATTERNS.items()}

ASPECT_TO_COMPLAINT = {
    "performance": "performance_issue", "quality_durability": "poor_durability", "installation_setup": "installation_difficulty",
    "price_value": "poor_value", "customer_service": "customer_service_issue", "compatibility": "compatibility_fit_issue",
    "ease_of_use": "usability_issue", "noise": "noise_issue", "safety": "safety_concern",
}
COMPLAINT_PATTERNS: dict[str, str] = {
    "poor_durability": r"poor durability|durab\w*|lasting only|stopped working|short time|broke\w*",
    "installation_difficulty": r"installation difficult\w*|difficult\w* (?:to )?install\w*|install\w*",
    "poor_value": r"poor value|value for money|\bvalue\b|overpriced|expensive",
    "product_defect": r"product defects?|defect\w*",
    "customer_service_issue": r"customer service",
    "performance_issue": r"performance issues?|performance",
    "compatibility_fit_issue": r"compatibility(?: fit)? issues?|compatib\w*|\bfit\w*",
    "usability_issue": r"usability|ease of use",
    "not_as_described": r"not as described|as advertised|advertised colou?r",
    "missing_parts": r"missing",
    "noise_issue": r"noise|noisy",
    "safety_concern": r"safety",
}
_COMPLAINT_RX = {c: re.compile(rf"\b(?:{p})", re.I) for c, p in COMPLAINT_PATTERNS.items()}

ASPECT_TO_CATEGORY = {
    "performance": "product_performance", "quality_durability": "quality_durability", "design_build": "design_build",
    "ease_of_use": "ease_of_use", "installation_setup": "installation", "compatibility": "compatibility",
    "price_value": "price_value", "customer_service": "customer_service", "shipping_packaging": "shipping_packaging",
    "size_capacity": "design_build", "noise": "product_performance", "safety": "other",
}
SENTIMENTS = ("positive", "negative", "neutral", "mixed")
POS_CUES = re.compile(r"\b(appreciat\w*|prais\w*|liked?|enjoy\w*|happy with|satisf\w*|positive aspects?)\b", re.I)
NEG_CUES = re.compile(r"\b(issues?|problems?|complain\w*|complaints?|concerns?|difficult\w*|poor|fail\w*|trouble|"
                      r"warrant attention|inconsistent|problematic|not durable|negative aspects?)\b", re.I)
NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
                "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20, "thirty": 30, "half": 0.5}
REVIEW_ID_RX = re.compile(r"\bR[0-9a-f]{12}\b")


def norm(text: str) -> str:
    """Normalisation used ONLY for matching (quotes, case, whitespace, punctuation); stored text is never changed."""
    t = unicodedata.normalize("NFKC", text or "").lower()
    t = t.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    t = t.replace("_", " ")
    t = re.sub(r"[^\w%.' ]+", " ", t)
    t = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", t)  # drop sentence dots, keep decimal points
    return re.sub(r"\s+", " ", t).strip()


def quote_in_texts(quote: str, texts: list[str], min_words: int = 3) -> bool:
    """True if ``quote`` (allowing '...' gaps) is an exact substring, after normalisation, of one of ``texts``."""
    q = (quote or "").strip()
    if not q:
        return False
    parts = [norm(p) for p in re.split(r"\.\.\.|…", q)]
    parts = [p for p in parts if p]
    if not parts or sum(len(p.split()) for p in parts) < min_words and not any(norm(q) == norm(t) for t in texts):
        return False
    for t in texts:
        nt = norm(t)
        pos = 0
        ok = True
        for p in parts:
            i = nt.find(p, pos)
            if i < 0:
                ok = False
                break
            pos = i + len(p)
        if ok:
            return True
    return False


def aspects_in(text: str) -> list[str]:
    t = (text or "").replace("_", " ")
    return [a for a, rx in _ASPECT_RX.items() if rx.search(t)]


def complaints_in(text: str) -> list[str]:
    t = (text or "").replace("_", " ")
    return [c for c, rx in _COMPLAINT_RX.items() if rx.search(t)]


# --------------------------------------------------------------------------- statistics view
@dataclass
class Stats:
    """Read-only view of one product's statistics (recomputed by ``pipeline.aggregation``) plus per-review joins."""

    total: int
    sentiment_counts: dict[str, int]
    sentiment_pct: dict[str, float]
    average_rating: float | None
    reviews_with_complaints: int
    complaint_counts: dict[str, int]
    complaint_pct: dict[str, float]
    aspects: dict[str, dict]  # aspect -> {mentions, positive, negative, neutral}
    among_sentiment: dict[str, dict[str, float]] = field(default_factory=dict)  # "negative" -> {label: pct}

    @classmethod
    def from_aggregate(cls, agg, records=None) -> "Stats":
        aspects = {a.aspect_category: {"mentions": a.mentions, "positive": a.positive, "negative": a.negative, "neutral": a.neutral}
                   for a in agg.aspect_stats}
        among: dict[str, dict[str, float]] = {}
        if records:
            ids = set(agg.review_ids)
            recs = [r for r in records if r.review_id in ids]
            for s in ("positive", "negative"):
                grp = [r for r in recs if str(getattr(r.overall_sentiment, "value", r.overall_sentiment)) == s]
                if not grp:
                    continue
                shares: dict[str, float] = {}
                labels = set()
                for r in grp:
                    labels |= {str(getattr(c, "value", c)) for c in r.complaint_categories}
                    labels |= {str(getattr(a.aspect_category, "value", a.aspect_category)) for a in r.aspects}
                for lab in labels:
                    n = sum(1 for r in grp if lab in {str(getattr(c, "value", c)) for c in r.complaint_categories}
                            or lab in {str(getattr(a.aspect_category, "value", a.aspect_category)) for a in r.aspects})
                    shares[lab] = round(100 * n / len(grp), 1)
                among[s] = shares
        return cls(total=agg.total_reviews, sentiment_counts=dict(agg.sentiment_counts), sentiment_pct=dict(agg.sentiment_percentages),
                   average_rating=agg.average_rating, reviews_with_complaints=agg.reviews_with_complaints,
                   complaint_counts=dict(agg.complaint_counts), complaint_pct=dict(agg.complaint_percentages), aspects=aspects,
                   among_sentiment=among)

    def neg_count(self, aspect: str) -> int:
        a = self.aspects.get(aspect, {})
        return max(a.get("negative", 0), self.complaint_counts.get(ASPECT_TO_COMPLAINT.get(aspect, ""), 0))

    def pos_count(self, aspect: str) -> int:
        return self.aspects.get(aspect, {}).get("positive", 0)

    def render(self) -> str:
        """Plain-text statistics block used as evidence for opinion sentences no deterministic pattern covers."""
        lines = [f"Total reviews: {self.total}",
                 "Overall sentiment: " + ", ".join(f"{s} {self.sentiment_counts.get(s, 0)} of {self.total} ({self.sentiment_pct.get(s, 0.0)}%)"
                                                   for s in SENTIMENTS),
                 f"Average star rating: {self.average_rating}",
                 f"Reviews with at least one complaint: {self.reviews_with_complaints} of {self.total}",
                 "Complaints (reviews mentioning each): " + "; ".join(f"{c.replace('_', ' ')} {n} of {self.total} ({self.complaint_pct.get(c, 0)}%)"
                                                                      for c, n in sorted(self.complaint_counts.items(), key=lambda kv: -kv[1])),
                 "Aspect mentions (positive / negative / neutral): " + "; ".join(
                     f"{a.replace('_', ' ')} {v['positive']} / {v['negative']} / {v['neutral']}"
                     for a, v in sorted(self.aspects.items(), key=lambda kv: -kv[1]["mentions"]))]
        return "\n".join(lines)


def _close(x: float, c: float) -> bool:
    return abs(x - c) <= 0.5 or (float(x).is_integer() and round(c) == x)


def _fmt(x: float) -> str:
    return str(int(x)) if float(x).is_integer() else str(x)


# --------------------------------------------------------------------------- subjects of numbers
def _clip_before(text: str, start: int) -> str:
    seg = text[max(0, start - 120):start].rstrip()
    if seg.endswith("("):
        seg = seg[:-1]
    cut = max(seg.rfind(","), seg.rfind("("), seg.rfind(")"), seg.rfind(";"), seg.rfind(". "))
    return seg[cut + 1:] if cut >= 0 else seg


def _clip_after(text: str, end: int) -> str:
    seg = re.sub(r"^\s*(?:reviews?\s*)?(?:each\s*)?\)", "", text[end:end + 120])  # skip the ")" of "(7 out of 9)"
    m = re.search(r"[,;()]|\.\s|\.$|\d", seg)
    return seg[: m.start()] if m else seg


def _subject(window: str) -> dict:
    w = window.replace("_", " ")
    subj = {"sentiment": [s for s in SENTIMENTS if re.search(rf"\b{s}\b", w, re.I)],
            "with_complaints": bool(re.search(r"(?:with|containing|had|have|having)\s+(?:at least one\s+)?complaints?", w, re.I)),
            "complaints": complaints_in(w), "aspects": aspects_in(w),
            "polarity": "positive" if POS_CUES.search(w) else "negative" if NEG_CUES.search(w) else None}
    subj["found"] = bool(subj["sentiment"] or subj["with_complaints"] or subj["complaints"] or subj["aspects"])
    return subj


def _count_candidates(subj: dict, st: Stats) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    if subj["with_complaints"]:
        out.append(("reviews_with_complaints", st.reviews_with_complaints))
    for s in subj["sentiment"]:
        out.append((f"{s} reviews", st.sentiment_counts.get(s, 0)))
    for c in subj["complaints"]:
        if c in st.complaint_counts:
            out.append((f"complaint {c}", st.complaint_counts[c]))
    for a in subj["aspects"]:
        if a in st.aspects:
            if subj["polarity"] != "negative":
                out.append((f"{a} positive mentions", st.pos_count(a)))
            if subj["polarity"] != "positive":
                out.append((f"{a} negative mentions", st.neg_count(a)))
    return out


def _pct_candidates(subj: dict, st: Stats, of_sentiment: str | None) -> list[tuple[str, float]]:
    out: list[tuple[str, float]] = []
    for s in subj["sentiment"]:
        out.append((f"{s} share of all reviews", st.sentiment_pct.get(s, 0.0)))
    if subj["with_complaints"]:
        out.append(("share of reviews with complaints", round(100 * st.reviews_with_complaints / st.total, 1) if st.total else 0.0))
    for c in subj["complaints"]:
        if c in st.complaint_pct:
            out.append((f"complaint {c} share", st.complaint_pct[c]))
    for a in subj["aspects"]:
        if a in st.aspects and st.total:
            out.append((f"{a} mention share", round(100 * st.aspects[a]["mentions"] / st.total, 1)))
    if of_sentiment and of_sentiment in st.among_sentiment:
        labels = set(subj["complaints"]) | set(subj["aspects"]) or set(st.among_sentiment[of_sentiment])
        for lab in sorted(labels):
            if lab in st.among_sentiment[of_sentiment]:
                out.append((f"share of {of_sentiment} reviews mentioning {lab}", st.among_sentiment[of_sentiment][lab]))
    if not out:  # no recognisable subject: the number must match SOME statistic of this product
        out = [(f"{s} share of all reviews", v) for s, v in st.sentiment_pct.items()] + \
              [(f"complaint {c} share", v) for c, v in st.complaint_pct.items()]
    return out


COUNT_RX = re.compile(r"\b(\d+)\s+(?:(positive|negative|neutral|mixed)\s+)?(?:reviews?\s+|users?\s+|customers?\s+)?(?:out\s+of|of)\s+(?:the\s+)?(\d+)\b", re.I)
PCT_RX = re.compile(r"(\d+(?:\.\d+)?)\s*%")
AVG_RX = re.compile(r"(?:average|avg\.?)\s+(?:star\s+)?rating\s+(?:of\s+|is\s+|was\s+)?(\d(?:\.\d+)?)", re.I)
COMPARATIVE_RX = re.compile(r"^\s*(?:cheaper|less|more|off|faster|slower|smaller|larger|longer|shorter|better|worse|savings?)\b", re.I)


def numeric_checks_statistics(text: str, st: Stats) -> list[NumericCheck]:
    """Counts, percentages, denominators and average ratings in a claim, re-computed from the product statistics."""
    checks: list[NumericCheck] = []
    used: list[tuple[int, int]] = []
    for m in COUNT_RX.finditer(text):
        n, sent_word, d = int(m.group(1)), m.group(2), int(m.group(3))
        if d < n or d == 0:
            continue
        used.append(m.span())
        in_paren = text[:m.start()].rstrip().endswith("(")
        after, before = _clip_after(text, m.end()), _clip_before(text, m.start())
        if sent_word:
            subj = _subject(sent_word)
        else:
            first, second = (before, after) if in_paren else (after, before)
            subj = _subject(first)
            if not subj["found"]:
                subj = _subject(second)
        each = bool(re.match(r"\s*reviews?\s+each|\s*each", text[m.end():m.end() + 14], re.I))
        cands = _count_candidates(subj, st)
        if not cands:
            cands = [("any review count", v) for v in [st.reviews_with_complaints, *st.sentiment_counts.values(), *st.complaint_counts.values()]]
        hits = [c for c in cands if c[1] == n]
        if each:  # "(6 of 19 reviews each)": every named complaint (or, if none, every subject) must have exactly n
            relevant = [c for c in cands if c[0].startswith("complaint ")] or cands
            num_ok = all(c[1] == n for c in relevant)
        else:
            num_ok = bool(hits)
        denoms_ok = {st.total} | {st.sentiment_counts.get(s, -1) for s in subj["sentiment"]}  # "2 of the 2 negative reviews"
        den_ok = d in denoms_ok
        calc = (f"claimed {n} of {d}; candidates for the subject: " + ", ".join(f"{k} = {v}" for k, v in cands[:6]) +
                f"; product has {st.total} reviews, {st.reviews_with_complaints} with complaints")
        expected = float(hits[0][1]) if hits else float(cands[0][1])
        if num_ok and den_ok:
            sev, ok = "ok", True
        elif num_ok and d == st.reviews_with_complaints and d != st.total:
            sev, ok = "misleading", False
            calc += f"; denominator {d} is the number of reviews WITH complaints, not the {st.total} reviews"
        else:
            sev, ok = "error", False
            if not den_ok:
                calc += f"; denominator {d} matches no review total (expected {st.total})"
        checks.append(NumericCheck(mention=m.group(0), kind="count", extracted_value=n, expected_value=expected,
                                   denominator=st.total, calculation=calc, passed=ok, severity=sev))
    for m in PCT_RX.finditer(text):
        if any(a <= m.start() < b for a, b in used):
            continue
        x = float(m.group(1))
        if COMPARATIVE_RX.match(text[m.end():]):
            continue  # "40% cheaper": a quoted comparison, checked against review text (number_checks_in_texts)
        after, before = _clip_after(text, m.end()), _clip_before(text, m.start())
        of_sent = None
        mo = re.match(r"\s*of\s+(?:the\s+)?(positive|negative)\s+reviews", text[m.end():], re.I)
        if mo:
            of_sent = mo.group(1).lower()
        subj = _subject(after)
        if not subj["found"]:
            subj = _subject(before)
        cands = _pct_candidates(subj, st, of_sent)
        if of_sent and not subj["sentiment"]:
            cands.insert(0, (f"{of_sent} share of all reviews", st.sentiment_pct.get(of_sent, 0.0)))
        hits = [c for c in cands if _close(x, c[1])]
        calc = f"claimed {_fmt(x)}%; " + ", ".join(f"{k} = {v}%" for k, v in cands[:8]) + f" (tolerance 0.5 points; n = {st.total})"
        checks.append(NumericCheck(mention=m.group(0), kind="percentage", extracted_value=x,
                                   expected_value=float(hits[0][1]) if hits else float(cands[0][1]), denominator=st.total,
                                   calculation=calc, passed=bool(hits), severity="ok" if hits else "error"))
    for m in AVG_RX.finditer(text):
        x = float(m.group(1))
        exp = st.average_rating
        ok = exp is not None and abs(round(exp, 2) - x) <= 0.05
        checks.append(NumericCheck(mention=m.group(0), kind="average_rating", extracted_value=x, expected_value=exp,
                                   denominator=st.total, calculation=f"mean star rating over {st.total} reviews = {exp}",
                                   passed=ok, severity="ok" if ok else "error"))
    return checks


def _numbers_in(text: str) -> list[tuple[str, float]]:
    t = REVIEW_ID_RX.sub(" ", text)
    out = []
    for m in re.finditer(r"(?<![\w.-])(\d+(?:\.\d+)?)(?![\w-]*\d{3,})(?=[\s%\-.,)]|$)", t):
        tok = m.group(1)
        if re.search(r"[A-Za-z]\d|\d[A-Za-z]", t[max(0, m.start() - 1):m.end() + 1]):
            continue  # part of a model number such as LT700P
        v = float(tok)
        if v >= 1000:
            continue  # part / model numbers, years
        out.append((m.group(0), v))
    for w, v in NUMBER_WORDS.items():
        if w == "half":
            continue
        for m in re.finditer(rf"\b{w}\b", t, re.I):
            out.append((m.group(0), float(v)))
    return out


def number_checks_in_texts(text: str, texts: list[str]) -> list[NumericCheck]:
    """Every specific number in a point (e.g. '40% cheaper', '5 minutes', 'two-year') must occur in the cited review text."""
    checks = []
    blob = " ".join(norm(t) for t in texts)
    present = {float(x) for x in re.findall(r"(?<![\w.])(\d+(?:\.\d+)?)", blob)}
    present |= {float(v) for w, v in NUMBER_WORDS.items() if re.search(rf"\b{w}\b", blob)}
    if re.search(r"\b(a|one) month\b", blob):
        present.add(1.0)
    seen = set()
    for mention, v in _numbers_in(text):
        if v in seen:
            continue
        seen.add(v)
        ok = v in present
        checks.append(NumericCheck(mention=mention, kind="number_in_cited_text", extracted_value=v, expected_value=v if ok else None,
                                   calculation=("found in the cited review text" if ok else "this number does not occur in any cited review text"),
                                   passed=ok, severity="ok" if ok else "misleading"))
    return checks


# --------------------------------------------------------------------------- qualitative statistical propositions
def _f(check: str, outcome: str, cap: ClaimStatus | None, detail: str) -> DeterministicFinding:
    return DeterministicFinding(check=check, outcome=outcome, status_cap=cap, detail=detail)


def _share(st: Stats, s: str) -> float:
    return st.sentiment_pct.get(s, 0.0)


def _direction(st: Stats, s: str, check: str, phrase: str, require_majority: bool, slight: bool = False) -> DeterministicFinding:
    opp = "negative" if s == "positive" else "positive"
    sh, osh = _share(st, s), _share(st, opp)
    calc = f"{s} {st.sentiment_counts.get(s, 0)}/{st.total} = {sh}%, {opp} {st.sentiment_counts.get(opp, 0)}/{st.total} = {osh}%"
    if require_majority:
        if sh > 50:
            if slight and sh >= 65:
                return _f(check, "misleading", ClaimStatus.PARTIALLY_SUPPORTED, f"'{phrase}': {calc}; 'slight' understates a {sh}% majority")
            return _f(check, "pass", None, f"'{phrase}': {calc}")
        cap = ClaimStatus.CONTRADICTED
        return _f(check, "fail", cap, f"'{phrase}' claims a {s} majority but {calc}")
    if sh > osh:
        if slight and sh >= 65:
            return _f(check, "misleading", ClaimStatus.PARTIALLY_SUPPORTED, f"'{phrase}': {calc}; 'slight' understates a {sh}% share")
        return _f(check, "pass", None, f"'{phrase}': {calc}")
    if sh == osh:
        return _f(check, "misleading", ClaimStatus.PARTIALLY_SUPPORTED, f"'{phrase}': {calc} (tie)")
    return _f(check, "fail", ClaimStatus.CONTRADICTED, f"'{phrase}' claims a {s} lean but {calc}")


def _clauses(text: str) -> list[str]:
    parts = re.split(r",?\s+\b(?:but|however|though|although|while|whereas)\b\s*|;\s*|\.\s+", text, flags=re.I)
    return [p for p in parts if p and p.strip()]


def qualitative_checks(text: str, st: Stats, *, polarity_hint: str | None = None, aspect_statements: bool = True) -> list[DeterministicFinding]:
    out: list[DeterministicFinding] = []
    t = text.replace("_", " ")
    covered: list[tuple[int, int]] = []

    def mark(m):
        covered.append(m.span())

    for m in re.finditer(r"\b(slight|small|narrow)?\s*majority of (?:the\s+)?(positive|negative) (?:reviews|feedback|ratings|sentiments?)", t, re.I):
        out.append(_direction(st, m.group(2).lower(), "sentiment_majority", m.group(0).strip(), True, bool(m.group(1))))
        mark(m)
    for m in re.finditer(r"\b(slight|small|narrow)?\s*majority of (?:the\s+)?(?:reviews|users|customers|reviewers|ratings|feedback)\s+"
                         r"(?:\(\d+ (?:out )?of \d+\)\s+)?(?:being\s+|are\s+|is\s+|were\s+)?(positive|negative)\b", t, re.I):
        out.append(_direction(st, m.group(2).lower(), "sentiment_majority", m.group(0).strip(), True, bool(m.group(1))))
        mark(m)
    for m in re.finditer(r"\bmajority of (?:the\s+)?(?:reviews|users|customers|reviewers)\s+(?:are\s+|were\s+)?"
                         r"(experiencing|reporting|mentioning|complaining about|citing|praising)\s+([^,.;]{3,90})", t, re.I):
        verb, obj = m.group(1).lower(), m.group(2)
        pos = verb == "praising"
        labs = aspects_in(obj)
        if not labs:
            continue
        mark(m)
        bad = []
        for a in labs:
            n = st.pos_count(a) if pos else st.neg_count(a)
            if n * 2 <= st.total:
                bad.append(f"{a}: {n}/{st.total} = {round(100 * n / st.total, 1) if st.total else 0}%")
        if bad:
            out.append(_f("aspect_majority", "fail", ClaimStatus.CONTRADICTED, f"'{m.group(0).strip()}' but " + "; ".join(bad) + " (not > 50%)"))
        else:
            out.append(_f("aspect_majority", "pass", None, f"'{m.group(0).strip()}': each > 50% of {st.total}"))
    for m in re.finditer(r"\b(predominantly|mostly|largely|overwhelmingly)\s+(positive|negative)\b", t, re.I):
        out.append(_direction(st, m.group(2).lower(), "sentiment_majority", m.group(0), True))
        mark(m)
    for m in re.finditer(r"\b(?:generally|overall|largely|widely)\s+well[- ]received\b|\breceived (?:a|an) (?:generally |overall )?positive\b|"
                         r"\bpositive overall sentiment\b|\bgenerally positive\b", t, re.I):
        f = _direction(st, "positive", "overall_positive", m.group(0), True)
        if f.outcome == "fail" and _share(st, "positive") > _share(st, "negative"):
            f = _f("overall_positive", "misleading", ClaimStatus.PARTIALLY_SUPPORTED, f.detail + " (plurality, not majority)")
        out.append(f)
        mark(m)
    for m in re.finditer(r"\bgenerally negative\b|\bnegative user experience\b|\bpredominantly negative\b", t, re.I):
        if any(a <= m.start() < b for a, b in covered):
            continue
        out.append(_direction(st, "negative", "overall_negative", m.group(0), False))
        mark(m)
    for m in re.finditer(r"\b(?:slightly|slight(?:ly)? leaning towards|slight lean towards|leaning towards|leaning)\s+(positive|negative)\b", t, re.I):
        out.append(_direction(st, m.group(1).lower(), "sentiment_lean", m.group(0), False, slight="slight" in m.group(0).lower()))
        mark(m)
    for m in re.finditer(r"\bmore (?:reviews|reviewers|customers) (?:are|were) (positive|negative) than (positive|negative)\b|"
                         r"\bmore (positive|negative) (?:than (positive|negative) )?reviews(?: than (positive|negative)(?: reviews)?)?\b", t, re.I):
        s = (m.group(1) or m.group(3)).lower()
        out.append(_direction(st, s, "sentiment_comparison", m.group(0), False))
        mark(m)
    for m in re.finditer(r"\bmixed(?: to (positive|negative))?\s+(?:overall\s+)?(?:reception|feedback|reviews|sentiment|response|opinions?)\b", t, re.I):
        mark(m)
        side = (m.group(1) or "").lower()
        top = max(("positive", "negative"), key=lambda s: _share(st, s))
        calc = f"positive {_share(st, 'positive')}%, negative {_share(st, 'negative')}%, mixed {_share(st, 'mixed')}%"
        if side:
            ok = side == top
            out.append(_f("mixed_sentiment", "pass" if ok else "fail", None if ok else ClaimStatus.CONTRADICTED, f"'{m.group(0)}': {calc}"))
        elif _share(st, top) >= 75:
            out.append(_f("mixed_sentiment", "misleading", ClaimStatus.PARTIALLY_SUPPORTED, f"'{m.group(0)}' but {calc}: {_share(st, top)}% {top} is one-sided"))
        else:
            out.append(_f("mixed_sentiment", "pass", None, f"'{m.group(0)}': {calc}"))
    # most frequently mentioned aspects
    for m in re.finditer(r"most (?:frequently|commonly|often) mentioned aspects? (?:were|are|was|is|include[sd]?)\s+([^.;]+)", t, re.I):
        mark(m)
        out.append(_frequent_aspects(aspects_in(m.group(1)), st, m.group(0)))
    # most common complaint
    for m in re.finditer(r"([^.;,]{3,80}?)\s+(?:is|are|was|were)\s+the most (?:common|frequent|frequently mentioned) (?:complaints?|issues?|problems?)|"
                         r"the most (?:common|frequent) (?:complaints?|issues?|problems?) (?:is|are|was|were)\s+([^.;]{3,80})", t, re.I):
        subj = m.group(1) or m.group(2)
        mark(m)
        labs = complaints_in(subj) or [ASPECT_TO_COMPLAINT.get(a, "") for a in aspects_in(subj)]
        labs = [x for x in labs if x]
        if not labs or not st.complaint_counts:
            continue
        top = max(st.complaint_counts.values())
        bad = [f"{c} = {st.complaint_counts.get(c, 0)}" for c in labs if st.complaint_counts.get(c, 0) < top]
        tops = [c for c, n in st.complaint_counts.items() if n == top]
        if bad:
            out.append(_f("most_common_complaint", "fail", ClaimStatus.CONTRADICTED, f"'{m.group(0).strip()}' but " + ", ".join(bad) + f"; most common = {tops} ({top})"))
        else:
            out.append(_f("most_common_complaint", "pass", None, f"most common complaint(s) {tops} = {top}"))
    # universal quantifiers ("all reviewers", "every user", "no customer complained")
    for m in re.finditer(r"\b(?:all|every) (?:reviewers?|users?|customers?|buyers?)\b|\beveryone\b|\bno (?:reviewers?|users?|customers?) "
                         r"(?:complain\w*|report\w*|mention\w*)", t, re.I):
        mark(m)
        pol = "positive" if POS_CUES.search(t) else "negative" if NEG_CUES.search(t) else polarity_hint
        if pol is None or pol == "neutral":  # "All reviewers agree that the part fits": polarity from negation words
            pol = "negative" if re.search(r"(?:not|no|never|n't|poor\w*|difficult|hard|broke\w*|late|damaged|noisy|unhelpful)|n't", t, re.I) else "positive"
        if m.group(0).lower().startswith("no "):
            pol = "positive"
        labs = aspects_in(t)
        opp = 0
        detail = []
        if labs:
            for a in labs:
                o = st.neg_count(a) if pol == "positive" else st.pos_count(a)
                opp += o
                detail.append(f"{a}: {st.pos_count(a)} positive / {st.neg_count(a)} negative")
        else:
            o_s = "negative" if pol == "positive" else "positive"
            opp = st.sentiment_counts.get(o_s, 0)
            detail.append(f"{o_s} reviews = {opp}")
        if pol and opp > 0:
            out.append(_f("universal_quantifier", "fail", ClaimStatus.CONTRADICTED, f"'{m.group(0)}' but the statistics show the opposite view: " + "; ".join(detail)))
        else:
            out.append(_f("universal_quantifier", "pass", None, f"'{m.group(0)}': " + "; ".join(detail)))
    if aspect_statements:
        out += aspect_statement_checks(t, st, covered)
    return out


def aspect_statement_checks(text: str, st: Stats, covered: list[tuple[int, int]] | None = None) -> list[DeterministicFinding]:
    """'Users appreciate X ... but face issues with Y': X needs positive and Y negative mentions in the statistics."""
    out = []
    for cl in _clauses(text):
        if re.search(r"most (?:frequently|commonly|often) mentioned|majority of|\d", cl, re.I):
            continue  # frequency / majority / numeric clauses are checked by the specific rules above
        pos, neg = bool(POS_CUES.search(cl)), bool(NEG_CUES.search(cl))
        if pos == neg:
            continue
        labs = aspects_in(cl)
        for a in labs:
            p, n = st.pos_count(a), st.neg_count(a)
            have, opp = (p, n) if pos else (n, p)
            word = "positive" if pos else "negative"
            calc = f"{a}: {p} positive / {n} negative mentions"
            if a not in st.aspects and not (neg and n):
                out.append(_f("aspect_statement", "fail", ClaimStatus.UNSUPPORTED, f"'{cl.strip()}': no review mentions {a}"))
            elif have == 0:
                out.append(_f("aspect_statement", "fail", ClaimStatus.CONTRADICTED if opp else ClaimStatus.UNSUPPORTED,
                              f"'{cl.strip()}' states a {word} view of {a} but {calc}"))
            elif have < opp and have / (have + opp) < 0.25:
                out.append(_f("aspect_statement", "misleading", ClaimStatus.PARTIALLY_SUPPORTED,
                              f"'{cl.strip()}': {calc}; the {word} view is a small minority"))
            else:
                out.append(_f("aspect_statement", "pass", None, f"{word} view of {a}: {calc}"))
    return out


def _frequent_aspects(listed: list[str], st: Stats, phrase: str) -> DeterministicFinding:
    ranked = [a for a, _ in sorted(st.aspects.items(), key=lambda kv: -kv[1]["mentions"])]
    k = len(listed) + 2
    missing = [a for a in listed if a not in st.aspects]
    low = [a for a in listed if a in st.aspects and a not in ranked[:k]]
    detail = "ranking by mentions: " + ", ".join(f"{a} {st.aspects[a]['mentions']}" for a in ranked[:k + 2])
    if missing:
        return _f("frequent_aspects", "fail", ClaimStatus.UNSUPPORTED, f"'{phrase[:80]}': never mentioned: {missing}; {detail}")
    if low:
        return _f("frequent_aspects", "misleading", ClaimStatus.PARTIALLY_SUPPORTED, f"'{phrase[:80]}': not among the top {k}: {low}; {detail}")
    return _f("frequent_aspects", "pass", None, detail)


def frequent_aspects_check(listed: list[str], st: Stats) -> list[DeterministicFinding]:
    norm_listed = [a.strip().lower().replace(" ", "_") for a in listed]
    out = [_frequent_aspects(list(dict.fromkeys(norm_listed)), st, "frequent_aspects: " + ", ".join(listed))]
    if len(set(norm_listed)) != len(norm_listed):
        out.append(_f("frequent_aspects_duplicates", "info", None, "the list repeats an aspect"))
    return out


def derived_claim_check(text: str, st: Stats, verified_negative: dict[str, str] | None = None) -> tuple[ClaimStatus | None, list[DeterministicFinding]]:
    """Improvement areas / recommendations imply a problem (summariser rule 6): they need negative evidence for the aspect
    they target, either in the statistics or in a complaint point of the same summary that the verifier found supported
    (``verified_negative``: aspect -> claim id). The second source matters when the extraction missed a complaint."""
    verified_negative = verified_negative or {}
    labs = aspects_in(text) or [a for c in complaints_in(text) for a, cc in ASPECT_TO_COMPLAINT.items() if cc == c]
    labs = list(dict.fromkeys(labs))
    if not labs:
        return None, [_f("derived_aspect_evidence", "info", None, "no product aspect recognised: advice that cannot be checked against the statistics")]
    findings = []
    supported = []
    for a in labs:
        n, p = st.neg_count(a), st.pos_count(a)
        if n > 0:
            supported.append(a)
            findings.append(_f("derived_aspect_evidence", "pass", None, f"{a}: {n} negative mention(s) / complaint(s), {p} positive"))
        elif a in verified_negative:
            supported.append(a)
            findings.append(_f("derived_aspect_evidence", "pass", None,
                               f"{a}: 0 negative mentions in the statistics, but complaint point {verified_negative[a]} was verified against its cited reviews"))
        else:
            findings.append(_f("derived_aspect_evidence", "fail", ClaimStatus.UNSUPPORTED,
                               f"targets {a}, but no review complains about it ({p} positive / 0 negative)"))
    if supported:
        # at least one targeted aspect has negative evidence: drop the cap of the others but keep them as information
        findings = [f if f.outcome == "pass" else _f(f.check, "info", None, f.detail) for f in findings]
        return ClaimStatus.SUPPORTED, findings
    return ClaimStatus.UNSUPPORTED, findings


# --------------------------------------------------------------------------- citations
def citation_evidence(cited: list[str], product_id: str, reviews: dict[str, dict]) -> tuple[list[EvidenceExcerpt], list[DeterministicFinding]]:
    ev, findings = [], []
    for rid in cited:
        r = reviews.get(rid)
        ok_p = bool(r) and r["product_id"] == product_id
        ev.append(EvidenceExcerpt(review_id=rid, exists=bool(r), belongs_to_product=ok_p, rating=(r or {}).get("rating")))
        if not r:
            findings.append(_f("citation_exists", "fail", None, f"{rid} does not exist in the review dataset"))
        elif not ok_p:
            findings.append(_f("citation_product", "fail", None, f"{rid} belongs to {r['product_id']}, not {product_id}"))
    valid = [e for e in ev if e.exists and e.belongs_to_product]
    if cited and not valid:
        findings.append(_f("citation_valid", "fail", ClaimStatus.UNSUPPORTED, "no cited review id is valid for this product"))
    elif len(valid) < len(cited):
        findings.append(_f("citation_valid", "fail", ClaimStatus.PARTIALLY_SUPPORTED, f"{len(cited) - len(valid)} of {len(cited)} cited ids are invalid"))
    else:
        findings.append(_f("citation_valid", "pass", None, f"{len(valid)} cited id(s) exist and belong to {product_id}"))
    return ev, findings


def rating_polarity_finding(polarity: str, evidence: list[EvidenceExcerpt]) -> DeterministicFinding | None:
    """Information only (star ratings are a weak signal for one aspect): praise citing only 1-2 star reviews, or a complaint
    citing only 5-star reviews."""
    rs = [e.rating for e in evidence if e.rating is not None]
    if not rs:
        return None
    if polarity == "positive" and max(rs) <= 2:
        return _f("citation_rating_polarity", "info", None, f"praise cites only low-rated reviews (ratings {rs})")
    if polarity == "negative" and min(rs) >= 5:
        return _f("citation_rating_polarity", "info", None, f"complaint cites only 5-star reviews (ratings {rs})")
    return None
