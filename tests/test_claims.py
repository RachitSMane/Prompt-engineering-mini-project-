"""Judge v4 deterministic layer: claim decomposition, numeric / statistical checks, citations, quote matching. No LLM."""

from __future__ import annotations

import pytest

from evaluation.claim_checks import (
    Stats, citation_evidence, derived_claim_check, frequent_aspects_check, number_checks_in_texts, numeric_checks_statistics,
    qualitative_checks, quote_in_texts,
)
from evaluation.claims import decompose, split_sentences
from schemas.claim_schema import ClaimStatus
from schemas.evaluation_schema import AspectStat, ProductAggregate, ProductSummaryContent, SummaryPoint


def make_agg(**kw) -> ProductAggregate:
    base = dict(product_id="P1", total_reviews=15, sentiment_counts={"positive": 13, "negative": 2, "neutral": 0, "mixed": 0},
                sentiment_percentages={"positive": 86.7, "negative": 13.3, "neutral": 0.0, "mixed": 0.0}, average_rating=4.2,
                aspect_stats=[AspectStat(aspect_category="performance", mentions=10, positive=8, negative=2, neutral=0, net_sentiment=0.6),
                              AspectStat(aspect_category="quality_durability", mentions=3, positive=3, negative=0, neutral=0, net_sentiment=1.0),
                              AspectStat(aspect_category="installation_setup", mentions=5, positive=4, negative=1, neutral=0, net_sentiment=0.6),
                              AspectStat(aspect_category="compatibility", mentions=4, positive=1, negative=3, neutral=0, net_sentiment=-0.5),
                              AspectStat(aspect_category="price_value", mentions=4, positive=3, negative=1, neutral=0, net_sentiment=0.5)],
                complaint_counts={"compatibility_fit_issue": 2, "poor_value": 1}, complaint_percentages={"compatibility_fit_issue": 13.3, "poor_value": 6.7},
                reviews_with_complaints=3, review_ids=["Ra", "Rb", "Rc"])
    base.update(kw)
    return ProductAggregate(**base)


ST = Stats.from_aggregate(make_agg())
NEG = Stats.from_aggregate(make_agg(total_reviews=10, sentiment_counts={"positive": 4, "negative": 5, "neutral": 1, "mixed": 0},
                                    sentiment_percentages={"positive": 40.0, "negative": 50.0, "neutral": 10.0, "mixed": 0.0}))


def sev(text, st=ST):
    return [n.severity for n in numeric_checks_statistics(text, st)]


# --------------------------------------------------------------------------- decomposition
CONTENT = ProductSummaryContent(
    overall_opinion="The product is well liked. 13 of the 15 reviews are positive. Users appreciate the installation.",
    appreciated=[SummaryPoint(point="Installation was easy.", supporting_review_ids=["Ra", "Rb"])],
    complaints=[SummaryPoint(point="It does not fit some models.", supporting_review_ids=["Rc"])],
    frequent_aspects=["performance", "installation_setup"], improvement_areas=["Improve compatibility"], recommendations=["Check the model number"])


def test_every_point_becomes_one_claim_with_its_citations():
    claims = decompose("S1", "P1", CONTENT)
    pts = [c for c in claims if c.kind == "point"]
    assert [(c.text, c.cited_review_ids, c.polarity) for c in pts] == [
        ("Installation was easy.", ["Ra", "Rb"], "positive"), ("It does not fit some models.", ["Rc"], "negative")]


def test_claims_are_verbatim_and_never_invented():
    claims = decompose("S1", "P1", CONTENT)
    blob = " ".join([CONTENT.overall_opinion, *(p.point for p in CONTENT.appreciated + CONTENT.complaints),
                     *CONTENT.improvement_areas, *CONTENT.recommendations])
    for c in claims:
        if c.source_field == "frequent_aspects":
            assert c.text.endswith(", ".join(CONTENT.frequent_aspects))
        else:
            assert c.text in blob
    assert len(claims) == 3 + 2 + 1 + 1 + 1


def test_claim_ids_stable_and_product_associated():
    a, b = decompose("S1", "P1", CONTENT), decompose("S1", "P1", CONTENT)
    assert [c.claim_id for c in a] == [c.claim_id for c in b]
    assert all(c.product_id == "P1" and c.summary_id == "S1" for c in a)
    assert len({c.claim_id for c in a}) == len(a)


def test_sentence_split_keeps_decimals():
    assert split_sentences("Average rating is 3.38 here. Second one.") == ["Average rating is 3.38 here.", "Second one."]


# --------------------------------------------------------------------------- numeric checks
def test_misattributed_percentages_fail():
    assert sev("The aspects were performance and ease of use, with positive sentiments from 80% of the reviews.") == ["error"]
    assert sev("with 67% of negative reviews mentioning these aspects.") == ["error"]


def test_correct_percentage_and_rounding_pass():
    assert sev("86.7% of the reviews are positive.") == ["ok"]
    assert sev("87% of the reviews are positive.") == ["ok"]  # rounded integer of 86.7
    assert sev("13.3% negative") == ["ok"]


def test_counts_and_denominators():
    assert sev("13 of the 15 reviews are positive.") == ["ok"]
    assert sev("3 positive reviews out of 15") == ["error"]          # wrong count
    assert sev("13 of the 20 reviews are positive.") == ["error"]    # wrong denominator
    assert sev("compatibility fit issues (2 out of 3 reviews)") == ["misleading"]  # 3 = reviews with complaints
    assert sev("3 out of 15 reviews containing complaints") == ["ok"]
    assert sev("both negative reviews (2 of 2) mentioning these aspects") == ["ok"]


def test_count_attributed_to_aspects_is_checked():
    assert sev("well received, as indicated by the majority of reviews (13 out of 15) praising its performance and price value.") == ["error"]
    assert sev("with 8 out of 15 reviews praising its performance") == ["ok"]


def test_average_rating():
    assert sev("an average star rating of 4.2") == ["ok"]
    assert sev("an average star rating of 3.9") == ["error"]


def test_comparative_percent_checked_against_review_text_not_statistics():
    assert numeric_checks_statistics("It is 40% cheaper than the brand filter.", ST) == []
    ok = number_checks_in_texts("It is 40% cheaper than the brand filter.", ["Saves us money, at 40% less expensive"])
    bad = number_checks_in_texts("It comes with a two-year warranty.", ["Works fine and arrived quickly."])
    assert [n.passed for n in ok] == [True] and [n.passed for n in bad] == [False]
    assert bad[0].severity == "misleading"


def test_model_numbers_and_review_ids_are_not_numbers():
    assert number_checks_in_texts("Compatible with LG LT700P, as in [Rf06c65f9d7b5].", ["fits"]) == []


# --------------------------------------------------------------------------- qualitative statistics
def outcomes(text, st=ST):
    return [(f.check, f.outcome) for f in qualitative_checks(text, st) if f.outcome != "info"]


def test_majority_direction():
    assert ("sentiment_majority", "fail") in outcomes("mixed feedback with a slight majority of positive reviews", NEG)
    assert ("sentiment_majority", "pass") in outcomes("a majority of positive reviews")
    assert ("overall_positive", "fail") in outcomes("the product is generally well-received", NEG)
    assert ("sentiment_comparison", "pass") in outcomes("More reviews are negative than positive.", NEG)
    assert ("sentiment_comparison", "fail") in outcomes("More reviews are positive than negative.", NEG)


def test_slight_lean_and_mixed():
    assert ("sentiment_lean", "fail") in outcomes("a neutral to slightly positive tone", NEG)
    assert ("mixed_sentiment", "misleading") in outcomes("The product received mixed reviews.")  # 86.7% positive is one-sided


def test_aspect_majority_claim():
    assert ("aspect_majority", "fail") in outcomes("a majority of users experiencing compatibility issues")


def test_aspect_statements():
    assert outcomes("Users appreciate the ease of installation, but face issues with compatibility.") == [
        ("aspect_statement", "pass"), ("aspect_statement", "pass")]
    assert ("aspect_statement", "fail") in outcomes("Users complain about durability.")  # 3 positive / 0 negative


def test_universal_quantifier():
    assert ("universal_quantifier", "fail") in outcomes("All reviewers agree that the part fits.")  # compatibility has negatives


def test_frequent_aspects():
    assert frequent_aspects_check(["performance", "installation_setup"], ST)[0].outcome == "pass"
    assert frequent_aspects_check(["noise"], ST)[0].outcome == "fail"


def test_derived_claim_needs_negative_evidence():
    assert derived_claim_check("Improve compatibility with more models", ST)[0] == ClaimStatus.SUPPORTED
    assert derived_claim_check("Enhance durability of the material", ST)[0] == ClaimStatus.UNSUPPORTED
    assert derived_claim_check("Enhance durability", ST, {"quality_durability": "S1:complaints:0"})[0] == ClaimStatus.SUPPORTED
    assert derived_claim_check("Read other reviews first", ST)[0] is None


# --------------------------------------------------------------------------- citations and quotes
REVIEWS = {"Ra": {"product_id": "P1", "rating": 5.0, "title": "Great", "text": "Easy to install in five minutes."},
           "Rx": {"product_id": "P2", "rating": 1.0, "title": "Bad", "text": "Broke."}}


def test_citation_validation():
    ev, f = citation_evidence(["Ra"], "P1", REVIEWS)
    assert ev[0].exists and ev[0].belongs_to_product and [x.outcome for x in f] == ["pass"]
    ev, f = citation_evidence(["Rx", "Rzz"], "P1", REVIEWS)
    assert any(x.status_cap == ClaimStatus.UNSUPPORTED for x in f)
    assert not ev[0].belongs_to_product and not ev[1].exists


@pytest.mark.parametrize("quote,ok", [
    ("Easy to install in five minutes", True), ("easy to   INSTALL in five minutes.", True), ("Easy to ... five minutes", True),
    ("Hard to install in five minutes", False), ("Easy", False), ("", False)])
def test_quote_must_be_exact_substring(quote, ok):
    assert quote_in_texts(quote, ["Easy to install in five minutes."]) is ok
