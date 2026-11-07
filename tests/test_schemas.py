import pytest
from pydantic import ValidationError

from schemas.review_schema import Aspect, LLMExtraction, evidence_is_grounded
from schemas.taxonomy import (
    ASPECT_CATEGORY_DEFINITIONS,
    COMPLAINT_CATEGORY_DEFINITIONS,
    OVERALL_SENTIMENT_DEFINITIONS,
    AspectCategory,
    ComplaintCategory,
    OverallSentiment,
    taxonomy_prompt_block,
)
from tests.conftest import REVIEW_TEXT, valid_extraction

CTX = {"review_text": REVIEW_TEXT}


def test_taxonomy_definitions_cover_every_enum_member():
    assert set(ASPECT_CATEGORY_DEFINITIONS) == set(AspectCategory)
    assert set(COMPLAINT_CATEGORY_DEFINITIONS) == set(ComplaintCategory)
    assert set(OVERALL_SENTIMENT_DEFINITIONS) == set(OverallSentiment)
    block = taxonomy_prompt_block()
    assert all(m.value in block for m in ComplaintCategory)


def test_valid_multi_aspect_mixed_review():
    ex = LLMExtraction.model_validate(valid_extraction(), context=CTX)
    assert ex.overall_sentiment == OverallSentiment.MIXED
    assert len(ex.aspects) == 3
    assert {a.sentiment.value for a in ex.aspects} == {"positive", "negative"}


def test_review_without_aspects_is_valid():
    ex = LLMExtraction.model_validate(valid_extraction(overall_sentiment="neutral", aspects=[], complaint_categories=[]), context=CTX)
    assert ex.aspects == [] and ex.complaint_categories == []


@pytest.mark.parametrize("bad", ["very good", "POSITIVE!", "", 3])
def test_invalid_overall_sentiment_rejected(bad):
    with pytest.raises(ValidationError):
        LLMExtraction.model_validate(valid_extraction(overall_sentiment=bad), context=CTX)


def test_aspect_sentiment_cannot_be_mixed():
    data = valid_extraction()
    data["aspects"][0]["sentiment"] = "mixed"
    with pytest.raises(ValidationError):
        LLMExtraction.model_validate(data, context=CTX)


def test_missing_required_field_rejected():
    data = valid_extraction()
    del data["confidence"]
    with pytest.raises(ValidationError, match="confidence"):
        LLMExtraction.model_validate(data, context=CTX)


def test_extra_fields_forbidden():
    with pytest.raises(ValidationError):
        LLMExtraction.model_validate(valid_extraction(reasoning="step 1 ..."), context=CTX)


def test_confidence_bounds():
    with pytest.raises(ValidationError):
        LLMExtraction.model_validate(valid_extraction(confidence=1.5), context=CTX)


def test_ungrounded_evidence_rejected():
    data = valid_extraction()
    data["aspects"][0]["evidence"] = "the water filter tastes amazing"
    with pytest.raises(ValidationError, match="not a verbatim quote"):
        LLMExtraction.model_validate(data, context=CTX)


def test_grounding_tolerates_case_and_punctuation():
    assert evidence_is_grounded("THE ICE MAKER works great!!", REVIEW_TEXT)
    assert evidence_is_grounded("works great ... very loud at night", REVIEW_TEXT)
    assert not evidence_is_grounded("terrible customer service", REVIEW_TEXT)


def test_grounding_skipped_without_context():
    data = valid_extraction()
    data["aspects"][0]["evidence"] = "not in any text"
    LLMExtraction.model_validate(data)  # no context -> structural validation only


def test_duplicate_aspects_and_complaints_removed():
    data = valid_extraction(complaint_categories=["noise_issue", "noise_issue"])
    data["aspects"].append(dict(data["aspects"][0]))
    ex = LLMExtraction.model_validate(data, context=CTX)
    assert len(ex.aspects) == 3
    assert ex.complaint_categories == [ComplaintCategory.NOISE_ISSUE]


def test_aspect_name_normalised():
    a = Aspect(aspect_name="  Ice   Maker ", aspect_category="performance", sentiment="positive", evidence="works great", confidence=0.5)
    assert a.aspect_name == "ice maker"
