"""JSON parsing, deterministic fixes and the LLM repair loop."""

import json

from llm.base import LLMError
from pipeline.extraction import ReviewExtractor
from pipeline.repair import generate_structured
from pipeline.validation import extract_json_candidate, normalize_extraction, parse_json
from prompts.strategies import get_strategy
from schemas.review_schema import ExtractionFailure, LLMExtraction, ReviewRecord
from tests.conftest import REVIEW_TEXT, ScriptedLLM, valid_extraction


# ---------------------------------------------------------------- parsing
def test_parse_plain_json(valid_json):
    obj, fixes, err = parse_json(valid_json)
    assert err is None and fixes == [] and obj["overall_sentiment"] == "mixed"


def test_parse_json_inside_prose_and_fence(valid_json):
    obj, fixes, err = parse_json(f"Sure, here you go:\n```json\n{valid_json}\n```\nHope this helps!")
    assert err is None and obj["confidence"] == 0.88
    assert "removed markdown code fence" in fixes


def test_parse_trailing_comma_fixed_locally():
    obj, fixes, err = parse_json('{"a": [1, 2,], "b": 3,}')
    assert err is None and obj == {"a": [1, 2], "b": 3}
    assert "removed trailing commas" in fixes


def test_braces_inside_strings_do_not_break_extraction():
    cand, _ = extract_json_candidate('{"evidence": "a } tricky { quote", "x": 1} trailing')
    assert json.loads(cand)["x"] == 1


def test_truncated_json_reports_error():
    obj, _, err = parse_json('{"overall_sentiment": "positive", "aspects": [')
    assert obj is None and "invalid JSON" in err


def test_no_json_at_all():
    obj, _, err = parse_json("I cannot help with that.")
    assert obj is None and "no JSON object" in err


def test_normalizer_fixes_casing_but_not_semantics():
    obj, notes = normalize_extraction({"overall_sentiment": "Positive", "complaint_categories": ["Noise Issue"], "aspects": None})
    assert obj["overall_sentiment"] == "positive"
    assert obj["complaint_categories"] == ["noise_issue"]
    assert obj["aspects"] == []
    assert notes
    obj, _ = normalize_extraction({"overall_sentiment": "very good"})
    assert obj["overall_sentiment"] == "very_good"  # still invalid -> must go to LLM repair


def test_normalizer_unwraps_wrapper(valid_json):
    obj, notes = normalize_extraction({"result": json.loads(valid_json)})
    assert "overall_sentiment" in obj and "unwrapped nested wrapper object" in notes


# ---------------------------------------------------------------- repair loop
def _run(client, max_repairs=2):
    return generate_structured(
        client, [], LLMExtraction, task="extraction", task_description="t", format_instructions="f",
        payload={"review_text": REVIEW_TEXT}, context={"review_text": REVIEW_TEXT}, source_text=REVIEW_TEXT,
        max_repairs=max_repairs, normalizer=normalize_extraction,
    )


def test_valid_first_time_no_repair(valid_json):
    client = ScriptedLLM([valid_json])
    res = _run(client)
    assert res.success and res.repair_attempts == 0 and len(client.calls) == 1


def test_invalid_json_repaired_by_llm(valid_json):
    client = ScriptedLLM(['{"overall_sentiment": "mixed", "aspects": [', valid_json])
    res = _run(client)
    assert res.success and res.repair_attempts == 1
    repair_call = client.calls[1]
    assert repair_call["task"] == "repair"
    prompt = "\n".join(m.content for m in repair_call["messages"])
    assert "invalid JSON" in prompt and '"aspects": [' in prompt  # original output + errors are sent


def test_wrong_enum_repaired(valid_json):
    client = ScriptedLLM([json.dumps(valid_extraction(overall_sentiment="very good")), valid_json])
    res = _run(client)
    assert res.success and res.repair_attempts == 1
    assert "overall_sentiment" in res.attempts[0].errors[0]


def test_missing_field_and_wrong_type_repaired(valid_json):
    bad = valid_extraction(confidence="high")
    del bad["complaint_categories"]
    client = ScriptedLLM([json.dumps(bad), valid_json])
    res = _run(client)
    assert res.success and any("confidence" in e for e in res.attempts[0].errors)


def test_incomplete_aspect_record_repaired(valid_json):
    bad = valid_extraction()
    del bad["aspects"][1]["sentiment"]
    del bad["aspects"][2]["evidence"]
    client = ScriptedLLM([json.dumps(bad), valid_json])
    res = _run(client)
    errs = " ".join(res.attempts[0].errors)
    assert res.success and "aspects.1.sentiment" in errs and "aspects.2.evidence" in errs


def test_hallucinated_evidence_triggers_repair(valid_json):
    bad = valid_extraction()
    bad["aspects"][0]["evidence"] = "customer service was amazing"
    client = ScriptedLLM([json.dumps(bad), valid_json])
    res = _run(client)
    assert res.success and "verbatim" in res.attempts[0].errors[0]


def test_controlled_failure_after_max_repairs():
    client = ScriptedLLM(["nope", "still nope", "{bad json"])
    res = _run(client, max_repairs=2)
    assert not res.success and res.value is None
    assert len(res.attempts) == 3 and res.repair_attempts == 2
    assert "still invalid after 2 repair" in res.error


def test_zero_repairs_configurable():
    client = ScriptedLLM(["nope"])
    res = _run(client, max_repairs=0)
    assert not res.success and len(client.calls) == 1


def test_provider_error_is_controlled():
    class Boom(ScriptedLLM):
        def invoke(self, *a, **k):
            raise LLMError("quota exceeded")

    res = _run(Boom([]))
    assert not res.success and "quota exceeded" in res.error and res.attempts[0].kind == "provider_error"


def test_token_and_latency_accounting(valid_json):
    res = _run(ScriptedLLM(["{", valid_json]))
    assert res.input_tokens == 200 and res.output_tokens == 100 and res.total_latency_s == 0.02


# ---------------------------------------------------------------- extractor
def test_extractor_returns_record_with_metadata(settings, review, valid_json):
    ex = ReviewExtractor(ScriptedLLM(["```json\n" + valid_json + "\n```"]), get_strategy("zero_shot"), settings)
    out = ex.extract(review)
    assert isinstance(out, ReviewRecord)
    md = out.extraction_metadata
    assert md.strategy == "zero_shot" and md.validation_success and md.local_repairs >= 1
    assert md.attempts[0].raw_output.startswith("```json")


def test_extractor_returns_controlled_failure(settings, review):
    ex = ReviewExtractor(ScriptedLLM(["x", "y", "z"]), get_strategy("cot"), settings)
    out = ex.extract(review)
    assert isinstance(out, ExtractionFailure)
    assert not out.extraction_metadata.validation_success and len(out.extraction_metadata.attempts) == 3
