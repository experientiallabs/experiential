"""Native decision request shape, strict JSON, and work bounds."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from exp.common.core.artifacts import JsonObject, JsonValue
from exp.runtime.gateway.contracts import GatewayApiSurface
from exp.runtime.gateway.decisions_contracts import (
    MAX_DECISION_INPUT_BYTES,
    MAX_DECISION_JSON_DEPTH,
    ChoiceQuestion,
    DecisionRequest,
    DecisionUsage,
    NoulQuestion,
    ScoreQuestion,
    decode_decision_request,
)


def _body() -> JsonObject:
    """Return three representative typed questions over synthetic state."""
    return {
        "model": "type-safe/jev-latest",
        "state": {"message": "Refund the duplicate invoice", "count": 3},
        "questions": {
            "truth": {"type": "noul", "instructions": "Is count positive?"},
            "team": {
                "type": "choice",
                "instructions": "Select the responsible team",
                "criteria": {"billing": "Invoices", "technical": None},
            },
            "quantity": {
                "type": "score",
                "instructions": "Classify count",
                "criteria": ["None", "One or two", "Three or more"],
            },
        },
    }


def test_decisions_are_native_and_preserve_question_identity() -> None:
    """State and criteria stay typed, without chat messages or tool translation."""
    body = _body()
    decoded = decode_decision_request(json.dumps(body))
    assert decoded.alias == "type-safe/jev-latest"
    assert decoded.request.surface is GatewayApiSurface.DECISIONS
    assert decoded.request.attribution_label is None
    assert decoded.request.provider_body("jev-latest") == {**body, "model": "jev-latest"}
    assert decoded.request.input_token_reservation > 3 * 1024
    assert decoded.request.output_token_reservation > 0
    assert not hasattr(decoded.request, "messages")


@pytest.mark.parametrize("extra", ["stream", "messages", "tools", "max_tokens", "user"])
def test_unrepresentable_public_fields_are_rejected(extra: str) -> None:
    """Unknown public fields cannot be silently ignored on this native surface."""
    with pytest.raises(ValueError, match="requires only"):
        decode_decision_request(json.dumps({**_body(), extra: True}))


@pytest.mark.parametrize("state", [None, True, 42, 1.5])
def test_scalar_nontext_states_are_rejected(state: JsonValue) -> None:
    """Only text, objects, and arrays are documented as provider state."""
    with pytest.raises(ValueError):
        decode_decision_request(json.dumps({**_body(), "state": state}))


@pytest.mark.parametrize("state", ["hello", {"fact": True}, ["a", 2, None]])
def test_native_state_variants_roundtrip(state: JsonValue) -> None:
    """Valid text and structured states survive canonical serialization."""
    request = decode_decision_request(json.dumps({**_body(), "state": state})).request
    assert request.state == state
    assert DecisionRequest.model_validate_json(request.model_dump_json()) == request


@pytest.mark.parametrize(
    "raw",
    [
        '{"model":"jev-latest","state":"x","questions":{"x":{"type":"noul","instructions":"a"},"x":{"type":"noul","instructions":"b"}}}',
        '{"model":"jev-latest","state":{"x":NaN},"questions":{"x":{"type":"noul","instructions":"a"}}}',
        '{"model":"jev-latest","state":{"x":Infinity},"questions":{"x":{"type":"noul","instructions":"a"}}}',
    ],
)
def test_duplicate_keys_and_nonfinite_json_are_rejected(raw: str) -> None:
    """No provider invocation can silently lose questions or carry non-JSON values."""
    with pytest.raises(ValueError):
        decode_decision_request(raw)


@pytest.mark.parametrize(
    "questions",
    [
        {},
        {"": {"type": "noul", "instructions": "a"}},
        {"x": {"type": "unknown", "instructions": "a"}},
        {"x": {"type": "choice", "instructions": "a", "criteria": {}}},
        {"x": {"type": "score", "instructions": "a", "criteria": ["one"]}},
        {"x": {"type": "noul", "instructions": 1}},
    ],
)
def test_invalid_questions_fail_before_admission(questions: JsonObject) -> None:
    """Invalid native schemas receive a local parameter error, not a provider call."""
    with pytest.raises(ValueError):
        decode_decision_request(json.dumps({**_body(), "questions": questions}))


@pytest.mark.parametrize("count", [2, 10, 11, 64])
def test_score_criteria_follow_the_documented_two_through_ten_limit(count: int) -> None:
    """Reject invalid score levels before reserving or sending provider work."""
    question = {
        "type": "score",
        "instructions": "Rate count",
        "criteria": [str(i) for i in range(count)],
    }
    body = json.dumps({**_body(), "questions": {"score": question}})
    if count <= 10:
        assert decode_decision_request(body).request.provider_body("jev-latest")["questions"] == {
            "score": question
        }
    else:
        with pytest.raises(ValueError):
            decode_decision_request(body)


def test_structured_criteria_and_instructions_roundtrip_without_stringification() -> None:
    """Native question definitions preserve objects, arrays, null choice labels, and text."""
    questions: JsonObject = {
        "choice": {
            "type": "choice",
            "instructions": {"task": ["Choose the team"]},
            "criteria": {
                "a": {"description": "Billing", "examples": [1, True, None]},
                "b": ["Technical", {"code": 2}],
                "c": None,
                "d": "Other",
            },
        },
        "score": {
            "type": "score",
            "instructions": ["Rate", {"scale": "ordinal"}],
            "criteria": ["Low", {"level": "medium"}, ["High", {"priority": 3}]],
        },
    }
    decoded = decode_decision_request(json.dumps({**_body(), "questions": questions}))
    assert decoded.request.provider_body("jev-latest")["questions"] == questions
    assert DecisionRequest.model_validate_json(decoded.request.model_dump_json()) == decoded.request


@pytest.mark.parametrize("criterion", [True, False, 1, 1.5])
def test_choice_criteria_reject_bare_boolean_and_numeric_descriptions(criterion: JsonValue) -> None:
    """Arbitrary JSON is nested content, not permission for scalar category descriptions."""
    with pytest.raises(ValueError, match="choice criteria must"):
        ChoiceQuestion(instructions="Choose", criteria={"a": criterion, "b": "Other"})


@pytest.mark.parametrize("criterion", [None, True, False, 1, 1.5])
def test_score_criteria_require_text_or_structured_descriptions(criterion: JsonValue) -> None:
    """Score levels cannot be null, boolean, or numeric scalar descriptions."""
    with pytest.raises(ValueError, match="score criteria must"):
        ScoreQuestion(instructions="Rate", criteria=("Low", criterion))


def _request_with_nested(value: JsonValue, location: str) -> JsonObject:
    """Place arbitrary JSON in every native request position that can carry it."""
    question: JsonObject = {"type": "noul", "instructions": "Decide"}
    body: JsonObject = {"model": "jev-latest", "state": "test", "questions": {"q": question}}
    if location == "state":
        body["state"] = {"nested": [value]}
    elif location == "instructions":
        question["instructions"] = [value]
    elif location == "choice":
        question.update(type="choice", criteria={"a": {"nested": value}, "b": None})
    else:
        question.update(type="score", criteria=["Low", {"nested": value}])
    return body


@pytest.mark.parametrize("location", ["state", "instructions", "choice", "score"])
@pytest.mark.parametrize("value", [-(2**63), 2**64 - 1])
def test_native_integer_endpoints_preserve_exact_values(location: str, value: int) -> None:
    """Signed minimum and unsigned maximum remain integers on the admitted wire."""
    body = _request_with_nested(value, location)
    request = decode_decision_request(json.dumps(body)).request
    assert request.provider_body("jev-latest") == body


@pytest.mark.parametrize("location", ["state", "instructions", "choice", "score"])
@pytest.mark.parametrize("value", [-(2**63) - 1, 2**64, 2**128])
def test_out_of_native_range_integers_fail_before_admission(location: str, value: int) -> None:
    """No oversized nested integer can be rounded to float or rejected only after accept."""
    with pytest.raises(ValueError, match="64-bit"):
        decode_decision_request(json.dumps(_request_with_nested(value, location)))


@pytest.mark.parametrize("location", ["state", "instructions", "choice", "score"])
@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan"), "\ud800"])
def test_nonfinite_and_invalid_unicode_nested_values_fail_cleanly(
    location: str,
    value: JsonValue,
) -> None:
    """Invalid nested JSON is always a local ValueError, never a native bridge failure."""
    with pytest.raises(ValueError):
        decode_decision_request(json.dumps(_request_with_nested(value, location)))


def test_exponent_overflow_is_rejected_even_when_json_syntax_is_standard() -> None:
    """A finite-looking numeric literal cannot overflow Python's float into infinity."""
    raw = json.dumps(_request_with_nested("EXPONENT", "state")).replace('"EXPONENT"', "1e400")
    with pytest.raises(ValueError, match="finite"):
        decode_decision_request(raw)


@pytest.mark.parametrize("location", ["alias", "state_key", "question_id", "choice_name"])
def test_lone_surrogates_in_identifiers_are_refused(location: str) -> None:
    """JSON object keys and aliases must be valid UTF-8 just like their values."""
    body = _request_with_nested("valid", "state")
    if location == "alias":
        body["model"] = "\udfff"
    elif location == "state_key":
        body["state"] = {"\udfff": "value"}
    elif location == "question_id":
        body["questions"] = {"\udfff": {"type": "noul", "instructions": "Decide"}}
    else:
        body["questions"] = {
            "q": {"type": "choice", "instructions": "Choose", "criteria": {"\udfff": "A", "b": "B"}}
        }
    with pytest.raises(ValueError, match="UTF-8"):
        decode_decision_request(json.dumps(body))


@pytest.mark.parametrize("depth", [MAX_DECISION_JSON_DEPTH + 1, 1500])
def test_excessive_json_nesting_is_a_clean_parameter_failure(depth: int) -> None:
    """Both bridge-depth overflow and Python decoder recursion fail before acceptance."""
    raw = (
        '{"model":"jev-latest","state":'
        + "[" * depth
        + '"nested"'
        + "]" * depth
        + ',"questions":{"q":{"type":"noul","instructions":"Decide"}}}'
    )
    with pytest.raises(ValueError, match="64-level"):
        decode_decision_request(raw)


@pytest.mark.parametrize("value", [2**64, float("inf"), "\ud800"])
def test_direct_request_construction_cannot_bypass_native_json_safety(value: JsonValue) -> None:
    """Programmatic callers get the same native representation checks as the HTTP decoder."""
    with pytest.raises(ValueError):
        DecisionRequest(
            state={"nested": [value]}, questions={"q": NoulQuestion(instructions="Decide")}
        )


def test_question_and_total_input_bounds_are_independent() -> None:
    """Tiny many-question requests and oversized single-question inputs both stop."""
    question = {"type": "noul", "instructions": "Is it true?"}
    with pytest.raises(ValueError):
        decode_decision_request(
            json.dumps({**_body(), "questions": {str(i): question for i in range(33)}})
        )
    with pytest.raises(ValueError, match="byte"):
        decode_decision_request(json.dumps({**_body(), "state": "x" * MAX_DECISION_INPUT_BYTES}))


@pytest.mark.parametrize(
    "usage",
    [
        {"input_tokens": 0, "output_tokens": 0},
        {"input_tokens": True, "output_tokens": 0},
        {"input_tokens": -1, "output_tokens": 1},
        {"input_tokens": 1.1, "output_tokens": 0},
        {"input_tokens": 1},
    ],
)
def test_decision_usage_never_coerces_or_invents_counts(usage: JsonObject) -> None:
    """Only paired, nonnegative integer token counts are settlement evidence."""
    with pytest.raises(ValidationError):
        DecisionUsage.model_validate(usage)
