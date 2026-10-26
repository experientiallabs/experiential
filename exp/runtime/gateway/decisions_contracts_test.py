"""Native decision request shape, strict JSON, and work bounds."""

from __future__ import annotations

import base64
import json
import os
import tracemalloc
from typing import cast

import pytest
from pydantic import ValidationError

from exp.common.core.artifacts import JsonObject, JsonValue
from exp.runtime.gateway import decisions_contracts
from exp.runtime.gateway.contracts import GatewayApiSurface
from exp.runtime.gateway.decisions_contracts import (
    DECISION_IMAGE_TOKEN_RESERVATION,
    MAX_DECISION_BODY_BYTES,
    MAX_DECISION_INPUT_BYTES,
    MAX_DECISION_JSON_DEPTH,
    ChoiceQuestion,
    DecisionRequest,
    DecisionRequestError,
    DecisionUsage,
    NoulQuestion,
    ScoreQuestion,
    decode_decision_request,
    public_decision_error,
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
    with pytest.raises(DecisionRequestError, match="accept only model, state, questions") as raised:
        decode_decision_request(json.dumps({**_body(), extra: True}))
    assert raised.value.param == extra


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


def test_long_context_state_roundtrips_without_truncation() -> None:
    """A synthetic 200k-token state reaches the provider with its final evidence intact."""
    state = "The deployment is healthy. " * 40_000 + "Final evidence: deployment failed."
    body = {**_body(), "state": state}
    request = decode_decision_request(json.dumps(body)).request
    assert request.state == state
    assert request.provider_body("clef")["state"] == state


def test_text_budget_counts_utf8_bytes() -> None:
    """Non-ASCII state consumes its encoded byte count, not its character count."""
    state = "\U0001f642" * (MAX_DECISION_INPUT_BYTES // 4)
    with pytest.raises(ValueError, match="byte input limit"):
        decode_decision_request(json.dumps({**_body(), "state": state}, ensure_ascii=False))


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


_FOUND = "a3f1c2e8-found"
_REF = "77b0d4a1-poster"


def _png() -> JsonObject:
    """Return one fresh synthetic image asset."""
    return {"type": "image", "format": "png", "data_base64": base64.b64encode(b"png").decode()}


def _assets() -> dict[str, JsonObject]:
    """Return the two images placed by :func:`_image_body`."""
    return {_FOUND: _png(), _REF: _png()}


def _image_body(
    *,
    assets: dict[str, JsonObject] | None = None,
    note: str = "found near the park",
    truth: JsonObject | None = None,
    team_criteria: JsonObject | None = None,
) -> JsonObject:
    """Place one image in the state and another inside a single question's instructions."""
    questions: JsonObject = {
        "truth": truth
        or {"type": "noul", "instructions": f"Is the dog in `photo` the same dog as in {_REF}?"},
        "team": {
            "type": "choice",
            "instructions": "Select the responsible team",
            "criteria": team_criteria or {"billing": "Invoices", "technical": None},
        },
        "quantity": {
            "type": "score",
            "instructions": "Classify count",
            "criteria": ["None", "One or two", "Three or more"],
        },
    }
    return {
        "model": "type-safe/jev-latest",
        "state": {"photo": _FOUND, "note": note},
        "questions": questions,
        "assets": cast("JsonObject", _assets() if assets is None else assets),
    }


def _refusal(body: JsonObject) -> tuple[str, str]:
    """Decode one invalid body and return its caller-safe parameter and message."""
    with pytest.raises(ValueError) as raised:
        decode_decision_request(json.dumps(body))
    return public_decision_error(raised.value)


def test_image_assets_pass_through_and_reserve_tokens_per_use() -> None:
    """Assets reach the provider body unchanged; each image use adds a planning allowance."""
    body = _image_body()
    request = decode_decision_request(json.dumps(body)).request
    assert request.has_images
    assert request.provider_body("jev-latest") == {**body, "model": "jev-latest"}
    text_only = DecisionRequest(state=request.state, questions=request.questions)
    # The state image is repeated per question (3) and the poster is used once.
    assert request.input_token_reservation == (
        text_only.input_token_reservation + 4 * DECISION_IMAGE_TOKEN_RESERVATION
    )


def test_text_only_requests_carry_no_assets_key() -> None:
    """Existing callers keep a byte-identical provider body."""
    request = decode_decision_request(json.dumps(_body())).request
    assert not request.has_images
    assert "assets" not in request.provider_body("jev-latest")


def test_image_data_does_not_count_toward_the_text_budget() -> None:
    """A large image is bounded independently of the text budget."""
    data = base64.b64encode(b"x" * MAX_DECISION_INPUT_BYTES).decode()
    big: JsonObject = {"format": "jpg", "data_base64": data}
    assets = {**_assets(), "c83a0e15-big": big}
    body = _image_body(assets=assets, note="see c83a0e15-big")
    request = decode_decision_request(json.dumps(body)).request
    assert request.assets is not None and len(request.assets) == 3


def test_decoded_image_bytes_are_bounded_per_request(monkeypatch: pytest.MonkeyPatch) -> None:
    """The decoded total, not the base64 length, is held to the image budget."""
    monkeypatch.setattr(decisions_contracts, "MAX_DECISION_IMAGE_BYTES", 5)
    assert _refusal(_image_body()) == (
        "assets",
        "decoded image data must total at most 33554432 bytes per request",
    )


def _with_asset(reference: str, asset: JsonObject) -> dict[str, JsonObject]:
    return {**_assets(), reference: asset}


def _poster(**fields: JsonValue) -> dict[str, JsonObject]:
    return {_FOUND: _png(), _REF: {**_png(), **fields}}


@pytest.mark.parametrize(
    ("assets", "phrase"),
    [
        (_with_asset("unused", _png()), "must appear"),
        (_with_asset("   ", _png()), "empty or whitespace"),
        (_with_asset("r" * 1025, _png()), "1024 bytes"),
        (_poster(part="poster"), "multipart"),
        (_poster(data_base64="AA AA"), "standard base64"),
        (_poster(data_base64="data:image/png;base64,AAAA"), "standard base64"),
        (_poster(data_base64="AAA"), "standard base64"),
        ({}, "at least one image"),
    ],
)
def test_asset_refusals_are_field_specific(assets: dict[str, JsonObject], phrase: str) -> None:
    """Unreferenced, malformed, multipart, and empty assets fail before admission on assets."""
    param, message = _refusal(_image_body(assets=assets))
    assert param == "assets"
    assert phrase in message


def test_a_reference_used_only_as_an_object_key_is_unreferenced() -> None:
    """Object keys stay text upstream, so a key alone does not place an image."""
    body = _image_body(
        truth={"type": "noul", "instructions": "Is it the same dog?"},
        team_criteria={_REF: "poster", "other": None},
    )
    assert _refusal(body) == (
        "assets",
        "every asset reference must appear in state, a question's instructions, or its criteria",
    )


def test_more_than_the_asset_cap_is_refused() -> None:
    """The reference scan stays bounded by a fixed asset count."""
    references = [f"ref-{index:04d}" for index in range(257)]
    assets = {**_assets(), **{reference: _png() for reference in references}}
    body = _image_body(assets=assets, note=" ".join(references))
    assert _refusal(body) == ("assets", "assets may contain at most 256 images")


def test_unsupported_asset_format_maps_to_a_format_message() -> None:
    """Pydantic literal failures become fixed text naming the allowed formats."""
    body = _image_body(assets=_poster(format="gif"))
    assert _refusal(body) == ("assets", "asset format must be png, jpg, or jpeg")


def test_noul_criteria_accept_structured_outcomes_with_example_images() -> None:
    """True and false outcomes may be objects or arrays that carry asset references."""
    truth: JsonObject = {
        "type": "noul",
        "instructions": "Is the dog in `photo` the poster dog?",
        "criteria": {
            "true": {"description": "Same dog", "examples": [_REF]},
            "false": "A different dog.",
        },
    }
    request = decode_decision_request(json.dumps(_image_body(truth=truth))).request
    assert request.questions["truth"].model_dump(mode="json", exclude_none=True) == truth
    numeric: JsonObject = {**truth, "criteria": {"true": 1}}
    assert _refusal(_image_body(truth=numeric, note=_REF))[0] == "questions"


@pytest.mark.parametrize(
    "raw",
    [
        '{"model":"m","state":"secret-state","questions":{"secret-id":{"type":"secret-type","instructions":"x"}}}',
        '{"model":"m","state":"secret-state","questions":{"secret-id":{"type":"noul","instructions":"x","secret-key":1}}}',
        '{"model":"m","state":"secret-state","questions":{"secret-id":{"type":"choice","instructions":"x","criteria":{"secret-only":"x"}}}}',
        '{"model":"m","state":true,"questions":{"secret-id":{"type":"noul","instructions":"x"}}}',
        '{"model":"m","state":"secret-state","questions":{"secret-id":{"type":"noul","instructions":"x"}},"secret-field":1}',
        '{"model":"m","state":"secret-state","questions":{"a":{"type":"noul","instructions":"x"}},"assets":{"secret-ref":{"format":"png","data_base64":"AAAA","secret-key":1}}}',
        '{"model":"m","state":"secret-state",',
    ],
)
def test_public_errors_never_echo_caller_content(raw: str) -> None:
    """Only field names and fixed text leave the decoder, never values or caller keys."""
    with pytest.raises(ValueError) as raised:
        decode_decision_request(raw)
    param, message = public_decision_error(raised.value)
    assert "secret" not in param + message


def test_a_one_character_reference_in_a_large_state_reserves_once_per_question() -> None:
    """Repeated substring matches cannot multiply the reservation past assets x questions."""
    question: JsonObject = {"type": "noul", "instructions": "Is `photo` a dog? a a a"}
    questions = {f"q{index}": question for index in range(32)}
    state = "a" * 200_000
    body: JsonObject = {
        "model": "jev-latest",
        "state": state,
        "questions": cast("JsonObject", questions),
        "assets": {"a": _png()},
    }
    request = decode_decision_request(json.dumps(body)).request
    text_only = DecisionRequest(state=state, questions=request.questions)
    assert request.input_token_reservation == (
        text_only.input_token_reservation + 32 * DECISION_IMAGE_TOKEN_RESERVATION
    )


@pytest.mark.parametrize(("data", "size"), [("AAAA", 3), ("AAA=", 2), ("AA==", 1)])
def test_decoded_size_is_computed_from_the_base64_text(data: str, size: int) -> None:
    """Padding is subtracted arithmetically; the image is never decoded to measure it."""
    request = decode_decision_request(json.dumps(_image_body(assets=_poster(data_base64=data))))
    assert request.request.assets is not None
    assert request.request.assets[_REF].decoded_bytes() == size


def test_each_image_is_held_to_typesafes_per_file_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    """A single oversized image is a gateway 400 on assets, not an upstream 413."""
    monkeypatch.setattr(decisions_contracts, "MAX_DECISION_ASSET_BYTES", 2)
    assert _refusal(_image_body()) == (
        "assets",
        "each image must be at most 16777216 bytes after base64 decoding",
    )


def test_a_caller_measured_byte_length_bounds_the_raw_body() -> None:
    """The native front's measured length is trusted, so the body is not re-encoded."""
    with pytest.raises(DecisionRequestError, match="request size limit"):
        decode_decision_request(json.dumps(_body()), MAX_DECISION_BODY_BYTES + 1)


def test_decoding_a_large_image_body_does_not_multiply_memory() -> None:
    """Validation, reservation, and the provider body share the parsed base64 text."""
    data = base64.b64encode(os.urandom(6 * 1024 * 1024)).decode()
    body = json.dumps(_image_body(assets=_poster(data_base64=data)))
    tracemalloc.start()
    try:
        request = decode_decision_request(body, len(body)).request
        request.provider_body("jev-latest")
        assert request.input_token_reservation > 0
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 1.5 * len(body)
