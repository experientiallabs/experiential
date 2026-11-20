"""Chat-only generation controls are ignored by ``count_tokens`` and refused by Messages."""

from __future__ import annotations

import pytest
from pydantic import JsonValue

from exp.common.core.artifacts import JsonObject
from exp.runtime.anthropic_protocol.count_neutral_fields import (
    COUNT_NEUTRAL_FIELDS,
    count_neutral_disclosure,
    drop_count_neutral_fields,
)
from exp.runtime.anthropic_protocol.requests import decode_messages, decode_messages_count_tokens
from exp.runtime.gateway.attempt_tokens import counted_input_tokens
from exp.runtime.openai_protocol.errors import OpenAIProtocolError


def _body(**overrides: JsonValue) -> JsonObject:
    """One minimal Messages body (``max_tokens`` set, so it suits both entrypoints)."""
    payload: JsonObject = {
        "model": "coding",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "foo"}],
    }
    payload.update(overrides)
    return payload


def test_drop_removes_each_field_with_disclosure() -> None:
    """Every count-neutral field leaves the body and is named in order."""
    payload = _body(service_tier="auto", reasoning_effort="high", verbosity="low")
    remaining, disclosures = drop_count_neutral_fields(payload)
    assert remaining == _body()
    assert disclosures == tuple(count_neutral_disclosure(field) for field in COUNT_NEUTRAL_FIELDS)


def test_drop_leaves_a_body_without_the_fields_untouched() -> None:
    """A body with none of the fields is returned as is."""
    payload = _body()
    remaining, disclosures = drop_count_neutral_fields(payload)
    assert remaining is payload
    assert disclosures == ()


def test_null_values_are_removed_silently() -> None:
    """A null value asks for nothing, so it is removed without a disclosure."""
    remaining, disclosures = drop_count_neutral_fields(_body(reasoning_effort=None))
    assert remaining == _body()
    assert disclosures == ()


def test_count_tokens_with_reasoning_effort_counts_the_same_prompt() -> None:
    """Claude Desktop 3p's ``reasoning_effort`` is ignored: same request, same count."""
    decoded = decode_messages_count_tokens(_body(reasoning_effort="high"))
    plain = decode_messages_count_tokens(_body())
    assert decoded.request.reasoning_effort is None
    assert decoded.request.ignored_parameters == ("reasoning_effort->ignored(count_tokens)",)
    assert counted_input_tokens(decoded.request) == counted_input_tokens(plain.request)


@pytest.mark.parametrize(
    ("field", "value"),
    [("reasoning_effort", "high"), ("verbosity", "low"), ("service_tier", "auto")],
)
def test_count_tokens_accepts_each_field(field: str, value: str) -> None:
    """Each Chat-only control decodes on ``count_tokens`` with its disclosure."""
    decoded = decode_messages_count_tokens(_body(**{field: value}))
    assert decoded.request.ignored_parameters == (count_neutral_disclosure(field),)


def test_count_tokens_keeps_the_messages_reasoning_object() -> None:
    """The Messages ``reasoning`` object is a real field and still decodes as one."""
    decoded = decode_messages_count_tokens(_body(reasoning={"effort": "low"}))
    assert decoded.request.reasoning_effort == "low"


@pytest.mark.parametrize(
    ("field", "value"),
    [("reasoning_effort", "high"), ("verbosity", "low"), ("service_tier", "auto")],
)
def test_messages_still_refuses_each_field(field: str, value: str) -> None:
    """Generation changes behavior on these fields, so ``/v1/messages`` keeps refusing them."""
    with pytest.raises(OpenAIProtocolError) as refused:
        decode_messages(_body(**{field: value}))
    assert refused.value.detail.param == field
