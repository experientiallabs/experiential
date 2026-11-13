"""DeepSeek's ``thinking`` switch on Responses folds into the reasoning control."""

from __future__ import annotations

import pytest
from pydantic import JsonValue

from exp.common.core.artifacts import JsonObject
from exp.runtime.openai_protocol.client_field_repairs import THINKING_DISPLAY_DROPPED
from exp.runtime.openai_protocol.errors import OpenAIProtocolError
from exp.runtime.openai_protocol.requests import decode_responses
from exp.runtime.openai_protocol.responses_thinking import (
    RESPONSES_THINKING_IGNORED,
    RESPONSES_THINKING_TRANSLATED,
    translate_responses_thinking,
)

_RESPONSES_BASE: JsonObject = {"model": "m", "input": "hi"}


def test_responses_thinking_disabled_turns_reasoning_off() -> None:
    """DeepSeek's ``thinking: {"type": "disabled"}`` is ``reasoning.effort: "none"``."""
    decoded = decode_responses({**_RESPONSES_BASE, "thinking": {"type": "disabled"}})
    assert decoded.request.reasoning_effort == "none"
    assert decoded.request.thinking_default_enable is False
    assert decoded.request.ignored_parameters == (RESPONSES_THINKING_TRANSLATED,)


@pytest.mark.parametrize("kind", ("enabled", "adaptive"))
def test_responses_thinking_enabled_asks_for_the_default_depth(kind: str) -> None:
    """An enable carries no depth, so the route resolves its default effort."""
    decoded = decode_responses({**_RESPONSES_BASE, "thinking": {"type": kind}})
    assert decoded.request.reasoning_effort is None
    assert decoded.request.thinking_default_enable is True
    assert decoded.request.ignored_parameters == (RESPONSES_THINKING_TRANSLATED,)


def test_responses_thinking_disabled_merges_into_the_reasoning_object() -> None:
    """An off switch merges into the caller's ``reasoning`` object, keeping its other fields."""
    repaired, disclosures, default_enable = translate_responses_thinking(
        {**_RESPONSES_BASE, "reasoning": {"summary": "auto"}, "thinking": {"type": "disabled"}}
    )
    assert "thinking" not in repaired
    assert repaired["reasoning"] == {"summary": "auto", "effort": "none"}
    assert disclosures == (RESPONSES_THINKING_TRANSLATED,)
    assert default_enable is False


@pytest.mark.parametrize(("kind", "effort"), (("enabled", "high"), ("disabled", "none")))
def test_responses_thinking_agreeing_with_an_explicit_effort_yields(kind: str, effort: str) -> None:
    """An explicit effort that agrees on on-versus-off decides the request; the switch yields."""
    decoded = decode_responses(
        {**_RESPONSES_BASE, "reasoning": {"effort": effort}, "thinking": {"type": kind}}
    )
    assert decoded.request.reasoning_effort == effort
    assert decoded.request.thinking_default_enable is False
    assert decoded.request.ignored_parameters == (RESPONSES_THINKING_IGNORED,)


@pytest.mark.parametrize(("kind", "effort"), (("enabled", "none"), ("disabled", "high")))
def test_responses_thinking_contradicting_an_explicit_effort_is_a_named_400(
    kind: str, effort: str
) -> None:
    """An explicit effort that contradicts the switch is a 400 naming ``thinking``."""
    with pytest.raises(OpenAIProtocolError) as error:
        decode_responses(
            {**_RESPONSES_BASE, "reasoning": {"effort": effort}, "thinking": {"type": kind}}
        )
    assert error.value.detail.code == "invalid_parameter"
    assert error.value.detail.param == "thinking"


def test_responses_thinking_null_is_consumed_silently() -> None:
    """``thinking: null`` asks for nothing and is removed without disclosure."""
    decoded = decode_responses({**_RESPONSES_BASE, "thinking": None})
    assert decoded.request.reasoning_effort is None
    assert decoded.request.thinking_default_enable is False
    assert decoded.request.ignored_parameters == ()


def test_responses_thinking_display_is_dropped_with_disclosure() -> None:
    """Anthropic's ``display`` preference drops with the Chat disclosure."""
    decoded = decode_responses(
        {**_RESPONSES_BASE, "thinking": {"type": "enabled", "display": "summarized"}}
    )
    assert decoded.request.ignored_parameters == (
        THINKING_DISPLAY_DROPPED,
        RESPONSES_THINKING_TRANSLATED,
    )


def test_responses_thinking_budget_keeps_a_named_refusal() -> None:
    """A numeric budget cannot be enforced on Responses and keeps a named refusal."""
    with pytest.raises(OpenAIProtocolError) as error:
        decode_responses(
            {**_RESPONSES_BASE, "thinking": {"type": "enabled", "budget_tokens": 2048}}
        )
    assert error.value.detail.code == "unsupported_parameter"
    assert error.value.detail.param == "thinking.budget_tokens"


@pytest.mark.parametrize(
    ("thinking", "param"),
    (
        (True, "thinking"),
        ("enabled", "thinking"),
        ({}, "thinking.type"),
        ({"type": "on"}, "thinking.type"),
        ({"type": "enabled", "level": 3}, "thinking.level"),
    ),
)
def test_responses_thinking_malformed_is_a_named_400(thinking: JsonValue, param: str) -> None:
    """Any other ``thinking`` shape is a 400 naming the offending field."""
    with pytest.raises(OpenAIProtocolError) as error:
        decode_responses({**_RESPONSES_BASE, "thinking": thinking})
    assert error.value.detail.code == "invalid_parameter"
    assert error.value.detail.param == param


@pytest.mark.parametrize("effort", ("bogus", {"level": "high"}, 3))
def test_responses_thinking_beside_an_unknown_effort_defers_to_the_wire_model(
    effort: JsonValue,
) -> None:
    """An unrecognized effort is refused by name, never reported as a ``thinking`` conflict."""
    with pytest.raises(OpenAIProtocolError) as error:
        decode_responses(
            {**_RESPONSES_BASE, "reasoning": {"effort": effort}, "thinking": {"type": "disabled"}}
        )
    assert error.value.detail.param == "reasoning.effort"


def test_responses_thinking_unknown_key_is_bounded_in_the_error() -> None:
    """An oversized unknown key still yields the typed 400, its name truncated."""
    with pytest.raises(OpenAIProtocolError) as error:
        decode_responses({**_RESPONSES_BASE, "thinking": {"type": "enabled", "k" * 600: 1}})
    assert error.value.detail.code == "invalid_parameter"
    assert error.value.detail.param == "thinking." + "k" * 64
