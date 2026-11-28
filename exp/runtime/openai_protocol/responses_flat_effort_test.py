"""Tests for folding a flat ``reasoning_effort`` into Responses ``reasoning.effort``."""

from __future__ import annotations

import pytest
from pydantic import JsonValue

from exp.common.core.artifacts import JsonObject
from exp.runtime.openai_protocol.errors import OpenAIProtocolError
from exp.runtime.openai_protocol.requests import decode_responses
from exp.runtime.openai_protocol.responses_flat_effort import (
    FLAT_EFFORT_IGNORED,
    FLAT_EFFORT_TRANSLATED,
    translate_responses_flat_effort,
)
from exp.runtime.openai_protocol.responses_thinking import RESPONSES_THINKING_IGNORED

_RESPONSES_BASE: JsonObject = {"model": "m", "input": "hi"}


@pytest.mark.parametrize("effort", ("none", "low", "medium", "high", "xhigh"))
def test_flat_effort_is_served_as_the_nested_responses_effort(effort: str) -> None:
    """A Chat-spelled effort on Responses decides the request with disclosure."""
    decoded = decode_responses({**_RESPONSES_BASE, "reasoning_effort": effort})
    assert decoded.request.reasoning_effort == effort
    assert decoded.request.ignored_parameters == (FLAT_EFFORT_TRANSLATED,)


def test_flat_effort_merges_into_a_reasoning_object_without_an_effort() -> None:
    """Other ``reasoning`` keys the caller sent are kept beside the translated effort."""
    repaired, disclosures = translate_responses_flat_effort(
        {**_RESPONSES_BASE, "reasoning_effort": "high", "reasoning": {"summary": "auto"}}
    )
    assert "reasoning_effort" not in repaired
    assert repaired["reasoning"] == {"summary": "auto", "effort": "high"}
    assert disclosures == (FLAT_EFFORT_TRANSLATED,)


@pytest.mark.parametrize("flat", ("high", "low"))
def test_explicit_nested_effort_wins_over_the_flat_spelling(flat: str) -> None:
    """The Responses-native ``reasoning.effort`` decides, agreeing or not."""
    decoded = decode_responses(
        {**_RESPONSES_BASE, "reasoning_effort": flat, "reasoning": {"effort": "high"}}
    )
    assert decoded.request.reasoning_effort == "high"
    assert decoded.request.ignored_parameters == (FLAT_EFFORT_IGNORED,)


def test_null_flat_effort_is_removed_silently() -> None:
    """``reasoning_effort: null`` asks for nothing."""
    repaired, disclosures = translate_responses_flat_effort(
        {**_RESPONSES_BASE, "reasoning_effort": None}
    )
    assert repaired == _RESPONSES_BASE
    assert disclosures == ()


def test_flat_effort_reaches_the_thinking_switch_resolution() -> None:
    """A translated flat effort is the explicit effort the ``thinking`` switch yields to."""
    decoded = decode_responses(
        {**_RESPONSES_BASE, "reasoning_effort": "none", "thinking": {"type": "disabled"}}
    )
    assert decoded.request.reasoning_effort == "none"
    assert decoded.request.thinking_default_enable is False
    assert decoded.request.ignored_parameters == (
        FLAT_EFFORT_TRANSLATED,
        RESPONSES_THINKING_IGNORED,
    )


@pytest.mark.parametrize("effort", ("hgih", "", 3, True, {"effort": "high"}))
def test_unknown_flat_effort_keeps_a_named_refusal(effort: JsonValue) -> None:
    """A value that names no tier is a caller error refused by name."""
    with pytest.raises(OpenAIProtocolError) as error:
        decode_responses({**_RESPONSES_BASE, "reasoning_effort": effort})
    assert error.value.detail.code == "invalid_parameter"
    assert error.value.detail.param == "reasoning_effort"
