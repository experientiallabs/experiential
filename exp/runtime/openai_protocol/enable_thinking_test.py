"""Tests for enable-thinking field translation to canonical reasoning_effort."""

from __future__ import annotations

from typing import cast

import pytest

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.contracts import GatewayRequest
from exp.runtime.openai_protocol.errors import OpenAIProtocolError
from exp.runtime.openai_protocol.requests import decode_chat


def _decode(**overrides: object) -> GatewayRequest:
    body: dict[str, object] = {
        "model": "coding",
        "messages": [{"role": "user", "content": "hi"}],
        **overrides,
    }
    return decode_chat(cast(JsonObject, body)).request


def test_nested_reasoning_effort_translates_to_flat_effort() -> None:
    request = _decode(reasoning={"effort": "high"})
    assert request.reasoning_effort == "high"
    assert request.thinking_default_enable is False
    assert request.ignored_parameters == ("reasoning->translated(reasoning_effort)",)


def test_thinking_enabled_defers_to_the_model_default() -> None:
    request = _decode(thinking={"type": "enabled"})
    assert request.reasoning_effort is None
    assert request.thinking_default_enable is True
    assert request.ignored_parameters == ("thinking->translated(reasoning_effort)",)


def test_thinking_enabled_budget_tokens_is_disclosed_not_carried() -> None:
    request = _decode(thinking={"type": "enabled", "budget_tokens": 4096})
    assert request.thinking_default_enable is True
    assert request.ignored_parameters == (
        "budget_tokens->dropped(not_carried)",
        "thinking->translated(reasoning_effort)",
    )


def test_chat_template_kwargs_enable_defers_to_the_model_default() -> None:
    request = _decode(chat_template_kwargs={"enable_thinking": True})
    assert request.reasoning_effort is None
    assert request.thinking_default_enable is True
    assert request.ignored_parameters == ("chat_template_kwargs->translated(reasoning_effort)",)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("thinking", {"type": "disabled"}),
        ("chat_template_kwargs", {"enable_thinking": False}),
    ],
)
def test_disable_shapes_translate_to_reasoning_none(field: str, value: object) -> None:
    request = _decode(**{field: value})
    assert request.reasoning_effort == "none"
    assert request.thinking_default_enable is False
    assert request.ignored_parameters == (f"{field}->translated(reasoning_effort)",)


def test_explicit_flat_reasoning_effort_wins_over_translate_fields() -> None:
    request = _decode(
        reasoning_effort="low",
        thinking={"type": "enabled"},
        chat_template_kwargs={"enable_thinking": True},
    )
    assert request.reasoning_effort == "low"
    assert request.thinking_default_enable is False
    assert request.ignored_parameters == (
        "thinking->ignored(explicit_reasoning_effort)",
        "chat_template_kwargs->ignored(explicit_reasoning_effort)",
    )


def test_conflicting_enable_and_disable_fields_are_rejected() -> None:
    with pytest.raises(OpenAIProtocolError):
        _decode(thinking={"type": "enabled"}, chat_template_kwargs={"enable_thinking": False})


def test_agreeing_enable_fields_prefer_the_nested_level() -> None:
    request = _decode(reasoning={"effort": "medium"}, thinking={"type": "enabled"})
    assert request.reasoning_effort == "medium"
    assert request.thinking_default_enable is False
