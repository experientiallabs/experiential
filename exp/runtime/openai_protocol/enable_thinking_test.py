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


def test_thinking_adaptive_defers_to_the_model_default() -> None:
    """Anthropic's 4.6+ on-mode is admitted on the Chat wire like ``enabled``.

    Claude-configured clients (Anthropic SDKs, Claude Code shims, Cherry
    Studio) pin ``thinking: {type: adaptive}`` on every model; 3,935 Chat
    requests over 7 days were refused at decode for it (2026-09-15).
    """
    request = _decode(thinking={"type": "adaptive"})
    assert request.reasoning_effort is None
    assert request.thinking_default_enable is True
    assert request.ignored_parameters == ("thinking->translated(reasoning_effort)",)


@pytest.mark.parametrize("mode", ("adaptive", "enabled"))
def test_numeric_thinking_budget_is_refused_not_discarded(mode: str) -> None:
    """Chat rejects a numerical budget its adapters cannot preserve."""
    with pytest.raises(OpenAIProtocolError) as error:
        _decode(thinking={"type": mode, "budget_tokens": 4096})
    assert error.value.detail.param == "thinking.budget_tokens"
    assert error.value.detail.code == "unsupported_parameter"


def test_thinking_unknown_type_names_the_members_not_the_json_type() -> None:
    """A non-member string is a value fault; the members are the useful fact.

    The old rendering ("expected one of 'enabled' or 'disabled', but got a
    string instead") told callers their string was not a string.
    """
    with pytest.raises(OpenAIProtocolError) as error:
        _decode(thinking={"type": "extended"})
    assert error.value.detail.param == "thinking.type"
    assert error.value.detail.message == (
        "Invalid value for 'thinking.type': expected one of 'enabled', 'disabled' or 'adaptive'."
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


@pytest.mark.parametrize(
    "off_fields",
    (
        {"thinking": {"type": "disabled"}},
        {"reasoning": {"enabled": False}},
        {"chat_template_kwargs": {"enable_thinking": False}},
        {"enable_thinking": False},
    ),
)
def test_flat_effort_cannot_override_an_explicit_off_setting(off_fields: JsonObject) -> None:
    """An active flat effort conflicts with any caller's explicit off switch."""
    with pytest.raises(OpenAIProtocolError) as error:
        _decode(reasoning_effort="high", **off_fields)
    assert error.value.detail.param == "reasoning_effort"


def test_conflicting_enable_and_disable_fields_are_rejected() -> None:
    with pytest.raises(OpenAIProtocolError):
        _decode(thinking={"type": "enabled"}, chat_template_kwargs={"enable_thinking": False})


def test_agreeing_enable_fields_prefer_the_nested_level() -> None:
    request = _decode(reasoning={"effort": "medium"}, thinking={"type": "enabled"})
    assert request.reasoning_effort == "medium"
    assert request.thinking_default_enable is False


def test_top_level_enable_thinking_translates_like_chat_template_kwargs() -> None:
    """DashScope's top-level switch enables at the model default or disables to none."""
    enabled = _decode(enable_thinking=True)
    assert enabled.reasoning_effort is None
    assert enabled.thinking_default_enable is True
    assert enabled.ignored_parameters == ("enable_thinking->translated(reasoning_effort)",)
    disabled = _decode(enable_thinking=False)
    assert disabled.reasoning_effort == "none"
    assert disabled.thinking_default_enable is False


def test_openrouter_reasoning_enabled_translates_to_the_canonical_control() -> None:
    """``reasoning.enabled`` is OpenRouter's literal on/off vote."""
    enabled = _decode(reasoning={"enabled": True})
    assert enabled.reasoning_effort is None
    assert enabled.thinking_default_enable is True
    assert enabled.ignored_parameters == ("reasoning->translated(reasoning_effort)",)
    disabled = _decode(reasoning={"enabled": False})
    assert disabled.reasoning_effort == "none"
    assert disabled.thinking_default_enable is False


@pytest.mark.parametrize("budget", (1024, 4096, 8192, 32768))
def test_openrouter_reasoning_budget_is_not_replaced_by_effort(budget: int) -> None:
    """A hard token budget cannot become an advisory effort level."""
    with pytest.raises(OpenAIProtocolError) as error:
        _decode(reasoning={"max_tokens": budget})
    assert error.value.detail.param == "reasoning.max_tokens"
    assert error.value.detail.code == "unsupported_parameter"


def test_openrouter_reasoning_exclude_is_disclosed_not_carried() -> None:
    """``exclude: true`` (hide reasoning in the response) is accepted and disclosed."""
    request = _decode(reasoning={"effort": "high", "exclude": True})
    assert request.reasoning_effort == "high"
    assert request.ignored_parameters == (
        "reasoning.exclude->dropped(not_carried)",
        "reasoning->translated(reasoning_effort)",
    )
    # exclude:false is the default and carries nothing to disclose.
    assert _decode(reasoning={"effort": "high", "exclude": False}).ignored_parameters == (
        "reasoning->translated(reasoning_effort)",
    )


def test_openrouter_reasoning_object_rejects_internal_contradictions_by_field() -> None:
    """``enabled: false`` beside a tier or budget, or a tier beside a budget, is named."""
    with pytest.raises(OpenAIProtocolError) as disagree:
        _decode(reasoning={"effort": "high", "enabled": False})
    assert disagree.value.detail.param == "reasoning.enabled"
    with pytest.raises(OpenAIProtocolError) as both:
        _decode(reasoning={"effort": "high", "max_tokens": 10})
    assert both.value.detail.param == "reasoning"
    assert "mutually exclusive" in both.value.detail.message


def test_explicit_flat_effort_reports_every_present_alternate_spelling() -> None:
    request = _decode(reasoning_effort="low", reasoning={"enabled": True}, enable_thinking=True)
    assert request.reasoning_effort == "low"
    assert request.ignored_parameters == (
        "reasoning->ignored(explicit_reasoning_effort)",
        "enable_thinking->ignored(explicit_reasoning_effort)",
    )


@pytest.mark.parametrize("field", ("thinking", "reasoning"))
def test_explicit_flat_effort_cannot_discard_a_numeric_budget(field: str) -> None:
    """A second effort channel cannot erase the caller's numerical bound."""
    value = (
        {"type": "enabled", "budget_tokens": 4096} if field == "thinking" else {"max_tokens": 2048}
    )
    with pytest.raises(OpenAIProtocolError) as error:
        _decode(reasoning_effort="low", **{field: value})
    assert error.value.detail.param == (
        "thinking.budget_tokens" if field == "thinking" else "reasoning.max_tokens"
    )
