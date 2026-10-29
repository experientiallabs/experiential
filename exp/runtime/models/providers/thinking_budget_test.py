"""Numeric thinking caps survive decoding, admission, dispatch and replay identity."""

from __future__ import annotations

from dataclasses import replace

import pytest

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.replay_identity import canonical_request_sha256
from exp.runtime.models.providers.base import GatewayWireProfile
from exp.runtime.models.providers.dialect_dispatch import dialect_stream_payload
from exp.runtime.models.providers.errors import ProviderParameterError
from exp.runtime.models.providers.streaming_requests import route_generation_parameter_requests
from exp.runtime.openai_protocol.errors import OpenAIProtocolError
from exp.runtime.openai_protocol.requests import decode_chat

_PROFILE = GatewayWireProfile(
    dialect="openai_compatible",
    url="https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1/chat/completions",
    model_id="qwen3.8-max",
    supports_reasoning=True,
    reasoning_wire_format="reasoning_effort",
    supported_reasoning_efforts=("low", "medium", "xhigh"),
)


def test_budget_reaches_qwen_without_an_advisory_effort() -> None:
    """An enable switch cannot cause the provider to receive a second depth dial."""
    request = decode_chat(
        {
            "model": "qwen3.8-max",
            "messages": [{"role": "user", "content": "hi"}],
            "thinking_budget": 4096,
            "enable_thinking": True,
            "max_output_tokens": 8192,
        }
    ).request
    public, provider = route_generation_parameter_requests((_PROFILE,), request)
    payload = dialect_stream_payload(_PROFILE, provider)
    assert payload["thinking_budget"] == 4096
    assert payload["enable_thinking"] is True
    assert payload["max_completion_tokens"] == 8192
    assert "max_tokens" not in payload
    assert "reasoning_effort" not in payload
    assert "reasoning" not in payload
    assert public.thinking_budget == 4096
    assert provider.thinking_default_enable is False
    changed = request.model_copy(update={"thinking_budget": 2048})
    assert canonical_request_sha256(request) != canonical_request_sha256(changed)


@pytest.mark.parametrize(
    "profile",
    (
        replace(_PROFILE, url="https://api.openai.com/v1/chat/completions"),
        replace(_PROFILE, url="https://openrouter.ai/api/v1/chat/completions"),
        replace(
            _PROFILE, url="https://dashscope-intl.aliyuncs.com.attacker.test/v1/chat/completions"
        ),
        replace(_PROFILE, dialect="openai_responses"),
        replace(_PROFILE, supports_reasoning=False, supported_reasoning_efforts=()),
        replace(_PROFILE, reasoning_effort_required=True, reasoning_effort="medium"),
        replace(_PROFILE, model_id="non-reasoning-model"),
    ),
)
def test_unsupported_routes_never_drop_the_budget(profile: GatewayWireProfile) -> None:
    """Both admission and direct dispatch reject incapable or unknown wires."""
    request = decode_chat(
        {
            "model": "qwen3.8-max",
            "messages": [{"role": "user", "content": "hi"}],
            "thinking_budget": 4096,
        }
    ).request
    with pytest.raises(ProviderParameterError) as admission:
        route_generation_parameter_requests((profile,), request)
    assert admission.value.param == "thinking_budget"
    with pytest.raises(ProviderParameterError) as dispatch:
        dialect_stream_payload(profile, request)
    assert dispatch.value.param == "thinking_budget"


@pytest.mark.parametrize("budget", (-2, True, "4096", 1.5, {}))
def test_invalid_budgets_fail_at_decode(budget: object) -> None:
    """Numeric bounds cannot be coerced from bools, strings or fractional values."""
    from typing import cast

    with pytest.raises(OpenAIProtocolError) as error:
        decode_chat(
            cast(
                JsonObject,
                {
                    "model": "qwen3.8-max",
                    "messages": [{"role": "user", "content": "hi"}],
                    "thinking_budget": budget,
                },
            )
        )
    assert error.value.detail.param == "thinking_budget"


@pytest.mark.parametrize(
    "control",
    (
        {"enable_thinking": False},
        {"reasoning_effort": "high"},
        {"reasoning": {"effort": "low"}},
        {"reasoning": {"enabled": False}},
        {"thinking": {"type": "disabled"}},
        {"chat_template_kwargs": {"enable_thinking": False}},
    ),
)
def test_budget_rejects_conflicting_controls(control: JsonObject) -> None:
    """An explicit off switch or effort must not erase the numeric budget."""
    with pytest.raises(OpenAIProtocolError) as error:
        decode_chat(
            {
                "model": "qwen3.8-max",
                "messages": [{"role": "user", "content": "hi"}],
                "thinking_budget": 4096,
                **control,
            }
        )
    assert error.value.detail.param == "thinking_budget"


@pytest.mark.parametrize(
    "model,total_cap",
    [
        ("qwen3.8-max", True),
        ("qwen3.8-max-0902", True),
        ("qwen3.8-flash", True),
        ("qwen3.8-27b", False),
        ("qwen3.8-2.4t-a95b", False),
        ("qwen3.7-plus-2026-05-26", True),
        ("qwen3.6-plus", True),
        ("qwen3.5-flash", True),
        ("qwen3-235b-a22b-thinking-2507", False),
        ("qwen3-vl-8b-thinking", False),
        ("glm-4.7", False),
        ("glm-5", False),
        ("glm-5.1", False),
        ("glm-5.2", False),
        ("kimi-k2-thinking", False),
        ("kimi-k2.5", False),
        ("kimi-k2.6", False),
        ("kimi-k2.7-code", False),
    ],
)
@pytest.mark.parametrize("nested", (False, True))
def test_qwen_cloud_budget_contracts(model: str, total_cap: bool, nested: bool) -> None:
    """Documented host/model pairs carry the exact budget inside a reserved total."""
    control: JsonObject = (
        {"thinking": {"type": "enabled", "budget_tokens": 2048}}
        if nested
        else {"thinking_budget": 2048}
    )
    request = decode_chat(
        {
            "model": "coding",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 4096,
            **control,
        }
    ).request
    profile = replace(_PROFILE, model_id=model)
    public, provider = route_generation_parameter_requests((profile,), request)
    payload = dialect_stream_payload(profile, provider)
    assert payload["thinking_budget"] == 2048
    assert payload["max_completion_tokens" if total_cap else "max_tokens"] == (
        4096 if total_cap else 2048
    )
    assert "thinking" not in payload
    assert "reasoning_effort" not in payload
    assert "reasoning" not in payload
    if model == "kimi-k2-thinking":
        assert "enable_thinking" not in payload
    else:
        assert payload["enable_thinking"] is True
    if nested:
        assert "thinking.budget_tokens->translated(thinking_budget)" in public.ignored_parameters
    if not total_cap:
        assert (
            "max_tokens->translated(total_output_minus_thinking_budget)"
            in public.ignored_parameters
        )


@pytest.mark.parametrize(
    "model",
    (
        "glm-5.3",
        "glm-5.3-flash",
        "kimi-k3",
        "qwen3.8-coder",
        "qwen9-max",
        "qwen3-32b-instruct",
        "qwen3.8-max-evil",
    ),
)
def test_qwen_model_names_do_not_imply_budget_support(model: str) -> None:
    """Ignoring models, unknown variants and future families stay closed."""
    request = decode_chat(
        {
            "model": "coding",
            "messages": [{"role": "user", "content": "hi"}],
            "thinking_budget": 2048,
        }
    ).request
    with pytest.raises(ProviderParameterError) as error:
        dialect_stream_payload(replace(_PROFILE, model_id=model), request)
    assert error.value.param == "thinking_budget"


def _gemini_profile(model: str) -> GatewayWireProfile:
    """Build a native Gemini profile whose budget contract is model-specific."""
    return GatewayWireProfile(
        dialect="gemini_generate_content",
        model_id=model,
        url=f"https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent?alt=sse",
        supports_reasoning=True,
        reasoning_wire_format="gemini_thinking",
    )


@pytest.mark.parametrize(
    "model,budget",
    [
        ("gemini-2.5-pro", 128),
        ("gemini-2.5-pro", 32768),
        ("gemini-2.5-pro", -1),
        ("gemini-2.5-flash", 0),
        ("gemini-2.5-flash", 1),
        ("gemini-2.5-flash", 24576),
        ("gemini-2.5-flash", -1),
        ("gemini-2.5-flash-lite", 512),
        ("gemini-2.5-flash-lite", 24576),
        ("gemini-2.5-flash-lite", 0),
        ("gemini-2.5-flash-lite", -1),
        ("models/gemini-2.5-flash", 1024),
    ],
)
def test_gemini_budget_ranges_and_sentinels(model: str, budget: int) -> None:
    """Native budgets never become Gemini 3 thinkingLevel controls."""
    request = decode_chat(
        {
            "model": "coding",
            "messages": [{"role": "user", "content": "hi"}],
            "thinking_budget": budget,
            "max_tokens": 40000,
        }
    ).request
    profile = _gemini_profile(model)
    public, provider = route_generation_parameter_requests((profile,), request)
    payload = dialect_stream_payload(profile, provider)
    generation = payload["generationConfig"]
    assert isinstance(generation, dict)
    assert generation["thinkingConfig"] == {"thinkingBudget": budget}
    assert generation["maxOutputTokens"] == 40000
    assert (
        "thinking_budget->translated(generationConfig.thinkingConfig.thinkingBudget)"
        in public.ignored_parameters
    )


@pytest.mark.parametrize(
    "model,budget",
    [
        ("gemini-2.5-pro", 0),
        ("gemini-2.5-pro", 127),
        ("gemini-2.5-pro", 32769),
        ("gemini-2.5-flash", 24577),
        ("gemini-2.5-flash-lite", 1),
        ("gemini-2.5-flash-lite", 511),
        ("gemini-2.5-flash-lite", 24577),
        ("gemini-3-pro", 1024),
        ("gemini-2.5-flash-image", 1024),
    ],
)
def test_gemini_rejects_unsupported_budgets_and_models(model: str, budget: int) -> None:
    """Per-model domains and unknown variants fail before a provider call."""
    request = decode_chat(
        {
            "model": "coding",
            "messages": [{"role": "user", "content": "hi"}],
            "thinking_budget": budget,
        }
    ).request
    with pytest.raises(ProviderParameterError) as error:
        dialect_stream_payload(_gemini_profile(model), request)
    assert error.value.param == "thinking_budget"


@pytest.mark.parametrize(
    "model",
    (
        "claude-sonnet-4-6",
        "claude-opus-4-6",
        "claude-haiku-4-5",
        "claude-opus-4-5",
        "claude-mythos-preview",
    ),
)
def test_top_level_budget_reaches_budget_capable_anthropic_models(model: str) -> None:
    """The shared numeric Chat field also supports the Anthropic native wire."""
    request = decode_chat(
        {
            "model": "coding",
            "messages": [{"role": "user", "content": "hi"}],
            "thinking_budget": 2048,
            "max_tokens": 4096,
        }
    ).request
    profile = GatewayWireProfile(
        dialect="anthropic_messages",
        model_id=model,
        url="https://api.anthropic.com/v1/messages",
        supports_reasoning=True,
        reasoning_wire_format="anthropic_adaptive",
    )
    public, provider = route_generation_parameter_requests((profile,), request)
    payload = dialect_stream_payload(profile, provider)
    assert payload["thinking"] == {"type": "enabled", "budget_tokens": 2048}
    assert "thinking_budget" not in payload
    assert "thinking_budget->translated(thinking.budget_tokens)" in public.ignored_parameters


@pytest.mark.parametrize("nested", (False, True))
def test_anthropic_numeric_budget_applies_reasoning_sampling_policy(nested: bool) -> None:
    """Top-level and nested budgets enable thinking before temperature/top-p admission."""
    control: JsonObject = (
        {"thinking": {"type": "enabled", "budget_tokens": 2048}}
        if nested
        else {"thinking_budget": 2048}
    )
    request = decode_chat(
        {
            "model": "coding",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 4096,
            "temperature": 0.5,
            "top_p": 0.9,
            **control,
        }
    ).request
    profile = GatewayWireProfile(
        dialect="anthropic_messages",
        model_id="claude-haiku-4-5",
        url="https://api.anthropic.com/v1/messages",
        supports_reasoning=True,
        reasoning_wire_format="anthropic_adaptive",
        supports_temperature=True,
        supports_top_p=True,
        sampling_requires_reasoning_none=True,
    )
    public, provider = route_generation_parameter_requests((profile,), request)
    assert provider.temperature is None and provider.top_p is None
    assert "temperature->dropped(set_reasoning_effort_none)" in public.ignored_parameters
    assert "top_p->dropped(set_reasoning_effort_none)" in public.ignored_parameters
    payload = dialect_stream_payload(profile, provider)
    assert payload["thinking"] == {"type": "enabled", "budget_tokens": 2048}
    assert "temperature" not in payload and "top_p" not in payload


@pytest.mark.parametrize("budget", (0, -1, 1024))
def test_gemini_budget_sentinels_join_replay_identity(budget: int) -> None:
    """Explicit zero, dynamic thinking and a numerical target are distinct requests."""
    base: JsonObject = {"model": "coding", "messages": [{"role": "user", "content": "hi"}]}
    request = decode_chat({**base, "thinking_budget": budget}).request
    assert canonical_request_sha256(request) != canonical_request_sha256(decode_chat(base).request)


@pytest.mark.parametrize("budget", (4096, 5000))
def test_split_qwen_budget_cannot_exceed_the_reserved_total(budget: int) -> None:
    """A separate answer cap cannot become zero or negative."""
    request = decode_chat(
        {
            "model": "coding",
            "messages": [{"role": "user", "content": "hi"}],
            "thinking_budget": budget,
            "max_tokens": 4096,
        }
    ).request
    with pytest.raises(ProviderParameterError) as error:
        dialect_stream_payload(replace(_PROFILE, model_id="kimi-k2.5"), request)
    assert error.value.param == "thinking_budget"


@pytest.mark.parametrize(
    "control",
    ({"enable_thinking": True}, {"enable_thinking": False}, {"thinking": {"type": "enabled"}}),
)
def test_zero_budget_cannot_conflict_with_an_enable_switch(control: JsonObject) -> None:
    """Provider-specific zero semantics require the numeric control to stand alone."""
    with pytest.raises(OpenAIProtocolError) as error:
        decode_chat(
            {
                "model": "coding",
                "messages": [{"role": "user", "content": "hi"}],
                "thinking_budget": 0,
                **control,
            }
        )
    assert error.value.detail.param == "thinking_budget"


@pytest.mark.parametrize("nested", (False, True))
def test_native_vertex_gemini_preserves_numeric_budget(nested: bool) -> None:
    """Vertex's generateContent dialect shares Google's numeric control contract."""
    control: JsonObject = (
        {"thinking": {"type": "enabled", "budget_tokens": 2048}}
        if nested
        else {"thinking_budget": 2048}
    )
    request = decode_chat(
        {"model": "coding", "messages": [{"role": "user", "content": "hi"}], **control}
    ).request
    profile = replace(
        _gemini_profile("gemini-2.5-pro"),
        url="https://us-central1-aiplatform.googleapis.com/v1/projects/test/locations/us-central1/publishers/google/models/gemini-2.5-pro:streamGenerateContent?alt=sse",
    )
    public, provider = route_generation_parameter_requests((profile,), request)
    payload = dialect_stream_payload(profile, provider)
    assert payload["generationConfig"] == {"thinkingConfig": {"thinkingBudget": 2048}}
    assert public.provider_thinking_config == request.provider_thinking_config
