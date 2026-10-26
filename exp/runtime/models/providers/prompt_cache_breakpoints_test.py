"""OpenAI prompt-cache controls reach each dialect as its own equivalent, or are disclosed."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from exp.common.core.artifacts import JsonObject, JsonValue
from exp.common.models import BillingSource, ModelSnapshot
from exp.runtime.gateway.contracts import GatewayRequest
from exp.runtime.models.providers.azure import AzureClient
from exp.runtime.models.providers.base import GatewayWireProfile
from exp.runtime.models.providers.dialect_dispatch import dialect_stream_payload
from exp.runtime.models.providers.openai import OpenAIClient
from exp.runtime.models.providers.openai_compatible import (
    OPENROUTER_BASE_URL,
    OpenAICompatibleClient,
    OpenRouterClient,
)
from exp.runtime.models.providers.prompt_cache_options import (
    PROMPT_CACHE_BREAKPOINT_TRANSLATED,
    PROMPT_CACHE_MODE_NOT_FORWARDED,
)
from exp.runtime.models.providers.streaming_requests import route_generation_parameter_requests
from exp.runtime.openai_protocol.requests import decode_chat, decode_responses

_BREAKPOINT = {"mode": "explicit"}


def _snapshot(provider: str, model_id: str = "gpt-5.6-sol") -> ModelSnapshot:
    """One frozen model identity for a client under test."""
    return ModelSnapshot(
        billing_source=BillingSource.HOST_MANAGED,
        provider=provider,
        model_id=model_id,
        revision="fixture-revision",
        capabilities_sha256="a" * 64,
        connection_sha256="a" * 64,
    )


def _openai() -> GatewayWireProfile:
    """OpenAI's own Responses rung."""
    return OpenAIClient(model=_snapshot("openai"), api_key="k").gateway_wire_profile()


def _novita_responses() -> GatewayWireProfile:
    """A Responses rung on an origin that does not document the controls."""
    return OpenAIClient(
        model=_snapshot("openai"), api_key="k", base_url="https://api.novita.ai/v3/openai"
    ).gateway_wire_profile()


def _openrouter() -> GatewayWireProfile:
    """OpenRouter's Chat rung, which carries both breakpoint spellings."""
    return OpenRouterClient(
        model=_snapshot("openrouter", "openai/gpt-5.6-sol"),
        base_url=OPENROUTER_BASE_URL,
        api_key="k",
    ).gateway_wire_profile()


def _azure(surface: str = "openai_deployments") -> GatewayWireProfile:
    """An Azure Chat rung on one of its two surfaces."""
    return AzureClient(
        model=_snapshot("azure"),
        endpoint="https://resource.openai.azure.com",
        api_key="k",
        api_version="v1" if surface == "openai_deployments" else "2024-05-01-preview",
        api_surface=surface,
    ).gateway_wire_profile()


def _vllm() -> GatewayWireProfile:
    """A customer-run OpenAI-compatible server (strict about unknown fields)."""
    return OpenAICompatibleClient(
        model=_snapshot("openai-compatible"), base_url="http://vllm.internal:8000/v1", api_key="k"
    ).gateway_wire_profile()


def _anthropic() -> GatewayWireProfile:
    """An Anthropic Messages rung."""
    return GatewayWireProfile(
        dialect="anthropic_messages",
        url="https://api.anthropic.com/v1/messages",
        model_id="claude-opus-4-8",
        maximum_output_tokens=4096,
    )


def _chat(**options: JsonValue) -> GatewayRequest:
    """A Chat request with a marked system prompt and a marked user block."""
    body: JsonObject = {
        "model": "gpt-5.6-sol",
        "max_tokens": 64,
        "messages": [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "rules", "prompt_cache_breakpoint": _BREAKPOINT}
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "doc "},
                    {"type": "text", "text": "body", "prompt_cache_breakpoint": _BREAKPOINT},
                    {"type": "text", "text": " question"},
                ],
            },
        ],
        "prompt_cache_options": {"mode": "explicit", **options},
    }
    return decode_chat(body).request.model_copy(update={"stream": True})


def _responses(**options: JsonValue) -> GatewayRequest:
    """The same conversation on the Responses surface."""
    body: JsonObject = {
        "model": "gpt-5.6-sol",
        "input": [
            {
                "role": "developer",
                "content": [
                    {"type": "input_text", "text": "rules", "prompt_cache_breakpoint": _BREAKPOINT}
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "doc "},
                    {"type": "input_text", "text": "body", "prompt_cache_breakpoint": _BREAKPOINT},
                    {"type": "input_text", "text": " question"},
                ],
            },
        ],
        "prompt_cache_options": {"mode": "explicit", **options},
    }
    return decode_responses(body).request.model_copy(update={"stream": True})


def _objects(value: JsonValue) -> list[JsonObject]:
    """A JSON list of objects, asserted."""
    assert isinstance(value, list)
    return [item for item in value if isinstance(item, dict)]


def _marked(parts: JsonValue, field: str) -> list[str]:
    """The texts of the parts carrying ``field``."""
    assert isinstance(parts, list)
    return [str(part["text"]) for part in parts if isinstance(part, dict) and field in part]


@pytest.mark.parametrize("request_factory", (_chat, _responses))
def test_openai_responses_rung_forwards_options_and_breakpoints_verbatim(
    request_factory: Callable[[], GatewayRequest],
) -> None:
    """Both surfaces reach OpenAI's Responses wire with mode and per-block breakpoints.

    A marked instruction rides an input item: OpenAI's guide says top-level
    ``instructions`` cannot contain an explicit breakpoint.
    """
    payload = dialect_stream_payload(_openai(), request_factory())
    assert payload["prompt_cache_options"] == {"mode": "explicit"}
    assert "instructions" not in payload
    system, user = _objects(payload["input"])
    assert _marked(system["content"], "prompt_cache_breakpoint") == ["rules"]
    assert _marked(user["content"], "prompt_cache_breakpoint") == ["body"]
    assert "cache_control" not in str(payload)


def test_responses_twelve_hour_lifetime_rides_beside_the_mode() -> None:
    """A request-level ``"12h"`` joins the mode; the breakpoints inherit it."""
    payload = dialect_stream_payload(_openai(), _responses(ttl="12h"))
    assert payload["prompt_cache_options"] == {"mode": "explicit", "ttl": "12h"}


@pytest.mark.parametrize("profile_factory", (_openrouter, lambda: _azure()))
def test_chat_rungs_that_carry_the_fields_forward_them_verbatim(
    profile_factory: Callable[[], GatewayWireProfile],
) -> None:
    """OpenRouter and Azure OpenAI Chat rungs send the caller's own spelling."""
    payload = dialect_stream_payload(profile_factory(), _chat())
    assert payload["prompt_cache_options"] == {"mode": "explicit"}
    system, user = _objects(payload["messages"])
    assert _marked(system["content"], "prompt_cache_breakpoint") == ["rules"]
    assert _marked(user["content"], "prompt_cache_breakpoint") == ["body"]
    # OpenRouter takes cache_control too; the caller's breakpoints are not doubled.
    assert "cache_control" not in str(payload)


def test_anthropic_rung_maps_breakpoints_to_ephemeral_cache_control() -> None:
    """Each breakpoint caches at the same block, at Anthropic's 5-minute default."""
    request = _chat()
    payload = dialect_stream_payload(_anthropic(), request)
    assert "prompt_cache_options" not in payload
    assert "prompt_cache_breakpoint" not in str(payload)
    assert _marked(payload["system"], "cache_control") == ["rules"]
    user = _objects(payload["messages"])[0]
    assert _marked(user["content"], "cache_control") == ["body"]
    assert all(
        part.get("cache_control") in (None, {"type": "ephemeral"})
        for part in _objects(user["content"])
    )
    public, _provider = route_generation_parameter_requests((_anthropic(),), request)
    assert PROMPT_CACHE_BREAKPOINT_TRANSLATED in public.ignored_parameters
    # Anthropic caches only at markers, so explicit mode already holds there.
    assert PROMPT_CACHE_MODE_NOT_FORWARDED not in public.ignored_parameters


@pytest.mark.parametrize(
    "profile_factory", (_vllm, _novita_responses, lambda: _azure("model_inference"))
)
def test_wires_without_the_fields_drop_them_and_disclose(
    profile_factory: Callable[[], GatewayWireProfile],
) -> None:
    """A strict or undocumented origin never receives them; the route discloses it."""
    profile = profile_factory()
    assert profile.forwards_prompt_cache_options is False
    request = _chat()
    payload = dialect_stream_payload(profile, request)
    assert "prompt_cache_options" not in payload
    assert "prompt_cache_breakpoint" not in str(payload)
    public, _provider = route_generation_parameter_requests((profile,), request)
    assert PROMPT_CACHE_MODE_NOT_FORWARDED in public.ignored_parameters
    assert any(
        entry.startswith("messages.content.prompt_cache_breakpoint->not_forwarded")
        for entry in public.ignored_parameters
    )


def test_mixed_route_keeps_the_controls_for_the_rung_that_honors_them() -> None:
    """Like every cache marker, the controls are disclosed only when no rung honors them."""
    public, _provider = route_generation_parameter_requests((_openai(), _vllm()), _chat())
    assert PROMPT_CACHE_MODE_NOT_FORWARDED not in public.ignored_parameters
    assert not any("prompt_cache_breakpoint" in entry for entry in public.ignored_parameters)


def test_implicit_mode_and_default_lifetime_are_never_sent() -> None:
    """Copilot's hardcoded implicit mode stays off the wire (older models refuse the field)."""
    request = decode_responses(
        {"model": "m", "input": "x", "prompt_cache_options": {"mode": "implicit", "ttl": "30m"}}
    ).request
    assert "prompt_cache_options" not in dialect_stream_payload(_openai(), request)


def test_tool_result_breakpoint_marks_the_whole_result_on_chat() -> None:
    """A tool message is one block upstream; its breakpoint marks the result's end."""
    request = decode_chat(
        {
            "model": "m",
            "messages": [
                {"role": "user", "content": "go"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "f", "arguments": "{}"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": [
                        {"type": "text", "text": "out", "prompt_cache_breakpoint": _BREAKPOINT}
                    ],
                },
            ],
            "tools": [{"type": "function", "function": {"name": "f", "parameters": {}}}],
        }
    ).request.model_copy(update={"stream": True})
    chat_tool = _objects(dialect_stream_payload(_azure(), request)["messages"])[-1]
    assert chat_tool["content"] == [
        {"type": "text", "text": "out", "prompt_cache_breakpoint": _BREAKPOINT}
    ]
    output = _objects(dialect_stream_payload(_openai(), request)["input"])[-1]
    assert output["output"] == [
        {"type": "input_text", "text": "out", "prompt_cache_breakpoint": _BREAKPOINT}
    ]
