"""Inline tests for the dialect dispatch seam.

The request-shaping behaviour is exercised through ``streaming_requests_test``
and every payload-builder suite; this module pins the disclosure wording that
callers read off the wire.
"""

from __future__ import annotations

import pytest
from pydantic import JsonValue

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.contracts import (
    GatewayApiSurface,
    GatewayMessage,
    GatewayRequest,
    GatewayToolDefinition,
)
from exp.runtime.models.providers.base import GatewayWireProfile
from exp.runtime.models.providers.dialect_dispatch import (
    CACHE_CONTROL_NOT_FORWARDED_SUFFIX,
    dialect_stream_payload,
)
from exp.runtime.models.providers.errors import ProviderCapabilityError


def test_cache_control_disclosure_names_where_cache_reads_show_up() -> None:
    """The unforwarded-marker disclosure is a stable wire string that never reads as "ignored".

    It travels in ``x-experiential-ignored-parameters`` beside a billed
    ``cache_read_input_tokens`` on OpenAI-compatible routes, so it has to say
    that caching is the provider's decision and where any reads are reported.
    """
    assert CACHE_CONTROL_NOT_FORWARDED_SUFFIX == (
        "->not_forwarded(provider_decides_caching;"
        " cache reads reported in usage.cache_read_input_tokens)"
    )
    assert "ignored" not in CACHE_CONTROL_NOT_FORWARDED_SUFFIX
    assert "usage.cache_read_input_tokens" in CACHE_CONTROL_NOT_FORWARDED_SUFFIX


@pytest.mark.parametrize("surface", list(GatewayApiSurface))
@pytest.mark.parametrize("stream", [False, True])
def test_connection_us_geography_overrides_caller_on_every_surface(
    surface: GatewayApiSurface, stream: bool
) -> None:
    """The trusted connection wins after surface translation, including a conflicting caller."""
    request = GatewayRequest(
        surface=surface,
        stream=stream,
        messages=(GatewayMessage(role="user", content="hello"),),
        inference_geo="global" if surface == GatewayApiSurface.MESSAGES else None,
    )
    profile = GatewayWireProfile(
        dialect="anthropic_messages",
        url="https://api.anthropic.com/v1/messages",
        model_id="claude-sonnet-4-6",
        maximum_output_tokens=128_000,
        inference_geo="us",
    )
    assert dialect_stream_payload(profile, request)["inference_geo"] == "us"
    assert request.inference_geo == ("global" if surface == GatewayApiSurface.MESSAGES else None)


@pytest.mark.parametrize("caller", [None, "us", "global"])
def test_unrestricted_connection_preserves_caller_geography(caller: str | None) -> None:
    """No operator setting leaves the existing request contract unchanged."""
    request = GatewayRequest(
        surface=GatewayApiSurface.MESSAGES,
        messages=(GatewayMessage(role="user", content="hello"),),
        inference_geo=caller,
    )
    profile = GatewayWireProfile(
        dialect="anthropic_messages",
        url="https://api.anthropic.com/v1/messages",
        model_id="claude-sonnet-4-6",
        maximum_output_tokens=128_000,
    )
    assert dialect_stream_payload(profile, request).get("inference_geo") == caller


def test_non_anthropic_dialect_refuses_geography_constraint() -> None:
    """An incorrectly constructed profile cannot silently drop the constraint."""
    profile = GatewayWireProfile(
        dialect="openai_compatible",
        url="https://example.test/v1/chat/completions",
        inference_geo="us",
    )
    request = GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(GatewayMessage(role="user", content="hi"),),
    )
    with pytest.raises(ProviderCapabilityError, match="inference_geo"):
        dialect_stream_payload(profile, request)


_DESCRIPTIONLESS_TOOL_PROFILES = (
    GatewayWireProfile(
        dialect="openai_compatible",
        url="https://api.mistral.ai/v1/chat/completions",
        model_id="ministral-3b-2512",
    ),
    GatewayWireProfile(
        dialect="openai_responses",
        url="https://api.openai.com/v1/responses",
        model_id="gpt-5.4",
    ),
    GatewayWireProfile(
        dialect="anthropic_messages",
        url="https://api.anthropic.com/v1/messages",
        model_id="claude-sonnet-4-6",
        maximum_output_tokens=128_000,
    ),
    GatewayWireProfile(
        dialect="gemini_generate_content",
        url="https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash",
        model_id="gemini-2.5-flash",
    ),
    GatewayWireProfile(
        dialect="bedrock_converse_stream",
        url="https://bedrock-runtime.us-east-1.amazonaws.com",
        model_id="mistral.ministral-3-3b-instruct",
    ),
)


def _null_paths(value: JsonValue, path: str = "") -> list[str]:
    """Return the JSON paths under ``value`` whose value is null.

    Args:
        value: Decoded JSON value to walk.
        path: Path of ``value`` itself.

    Returns:
        Every path holding ``None``, in walk order.
    """
    if value is None:
        return [path]
    if isinstance(value, dict):
        return [hit for key, item in value.items() for hit in _null_paths(item, f"{path}.{key}")]
    if isinstance(value, list):
        return [
            hit for index, item in enumerate(value) for hit in _null_paths(item, f"{path}[{index}]")
        ]
    return []


def _tools_of(payload: JsonObject) -> JsonValue:
    """Return the tool declarations of one dialect payload, wherever the dialect keeps them."""
    if "toolConfig" in payload:
        tool_config = payload["toolConfig"]
        assert isinstance(tool_config, dict)
        if "tools" in tool_config:
            return tool_config["tools"]
    return payload["tools"]


@pytest.mark.parametrize(
    "surface",
    [GatewayApiSurface.CHAT_COMPLETIONS, GatewayApiSurface.RESPONSES, GatewayApiSurface.MESSAGES],
)
@pytest.mark.parametrize(
    "profile", _DESCRIPTIONLESS_TOOL_PROFILES, ids=lambda profile: profile.dialect
)
def test_descriptionless_tool_never_serializes_a_null_field(
    profile: GatewayWireProfile, surface: GatewayApiSurface
) -> None:
    """A tool without a description omits the field instead of sending null, on every dialect.

    Mistral rejects ``"description": null`` with "Input should be a valid
    string" (live 2026-10-06, ministral-3b-2512), directly and behind
    OpenRouter and Azure; no wire treats null as meaning absent.
    """
    request = GatewayRequest(
        surface=surface,
        messages=(GatewayMessage(role="user", content="Weather in Paris?"),),
        tools=(
            GatewayToolDefinition(
                name="get_weather",
                parameters={"type": "object", "properties": {"city": {"type": "string"}}},
            ),
        ),
    )
    tools = _tools_of(dialect_stream_payload(profile, request))
    assert _null_paths(tools) == []
    assert "get_weather" in str(tools)


@pytest.mark.parametrize("dialect", ["openai_compatible", "openai_responses"])
def test_openai_family_tool_keeps_a_supplied_description_and_explicit_strict(dialect: str) -> None:
    """A supplied description is forwarded verbatim; ``strict`` stays an explicit boolean.

    Responses defaults an omitted ``strict`` to true, so the gateway always
    states it rather than relying on a wire default.
    """
    profile = next(item for item in _DESCRIPTIONLESS_TOOL_PROFILES if item.dialect == dialect)
    request = GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(GatewayMessage(role="user", content="Weather in Paris?"),),
        tools=(
            GatewayToolDefinition(
                name="get_weather", description="Look up weather.", parameters={"type": "object"}
            ),
        ),
    )
    tools = dialect_stream_payload(profile, request)["tools"]
    assert isinstance(tools, list)
    (tool,) = tools
    assert isinstance(tool, dict)
    declaration = tool["function"] if dialect == "openai_compatible" else tool
    assert isinstance(declaration, dict)
    assert declaration["description"] == "Look up weather."
    assert declaration["strict"] is False
    assert declaration["parameters"] == {"type": "object"}
