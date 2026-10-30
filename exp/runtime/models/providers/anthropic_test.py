"""Tests for the Anthropic official-origin check and its safeguards gate."""

from __future__ import annotations

from typing import Literal

import pytest

from exp.common.models import ToolChoice
from exp.common.tasks import ToolSchema
from exp.runtime.gateway.contracts import GatewayApiSurface, GatewayMessage, GatewayRequest
from exp.runtime.models.providers.anthropic import (
    AnthropicClient,
    anthropic_messages_request,
    anthropic_official_origin,
    safeguards_for_upstream,
)
from exp.runtime.models.providers.errors import ProviderCapabilityError
from exp.runtime.models.providers.openai_compatible_test import _request, _snapshot
from exp.runtime.models.providers.transport import JsonHttpResponse, ScriptedJsonTransport


@pytest.mark.parametrize(
    ("url", "official"),
    [
        ("https://api.anthropic.com/v1/messages", True),
        ("https://api.anthropic.com:443/v1/messages", True),
        ("https://x.services.ai.azure.com/anthropic/v1/messages", False),
        ("https://claude-proxy.example.com/v1/messages", False),
        ("http://api.anthropic.com/v1/messages", False),
        ("https://api.anthropic.com.evil.test/v1/messages", False),
        ("https://user@api.anthropic.com/v1/messages", False),
        ("https://api.anthropic.com:8443/v1/messages", False),
        ("https://api.anthropic.com/v10/messages", False),
    ],
)
def test_only_anthropics_own_api_is_the_official_origin(url: str, official: bool) -> None:
    """Exact scheme, host, port and path prefix; no lookalike passes."""
    assert anthropic_official_origin(url) is official


def test_safeguards_survive_only_on_the_official_origin() -> None:
    """The gate clears the field elsewhere and leaves other requests untouched."""
    request = GatewayRequest(
        surface=GatewayApiSurface.MESSAGES,
        messages=(GatewayMessage(role="user", content="hi"),),
        safeguards=({"type": "dangerous_tool_use"},),
    )
    official = safeguards_for_upstream("https://api.anthropic.com/v1/messages", request)
    assert official is request
    azure = safeguards_for_upstream(
        "https://x.services.ai.azure.com/anthropic/v1/messages", request
    )
    assert azure.safeguards is None
    plain = request.model_copy(update={"safeguards": None})
    assert safeguards_for_upstream("https://proxy.test/v1/messages", plain) is plain


@pytest.mark.parametrize("declared", [None, True])
@pytest.mark.parametrize(
    ("choice", "wire"),
    [
        (ToolChoice(name="create_ticket"), {"type": "tool", "name": "create_ticket"}),
        ("required", {"type": "any"}),
    ],
)
def test_a_model_not_declared_unable_to_force_gets_the_forced_choice(
    declared: bool | None, choice: ToolChoice | Literal["required"], wire: dict[str, str]
) -> None:
    """Unknown and declared support both send the caller's forced choice unchanged."""
    payload = anthropic_messages_request(
        "claude-sonnet-5", _request(tool_choice=choice), supports_forced_tool_choice=declared
    )

    assert payload["tool_choice"] == wire
    assert payload["system"] == "You are precise."


@pytest.mark.parametrize("choice", [ToolChoice(name="create_ticket"), "required"])
def test_a_model_declared_unable_to_force_is_steered_to_its_only_tool(
    choice: ToolChoice | Literal["required"],
) -> None:
    """The forced choice becomes one automatic call, and the instruction names the tool."""
    payload = anthropic_messages_request(
        "claude-sonnet-5-5", _request(tool_choice=choice), supports_forced_tool_choice=False
    )

    assert payload["tool_choice"] == {"type": "auto", "disable_parallel_tool_use": True}
    assert payload["system"] == (
        "You are precise.\n\nRespond only by calling the create_ticket tool."
    )


@pytest.mark.parametrize("choice", ["auto", "none", None])
def test_an_unforced_choice_is_untouched_on_a_model_that_cannot_force(
    choice: Literal["auto", "none"] | None,
) -> None:
    """Only a forced choice is rewritten; no steering instruction is added otherwise."""
    payload = anthropic_messages_request(
        "claude-sonnet-5-5", _request(tool_choice=choice), supports_forced_tool_choice=False
    )

    assert payload.get("tool_choice") == (None if choice is None else {"type": choice})
    assert payload["system"] == "You are precise."


def test_a_forced_choice_among_several_tools_is_refused_before_dispatch() -> None:
    """With a second tool, auto could pick the other one, so nothing is sent."""
    one_tool = _request(tool_choice=ToolChoice(name="create_ticket"))
    second = ToolSchema(
        name="close_ticket", description="Close one ticket.", input_schema={"type": "object"}
    )

    for choice in (ToolChoice(name="create_ticket"), "required"):
        request = one_tool.model_copy(
            update={"tools": (*one_tool.tools, second), "tool_choice": choice}
        )
        with pytest.raises(ProviderCapabilityError) as caught:
            anthropic_messages_request(
                "claude-sonnet-5-5", request, supports_forced_tool_choice=False
            )
        assert caught.value.capability == "forced_tool_choice"


@pytest.mark.parametrize(
    ("model_id", "declared", "wire"),
    [
        ("claude-sonnet-5", None, {"type": "tool", "name": "create_ticket"}),
        ("claude-sonnet-5-5", False, {"type": "auto", "disable_parallel_tool_use": True}),
    ],
)
def test_the_client_sends_the_declared_choice_and_returns_the_one_tool_call(
    model_id: str, declared: bool | None, wire: dict[str, object]
) -> None:
    """The catalog's declaration reaches the wire; the returned call is parsed either way."""
    transport = ScriptedJsonTransport(
        [
            JsonHttpResponse(
                status_code=200,
                body={
                    "model": "claude-sonnet-5-5",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": "create_ticket",
                            "input": {"priority": "high"},
                        }
                    ],
                    "stop_reason": "tool_use",
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 5,
                        "cache_read_input_tokens": 0,
                        "cache_creation_input_tokens": 0,
                    },
                },
            )
        ]
    )
    client = AnthropicClient(
        model=_snapshot("anthropic", model_id),
        api_key="fixture",
        transport=transport,
        supports_forced_tool_choice=declared,
    )

    response = client.complete(_request(tool_choice=ToolChoice(name="create_ticket")))

    assert transport.requests[0][2]["tool_choice"] == wire
    (call,) = response.output.tool_calls
    assert (call.name, call.arguments) == ("create_ticket", {"priority": "high"})


@pytest.mark.parametrize(
    "model_id", ["claude-sonnet-5-5", "claude-opus-5-5-20261001", "claude-fable-5-1@20260901"]
)
def test_an_undeclared_model_known_to_reject_forcing_names_the_catalog_field(
    model_id: str,
) -> None:
    """The catalog stays the only gate: nothing is rewritten, the operator is told what to set."""
    with pytest.raises(
        ProviderCapabilityError, match="declare supports_forced_tool_choice = false"
    ):
        anthropic_messages_request(model_id, _request(tool_choice="required"))

    unforced = anthropic_messages_request(model_id, _request(tool_choice="auto"))
    assert unforced["tool_choice"] == {"type": "auto"}
