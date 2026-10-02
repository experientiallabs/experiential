"""Tool ownership preserves public types and serialized request identity."""

from __future__ import annotations

import pytest

from exp.runtime.gateway import contracts, tool_contracts
from exp.runtime.gateway.contracts import GatewayApiSurface, GatewayMessage, GatewayRequest
from exp.runtime.gateway.replay_identity import canonical_request_sha256


def test_public_tool_exports_are_the_owned_contract_types() -> None:
    """Public imports retain the same class objects, not aliases with different validation."""
    assert contracts.GatewayToolDefinition is tool_contracts.GatewayToolDefinition
    assert contracts.GatewayProviderNativeTool is tool_contracts.GatewayProviderNativeTool
    assert contracts.GatewayNamedToolChoice is tool_contracts.GatewayNamedToolChoice


def test_tool_defaults_and_carrier_serialization_are_unchanged() -> None:
    """Provider carriers stay excluded from the tool document while retaining exact values."""
    tool = tool_contracts.GatewayToolDefinition(
        name="lookup",
        parameters={"type": "object"},
        cache_control={"type": "ephemeral"},
        eager_input_streaming=False,
        defer_loading=False,
        allowed_callers=("direct",),
        input_examples=({"id": "one"},),
    )
    assert tool.model_dump(mode="json") == {
        "name": "lookup",
        "description": None,
        "parameters": {"type": "object"},
        "strict": False,
    }
    assert tool.has_anthropic_tool_carriers()
    assert not tool_contracts.GatewayToolDefinition(
        name="lookup", parameters={}
    ).has_anthropic_tool_carriers()
    assert tool_contracts.GatewayProviderNativeTool(
        index=0, tool={"type": "web_search"}
    ).model_dump() == {
        "index": 0,
        "tool": {"type": "web_search"},
    }
    assert tool_contracts.GatewayNamedToolChoice(name="lookup").model_dump() == {"name": "lookup"}
    with pytest.raises(ValueError):
        tool_contracts.GatewayNamedToolChoice(name="")


def test_reexported_tools_produce_the_same_replay_identity() -> None:
    """Moving the owner cannot change canonical request identity or provider tool carriers."""
    values = {"name": "lookup", "parameters": {"type": "object"}, "input_examples": [{"id": "one"}]}
    owned = tool_contracts.GatewayToolDefinition.model_validate(values)
    exported = contracts.GatewayToolDefinition.model_validate(values)
    request = GatewayRequest(
        surface=GatewayApiSurface.MESSAGES,
        messages=(GatewayMessage(role="user", content="hi"),),
        tools=(owned,),
    )
    equivalent = request.model_copy(update={"tools": (exported,)})
    assert canonical_request_sha256(request) == canonical_request_sha256(equivalent)
    without_example = request.model_copy(
        update={"tools": (owned.model_copy(update={"input_examples": None}),)}
    )
    assert canonical_request_sha256(request) != canonical_request_sha256(without_example)
