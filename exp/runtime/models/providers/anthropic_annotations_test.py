"""Tests for Anthropic-native annotation disclosures."""

from __future__ import annotations

from exp.common.models.model import ToolCall
from exp.runtime.gateway.contracts import GatewayApiSurface, GatewayMessage, GatewayRequest
from exp.runtime.models.providers.anthropic_annotations import (
    TOOL_USE_CALLER_DROPPED,
    anthropic_annotation_disclosures,
)
from exp.runtime.models.providers.base import GatewayWireProfile

_ANTHROPIC = GatewayWireProfile(
    dialect="anthropic_messages", url="https://api.anthropic.com/v1/messages", model_id="claude"
)
_CHAT = GatewayWireProfile(
    dialect="openai_compatible", url="https://api.openai.com/v1/chat/completions", model_id="gpt"
)


def _request() -> GatewayRequest:
    call = ToolCall(
        call_id="toolu_1",
        name="Bash",
        arguments={},
        provider_anthropic_caller={"type": "direct"},
    )
    return GatewayRequest(
        surface=GatewayApiSurface.MESSAGES,
        messages=(
            GatewayMessage(role="user", content="go"),
            GatewayMessage(role="assistant", tool_calls=(call,)),
            GatewayMessage(role="tool", tool_call_id="toolu_1", content="ok"),
        ),
    )


def test_an_echoed_caller_is_disclosed_only_off_the_anthropic_wire() -> None:
    """Anthropic rungs carry the caller; a route with any other rung discloses its drop."""
    request = _request()
    assert anthropic_annotation_disclosures((_ANTHROPIC,), request) == ()
    assert anthropic_annotation_disclosures((_CHAT,), request) == (TOOL_USE_CALLER_DROPPED,)
    assert anthropic_annotation_disclosures((_ANTHROPIC, _CHAT), request) == (
        TOOL_USE_CALLER_DROPPED,
    )
