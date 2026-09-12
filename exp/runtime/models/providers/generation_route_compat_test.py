"""Tests for per-deployment generation-parameter route compatibility."""

from __future__ import annotations

from exp.common.models.model import ReasoningEffort
from exp.runtime.gateway.contracts import GatewayApiSurface, GatewayMessage, GatewayRequest
from exp.runtime.models.providers.base import GatewayWireProfile
from exp.runtime.models.providers.generation_route_compat import (
    request_with_reasoning_effort,
    snap_effort_onto_route,
)


def _messages_request(**overrides: object) -> GatewayRequest:
    """Build one Messages-surface request with overrides applied."""
    request = GatewayRequest(
        surface=GatewayApiSurface.MESSAGES,
        messages=(GatewayMessage(role="user", content="go"),),
    )
    return request.model_copy(update=dict(overrides))


def _ladder_profile(*efforts: ReasoningEffort) -> GatewayWireProfile:
    """Build one OpenAI-compatible reasoning rung with an explicit effort ladder."""
    return GatewayWireProfile(
        dialect="openai_compatible",
        url="https://provider.test",
        model_id="deepseek-v4-flash",
        supports_reasoning=True,
        reasoning_wire_format="reasoning_effort",
        supported_reasoning_efforts=tuple(efforts),
    )


def test_request_with_reasoning_effort_rewrites_every_effort_channel() -> None:
    """The effort lands on ``reasoning_effort`` and on a stated ``output_config.effort``."""
    both = request_with_reasoning_effort(
        _messages_request(
            reasoning_effort="medium",
            provider_output_config={"effort": "medium", "format": {"type": "text"}},
        ),
        "high",
    )
    assert both.reasoning_effort == "high"
    assert both.provider_output_config == {"effort": "high", "format": {"type": "text"}}
    # An output_config without an effort key is left exactly as sent.
    untouched = request_with_reasoning_effort(
        _messages_request(reasoning_effort="medium", provider_output_config={"format": {}}),
        "high",
    )
    assert untouched.reasoning_effort == "high"
    assert untouched.provider_output_config == {"format": {}}


def test_snap_effort_onto_route_takes_the_nearest_served_level() -> None:
    """A [high, xhigh] rung serves ``medium`` as ``high`` (nearer) with one disclosure."""
    route = (_ladder_profile("high", "xhigh"),)
    request = _messages_request(
        reasoning_effort="medium", provider_output_config={"effort": "medium"}
    )
    snapped = snap_effort_onto_route(route, request, "medium", {"high", "xhigh"}, admits=None)
    assert snapped is not None
    snapped_request, disclosure = snapped
    assert snapped_request.reasoning_effort == "high"
    assert snapped_request.provider_output_config == {"effort": "high"}
    assert disclosure == "reasoning_effort->high"
    # The caller's probe can veto a nearer candidate; the next one is offered.
    farther = snap_effort_onto_route(
        route,
        request,
        "medium",
        {"high", "xhigh"},
        admits=lambda candidate: candidate.reasoning_effort == "xhigh",
    )
    assert farther is not None
    assert farther[0].reasoning_effort == "xhigh"
    # No candidate that serves means no snap, never a guess.
    assert snap_effort_onto_route(route, request, "medium", set(), admits=None) is None
