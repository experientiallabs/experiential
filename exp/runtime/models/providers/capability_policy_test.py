"""Tests for the route capability-preservation policy."""

from __future__ import annotations

from exp.runtime.gateway.contracts import (
    GatewayApiSurface,
    GatewayMessage,
    GatewayRequest,
    GatewayToolDefinition,
)
from exp.runtime.models.providers.base import GatewayWireProfile
from exp.runtime.models.providers.capability_policy import (
    coerce_capability,
    coerce_generation_parameters,
    route_capability_failure_message,
)
from exp.runtime.models.providers.reasoning_compat import nearest_supported_effort


def _request(**overrides: object) -> GatewayRequest:
    """Build one canonical request with overrides applied."""
    request = GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(GatewayMessage(role="user", content="go"),),
    )
    return request.model_copy(update=dict(overrides)) if overrides else request


def _reasoning_profile(model_id: str) -> GatewayWireProfile:
    """Build one reasoning-capable OpenAI-compatible wire profile."""
    return GatewayWireProfile(
        dialect="openai_compatible",
        url="https://provider.test",
        model_id=model_id,
        supports_reasoning=True,
        reasoning_wire_format="reasoning_effort",
    )


def test_nearest_effort_prefers_the_lower_level_on_ties() -> None:
    """Distance is ladder positions; a tie never spends more than requested."""
    assert nearest_supported_effort("ultra", ("low", "medium", "high", "xhigh", "max")) == "xhigh"
    assert nearest_supported_effort("medium", ("low", "high")) == "low"
    assert nearest_supported_effort("minimal", ("high",)) == "high"
    assert nearest_supported_effort("high", ("high",)) == "high"
    assert nearest_supported_effort("high", ()) is None
    assert nearest_supported_effort("bogus", ("high",)) is None


def test_effort_snaps_to_the_nearest_route_level_with_disclosure() -> None:
    """An unsupported effort snaps onto the route ladder, disclosed by name."""
    # gpt-5.1 supports none/low/medium/high; xhigh must snap down to high.
    coercion = coerce_generation_parameters(
        (_reasoning_profile("gpt-5.1"),),
        _request(reasoning_effort="xhigh"),
    )
    assert coercion is not None
    assert coercion.request.reasoning_effort == "high"
    assert coercion.disclosures == ("reasoning_effort->high",)


def test_effort_none_drops_on_a_route_with_no_reasoning_at_all() -> None:
    """A non-reasoning route already delivers what 'none' asks for."""
    bare = GatewayWireProfile(dialect="openai_compatible", url="https://provider.test")
    coercion = coerce_generation_parameters((bare,), _request(reasoning_effort="none"))
    assert coercion is not None
    assert coercion.request.reasoning_effort is None
    assert coercion.disclosures == ("reasoning_effort",)

    # Any real effort on a zero-reasoning route stays a named rejection:
    # deleting the feature is not a nearest level.
    assert coerce_generation_parameters((bare,), _request(reasoning_effort="high")) is None


def test_portable_effort_is_never_snapped() -> None:
    """A failure elsewhere must not trigger an effort substitution."""
    coercion = coerce_generation_parameters(
        (_reasoning_profile("gpt-5.1"),),
        _request(reasoning_effort="high", temperature=1.9),
    )
    assert coercion is None
    assert coerce_generation_parameters((_reasoning_profile("gpt-5.1"),), _request()) is None


def test_strict_tools_degrade_only_as_a_disclosed_drop() -> None:
    """strict:true weakens to best-effort schemas with the drop disclosed."""
    request = _request(
        tools=(
            GatewayToolDefinition(name="lookup", parameters={"type": "object"}, strict=True),
            GatewayToolDefinition(name="plain", parameters={"type": "object"}),
        )
    )
    coercion = coerce_capability("strict_tools", request)
    assert coercion is not None
    assert tuple(tool.strict for tool in coercion.request.tools) == (False, False)
    assert coercion.disclosures == ("tools.strict->false",)

    # Every other capability names a feature with no approximation.
    assert coerce_capability("developer_messages", request) is None
    assert coerce_capability("strict_tools", _request()) is None


def test_fail_closed_message_names_the_capability_and_the_route_gap() -> None:
    """The terminal rejection states the exact gap and how to resolve it."""
    message = route_capability_failure_message("strict_tools", 3)
    assert "'strict_tools'" in message
    assert "3 deployments" in message
    assert "choose an alias" in message
    assert "1 deployment " in route_capability_failure_message("developer_messages", 1)
