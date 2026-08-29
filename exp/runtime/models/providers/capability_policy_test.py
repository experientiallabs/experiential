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
)
from exp.runtime.models.providers.reasoning_compat import efforts_by_nearness


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


def test_efforts_order_by_nearness_and_prefer_the_lower_level_on_ties() -> None:
    """Distance is ladder positions; a tie never spends more than requested."""
    assert efforts_by_nearness("ultra", ("low", "medium", "high", "xhigh", "max")) == (
        "xhigh",
        "max",
        "high",
        "medium",
        "low",
    )
    assert efforts_by_nearness("medium", ("low", "high")) == ("low", "high")
    assert efforts_by_nearness("minimal", ("high",)) == ("high",)
    assert efforts_by_nearness("high", ()) == ()
    assert efforts_by_nearness("bogus", ("high",)) == ()


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


def test_effort_snap_skips_levels_that_admit_no_rung() -> None:
    """The snap is the nearest level that actually serves, not the nearest
    level on paper: a rung carrying the closer effort may reject another
    requested control, and the coercion must not dead-end there."""
    xhigh_only_no_temperature = GatewayWireProfile(
        dialect="openai_compatible",
        url="https://a.test",
        model_id="gpt-5.2-pro",
        supports_temperature=False,
        supports_reasoning=True,
        reasoning_wire_format="reasoning_effort",
    )
    request = _request(reasoning_effort="ultra", temperature=0.5)
    coercion = coerce_generation_parameters(
        (xhigh_only_no_temperature, _reasoning_profile("gpt-5.1")),
        request,
    )
    assert coercion is not None
    # xhigh is nearer to ultra but only the temperature-rejecting rung has
    # it; high is the closest level that admits a serving rung.
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


def test_route_wide_capability_requires_unanimous_rejection() -> None:
    """Mixed per-rung rejections never produce the route-wide claim."""
    from exp.runtime.models.providers.capability_policy import route_wide_capability
    from exp.runtime.models.providers.errors import (
        ProviderCapabilityError,
        ProviderParameterError,
    )

    strict = ProviderCapabilityError(capability="strict_tools")
    developer = ProviderCapabilityError(capability="developer_messages")
    parameter = ProviderParameterError(
        message="The value 3 for 'top_k' is not supported.",
        param="top_k",
        code="invalid_parameter",
    )
    assert route_wide_capability((strict, strict), 2) == "strict_tools"
    assert route_wide_capability((strict, developer), 2) is None
    assert route_wide_capability((strict, parameter), 2) is None
    assert route_wide_capability((strict,), 2) is None
    assert route_wide_capability((), 0) is None


def test_effort_snap_requires_route_wide_construction_to_survive() -> None:
    """The snapped candidate must survive route-wide construction, not just
    per-rung admission: the narrowed rung set changes with the candidate, and
    the homogeneous encrypted-reasoning gate rejects a mixed Responses and
    Fireworks set that a farther candidate narrows past."""
    responses_medium = GatewayWireProfile(
        dialect="openai_responses",
        url="https://a.test",
        model_id="gpt-5.1",
        supports_reasoning=True,
        reasoning_wire_format="reasoning_effort",
        supported_reasoning_efforts=("medium",),
    )
    fireworks_medium_high = GatewayWireProfile(
        dialect="openai_compatible",
        url="https://b.test",
        model_id="kimi-k3",
        supports_reasoning=True,
        reasoning_wire_format="reasoning_effort",
        supported_reasoning_efforts=("medium", "high"),
        fireworks_reasoning_route_sha256="a" * 64,
    )
    request = GatewayRequest(
        surface=GatewayApiSurface.RESPONSES,
        messages=(GatewayMessage(role="user", content="go"),),
        reasoning_effort="low",
        include_encrypted_reasoning=True,
        response_store=False,
    )
    coercion = coerce_generation_parameters((responses_medium, fireworks_medium_high), request)
    assert coercion is not None
    # medium is nearer to low but admits both rungs, and the mixed set fails
    # the homogeneous encrypted-reasoning channel; high narrows to the
    # Fireworks rung alone and serves.
    assert coercion.request.reasoning_effort == "high"
    assert coercion.disclosures == ("reasoning_effort->high",)
