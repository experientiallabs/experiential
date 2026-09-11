"""Tests for gateway deployment capability declarations."""

from __future__ import annotations

import pytest

from exp.common.models.gateway_capabilities import GatewayDeploymentCapabilities
from exp.common.models.model import ReasoningEffort


@pytest.mark.parametrize(
    "values",
    (("high", "low"), ("high", "high")),
)
def test_gateway_reasoning_efforts_require_unique_canonical_order(
    values: tuple[ReasoningEffort, ...],
) -> None:
    """Ambiguous provider effort sets fail when the catalog is authored."""
    with pytest.raises(ValueError):
        GatewayDeploymentCapabilities(supported_reasoning_efforts=values)


def test_required_gateway_reasoning_effort_needs_supported_values() -> None:
    """A mandatory wire parameter cannot omit its provider value domain."""
    with pytest.raises(ValueError, match="at least one supported reasoning effort"):
        GatewayDeploymentCapabilities(reasoning_effort_required=True)


def test_required_gateway_reasoning_effort_needs_an_explicit_default() -> None:
    """A mandatory wire parameter cannot force admission to guess its value."""
    with pytest.raises(ValueError, match="needs reasoning_default_effort"):
        GatewayDeploymentCapabilities(
            supported_reasoning_efforts=("low", "high"),
            reasoning_effort_required=True,
        )


def test_gateway_reasoning_default_must_be_supported() -> None:
    """A provider default outside the exact domain fails catalog loading."""
    with pytest.raises(ValueError, match="must be one of the supported"):
        GatewayDeploymentCapabilities(
            supported_reasoning_efforts=("low", "high"),
            reasoning_default_effort="max",
        )


def test_astra_responses_capability_slots_default_off() -> None:
    """The three GPT-6 Astra Responses capability slots exist and default off.

    These are declaration slots for async function calling, mid-turn steering,
    and mid-conversation reasoning-effort updates. They default False (no
    deployment advertises a behavior the decoder/turn lifecycle does not yet
    honor) and, being defaulted, stay identity-invisible (see
    gateway_catalog_test's identity-digest pin). The platform's
    generation-capability vocabulary is drift-locked to these field names, so
    they must remain present for that projection to admit the keys.
    """
    caps = GatewayDeploymentCapabilities()
    assert caps.supports_async_tools is False
    assert caps.supports_mid_turn_steering is False
    assert caps.supports_reasoning_effort_update is False
    # Defaulted addition contributes zero identity bytes.
    assert caps.model_dump(mode="json", by_alias=True, exclude_defaults=True) == {}


def test_schema_drift_disclosure_slots_default_off() -> None:
    """The five schema-drift disclosure slots exist and default off.

    False means the capability is not declared. Defaulted fields stay
    identity-invisible, so the pinned catalog digest does not move.
    """
    caps = GatewayDeploymentCapabilities()
    assert caps.supports_prompt_cache_boundaries is False
    assert caps.supports_custom_tools is False
    assert caps.supports_grammar_tools is False
    assert caps.supports_tool_call_limit is False
    assert caps.reports_model_status is False
    assert caps.reports_reasoning_tokens is False
    assert caps.model_dump(mode="json", by_alias=True, exclude_defaults=True) == {}


def test_grammar_tools_require_custom_tools() -> None:
    """Grammar-tool support cannot be declared without custom-tool support."""
    with pytest.raises(ValueError, match="requires supports_custom_tools=true"):
        GatewayDeploymentCapabilities(supports_grammar_tools=True)


def test_custom_and_grammar_tools_can_be_declared_together() -> None:
    """Grammar-tool support is accepted when custom-tool support is also declared."""
    caps = GatewayDeploymentCapabilities(
        supports_custom_tools=True,
        supports_grammar_tools=True,
    )
    assert caps.supports_custom_tools is True
    assert caps.supports_grammar_tools is True
