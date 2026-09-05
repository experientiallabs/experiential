"""Tests for wire-profile capability intersection and deployment identity checks."""

from __future__ import annotations

from typing import cast

import pytest

from exp.common.models import BillingSource, ModelCapabilities, ModelSnapshot
from exp.common.models.catalog import (
    GatewayDeploymentCapabilities,
    GatewayDeploymentMetadata,
    GatewayTokenPrices,
)
from exp.common.models.gateway_catalog import ExactModelDeployment
from exp.runtime.gateway.execution_resolution import (
    GatewayWireContractError,
    _require_deployment_identity,
    _resolved_wire_profile,
)
from exp.runtime.models.providers.base import GatewayWireProfile
from exp.runtime.models.registry import ModelClient, ResolvedModel


def _snapshot() -> ModelSnapshot:
    """Build one frozen identity fixture for a resolved deployment."""
    return ModelSnapshot(
        billing_source=BillingSource.CUSTOMER_MANAGED,
        provider="openai-compatible",
        model_id="provider-model",
        revision=None,
        capabilities_sha256="a" * 64,
        connection_sha256="b" * 64,
    )


def _deployment(
    capabilities: ModelCapabilities | None,
    gateway_capabilities: GatewayDeploymentCapabilities | None = None,
) -> ExactModelDeployment:
    """Build one exact deployment carrying the given catalog capability contract."""
    return ExactModelDeployment(
        deployment_id="primary",
        source_alias="primary",
        exact_model_id="exact-one",
        connection="connection-primary",
        provider="openai-compatible",
        provider_model="provider-model",
        connection_sha256="b" * 64,
        capabilities_sha256="a" * 64,
        capabilities=capabilities,
        gateway=GatewayDeploymentMetadata(
            capabilities=gateway_capabilities or GatewayDeploymentCapabilities(),
            prices=GatewayTokenPrices(),
            pricing_source="test",
        ),
    )


class _NativeClient:
    """Minimal structural NativeWireClient returning one fixed wire profile."""

    def __init__(self, profile: GatewayWireProfile) -> None:
        """Store the profile handed back by every ``gateway_wire_profile`` call."""
        self._profile = profile

    def gateway_wire_profile(self) -> GatewayWireProfile:
        """Return the stored wire profile."""
        return self._profile


class _StreamOnlyClient:
    """A client without a native wire profile, as legacy injected clients were."""


def _resolved(
    client: object,
    capabilities: ModelCapabilities,
) -> ResolvedModel:
    """Bind one client and capability contract into a resolved model."""
    return ResolvedModel(
        alias="primary",
        snapshot=_snapshot(),
        capabilities=capabilities,
        client=cast(ModelClient, client),
        embedding_client=None,
    )


def test_profile_ranges_intersect_with_the_catalog_contract() -> None:
    """Catalog limits tighten the client profile without widening any range."""
    profile = GatewayWireProfile(
        dialect="openai_compatible",
        url="https://example.test/v1/chat/completions",
        maximum_output_tokens=4096,
    )
    capabilities = ModelCapabilities(
        minimum_temperature=0.5,
        maximum_temperature=1.5,
        maximum_output_tokens=128,
        chat_max_tokens_field="max_completion_tokens",
    )

    resolved = _resolved_wire_profile(
        _deployment(capabilities),
        _resolved(_NativeClient(profile), capabilities),
    )

    assert resolved.minimum_temperature == 0.5
    assert resolved.maximum_temperature == 1.5
    assert resolved.maximum_output_tokens == 128
    assert resolved.token_limit_key == "max_completion_tokens"
    assert resolved.model_id == "provider-model"


def test_profile_resolution_applies_exact_gateway_reasoning_values() -> None:
    """Deployment metadata replaces a family guess with provider-published values."""
    capabilities = ModelCapabilities(
        supports_reasoning=True,
        reasoning_effort="max",
    )
    profile = GatewayWireProfile(
        dialect="openai_compatible",
        url="https://example.test/v1/chat/completions",
        supports_reasoning=True,
        reasoning_wire_format="reasoning",
        reasoning_effort="max",
    )
    gateway_capabilities = GatewayDeploymentCapabilities(
        supported_reasoning_efforts=("low", "high", "max"),
        reasoning_default_effort="high",
        reasoning_effort_required=True,
    )

    resolved = _resolved_wire_profile(
        _deployment(capabilities, gateway_capabilities),
        _resolved(_NativeClient(profile), capabilities),
    )

    assert resolved.supported_reasoning_efforts == ("low", "high", "max")
    assert resolved.reasoning_effort == "high"
    assert resolved.reasoning_effort_required is True


def test_profile_resolution_rejects_provider_reasoning_contract_conflict() -> None:
    """A provider profile cannot silently weaken frozen gateway reasoning metadata."""
    capabilities = ModelCapabilities(supports_reasoning=True)
    profile = GatewayWireProfile(
        dialect="openai_compatible",
        url="https://example.test/v1/chat/completions",
    )
    gateway_capabilities = GatewayDeploymentCapabilities(
        supported_reasoning_efforts=("medium",),
        reasoning_default_effort="medium",
        reasoning_effort_required=True,
    )

    with pytest.raises(GatewayWireContractError, match="reasoning metadata conflicts"):
        _resolved_wire_profile(
            _deployment(capabilities, gateway_capabilities),
            _resolved(_NativeClient(profile), capabilities),
        )


def test_profile_resolution_requires_a_native_wire_client() -> None:
    """A resolved client without a native profile fails before any dispatch."""
    capabilities = ModelCapabilities()

    with pytest.raises(TypeError, match="native wire profile"):
        _resolved_wire_profile(
            _deployment(capabilities),
            _resolved(_StreamOnlyClient(), capabilities),
        )


def test_identity_check_accepts_the_matching_deployment() -> None:
    """A resolution that matches the frozen deployment passes silently."""
    capabilities = ModelCapabilities()
    profile = GatewayWireProfile(dialect="openai_compatible", url="https://example.test")

    _require_deployment_identity(
        _deployment(None),
        _resolved(_NativeClient(profile), capabilities),
    )


def test_identity_check_rejects_a_drifted_provider() -> None:
    """A provider drift between catalog and runtime fails closed."""
    capabilities = ModelCapabilities()
    profile = GatewayWireProfile(dialect="openai_compatible", url="https://example.test")
    deployment = _deployment(None).model_copy(update={"provider": "openai"})

    with pytest.raises(ValueError, match="frozen gateway deployment"):
        _require_deployment_identity(
            deployment,
            _resolved(_NativeClient(profile), capabilities),
        )


def test_profile_resolution_binds_the_deployment_billing_source() -> None:
    """The wire profile learns whether its rung dispatches BYOK credentials.

    Tier forwarding (service_tier) keys on this flag: the caller pays the
    provider directly only on customer-managed rungs.
    """
    base = GatewayWireProfile(dialect="openai_compatible", url="https://provider.test")
    byok = _resolved_wire_profile(
        _deployment(None), _resolved(_NativeClient(base), ModelCapabilities())
    )
    assert byok.billing_customer_managed is True

    hosted = _deployment(None).model_copy(update={"billing_source": BillingSource.HOST_MANAGED})
    house = _resolved_wire_profile(hosted, _resolved(_NativeClient(base), ModelCapabilities()))
    assert house.billing_customer_managed is False
    # A house rung stays untiered by default, so service_tier is not forwarded.
    assert house.service_tier_pricing_enabled is False
    assert house.forwards_service_tier is False


def test_profile_resolution_forwards_service_tier_on_a_tier_priced_house_lane() -> None:
    """A host-funded rung whose model carries per-tier pass-through pricing
    forwards service_tier even though the caller does not pay the provider
    directly; settlement bills the served tier at cost."""
    base = GatewayWireProfile(dialect="openai_compatible", url="https://provider.test")
    hosted = _deployment(None).model_copy(update={"billing_source": BillingSource.HOST_MANAGED})
    tiered = _resolved_wire_profile(
        hosted,
        _resolved(_NativeClient(base), ModelCapabilities(service_tier_pricing_enabled=True)),
    )
    assert tiered.billing_customer_managed is False
    assert tiered.service_tier_pricing_enabled is True
    assert tiered.forwards_service_tier is True
