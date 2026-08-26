"""Tests for wire-profile capability intersection and deployment identity checks."""

from __future__ import annotations

from typing import cast

import pytest

from exp.common.models import BillingSource, ModelCapabilities, ModelSnapshot
from exp.common.models.catalog import GatewayDeploymentMetadata, GatewayTokenPrices
from exp.common.models.gateway_catalog import ExactModelDeployment
from exp.runtime.gateway.execution_resolution import (
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


def _deployment(capabilities: ModelCapabilities | None) -> ExactModelDeployment:
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
