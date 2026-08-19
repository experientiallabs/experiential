"""Tests for provider capability preflight and gateway exclusions."""

from __future__ import annotations

import pytest

from wmo.common.models.catalog import GatewayDeploymentCapabilities
from wmo.runtime.gateway.contracts import (
    GatewayApiSurface,
    GatewayMessage,
    GatewayRequest,
    GatewayToolDefinition,
)
from wmo.runtime.models.providers.errors import ProviderCapabilityError
from wmo.runtime.models.providers.protocol import (
    preflight_gateway_request,
    require_gateway_provider,
)


def test_preflight_rejects_unsupported_semantics_before_dispatch() -> None:
    """A deployment that cannot preserve strict tools fails before provider construction."""
    request = GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(GatewayMessage(role="user", content="hi"),),
        tools=(
            GatewayToolDefinition(
                name="lookup",
                parameters={"type": "object"},
                strict=True,
            ),
        ),
    )

    with pytest.raises(ProviderCapabilityError, match="strict_tools"):
        preflight_gateway_request(request, GatewayDeploymentCapabilities())


def test_tinker_is_explicitly_excluded_from_gateway_execution() -> None:
    """Tinker remains optimizer-only until it has a cancellable stream contract."""
    with pytest.raises(ProviderCapabilityError, match="tinker_gateway_execution"):
        require_gateway_provider("tinker")

    require_gateway_provider("openai")
