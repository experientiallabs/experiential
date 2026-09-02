"""Rung-preference units; admission coercions are exercised e2e in native_bridge_test.py."""

from typing import Literal, cast

from exp.common.models.catalog import GatewayDeploymentMetadata
from exp.common.models.gateway_catalog import ExactModelDeployment
from exp.runtime.gateway.contracts import (
    AuthorizationSnapshot,
    DirectTarget,
    ExecutionSnapshot,
    GatewayApiSurface,
    GatewayMessage,
    GatewayRequest,
)
from exp.runtime.gateway.native_admission import _prefer_cache_capable_rungs
from exp.runtime.gateway.native_dispatch import NativeWireClient
from exp.runtime.gateway.routing import GatewayRoute
from exp.runtime.models.providers.base import GatewayWireProfile


def _deployment(deployment_id: str) -> ExactModelDeployment:
    """Build one exact deployment for rung-preference tests."""
    return ExactModelDeployment(
        deployment_id=deployment_id,
        source_alias=deployment_id,
        exact_model_id="exact-one",
        connection=f"connection-{deployment_id}",
        provider="openai-compatible",
        provider_model="provider-model",
        connection_sha256="b" * 64,
        capabilities_sha256="c" * 64,
        gateway=GatewayDeploymentMetadata(),
    )


def _mixed_route(failover_mode: str) -> GatewayRoute:
    """Build one two-rung route whose FIRST rung drops cache markers."""
    deployments = (_deployment("shim"), _deployment("native"))
    authorization = AuthorizationSnapshot(
        request_id="request-one",
        organization_id="organization-one",
        identity_id="identity-one",
        virtual_key_id="key-one",
        alias="public-model",
        alias_revision_id="revision-one",
        target=DirectTarget(pool_id="pool-one"),
        surface=GatewayApiSurface.MESSAGES,
        catalog_sha256="a" * 64,
        canonical_request_sha256="d" * 64,
        deadline_monotonic=1.0,
    )
    return GatewayRoute(
        snapshot=ExecutionSnapshot(
            authorization=authorization,
            exact_model_id="exact-one",
            pool_id="pool-one",
            deployment_ids=tuple(item.deployment_id for item in deployments),
            failover_mode=cast(Literal["maximize_availability", "maximize_cache"], failover_mode),
        ),
        deployment=deployments[0],
        fallback_deployments=deployments[1:],
        route_reason="direct",
    )


def _wires() -> tuple[tuple[GatewayWireProfile, NativeWireClient], ...]:
    """Pair one marker-dropping and one marker-honoring rung, shim first."""
    shim = GatewayWireProfile(dialect="openai_compatible", url="https://shim.test")
    native = GatewayWireProfile(dialect="anthropic_messages", url="https://anthropic.test")
    client = cast(NativeWireClient, object())
    return ((shim, client), (native, client))


def _marked_request() -> GatewayRequest:
    """Build one Messages request carrying a system cache marker."""
    return GatewayRequest(
        surface=GatewayApiSurface.MESSAGES,
        messages=(
            GatewayMessage(
                role="system",
                content="cached prompt",
                provider_text_blocks=(
                    {
                        "type": "text",
                        "text": "cached prompt",
                        "cache_control": {"type": "ephemeral"},
                    },
                ),
            ),
            GatewayMessage(role="user", content="hi"),
        ),
    )


def test_cache_marked_requests_dispatch_marker_honoring_rungs_first() -> None:
    """maximize_cache pools put the marker-carrying wire ahead of the shim.

    The haiku-4.5 incident shape: a certified waterfall paired a native
    Anthropic rung with an aggregator shim, and every marked session that
    dispatched on the shim billed its full context uncached (~10x). The
    pool's whole policy is prefix-cache preservation, so the marker-honoring
    rung dispatches first; certified order still decides everything else.
    """
    route, wires = _prefer_cache_capable_rungs(
        _mixed_route("maximize_cache"), _wires(), _marked_request()
    )
    assert route.deployment.deployment_id == "native"
    assert tuple(item.deployment_id for item in route.fallback_deployments) == ("shim",)
    assert wires[0][0].dialect == "anthropic_messages"

    # A markerless request keeps the certified order.
    plain = GatewayRequest(
        surface=GatewayApiSurface.MESSAGES,
        messages=(GatewayMessage(role="user", content="hi"),),
    )
    route, wires = _prefer_cache_capable_rungs(_mixed_route("maximize_cache"), _wires(), plain)
    assert route.deployment.deployment_id == "shim"

    # maximize_availability pools keep their certified order untouched.
    route, wires = _prefer_cache_capable_rungs(
        _mixed_route("maximize_availability"), _wires(), _marked_request()
    )
    assert route.deployment.deployment_id == "shim"

    # A route with no marker-honoring rung (or only such rungs) is unchanged;
    # the dropped markers are disclosed elsewhere.
    client = cast(NativeWireClient, object())
    shim = (GatewayWireProfile(dialect="openai_compatible", url="https://shim.test"), client)
    route, wires = _prefer_cache_capable_rungs(
        _mixed_route("maximize_cache"),
        (shim, shim),
        _marked_request(),
    )
    assert route.deployment.deployment_id == "shim"
