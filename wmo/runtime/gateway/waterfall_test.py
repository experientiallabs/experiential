"""Ordered exact-model routing and isolated deployment-health regressions."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from wmo.common.models.catalog import (
    GatewayDeploymentMetadata,
    GatewayEquivalenceCertification,
)
from wmo.common.models.gateway_catalog import (
    ExactModelDeployment,
    ExactModelPool,
    NormalizedGatewayCatalog,
)
from wmo.runtime.gateway.contracts import (
    AuthorizationSnapshot,
    DirectTarget,
    GatewayApiSurface,
    GatewayFailure,
    GatewayFailureClass,
    GatewayMessage,
    GatewayRequest,
)
from wmo.runtime.gateway.health import DeploymentHealthRegistry
from wmo.runtime.gateway.routing import CatalogRouteResolver

_DIGEST = "a" * 64


def test_direct_certified_pool_preserves_operational_deployment_order() -> None:
    """Direct resolution exposes every certified deployment in authored priority order."""
    first = _deployment("route-a", connection_sha256="b" * 64)
    second = _deployment("route-b", connection_sha256="c" * 64)
    certification = GatewayEquivalenceCertification(
        certification_id="certification-one",
        provenance="operator comparison run 2026-08-18",
        evidence_sha256=_DIGEST,
        certified_at=datetime(2026, 8, 18, tzinfo=UTC),
    )
    catalog = NormalizedGatewayCatalog(
        deployments=(first, second),
        pools=(
            ExactModelPool(
                pool_id="pool-one",
                exact_model_id="exact-one",
                deployment_ids=(first.deployment_id, second.deployment_id),
                equivalence=certification,
            ),
        ),
    )
    catalog_sha256 = catalog.identity_sha256()
    resolver = CatalogRouteResolver({("revision-one", catalog_sha256): catalog})

    async def scenario() -> None:
        """Resolve the direct target on one event loop."""
        route = await resolver.resolve(
            authorization=_authorization(catalog_sha256),
            request=GatewayRequest(
                surface=GatewayApiSurface.CHAT_COMPLETIONS,
                messages=(GatewayMessage(role="user", content="hello"),),
            ),
            episode_namespace=("org", "identity", "revision-one", "episode"),
        )

        assert route.deployments == (first, second)
        assert route.snapshot.deployment_ids == ("route-a", "route-b")

    asyncio.run(scenario())


def test_circuit_identity_is_catalog_and_connection_scoped() -> None:
    """One failed revision cannot suppress a same-named deployment in another catalog."""
    now = [100.0]
    health = DeploymentHealthRegistry(
        failure_threshold=1,
        open_seconds=10,
        throttle_seconds=5,
        clock=lambda: now[0],
    )
    first = ("catalog-a", "deployment", "connection-a")
    second = ("catalog-b", "deployment", "connection-b")

    assert health.claim(first)
    health.failed(
        first,
        GatewayFailure(
            failure_class=GatewayFailureClass.PROVIDER_AUTHENTICATION,
            safe_message="provider authentication failed",
            failover_eligible=True,
        ),
    )

    assert not health.claim(first)
    assert health.claim(second)
    health.succeeded(second)
    now[0] += 11
    assert health.claim(first)


def test_throttle_window_recovers_without_opening_the_failure_circuit() -> None:
    """A throttle suppresses dispatch only for its independent bounded window."""
    now = [100.0]
    health = DeploymentHealthRegistry(
        failure_threshold=2,
        open_seconds=30,
        throttle_seconds=5,
        clock=lambda: now[0],
    )
    key = ("catalog", "deployment", "connection")

    assert health.claim(key)
    health.failed(
        key,
        GatewayFailure(
            failure_class=GatewayFailureClass.THROTTLED,
            safe_message="provider throttled the request",
            failover_eligible=True,
        ),
    )
    assert not health.claim(key)
    now[0] += 6
    assert health.claim(key)


def _deployment(deployment_id: str, *, connection_sha256: str) -> ExactModelDeployment:
    """Build one deployment in the shared certified exact-model pool."""
    return ExactModelDeployment(
        deployment_id=deployment_id,
        source_alias=deployment_id,
        exact_model_id="exact-one",
        connection=f"connection-{deployment_id}",
        provider="openai",
        provider_model="provider-model",
        connection_sha256=connection_sha256,
        capabilities_sha256="d" * 64,
        gateway=GatewayDeploymentMetadata(),
    )


def _authorization(catalog_sha256: str) -> AuthorizationSnapshot:
    """Build one direct authority snapshot pinned to the test catalog."""
    return AuthorizationSnapshot(
        request_id="request-one",
        organization_id="organization-one",
        identity_id="identity-one",
        virtual_key_id="key-one",
        alias="public-model",
        alias_revision_id="revision-one",
        target=DirectTarget(pool_id="pool-one"),
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        catalog_sha256=catalog_sha256,
        canonical_request_sha256=_DIGEST,
        deadline_monotonic=1.0,
    )
