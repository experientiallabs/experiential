"""Tests for the per-reservation rung-policy decisions the accounting bridge applies."""

from __future__ import annotations

from exp.common.models.catalog import (
    GatewayDeploymentCapabilities,
    GatewayDeploymentMetadata,
    GatewayRungDispatchPolicy,
)
from exp.common.models.gateway_catalog import ExactModelDeployment, FailoverMode
from exp.runtime.gateway.contracts import (
    AuthorizationSnapshot,
    DirectTarget,
    ExecutionSnapshot,
    GatewayApiSurface,
    GatewayFailure,
    GatewayFailureClass,
    GatewayMessage,
    GatewayRequest,
)
from exp.runtime.gateway.health import DeploymentHealthRegistry
from exp.runtime.gateway.native_execution import InflightRequest, deployment_health_key
from exp.runtime.gateway.native_rung_policy import failed_dispatch_candidate, reserve_rung_slot
from exp.runtime.gateway.routing import GatewayRoute
from exp.runtime.gateway.rung_admission import RungLoadRegistry, RungShed
from exp.runtime.gateway.sticky_affinity import StickySpillRegistry


def _deployment(
    deployment_id: str,
    *,
    connection_sha256: str,
    dispatch: GatewayRungDispatchPolicy | None = None,
) -> ExactModelDeployment:
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
        gateway=GatewayDeploymentMetadata(
            capabilities=GatewayDeploymentCapabilities(supports_streaming=True),
            dispatch=dispatch,
        ),
    )


def _entry(
    deployments: tuple[ExactModelDeployment, ...],
    *,
    failover_mode: FailoverMode = "maximize_availability",
    throttle_cache_threshold: float | None = None,
    affinity_fingerprint: bytes | None = None,
) -> InflightRequest:
    """Build one admitted request over the given rung ladder."""
    authorization = AuthorizationSnapshot(
        request_id="request-one",
        organization_id="organization-one",
        identity_id="identity-one",
        virtual_key_id="key-one",
        alias="public-model",
        alias_revision_id="revision-one",
        target=DirectTarget(pool_id="pool-one"),
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        catalog_sha256="a" * 64,
        canonical_request_sha256="d" * 64,
        deadline_monotonic=1.0,
    )
    route = GatewayRoute(
        snapshot=ExecutionSnapshot(
            authorization=authorization,
            exact_model_id="exact-one",
            pool_id="pool-one",
            deployment_ids=tuple(item.deployment_id for item in deployments),
            failover_mode=failover_mode,
            throttle_cache_threshold=throttle_cache_threshold,
        ),
        deployment=deployments[0],
        fallback_deployments=deployments[1:],
        route_reason="direct",
    )
    return InflightRequest(
        authorization=authorization,
        route=route,
        request=GatewayRequest(
            surface=GatewayApiSurface.CHAT_COMPLETIONS,
            messages=(GatewayMessage(role="user", content="hello"),),
        ),
        deadline_monotonic=1.0,
        attempt_counts=[1, 0],
        total_attempts=1,
        affinity_fingerprint=affinity_fingerprint,
    )


_THROTTLE = GatewayFailure(
    failure_class=GatewayFailureClass.THROTTLED,
    safe_message="provider throttled the request",
    failover_eligible=True,
)


def test_reserve_rung_slot_is_inert_without_an_admission_policy() -> None:
    """A rung authoring no bound or rate window reserves nothing and sheds nothing."""
    loads = RungLoadRegistry()
    deployment = _deployment("deployment-a", connection_sha256="b" * 64)
    entry = _entry((deployment,))
    assert (
        reserve_rung_slot(
            loads, StickySpillRegistry(), entry, deployment, reserved_tokens=10, force=False
        )
        is None
    )
    assert loads.inflight(("deployment-a", "b" * 64)) == 0


def test_reserve_rung_slot_sheds_fresh_sessions_early_only_with_warm_standing_absent() -> None:
    """Under affinity a fingerprint without a live binding sheds at the early threshold."""
    loads = RungLoadRegistry()
    sticky = StickySpillRegistry()
    deployment = _deployment(
        "deployment-a",
        connection_sha256="b" * 64,
        dispatch=GatewayRungDispatchPolicy(
            concurrency_bound=4, fresh_session_spill_fraction=0.5, sticky_spill_seconds=60
        ),
    )
    warm = _entry(
        (deployment,), failover_mode="maximize_cache_affinity", affinity_fingerprint=b"warm"
    )
    sticky.bind(b"warm", "deployment-a", ttl_seconds=60.0)
    fresh = _entry(
        (deployment,), failover_mode="maximize_cache_affinity", affinity_fingerprint=b"fresh"
    )
    for _ in range(2):
        assert isinstance(
            reserve_rung_slot(loads, sticky, warm, deployment, reserved_tokens=0, force=False),
            str,
        )
    shed = reserve_rung_slot(loads, sticky, fresh, deployment, reserved_tokens=0, force=False)
    assert isinstance(shed, RungShed) and shed.reason == "fresh_session_spill"
    assert isinstance(
        reserve_rung_slot(loads, sticky, warm, deployment, reserved_tokens=0, force=False), str
    )


def test_failed_dispatch_candidate_reads_the_organizations_cache_on_the_failed_rung() -> None:
    """The disposition follows the requesting organization's EWMA on the throttled rung."""
    deployments = (
        _deployment("deployment-a", connection_sha256="b" * 64),
        _deployment("deployment-b", connection_sha256="c" * 64),
    )
    entry = _entry(deployments, throttle_cache_threshold=0.5)
    health = DeploymentHealthRegistry()
    keys = tuple(deployment_health_key(entry.authorization, item) for item in deployments)
    loads = RungLoadRegistry()

    # No evidence: fail over cold.
    assert failed_dispatch_candidate(
        health=health, loads=loads, keys=keys, entry=entry, failure=_THROTTLE, current_depth=0
    ) == (1, "throttle_failover_cold")
    # Another organization's warm cache on the rung does not count.
    loads.record_settle(
        ("deployment-a", "b" * 64), "organization-other", cached_tokens=1, input_tokens=1
    )
    assert failed_dispatch_candidate(
        health=health, loads=loads, keys=keys, entry=entry, failure=_THROTTLE, current_depth=0
    ) == (1, "throttle_failover_cold")
    # The requesting organization's own warm cache surfaces the throttle.
    loads.record_settle(
        ("deployment-a", "b" * 64), "organization-one", cached_tokens=9, input_tokens=10
    )
    assert failed_dispatch_candidate(
        health=health, loads=loads, keys=keys, entry=entry, failure=_THROTTLE, current_depth=0
    ) == (None, "throttle_surfaced_cache_preserving")
    # Without a threshold the same warm cache is inert and the mode rules.
    plain = _entry(deployments, failover_mode="maximize_cache")
    assert failed_dispatch_candidate(
        health=health, loads=loads, keys=keys, entry=plain, failure=_THROTTLE, current_depth=0
    ) == (None, None)
