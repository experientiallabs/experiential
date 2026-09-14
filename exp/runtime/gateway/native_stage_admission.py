"""Stage-local cache scheduling and forward-only session recovery at native admission."""

from __future__ import annotations

from exp.runtime.gateway.affinity import (
    affinity_fingerprint,
    affinity_seed_material,
    rendezvous_order,
)
from exp.runtime.gateway.attempt_tokens import worst_case_input_tokens, worst_case_output_tokens
from exp.runtime.gateway.contracts import AuthorizationSnapshot, GatewayRequest
from exp.runtime.gateway.native_accounting import NativeAttemptAccounting
from exp.runtime.gateway.native_execution import (
    deployment_health_key,
    reorder_route_deployments,
    request_carries_cache_markers,
    select_route_deployments,
)
from exp.runtime.gateway.native_recovery import recovery_prefix_digest
from exp.runtime.gateway.native_responses import ContinuationContext
from exp.runtime.gateway.recovery import SessionCacheKey
from exp.runtime.gateway.routing import GatewayRoute
from exp.runtime.gateway.sticky_affinity import AffinityPlacement
from exp.runtime.models.providers.base import GatewayWireProfile
from exp.runtime.models.providers.protocol import NativeWireClient


def stage_affinity_ordered_rungs(
    route: GatewayRoute,
    wires: tuple[tuple[GatewayWireProfile, NativeWireClient], ...],
    request: GatewayRequest,
    *,
    accounting: NativeAttemptAccounting,
    authorization: AuthorizationSnapshot,
    continuation: ContinuationContext | None,
) -> tuple[
    GatewayRoute, tuple[tuple[GatewayWireProfile, NativeWireClient], ...], AffinityPlacement
]:
    """Schedule each provider segment independently, then choose one request start cursor.

    Recovery is evaluated exactly once against this frozen graph. Starting at a
    sticky descendant discards every preceding leaf, rather than re-expanding that
    model as a new root. Canonical ancestors and repeated-reference skips therefore
    cannot resurrect a primary. No settlement changes an in-flight cursor.
    """
    fingerprint = affinity_fingerprint(
        organization_id=authorization.organization_id,
        identity_id=authorization.identity_id,
        material=affinity_seed_material(
            request,
            continuation_episode_key=None if continuation is None else continuation.episode_key,
            request_id=authorization.request_id,
        ),
    )
    order: list[int] = []
    offset = 0
    stages = route.snapshot.model_stages or (route.snapshot.stage_for_depth(0),)
    for stage in stages:
        indexes = tuple(range(offset, offset + len(stage.deployment_ids)))
        if stage.failover_mode == "maximize_cache_affinity":
            weighted = tuple(
                (
                    deployment.deployment_id,
                    1.0
                    if deployment.gateway.dispatch is None
                    or deployment.gateway.dispatch.affinity_weight is None
                    else deployment.gateway.dispatch.affinity_weight,
                )
                for i in indexes
                for deployment in (route.deployments[i],)
            )
            indexes = tuple(offset + index for index in rendezvous_order(fingerprint, weighted))
        if stage.failover_mode != "maximize_availability" and request_carries_cache_markers(
            request
        ):
            indexes = tuple(
                i for i in indexes if wires[i][0].dialect == "anthropic_messages"
            ) + tuple(i for i in indexes if wires[i][0].dialect != "anthropic_messages")
        order.extend(indexes)
        offset += len(stage.deployment_ids)
    route = reorder_route_deployments(route, tuple(order))
    wires = tuple(wires[i] for i in order)
    registry = accounting.recovery
    host = accounting.recovery_host
    # Without an explicit credential scope there is no reliable cache-sharing
    # identity. Missing host data means normal bounded routing, not guessed warmth.
    prefix = recovery_prefix_digest(request)
    if host is None or prefix is None:
        return route, wires, AffinityPlacement(fingerprint=fingerprint)
    candidates = tuple(
        (d.deployment_id, registry.scope(d, authorization.organization_id, host))
        for d in route.deployments
    )
    by_id = {d.deployment_id: d for d in route.deployments}

    def eligible(deployment_id: str) -> bool:
        """Require current graph membership and unsuppressed actual deployment health."""
        return not accounting.health.suppressed(
            deployment_health_key(authorization, by_id[deployment_id])
        )

    input_tokens = worst_case_input_tokens(request)

    def headroom(deployment_id: str) -> bool:
        """Require actual local concurrency, rate and fair-share headroom for a trial."""
        deployment = by_id[deployment_id]
        policy = deployment.gateway.dispatch
        if policy is None:
            return True
        return accounting.loads.can_admit(
            (deployment.deployment_id, deployment.connection_sha256),
            organization_id=authorization.organization_id,
            weight=authorization.fair_share_weight,
            bound=policy.concurrency_bound,
            fair_share=policy.fair_share,
            requests_per_minute=policy.requests_per_minute,
            tokens_per_minute=policy.tokens_per_minute,
            cache_priority_alpha=policy.cache_priority_alpha,
            reserved_tokens=input_tokens + worst_case_output_tokens(request, deployment),
        )

    decision = registry.choose(
        SessionCacheKey(
            authorization.organization_id,
            authorization.identity_id,
            fingerprint,
            prefix,
        ),
        candidates,
        eligible=eligible,
        snapshot=host.snapshot(),
        local_capacity=headroom,
    )
    if decision.deployment_id is not None:
        start = next(
            i for i, d in enumerate(route.deployments) if d.deployment_id == decision.deployment_id
        )
        if (
            start
            and route.snapshot.stage_for_depth(start).exact_model_id
            != route.snapshot.exact_model_id
            and not authorization.descendant_start_authorized
        ):
            return route, wires, AffinityPlacement(fingerprint=fingerprint)
        if start:
            indexes = tuple(range(start, len(route.deployments)))
            route = select_route_deployments(route, indexes)
            wires = wires[start:]
    return (
        route,
        wires,
        AffinityPlacement(
            fingerprint=fingerprint,
            sticky_preferred=decision.reason == "retained_warm_fallback",
            recovery_reason=decision.reason,
            sticky_deployment_id=decision.deployment_id
            if decision.reason == "retained_warm_fallback"
            else None,
        ),
    )
