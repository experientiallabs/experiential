"""Stage-local cache scheduling and forward-only session recovery at native admission."""

from __future__ import annotations

import importlib
import logging
import time

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
    rung_load_key,
    select_route_deployments,
)
from exp.runtime.gateway.native_fallback_rules import rung_rules
from exp.runtime.gateway.native_recovery import recovery_prefix_digest
from exp.runtime.gateway.native_responses import ContinuationContext
from exp.runtime.gateway.recovery import SessionCacheKey
from exp.runtime.gateway.recovery_binding import validated_recovery_binding
from exp.runtime.gateway.routing import GatewayRoute, GatewayRoutingError
from exp.runtime.gateway.sticky_affinity import AffinityPlacement
from exp.runtime.models.providers.base import GatewayWireProfile
from exp.runtime.models.providers.protocol import NativeWireClient

_logger = logging.getLogger(__name__)


def require_native_model_stage_contract(route: GatewayRoute) -> None:
    """Refuse staged admission when the loaded data plane cannot consume its wire contract.

    Routes without model stages need no marker or extension import. Resolve the
    loaded extension only at this feature boundary, as native serving does.
    Missing or unknown markers never fall back to a package version or generic
    export check, which cannot prove stage support.
    """
    if not route.snapshot.model_stages:
        return
    native = importlib.import_module("exp_gateway_native")
    contract = getattr(native, "MODEL_STAGE_CONTRACT_VERSION", None)
    if type(contract) is not int or contract != 1:
        raise GatewayRoutingError(
            "ordered model stages require native MODEL_STAGE_CONTRACT_VERSION=1; "
            "install the coordinated stage-capable native package before activating chains"
        )


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
    placement = AffinityPlacement(fingerprint=fingerprint, recovery_scoped=True)
    if route.resolved_route_id is not None:
        return route, wires, placement
    if route.reasoning_pinned_deployment_id is not None and any(
        deployment.deployment_id == route.reasoning_pinned_deployment_id
        for deployment in route.deployments
    ):
        return route, wires, placement
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
            indexes = tuple(i for i in indexes if wires[i][0].preserves_cache_control) + tuple(
                i for i in indexes if not wires[i][0].preserves_cache_control
            )
        order.extend(indexes)
        offset += len(stage.deployment_ids)
    route = reorder_route_deployments(route, tuple(order))
    wires = tuple(wires[i] for i in order)
    host = accounting.recovery_host
    # Without an explicit credential scope there is no reliable cache-sharing
    # identity. Missing host data means normal bounded routing, not guessed warmth.
    if host is None:
        return route, wires, placement
    # Sticky placement is exclusive to affinity stages. A descendant may not
    # skip an earlier availability/cache stage or an unauthorized model boundary.
    recovery_depth = 0
    for depth in range(len(route.deployments)):
        stage = route.snapshot.stage_for_depth(depth)
        if stage.failover_mode != "maximize_cache_affinity" or (
            stage.exact_model_id != route.snapshot.exact_model_id
            and not authorization.descendant_start_authorized
        ):
            break
        recovery_depth += 1
    if not recovery_depth:
        return route, wires, placement
    prefix = recovery_prefix_digest(request)
    if prefix is None:
        return route, wires, placement
    key = SessionCacheKey(
        authorization.organization_id, authorization.identity_id, fingerprint, prefix
    )
    registry = accounting.recovery
    if not registry.has_retained_history(key):
        return route, wires, placement
    candidates = tuple(
        (deployment.deployment_id, binding.scope)
        for deployment, (profile, _) in zip(
            route.deployments[:recovery_depth], wires[:recovery_depth], strict=True
        )
        if (
            binding := validated_recovery_binding(
                deployment, profile, authorization.organization_id
            )
        )
        is not None
    )
    if not candidates:
        return route, wires, placement
    by_id = {d.deployment_id: d for d in route.deployments[:recovery_depth]}

    def eligible(deployment_id: str) -> bool:
        """Require a first-dial route; historical failures never activate conditional rungs."""
        return rung_rules(by_id[deployment_id]) is None and not accounting.health.suppressed(
            deployment_health_key(authorization, by_id[deployment_id])
        )

    # Only token-window headroom needs BPE. Compute once outside the recovery
    # lock, never lazily from the callback choose invokes under that lock.
    input_tokens = (
        worst_case_input_tokens(request)
        if any(
            d.gateway.dispatch is not None and d.gateway.dispatch.tokens_per_minute is not None
            for d in by_id.values()
        )
        and registry.has_retained_history(key, live_only=True)
        else None
    )

    def headroom(deployment_id: str) -> bool:
        """Require actual local concurrency, rate and fair-share headroom for a trial."""
        deployment = by_id[deployment_id]
        policy = deployment.gateway.dispatch
        if policy is None and accounting.loads.default_bound is None:
            return True
        reserved_tokens = 0
        if policy is not None and policy.tokens_per_minute is not None:
            # History can arrive after the preflight. Defer its elective trial
            # rather than checking a token window with an underestimated prompt.
            if input_tokens is None:
                return False
            reserved_tokens = input_tokens + worst_case_output_tokens(request, deployment)
        return accounting.loads.can_admit(
            rung_load_key(deployment),
            organization_id=authorization.organization_id,
            weight=authorization.fair_share_weight,
            bound=accounting.loads.default_bound
            if policy is None or policy.concurrency_bound is None
            else policy.concurrency_bound,
            fair_share=policy is not None and policy.fair_share,
            requests_per_minute=None if policy is None else policy.requests_per_minute,
            tokens_per_minute=None if policy is None else policy.tokens_per_minute,
            cache_priority_alpha=None if policy is None else policy.cache_priority_alpha,
            reserved_tokens=reserved_tokens,
        )

    if request_carries_cache_markers(request):
        # Recovery may not discard an eligible marker-honoring provider in the
        # same segment merely because a later non-honoring wire was retained.
        allowed: set[str] = set()
        for deployment_id, _scope in candidates:
            depth = route.snapshot.deployment_ids.index(deployment_id)
            stage = route.snapshot.stage_for_depth(depth)
            if wires[depth][0].preserves_cache_control or not any(
                wires[prior][0].preserves_cache_control
                and route.snapshot.stage_for_depth(prior).stage_index == stage.stage_index
                and eligible(route.deployments[prior].deployment_id)
                and headroom(route.deployments[prior].deployment_id)
                for prior in range(depth)
            ):
                allowed.add(deployment_id)
        candidates = tuple(item for item in candidates if item[0] in allowed)
        if not candidates:
            return route, wires, placement
    decision_started = time.monotonic()
    try:
        observation = host.snapshot()
    except Exception:  # noqa: BLE001 - only the optional advisory fetch may fail open.
        _logger.warning("Recovery snapshot unavailable; elective recovery skipped")
        return route, wires, placement
    decision = registry.choose(
        key,
        candidates,
        eligible=eligible,
        snapshot=observation,
        local_capacity=headroom,
    )
    if decision.deployment_id is not None:
        start = next(
            i for i, d in enumerate(route.deployments) if d.deployment_id == decision.deployment_id
        )
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
            verified_warm_deployment_id=decision.deployment_id,
            verified_warm_until_monotonic=decision_started + decision.warm_remaining_seconds,
            recovery_scoped=True,
        ),
    )
