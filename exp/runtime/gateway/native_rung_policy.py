"""Per-reservation dispatch-policy decisions for the native waterfall.

The accounting bridge reserves every physical dispatch immediately before
network work; these helpers make the two policy decisions it needs at that
moment without owning state of their own. ``reserve_rung_slot`` asks the
worker's load registry whether a policy-bounded rung admits the dispatch or
sheds it sideways, folding in the affinity pool's warm-session standing.
``failed_dispatch_candidate`` turns a classified failure into the ladder's
next candidate, reading the requesting organization's observed cached
fraction on the failed rung so a pool authoring ``throttle_cache_threshold``
can dispose of a throttle by the cache actually at stake.
"""

from __future__ import annotations

import logging

from exp.common.models.gateway_catalog import ExactModelDeployment
from exp.runtime.gateway.contracts import GatewayFailure
from exp.runtime.gateway.health import DeploymentHealthKey, DeploymentHealthRegistry
from exp.runtime.gateway.native_execution import (
    InflightRequest,
    ThrottleDisposition,
    next_route_candidate,
    rung_load_key,
    throttle_disposition,
)
from exp.runtime.gateway.rung_admission import RungLoadRegistry, RungShed
from exp.runtime.gateway.sticky_affinity import StickySpillRegistry

_logger = logging.getLogger(__name__)


def reserve_rung_slot(
    loads: RungLoadRegistry,
    sticky: StickySpillRegistry,
    entry: InflightRequest,
    deployment: ExactModelDeployment,
    *,
    reserved_tokens: int,
    force: bool,
) -> str | RungShed | None:
    """Reserve one policy-bounded slot on a rung, or report the shed.

    Args:
        loads: The worker's per-rung in-flight and rate-window registry.
        sticky: The worker-local conversation-to-rung bindings.
        entry: The owning in-flight request (organization and weight).
        deployment: The claimed rung about to dispatch.
        reserved_tokens: Worst-case tokens this dispatch reserves, counted
            against the rung's token window when one is authored.
        force: Admit past every policy limit because no other rung can
            serve.

    Returns:
        An opaque reservation ticket, the shed disclosure, or ``None``
        when the rung authors no admission policy (the untouched default).
    """
    policy = deployment.gateway.dispatch
    if policy is None or (
        policy.concurrency_bound is None
        and policy.requests_per_minute is None
        and policy.tokens_per_minute is None
    ):
        return None
    # Warm standing: the request's affinity fingerprint holds a live sticky
    # binding on THIS rung, so its provider cache lives here and the
    # fresh-session early threshold does not apply to it. The early threshold
    # only exists on affinity pools AND for requests that carry a fingerprint
    # (chat/Responses admission): a surface with no session concept
    # (embeddings, images) must never be classed fresh wholesale.
    fresh_fraction = (
        policy.fresh_session_spill_fraction
        if entry.route.snapshot.failover_mode == "maximize_cache_affinity"
        and entry.affinity_fingerprint is not None
        else None
    )
    warm_session = True
    if fresh_fraction is not None and entry.affinity_fingerprint is not None:
        warm_session = sticky.bound_deployment(entry.affinity_fingerprint) == (
            deployment.deployment_id
        )
    result = loads.reserve(
        rung_load_key(deployment),
        organization_id=entry.authorization.organization_id,
        weight=entry.authorization.fair_share_weight,
        bound=policy.concurrency_bound,
        fair_share=policy.fair_share,
        requests_per_minute=policy.requests_per_minute,
        tokens_per_minute=policy.tokens_per_minute,
        cache_priority_alpha=policy.cache_priority_alpha,
        reserved_tokens=reserved_tokens if policy.tokens_per_minute is not None else 0,
        warm_session=warm_session,
        fresh_spill_fraction=fresh_fraction,
        force=force,
    )
    if isinstance(result, RungShed) and result.reason == "rate_limit":
        _logger.debug(
            "gateway rate-limit shed on deployment %r (learned ceiling %s/min)",
            deployment.deployment_id,
            result.learned_requests_per_minute,
        )
    return result


def failed_dispatch_candidate(
    *,
    health: DeploymentHealthRegistry,
    loads: RungLoadRegistry,
    keys: tuple[DeploymentHealthKey, ...],
    entry: InflightRequest,
    failure: GatewayFailure,
    current_depth: int,
) -> tuple[int | None, ThrottleDisposition | None]:
    """Choose the ladder's next candidate after one classified failure.

    Reads the cache at stake on the failed rung (the requesting organization's
    EWMA of its settled cached fraction there, zero without evidence) and
    hands it with the pool's authored ``throttle_cache_threshold`` to the
    frozen candidate policy, so a throttle is surfaced or failed over by the
    warm cache it would abandon. Without a threshold the fraction is inert.

    Args:
        health: Revision-isolated circuit and throttle registry.
        loads: The worker's per-rung load registry holding the cache EWMA.
        keys: One health key per ordered route deployment.
        entry: The owning in-flight request.
        failure: The classified failure that ended the previous dispatch.
        current_depth: Route position of the failed dispatch.

    Returns:
        ``(candidate, disposition)``: the claimed route index or ``None``
        when the ladder is exhausted, and the throttle disposition when the
        failure was a throttle on a threshold-authoring pool (else ``None``).
    """
    route = entry.route
    threshold = route.snapshot.throttle_cache_threshold
    deployment = route.deployments[current_depth]
    cached_fraction = loads.cached_fraction(
        rung_load_key(deployment), entry.authorization.organization_id
    )
    candidate = next_route_candidate(
        health=health,
        keys=keys,
        failure=failure,
        current_depth=current_depth,
        attempt_counts=entry.attempt_counts,
        total_attempts=entry.total_attempts,
        refusal_failover=entry.authorization.refusal_failover,
        failover_mode=route.snapshot.failover_mode,
        throttle_cache_threshold=threshold,
        cached_fraction=cached_fraction,
    )
    disposition = throttle_disposition(
        failure,
        throttle_cache_threshold=threshold,
        cached_fraction=cached_fraction,
    )
    if disposition is not None:
        _logger.debug(
            "gateway throttle on deployment %r disposed %s (cached fraction %.3f, threshold %s)",
            deployment.deployment_id,
            disposition,
            cached_fraction,
            threshold,
        )
    return candidate, disposition
