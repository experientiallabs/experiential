"""Per-reservation dispatch-policy decisions for the native waterfall.

The accounting bridge reserves every physical dispatch immediately before
network work; these helpers make the two policy decisions it needs at that
moment without owning state of their own. ``reserve_rung_slot`` asks the
worker's load registry whether a policy-bounded rung admits the dispatch or
sheds it sideways, folding in the affinity pool's warm-session standing.
``failed_dispatch_candidate`` turns a classified failure into the ladder's
next candidate, reading the requesting organization's observed cached
fraction on the failed rung so a pool authoring ``throttle_cache_threshold``
can dispose of a throttle by the cache actually at stake, and honoring a
post-backoff redial under an authored ``throttle_redial`` schedule.
``throttle_redial_budgets`` applies the same cache-stakes gate at
admission, per rung, so the data plane knows how long each rung is worth
waiting on before the first throttle arrives.
"""

from __future__ import annotations

import dataclasses
import logging
import threading
import time
from collections.abc import Callable

from exp.common.models.gateway_catalog import ExactModelDeployment
from exp.runtime.gateway.contracts import GatewayEvent, GatewayFailure, GatewayFailureClass
from exp.runtime.gateway.health import DeploymentHealthKey, DeploymentHealthRegistry
from exp.runtime.gateway.native_execution import (
    THROTTLE_BACKOFF,
    THROTTLE_FAILOVER_COLD,
    InflightRequest,
    ThrottleDisposition,
    next_route_candidate,
    rung_load_key,
    throttle_disposition,
)
from exp.runtime.gateway.native_fallback_rules import route_fallback_rules
from exp.runtime.gateway.routing import GatewayRoute
from exp.runtime.gateway.rung_admission import RungLoadRegistry, RungShed, RungShedReason
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
    rate_retry: bool = False,
) -> str | RungShed | None:
    """Reserve one policy-bounded slot on a rung, or report the shed.

    Args:
        loads: The worker's per-rung in-flight and rate-window registry.
        sticky: The worker-local conversation-to-rung bindings.
        entry: The owning in-flight request (organization and weight).
        deployment: The claimed rung about to dispatch.
        reserved_tokens: Worst-case tokens this dispatch reserves, counted
            against the rung's token window when one is authored.
        force: Admit past soft policy limits when overflow is explicitly allowed.
        rate_retry: Skip only rate-window checks after scheduled backoff.

    Returns:
        An opaque reservation ticket, the shed disclosure, or ``None``
        when the rung authors no admission policy (the untouched default).
    """
    policy = deployment.gateway.dispatch
    authored = policy is not None and (
        policy.concurrency_bound is not None
        or policy.requests_per_minute is not None
        or policy.tokens_per_minute is not None
    )
    if not authored and loads.default_bound is None:
        return None
    # An unauthored bound falls back to the worker's default lane share.
    applies_default = policy is None or policy.concurrency_bound is None
    bound = loads.default_bound if applies_default else policy.concurrency_bound
    # Scoped routes carry admission-verified warmth for THIS rung. Ordinary
    # direct affinity routes instead read their live conversation binding. Never
    # let that unscoped binding stand in for tenant/prefix/credential evidence.
    # Either warm standing bypasses only the fresh-session early threshold.
    # The early threshold
    # only exists on affinity pools AND for requests that carry a fingerprint
    # (chat/Responses admission): a surface with no session concept
    # (embeddings, images) must never be classed fresh wholesale.
    fresh_fraction = (
        policy.fresh_session_spill_fraction
        if policy is not None
        and entry.route.snapshot.stage_for_depth(
            entry.route.snapshot.deployment_ids.index(deployment.deployment_id)
        ).failover_mode
        == "maximize_cache_affinity"
        and entry.affinity_fingerprint is not None
        else None
    )
    warm_session = True
    if fresh_fraction is not None and entry.affinity_fingerprint is not None:
        warm_session = (
            entry.verified_warm_deployment_id == deployment.deployment_id
            and time.monotonic() < entry.verified_warm_until_monotonic
            if entry.recovery_scoped
            else sticky.bound_deployment(entry.affinity_fingerprint) == deployment.deployment_id
        )
    tokens_per_minute = None if policy is None else policy.tokens_per_minute
    result = loads.reserve(
        rung_load_key(deployment),
        organization_id=entry.authorization.organization_id,
        weight=entry.authorization.fair_share_weight,
        bound=bound,
        fair_share=policy is not None and policy.fair_share,
        requests_per_minute=None if policy is None else policy.requests_per_minute,
        tokens_per_minute=tokens_per_minute,
        cache_priority_alpha=None if policy is None else policy.cache_priority_alpha,
        reserved_tokens=reserved_tokens if tokens_per_minute is not None else 0,
        warm_session=warm_session,
        fresh_spill_fraction=fresh_fraction,
        force=force,
        hard_bound=applies_default or (policy is not None and policy.saturation == "refuse"),
        rate_retry=rate_retry,
    )
    if isinstance(result, RungShed) and applies_default and result.reason == "queue_bound":
        result = dataclasses.replace(result, default_bound=True)
    if isinstance(result, RungShed) and result.reason == "rate_limit":
        _logger.debug(
            "gateway rate-limit shed on deployment %r (learned ceiling %s/min)",
            deployment.deployment_id,
            result.learned_requests_per_minute,
        )
    return result


def bind_sticky_dispatch(
    sticky: StickySpillRegistry,
    entry: InflightRequest,
    deployment: ExactModelDeployment,
) -> None:
    """Refresh ordinary direct affinity placement, never scoped cache evidence."""
    if entry.route.snapshot.model_stages or entry.affinity_fingerprint is None:
        return
    if entry.route.snapshot.failover_mode != "maximize_cache_affinity":
        return
    policy = deployment.gateway.dispatch
    if policy is None or policy.sticky_spill_seconds is None:
        return
    sticky.bind(
        entry.affinity_fingerprint,
        deployment.deployment_id,
        ttl_seconds=float(policy.sticky_spill_seconds),
    )


def failed_dispatch_candidate(
    *,
    health: DeploymentHealthRegistry,
    loads: RungLoadRegistry,
    keys: tuple[DeploymentHealthKey, ...],
    entry: InflightRequest,
    failure: GatewayFailure,
    current_depth: int,
    throttle_backoff: bool = False,
) -> tuple[int | None, ThrottleDisposition | None]:
    """Choose the ladder's next candidate after one classified failure.

    Reads the cache at stake on the failed rung (the requesting organization's
    EWMA of its settled cached fraction there, zero without evidence) and
    hands it with the pool's authored ``throttle_cache_threshold`` to the
    frozen candidate policy, so a throttle is surfaced or failed over by the
    warm cache it would abandon. Without a threshold the fraction is inert.
    On a pool authoring ``throttle_redial`` a throttle instead redials the
    warm rung when the data plane has waited the backoff, advances cold once
    the redials are spent, and the disposition names which happened.

    Args:
        health: Revision-isolated circuit and throttle registry.
        loads: The worker's per-rung load registry holding the cache EWMA.
        keys: One health key per ordered route deployment.
        entry: The owning in-flight request.
        failure: The classified failure that ended the previous dispatch.
        current_depth: Route position of the failed dispatch.
        throttle_backoff: Whether the data plane waited the pool's backoff
            and asks to redial the throttled rung.

    Returns:
        ``(candidate, disposition)``: the claimed route index or ``None``
        when the ladder is exhausted, and the throttle disposition when the
        failure was a throttle on a threshold- or schedule-authoring pool
        (else ``None``).
    """
    route = entry.route
    stage = route.snapshot.stage_for_depth(current_depth)
    threshold = stage.throttle_cache_threshold
    redial = stage.throttle_redial
    deployment = route.deployments[current_depth]
    cached_fraction = loads.cached_fraction(
        rung_load_key(deployment), entry.authorization.organization_id
    )
    candidate = next_route_candidate(
        health=health,
        keys=keys,
        failure=failure,
        current_depth=current_depth,
        attempt_counts=entry.ordinary_attempt_counts,
        total_attempts=entry.total_attempts,
        refusal_failover=entry.authorization.refusal_failover,
        failover_mode=stage.failover_mode,
        throttle_cache_threshold=threshold,
        cached_fraction=cached_fraction,
        throttle_redial=redial,
        throttle_backoff=throttle_backoff,
        throttle_redial_budget=(
            entry.throttle_redial_budgets[current_depth] - entry.throttle_redials[current_depth]
        ),
        fallback_rules=route_fallback_rules(route),
        maximum_total_attempts=entry.attempt_policy.maximum_total_attempts,
        maximum_same_deployment_attempts=entry.attempt_policy.maximum_same_deployment_attempts,
        physical_route_cap=entry.attempt_policy.physical_route_cap,
        physical_attempt_counts=entry.attempt_counts,
    )
    disposition = throttle_disposition(
        failure,
        throttle_cache_threshold=threshold,
        cached_fraction=cached_fraction,
    )
    if redial is not None and failure.failure_class == GatewayFailureClass.THROTTLED:
        # With a schedule authored a throttle never surfaces mid-ladder: it
        # either redials the warm rung or advances cold past it, and an
        # exhausted ladder is a plain exhausted throttle.
        if candidate == current_depth:
            disposition = THROTTLE_BACKOFF
        elif candidate is not None:
            disposition = THROTTLE_FAILOVER_COLD
        else:
            disposition = None
    if disposition is not None:
        _logger.debug(
            "gateway throttle on deployment %r disposed %s (cached fraction %.3f, threshold %s)",
            deployment.deployment_id,
            disposition,
            cached_fraction,
            threshold,
        )
    return candidate, disposition


def shed_keeps_pin(route: GatewayRoute, candidate: int) -> bool:
    """Whether a policy shed of ``candidate`` must force-admit it rather than spill sideways.

    True only for the issuing rung of a reasoning-pinned route. Its fallbacks
    dispatch without the request's sealed reasoning, a loss reserved for a real
    failover-eligible failure on the pinned rung (a throttle once its redial
    budget is spent, provider quota, unavailability, transport), never for a
    per-worker rate or concurrency shed the rung itself authored, which trips
    under ordinary load. The shed is disclosed as ``saturated_overflow`` exactly
    as a one-rung ladder's is.

    Args:
        route: The admitted route.
        candidate: Route position of the rung that shed.

    Returns:
        Whether the accounting keeps the candidate and admits it past the policy.
    """
    return route.reasoning_pinned_deployment_id is not None and not route.requires_reasoning_strip(
        route.deployments[candidate]
    )


def shed_keeps_rung(
    route: GatewayRoute,
    candidate: int,
    redial_depth: int | None,
    last_failure: GatewayFailure | None,
    shed_reason: RungShedReason,
) -> bool:
    """Whether a policy shed of ``candidate`` force-admits it instead of spilling sideways.

    Two cases keep the rung. A post-backoff throttle redial
    (``candidate == redial_depth``) shed by the rung's RATE WINDOW
    (``rate_limit``, the authored per-worker ``requests_per_minute`` or
    ``tokens_per_minute``): the per-minute windows are pacing, and a redial
    that already waited the pool's ``throttle_redial`` schedule has paid its
    pacing on the provider's own 429 clock, so converting it into a cold
    failover would abandon the cache the caller waited to keep for a prompt a
    fallback may never finish within its first-byte allowance. The redial
    count stays bounded by the rung's admission-time budget and the request's
    attempt cap. The rung's ``concurrency_bound`` (``queue_bound``, and the
    ``fresh_session_spill`` early threshold on it) and its ``fair_share_shed``
    are NOT bypassed by a redial: the bound is the per-worker hard ceiling that
    protects the provider connection and the other tenants on the rung, and it
    stays hard for everyone, so a redial shed by it spills sideways exactly
    like any other dispatch. The other case is the issuing rung of a
    reasoning-pinned route on its first dispatch (``shed_keeps_pin``), before
    any real failure on it, for every shed reason. Every other shed spills.

    Args:
        route: The admitted route.
        candidate: Route position of the rung that shed.
        redial_depth: The rung a post-backoff redial re-dials, or ``None``
            when this reservation is not a redial.
        last_failure: The classified failure that ended the previous
            dispatch, or ``None`` on the request's first reservation.
        shed_reason: Why the rung's dispatch policy refused the reservation.

    Returns:
        Whether the accounting keeps the candidate and admits it past the policy.
    """
    if candidate == redial_depth and shed_reason == "rate_limit":
        return True
    return last_failure is None and shed_keeps_pin(route, candidate)


def throttle_redial_budgets(
    loads: RungLoadRegistry,
    route: GatewayRoute,
    organization_id: str,
    *,
    sticky_deployment_id: str | None = None,
) -> tuple[int, ...]:
    """Size, per rung, how many post-backoff redials this request may spend there.

    Read once at admission so the data plane knows before the first throttle
    how long each rung is worth waiting on. Every budget is zero on a pool
    without a ``throttle_redial`` schedule (the historical failover-only
    throttle). With a schedule and no ``throttle_cache_threshold`` every rung
    gets the schedule's full ``max_attempts``: the operator asked for backoff
    on this pool. With both, the budget is decided rung by rung under three
    rules, in this order:

    1. No cold alternative: the LAST rung of the admitted route gets the full
       ``max_attempts`` regardless of cache evidence. ``route`` is the route
       as admitted, already narrowed to the rungs that are live and can serve
       this request, and a throttle advances cold only to later rungs, so a
       throttle on the last rung has nowhere to fail over. A zero budget
       there would surface the 429 at once while a bounded wait could still
       have served the request. A single-rung route is the same case.
    2. Warm sticky session: the rung the request's affinity fingerprint is
       bound to in the worker's ``StickySpillRegistry``
       (``sticky_deployment_id``) gets the full ``max_attempts``. The binding
       is direct evidence that the conversation's provider cache lives on
       that rung, the same warm standing ``reserve_rung_slot`` honors. The
       issuing rung of a reasoning continuation
       (``route.reasoning_pinned_deployment_id``) is the same case: it alone
       can replay the request's thinking, so failing over past it costs the
       turn's continuity as well as its cache, and it waits the whole
       schedule before its fallbacks are tried.
    3. Otherwise the budget scales with the cache at stake: the full
       ``max_attempts`` when the requesting organization's observed cached
       fraction on the rung meets the threshold, a proportional share
       (``floor(max_attempts * fraction / threshold)``) below it, and zero
       with no cache evidence, so a request with little to lose fails over
       sooner and one with nothing to lose fails over at once.

    Rules 1 and 2 exist because the fraction rule 3 reads is the WORKER-LOCAL
    time-decayed EWMA of the organization's settled cached fraction on the
    rung: it is zero when the organization has no live sample on this worker
    (a throttled attempt settles without usage, and a conversation trickling
    a few requests per hour across many workers leaves most of them without
    one) even when the same conversation is over ninety percent cached at the
    provider. Missing evidence must therefore never zero the budget when
    waiting is the only move (rule 1) or the obviously right one (rule 2).
    The fraction is the admission-time EWMA, at most seconds older than the
    reading a failure-time decision would take.

    Args:
        loads: The worker's per-rung load registry holding the cache EWMA.
        route: The admitted route, narrowed to the rungs that can serve this
            request, in dispatch order.
        organization_id: The requesting organization.
        sticky_deployment_id: The rung the request's affinity fingerprint
            holds a live sticky binding to, or ``None`` without one.

    Returns:
        One redial budget per route deployment, in route order.
    """
    snapshot = route.snapshot
    last_depth = len(route.deployments) - 1
    pinned_deployment_id = route.reasoning_pinned_deployment_id
    budgets: list[int] = []
    for depth, deployment in enumerate(route.deployments):
        stage = snapshot.stage_for_depth(depth)
        schedule = stage.throttle_redial
        threshold = stage.throttle_cache_threshold
        if schedule is None:
            budgets.append(0)
            continue
        if (
            threshold is None
            or threshold <= 0
            or depth == last_depth
            or deployment.deployment_id == sticky_deployment_id
            or deployment.deployment_id == pinned_deployment_id
        ):
            budgets.append(schedule.max_attempts)
            continue
        fraction = loads.cached_fraction(rung_load_key(deployment), organization_id)
        share = min(1.0, fraction / threshold)
        budgets.append(int(schedule.max_attempts * share))
    return tuple(budgets)


def record_cache_fraction(
    loads: RungLoadRegistry,
    entry: InflightRequest,
    attempt_id: str,
    terminal: GatewayEvent,
    *,
    lock: threading.Lock,
    sample_gate: Callable[[str], bool] | None,
) -> None:
    """Fold only actual provider cache evidence into the rung fairness sample once.

    Args:
        loads: Existing worker admission registry receiving the observed sample.
        entry: Request-local attempt ownership and completed-sample membership.
        attempt_id: Durable physical attempt being settled.
        terminal: Final event; estimated disconnect usage never establishes cache warmth.
        lock: Accounting lock protecting the existing membership test and mark.
        sample_gate: Optional hosted funding eligibility check, run outside the lock.
    """
    usage = terminal.usage
    if usage is None or usage.input_tokens is None or terminal.usage_estimated:
        return
    depth = entry.attempt_depths.get(attempt_id)
    if depth is None:
        return
    if sample_gate is not None:
        # Promo-funded (or otherwise excluded) attempts must not buy
        # fair-share weight; an erroring gate skips the sample rather
        # than admit one the host meant to exclude.
        try:
            if not sample_gate(attempt_id):
                return
        except Exception:  # noqa: BLE001 - the sample is telemetry, never worth failing settle.
            return
    with lock:
        if attempt_id in entry.cache_recorded_attempts:
            return
        entry.cache_recorded_attempts.add(attempt_id)
    cached = usage.cached_input_tokens
    loads.record_settle(
        rung_load_key(entry.route.deployments[depth]),
        entry.authorization.organization_id,
        cached_tokens=0 if cached is None else cached,
        input_tokens=usage.input_tokens,
    )
