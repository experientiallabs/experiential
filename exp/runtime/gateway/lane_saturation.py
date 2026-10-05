"""Worker protection against one slow lane holding every admission permit.

A gateway worker admits at most ``max_active_requests`` requests at once (the
data plane's permit semaphore); a request past that waits for a permit until
its own deadline. On 2026-09-19 one tier-4 organization sent ~100 requests a
minute to a model whose lead rung degraded to a two-minute first token; the
rung authored no ``concurrency_bound``, so 270-430 of its requests sat in
flight, held every permit on every worker, and EVERY route on the gateway
(other models, ``/v1/models``) waited minutes at the edge while CPU stayed at
30-60% and nothing scaled.

Two rules close that:

1. The DEFAULT LANE BOUND. A rung that authors no ``concurrency_bound`` is
   bounded anyway, per worker, at ``default_lane_bound(max_active_requests)``:
   a fixed share (``DEFAULT_LANE_SHARE``) of the worker's permits, so one
   physical lane can never hold them all. Past it the request spills to the
   next rung exactly like an authored bound (``queue_bound``). An authored
   ``concurrency_bound`` replaces the default on its rung, higher or lower.
2. REFUSAL INSTEAD OF OVERFLOW. When every rung of a pool is at its bound the
   accounting used to force-admit past the first shed rung
   (``saturated_overflow``: "policy never manufactures a failure"). That is
   still the default for an AUTHORED bound, and an authored rung may opt into
   ``saturation="refuse"``; the default lane bound refuses too. A priority
   caller (the host's paying and Pro organizations) is the exception on both:
   its shed overflows, capped at 1.25x / 1.5x the bound so the worker stays
   protected. A refusal is a fast,
   retryable 429 (``lane_saturated_failure``) with the protocol's throttle
   Retry-After, answered
   before any dispatch, so the caller's retry lands when a slot frees rather
   than queueing behind the slow lane.
"""

from __future__ import annotations

import math

from exp.runtime.gateway.routing import GatewayRoute
from exp.runtime.gateway.rung_admission import RungShed
from exp.runtime.gateway.stream_contracts import GatewayFailure, GatewayFailureClass
from exp.runtime.openai_protocol.errors import THROTTLED_RETRY_AFTER_SECONDS

# The share of a worker's admission permits one unauthored lane may hold.
# Half: a saturated lane leaves at least half the worker for every other
# model, and a pool of two saturated lanes still cannot hold more than the
# worker (its third lane would). Hosts tune it through the bound they pass.
DEFAULT_LANE_SHARE = 0.5

# What a refused caller is told to wait: the protocol's throttle floor (the
# renderer never emits a shorter Retry-After, so the message, the payload and
# the header agree). Slots free at the pace the slow lane finishes, so a
# retry after it lands on a freed slot instead of stacking a queue the
# request deadline would have to drain.
LANE_SATURATED_RETRY_AFTER_SECONDS = THROTTLED_RETRY_AFTER_SECONDS

# How far past a refusing AUTHORED bound each
# ``AuthorizationSnapshot.priority_admission`` level may overflow, as a multiple
# of the bound: free callers never, paying callers to 1.25x, Pro callers to
# 1.5x. The Pro cap stays at 1.5x because one organization's long cold
# prompts at twice a self-hosted origin's bound saturate the origin itself.
PRIORITY_OVERFLOW_FACTORS = (1.0, 1.25, 1.5)

# The same for the worker's DEFAULT lane bound, which is already a share of the
# worker's permits (DEFAULT_LANE_SHARE): the Pro factor stays strictly below
# 1 / DEFAULT_LANE_SHARE, so one lane's priority traffic can never hold every
# permit (1.5 x half the permits leaves a quarter for every other lane).
DEFAULT_BOUND_OVERFLOW_FACTORS = (1.0, 1.25, 1.5)


def priority_overflow_ceiling(
    bound: int | None,
    priority_admission: int,
    *,
    default_bound: bool,
    authored: tuple[float | None, float | None] = (None, None),
) -> float | None:
    """The in-flight ceiling a forced priority overflow may not exceed, or ``None``.

    Args:
        bound: The rung's effective bound (authored or the worker default).
        priority_admission: The caller's level (0 free, 1 paying, 2 Pro).
        default_bound: Whether ``bound`` is the worker's default lane bound.
        authored: The rung's authored ``(priority_overflow_paying,
            priority_overflow_pro)`` multiples; ``None`` keeps the default.
            Ignored under the default bound (its factors protect the worker).

    Returns:
        ``bound * factor`` for a priority caller on a bounded rung, else
        ``None``. The admitted count floors against it (1.5x of 5 holds 7).
    """
    if bound is None or not priority_admission:
        return None
    if default_bound:
        return bound * DEFAULT_BOUND_OVERFLOW_FACTORS[priority_admission]
    paying, pro = (
        PRIORITY_OVERFLOW_FACTORS[level] if value is None else value
        for level, value in ((1, authored[0]), (2, authored[1]))
    )
    # Paying never overflows past Pro, even when only one side is authored.
    return bound * (pro if priority_admission == 2 else min(paying, pro))


def default_lane_bound(max_active_requests: int, share: float = DEFAULT_LANE_SHARE) -> int:
    """The per-worker in-flight cap for rungs that author no ``concurrency_bound``.

    Args:
        max_active_requests: The worker's admission permit count (the data
            plane's ``max_active_requests``).
        share: The fraction of those permits one lane may hold, in ``(0, 1]``.

    Returns:
        ``ceil(max_active_requests * share)``, never below one.

    Raises:
        ValueError: The permit count is not positive or the share is outside
            ``(0, 1]``.
    """
    if max_active_requests < 1:
        raise ValueError("max_active_requests must be at least one")
    if not 0 < share <= 1:
        raise ValueError("share must be in (0, 1]")
    return max(1, math.ceil(max_active_requests * share))


def lane_saturated_failure() -> GatewayFailure:
    """The fail-fast refusal for a pool whose every rung is at its in-flight bound.

    Throttled, not provider-internal: nothing is down, the pool is full on
    this worker and the caller should retry shortly. The class renders as the
    caller-facing 429 ``unavailable_route`` with ``Retry-After: 5``, exactly
    like a pool whose every rung sits in a provider throttle window. The
    message is consumer copy (it reaches end users verbatim through clients),
    so it names capacity and the Pro priority benefit, never worker internals.
    """
    return GatewayFailure(
        failure_class=GatewayFailureClass.THROTTLED,
        safe_message=(
            "This model is at capacity right now. Please retry in a few seconds. "
            "Pro subscribers get priority access when models are busy."
        ),
        retry_after_seconds=LANE_SATURATED_RETRY_AFTER_SECONDS,
    )


def overflow_target(
    route: GatewayRoute,
    policy_sheds: list[tuple[int, str]],
    shed_records: dict[int, RungShed],
) -> int | None:
    """Where a ladder exhausted only by policy sheds force-admits, or ``None`` to refuse.

    The historical target is the first bypassed rung in ladder order. It is
    refused when that rung's shed came from the worker's default lane bound
    (the default protects the worker) or when the rung authors
    ``saturation="refuse"``, unless the caller is a priority caller
    (``AuthorizationSnapshot.priority_admission``): a priority request always
    overflows, so on a saturated lane only non-priority callers are turned
    away. The reservation caps that overflow (``priority_overflow_ceiling``);
    a rung whose forced admission hit its cap (``RungShed.overflow_ceiling``)
    is skipped for the next bypassed rung, and the request is refused once
    every bypassed rung is capped. A bypass that was not a registry shed
    (a cold throttle failover) keeps the historical overflow.

    Args:
        route: The admitted route.
        policy_sheds: ``(depth, reason)`` for every policy bypass this
            reservation, in ladder order.
        shed_records: The registry's shed per bypassed depth, for the bypasses
            that were reservations.

    Returns:
        The route depth to force-admit, or ``None`` when the request is refused.
    """
    if not policy_sheds:
        return None
    if route.snapshot.authorization.priority_admission:
        # The first bypassed rung still below its priority ceiling; a rung
        # whose forced admission already hit the ceiling is skipped, never
        # retried, so the walk ends in a refusal once every rung is capped.
        # Only a CAPACITY shed earns the priority exception: a rate-window
        # shed keeps the free caller's rule (a selected route never overflows
        # it), so priority never forces a rate cap the free rule would refuse.
        for depth, _reason in policy_sheds:
            shed = shed_records.get(depth)
            if shed is not None and shed.overflow_ceiling:
                continue
            if shed is not None and shed.reason == "rate_limit":
                if route.resolved_route_id is not None:
                    return None
                return _historical_target(route, depth, shed)
            return depth
        return None
    depth = policy_sheds[0][0]
    return _historical_target(route, depth, shed_records.get(depth))


def _historical_target(route: GatewayRoute, depth: int, shed: RungShed | None) -> int | None:
    """The free caller's overflow rule for one bypassed rung: refuse a refusing bound.

    Args:
        route: The admitted route.
        depth: The bypassed rung's route depth.
        shed: The registry's shed on that rung, when the bypass was a reservation.

    Returns:
        ``depth`` to force-admit, or ``None`` when the rung's bound refuses.
    """
    if shed is not None and shed.default_bound:
        return None
    policy = route.deployments[depth].gateway.dispatch
    if policy is not None and policy.saturation == "refuse":
        return None
    return depth
