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
   ``saturation="refuse"``; the default lane bound always refuses, because a
   protective bound that overflows protects nothing. A refusal is a fast,
   retryable 429 (``lane_saturated_failure``) with Retry-After, answered
   before any dispatch, so the caller's retry lands when a slot frees rather
   than queueing behind the slow lane.
"""

from __future__ import annotations

import math

from exp.runtime.gateway.routing import GatewayRoute
from exp.runtime.gateway.rung_admission import RungShed
from exp.runtime.gateway.stream_contracts import GatewayFailure, GatewayFailureClass

# The share of a worker's admission permits one unauthored lane may hold.
# Half: a saturated lane leaves at least half the worker for every other
# model, and a pool of two saturated lanes still cannot hold more than the
# worker (its third lane would). Hosts tune it through the bound they pass.
DEFAULT_LANE_SHARE = 0.5

# What a refused caller is told to wait. Slots free at the pace the slow
# lane finishes, seconds apart, so a short retry lands on a freed slot
# instead of stacking a queue the request deadline would have to drain.
LANE_SATURATED_RETRY_AFTER_SECONDS = 2


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
    caller-facing 429 ``unavailable_route`` with the Retry-After the message
    states, exactly like a pool whose every rung sits in a provider throttle
    window.
    """
    return GatewayFailure(
        failure_class=GatewayFailureClass.THROTTLED,
        safe_message=(
            "every lane for this model is at its in-flight bound on this gateway worker; "
            f"retry in {LANE_SATURATED_RETRY_AFTER_SECONDS} seconds"
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
    (never force-admitted: the default protects the worker), or when the rung
    authors ``saturation="refuse"``. A bypass that was not a registry shed
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
    depth = policy_sheds[0][0]
    shed = shed_records.get(depth)
    if shed is not None and shed.default_bound:
        return None
    policy = route.deployments[depth].gateway.dispatch
    if policy is not None and policy.saturation == "refuse":
        return None
    return depth
