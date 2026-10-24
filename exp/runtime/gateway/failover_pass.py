"""One failover reservation walk: the untried ladder, its sideways spill, and its forced pass.

When a dispatched rung fails before commitment with a failover-eligible
failure, the accounting bridge reserves a successor. Two production failures
(2026-10-21, a route of two self-hosted lanes with refusing bounds) shaped
this walk:

- The successor's rung slot shed on its own refusing bound and the walk ended
  as exhausted, answering the caller the ORIGINAL 502 although a live rung
  had only been full for a moment. A refusing bound protects the worker from
  a slow lane holding every admission permit, which matters for a fresh
  request; a request whose rung already failed has nowhere else to go. So a
  failover first spills sideways to every other untried rung like any shed,
  and only when every one of them shed does it FORCE a pass on the first shed
  rung. Forced failover passes are counted per rung and capped
  (``lane_saturation.failover_overflow_cap``: half the bound in flight on top
  of it), so one dead lane cannot shift its whole load onto the backup at
  once. A rung at that cap, a hard class cap or ``:fast`` reserve, or a
  rate-window shed (pacing the operator authored to stay under the provider's
  own limit) is skipped for the next shed rung. When every shed rung refuses,
  the request ends with the fast retryable ``lane_saturated`` 429, never the
  dead rung's 502: live capacity exists and a retry in seconds lands on it.
  Only a DEAD-LANE failure (``transport`` or ``provider_internal``: refused,
  5xx) earns a forced pass. A throttle or a timeout fails over as usual but
  never forces past a refusing bound: the failed rung is alive, and a
  GPU-bound self-hosted sibling must not be over-queued for it.
- A rung skipped at the request's first dial (its bound was full) was never
  revisited when the rung dialed instead failed: successors were only later
  rungs. The ladder is every depth this request has not dispatched, earlier
  ones included, in route order (``native_fallback_rules.untried_depths``),
  and the sideways spill walks the rest of that ladder rather than only the
  depths after the shed one.

A first dial keeps its own rules unchanged (``lane_saturation``): a fresh
request is still refused at a full refusing lane.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from exp.runtime.gateway.contracts import (
    AuthorizationSnapshot,
    GatewayFailure,
    GatewayFailureClass,
)
from exp.runtime.gateway.health import DeploymentHealthKey, DeploymentHealthRegistry
from exp.runtime.gateway.lane_ejection import DEAD_LANE_FAILURES
from exp.runtime.gateway.lane_saturation import lane_saturated_failure
from exp.runtime.gateway.native_execution import InflightRequest, claim_route_from
from exp.runtime.gateway.native_fallback_rules import route_fallback_rules, untried_depths
from exp.runtime.gateway.rung_admission import RungShed


@dataclass
class FailoverPass:
    """The successor walk of one failover reservation.

    Attributes:
        untried: Every depth the failover may land on, fixed at the walk's
            start: not dispatched yet in this request and admitted by the
            ``failover_only_on`` rules, in route order.
        ladder: The untried depths not yet shed or skipped in this walk.
        may_force: Whether the triggering failure was a dead lane, the only
            failure that earns a forced pass.
        forced: Depths this walk already tried to force past their bound.
        forcing: Whether the current candidate is a forced pass.
    """

    untried: tuple[int, ...]
    ladder: tuple[int, ...]
    may_force: bool = False
    forced: set[int] = field(default_factory=set)
    forcing: bool = False

    @classmethod
    def begin(
        cls, entry: InflightRequest, failure: GatewayFailure | None, current_depth: object
    ) -> FailoverPass | None:
        """Start the walk for a failure that may advance, else ``None``.

        A first dial (no failure, no depth) is not a failover. A failure the
        route policy would not advance (a client error, an unopted refusal)
        keeps the historical walk: same-rung redials only.

        Args:
            entry: The owning in-flight request.
            failure: The classified failure the data plane reported, if any.
            current_depth: The depth that failed, as the data plane sent it.

        Returns:
            The walk, or ``None`` when this reservation is not a failover.
        """
        if failure is None or not isinstance(current_depth, int):
            return None
        refusal = (
            failure.failure_class == GatewayFailureClass.REFUSAL
            and entry.authorization.refusal_failover
        )
        if not (failure.failover_eligible or refusal):
            return None
        untried = untried_depths(
            route_fallback_rules(entry.route), failure, entry.attempt_counts, current_depth
        )
        return cls(
            untried=untried,
            ladder=untried,
            may_force=failure.failure_class in DEAD_LANE_FAILURES,
        )

    def sideways(
        self,
        health: DeploymentHealthRegistry,
        keys: tuple[DeploymentHealthKey, ...],
        candidate: int,
    ) -> int | None:
        """Claim the next untried depth after ``candidate`` shed or was skipped.

        A forced pass that shed again ends its own try: the caller moves on
        to the next forced target rather than spilling once more.

        Args:
            health: The worker's circuit registry.
            keys: One health key per route depth.
            candidate: The depth that just shed or was skipped.

        Returns:
            The next claimed depth, or ``None`` when the ladder is spent.
        """
        if self.forcing:
            self.forcing = False
            return None
        self.ladder = tuple(depth for depth in self.ladder if depth != candidate)
        return claim_route_from(health, keys, 0, self.ladder) if self.ladder else None

    def forced_target(
        self,
        policy_sheds: list[tuple[int, str]],
        shed_records: dict[int, RungShed],
        health: DeploymentHealthRegistry,
        keys: tuple[DeploymentHealthKey, ...],
    ) -> int | None:
        """The next shed untried depth to force past its bound, or ``None`` to give up.

        The shed released the rung's health claim, so a forced pass claims it
        again (healthy, probe, or forced through an open circuit, never
        through a provider throttle window) before dialing. A depth whose
        claim fails is marked forced and NOT retried in this walk: it sits in
        a provider throttle window, and the walk moves on to the next shed
        rung (or ends) rather than waiting it out.

        Args:
            policy_sheds: ``(depth, reason)`` for every policy bypass this
                reservation, in walk order.
            shed_records: The registry's shed per bypassed depth.
            health: The worker's circuit registry.
            keys: One health key per route depth.

        Returns:
            The depth to force-admit, marking this walk's pass as forced.
        """
        if not self.may_force:
            return None
        for depth, _reason in policy_sheds:
            shed = shed_records.get(depth)
            if (
                depth in self.forced
                or depth not in self.untried
                or shed is None
                or shed.overflow_ceiling
                or shed.reason == "rate_limit"
            ):
                continue
            self.forced.add(depth)
            if claim_route_from(health, keys, depth, (depth,)) != depth:
                continue
            self.forcing = True
            return depth
        return None


def failover_refusal(
    failover: FailoverPass | None,
    authorization: AuthorizationSnapshot,
    shed_records: dict[int, RungShed],
) -> GatewayFailure | None:
    """The failure a failover walk ends with once nothing more can be forced.

    Args:
        failover: The failover walk, or ``None`` when this is not a failover.
        authorization: The caller, whose priority level picks the 429's copy.
        shed_records: The registry's shed per bypassed depth.

    Returns:
        The retryable ``lane_saturated`` 429 when a dead-lane failover found
        an untried rung that refused (full past its failover cap, a hard cap,
        or a rate window); ``None`` otherwise, so the original failure stands
        (nothing untried remained, or the trigger never earns a forced pass).
    """
    if (
        failover is None
        or not failover.may_force
        or not any(depth in shed_records for depth in failover.untried)
    ):
        return None
    return lane_saturated_failure(authorization)


def forced_failover_target(
    failover: FailoverPass | None,
    policy_sheds: list[tuple[int, str]],
    shed_records: dict[int, RungShed],
    health: DeploymentHealthRegistry,
    keys: tuple[DeploymentHealthKey, ...],
) -> int | None:
    """The failover's next forced pass (``FailoverPass.forced_target``), or ``None``.

    Args:
        failover: The failover walk, or ``None`` when this is not a failover.
        policy_sheds: ``(depth, reason)`` for every policy bypass this reservation.
        shed_records: The registry's shed per bypassed depth.
        health: The worker's circuit registry.
        keys: One health key per route depth.

    Returns:
        The depth to force-admit, or ``None`` when the walk ends.
    """
    if failover is None:
        return None
    return failover.forced_target(policy_sheds, shed_records, health, keys)


def sideways_claim(
    failover: FailoverPass | None,
    health: DeploymentHealthRegistry,
    keys: tuple[DeploymentHealthKey, ...],
    ladder: tuple[int, ...],
    candidate: int,
) -> int | None:
    """Claim where a shed or skipped ``candidate`` spills.

    Args:
        failover: The failover walk, or ``None`` for a first dial or a walk
            that never fails over (the historical later-only spill).
        health: The worker's circuit registry.
        keys: One health key per route depth.
        ladder: The reservation's eligible depths (used without a failover).
        candidate: The depth that shed or was skipped.

    Returns:
        The next claimed depth, or ``None`` when nothing is left.
    """
    if failover is None:
        return claim_route_from(health, keys, candidate + 1, ladder)
    return failover.sideways(health, keys, candidate)
