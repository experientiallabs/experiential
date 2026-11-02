"""Fleet-wide lane admission: one rate and concurrency budget shared by every worker.

``GatewayRungDispatchPolicy.requests_per_minute`` and ``concurrency_bound`` are
per worker process, so a fleet of N workers sends up to N times the authored
value and the realized rate swings with how traffic lands on workers.
``GatewayRungDispatchPolicy.fleet`` is one budget for the whole fleet. The engine
cannot hold that budget itself (each worker is a separate process), so the
host injects a :class:`FleetLaneGate` backed by a store every worker shares,
and this module calls it at the one moment that matters: right after the
worker's own admission policy reserved a slot on the rung and right before
the dispatch.

The gate answers a lease (admitted: the dispatch is counted against the
lane's fleet budget until the lease is released) or a :class:`FleetLaneShed`
(the lane is at its fleet pace or its fleet concurrency). The fleet rate is
the operator's own HARD limit, never a provider hint: it must hold even when
the provider accepts everything. So a fleet shed spills sideways like any
policy shed, and it is marked a hit overflow ceiling: no forced overflow,
post-backoff redial or failover pass ever lands back on the capped lane. When
every rung sheds, the waterfall force-admits a sibling that allows overflow,
or answers ``lane_saturated`` when none does. A lane authoring
``saturation="refuse"`` answers ``lane_saturated`` at once instead of
spilling; ``queue`` is the host's to honor (it may hold the dispatch briefly
for its next slot before answering).

Every gate call is guarded: a gate that raises admits the dispatch (the
worker's own per-worker policy still applies) and is counted, so a store
outage can never stop traffic to a lane. The host owns the store, the pacing
algorithm and its fail-open fallback; this module owns only the bookkeeping
that ties each lease to its reservation ticket and durable attempt so the
lease is released exactly once, with the attempt's measured duration and
whether the provider throttled it.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Protocol

from exp.common.models.dispatch_policy import GatewayFleetLanePolicy

_logger = logging.getLogger(__name__)

FleetLaneShedReason = Literal["rate", "concurrency"]


@dataclass(frozen=True)
class FleetLane:
    """One physical lane and its authored fleet policy.

    Attributes:
        deployment_id: The rung's deployment id (stable across catalog revisions).
        connection: The rung's credentialed connection identity (the second half
            of ``rung_admission.RungLoadKey``): a different account is a
            different lane with its own provider limits.
        policy: The rung's authored ``GatewayRungDispatchPolicy.fleet``.
    """

    deployment_id: str
    connection: str
    policy: GatewayFleetLanePolicy


@dataclass(frozen=True)
class FleetLaneShed:
    """The fleet gate refused one dispatch.

    Attributes:
        reason: ``rate`` when the lane is ahead of its fleet pace, ``concurrency``
            when the fleet already holds as many in-flight dispatches on it as
            the gate allows.
    """

    reason: FleetLaneShedReason


class FleetLaneGate(Protocol):
    """Host-provided fleet-wide admission for lanes authoring a fleet rate.

    Both calls run synchronously on the dispatch path, so an implementation
    must answer in bounded time (a short store timeout) and may raise: the
    engine treats an ``acquire`` exception as admission without a lease.
    Every lease MUST also expire on its own after a host-chosen lifetime longer
    than any request: ``release`` is best-effort (a raising release is logged
    and not retried, and a crashed worker never releases at all), so an
    implementation whose leases never expire would leak capacity.
    """

    def acquire(
        self, lane: FleetLane, *, reserved_tokens: int, deadline_seconds: float
    ) -> str | FleetLaneShed:
        """Admit one dispatch onto ``lane`` fleet-wide.

        Args:
            lane: The lane and its authored fleet policy.
            reserved_tokens: The dispatch's worst-case input plus output tokens.
            deadline_seconds: What the request's deadline leaves: a ``queue``
                lane must never hold a dispatch past it.

        Returns:
            An opaque lease id on admission, else the shed.
        """
        ...

    def release(
        self,
        lane: FleetLane,
        lease: str,
        *,
        duration_seconds: float | None,
        throttled: bool,
        used_tokens: int | None,
        output_tokens: int | None,
        first_token_seconds: float | None,
    ) -> None:
        """Return one lease.

        Args:
            lane: The lane the lease was acquired on.
            lease: The id ``acquire`` returned.
            duration_seconds: How long the dispatch held the lane, ``None``
                when the reservation never dispatched.
            throttled: The provider answered this dispatch with a throttle. A
                gate may slow the lane further on it; it must never let the
                lane exceed the authored rate.
            used_tokens: The settled input plus output tokens, ``None`` when
                unknown (never dispatched, abandoned, or no usage reported):
                the reservation then stays counted in full.
            output_tokens: The settled output tokens, when reported (the
                lane's decode-rate estimate).
            first_token_seconds: Seconds from the lease to the dispatch's first
                token, when the data plane reported one (the lane's TTFT).
        """
        ...


@dataclass(frozen=True)
class FleetUsage:
    """What one settled dispatch reports back to the gate (all ``None`` when unknown).

    Attributes:
        used_tokens: Settled input plus output tokens; ``None`` when the
            settlement carried no credible usage.
        output_tokens: Settled output tokens (the lane's decode-rate sample);
            ``None`` without credible usage.
        first_token_at: Wall-clock (epoch seconds) time of the first token, when
            the data plane reported one.
    """

    used_tokens: int | None = None
    output_tokens: int | None = None
    first_token_at: float | None = None


@dataclass
class _Held:
    """One lease this worker holds for a rung reservation.

    Attributes:
        lane: The lane the lease was acquired on.
        lease: The gate's opaque lease id.
        acquired_at: Monotonic acquisition time (the dispatch's duration base).
        acquired_wall: Wall-clock acquisition time (the first-token base).
    """

    lane: FleetLane
    lease: str
    acquired_at: float
    acquired_wall: float


class FleetLeases:
    """Ties fleet leases to rung tickets and attempts; releases each exactly once.

    Owned by the worker's ``RungLoadRegistry`` (``attach_fleet_gate``). Every
    gate call is made outside the registry's lock and outside this object's
    lock, so a slow store never serializes the worker's other admissions.
    """

    def __init__(self, gate: FleetLaneGate, *, clock: Callable[[], float] = time.monotonic) -> None:
        """Bind the host gate.

        Args:
            gate: The host's fleet-wide admission.
            clock: Monotonic clock measuring each dispatch's duration.
        """
        self._gate = gate
        self._clock = clock
        self._lock = threading.Lock()
        self._by_ticket: dict[str, _Held] = {}
        self._ticket_by_attempt: dict[str, str] = {}
        self.gate_errors = 0
        self.sheds = 0

    def acquire(
        self,
        ticket: str,
        lane: FleetLane,
        *,
        reserved_tokens: int,
        deadline_seconds: float = float("inf"),
    ) -> FleetLaneShed | None:
        """Ask the gate for one dispatch on ``lane`` held by ``ticket``.

        Args:
            ticket: The rung reservation this dispatch already holds.
            lane: The lane and its fleet policy.
            reserved_tokens: The dispatch's worst-case input plus output tokens.
            deadline_seconds: What the request's deadline leaves.

        Returns:
            The shed, or ``None`` when admitted (with or without a lease).
        """
        try:
            verdict = self._gate.acquire(
                lane, reserved_tokens=reserved_tokens, deadline_seconds=deadline_seconds
            )
        except Exception:  # noqa: BLE001 - a store outage must never stop the lane.
            with self._lock:
                self.gate_errors += 1
            _logger.warning(
                "fleet lane gate failed; admitting on the per-worker policy (deployment %r)",
                lane.deployment_id,
                exc_info=True,
            )
            return None
        if isinstance(verdict, FleetLaneShed):
            with self._lock:
                self.sheds += 1
            return verdict
        with self._lock:
            self._by_ticket[ticket] = _Held(lane, verdict, self._clock(), time.time())
        return None

    def bind(self, ticket: str, attempt_id: str) -> None:
        """Attach a ticket's lease to the durable attempt that now holds it."""
        with self._lock:
            if ticket in self._by_ticket:
                self._ticket_by_attempt[attempt_id] = ticket

    def release_ticket(self, ticket: str) -> None:
        """Release the lease of a reservation that never dispatched; idempotent."""
        with self._lock:
            held = self._by_ticket.pop(ticket, None)
        if held is not None:
            self._release(held, dispatched=False, throttled=False, usage=FleetUsage())

    def release_attempt(
        self, attempt_id: str, *, throttled: bool, usage: FleetUsage | None = None
    ) -> None:
        """Release the lease held by one settled or abandoned attempt; idempotent."""
        with self._lock:
            ticket = self._ticket_by_attempt.pop(attempt_id, None)
            held = None if ticket is None else self._by_ticket.pop(ticket, None)
        if held is not None:
            self._release(held, dispatched=True, throttled=throttled, usage=usage or FleetUsage())

    def held(self) -> int:
        """Return how many leases this worker holds (metrics and tests)."""
        with self._lock:
            return len(self._by_ticket)

    def _release(
        self, held: _Held, *, dispatched: bool, throttled: bool, usage: FleetUsage
    ) -> None:
        duration = max(0.0, self._clock() - held.acquired_at) if dispatched else None
        first = usage.first_token_at
        first_token_seconds = (
            None if first is None or not dispatched else max(0.0, first - held.acquired_wall)
        )
        try:
            self._gate.release(
                held.lane,
                held.lease,
                duration_seconds=duration,
                throttled=throttled,
                used_tokens=usage.used_tokens,
                output_tokens=usage.output_tokens,
                first_token_seconds=first_token_seconds,
            )
        except Exception:  # noqa: BLE001 - the host's lease expiry reclaims it.
            with self._lock:
                self.gate_errors += 1
            _logger.warning(
                "fleet lane gate release failed (deployment %r); the lease expires on its own",
                held.lane.deployment_id,
                exc_info=True,
            )
