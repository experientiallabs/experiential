"""Per-worker, per-lane ejection of a rung whose box answers nothing but dead-lane failures.

On 2026-10-21 a self-hosted rung's box refused every connection (or answered
an instant 502) for twenty minutes, and every request on the route dialed it
first: 7,000+ wasted dials. The revision-scoped health circuit
(``DeploymentHealthRegistry``) keys on the authorization's catalog digest, so
its memory restarts per revision view, and it counts timeouts and malformed
streams alike. This tracker keys on the physical lane, the ``rung_load_key``
convention ``(deployment_id, connection_sha256)``, across every revision view,
so every request on the worker learns from every other one. The deployment id
is the one the plan carries (``mp-<model_provider uuid>`` for a host lane); a
BYOK variant carries its own id (``mp-<row>-c-<connection>``) and so never
inherits its host lane's ejection, nor the host lane a variant's. It counts
only the failures that say the box itself is down: transport failures
(refused, reset, unreachable) and provider 5xx (``provider_internal``: 500,
502, 503, 504), and, on a separate and slower streak, header timeouts (a box
that accepts connections and never answers its headers holds every request
for its whole authored allowance, which can be minutes; only a timeout on the
lane's own allowance counts, never one the request's shared budget cut). A client 4xx, a
throttle, or any other timeout says nothing here (each has its own handling):
it neither extends nor resets a streak and never allocates state; a success,
or an open stream, closes both streaks.

``failures`` dead-lane failures inside ``window_seconds``, or
``header_timeouts`` header timeouts inside ``header_window_seconds`` (no
success between them), eject the lane for ``cooldown_seconds``. An ejected lane is never removed:
the claim walk orders it LAST among a route's candidates, so a single-rung
route still dials it and fails fast, and a route whose other rungs are all
unclaimable still reaches it. After the cooldown the lane is half-open: the
first request that claims it is its probe (other requests keep ordering it
last), a success closes the ejection, and a counted failure re-ejects it for
another cooldown. A probe that never reports back expires after one cooldown
so the lane cannot stay ejected forever on a lost settlement.

State is in memory, per worker, and never published to the catalog. Never
account-wide or provider-wide: a dead lane ejects only itself.
"""

from __future__ import annotations

import os
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field

from exp.runtime.gateway.contracts import (
    GatewayFailure,
    GatewayFailureClass,
    GatewayHeaderTimeout,
)

LaneKey = tuple[str, str]
"""``(deployment_id, connection_sha256)``: one physical lane on this worker."""

DEAD_LANE_FAILURES = frozenset(
    {GatewayFailureClass.TRANSPORT, GatewayFailureClass.PROVIDER_INTERNAL}
)
"""The failure classes that count toward an ejection: the box itself is down."""

DEFAULT_EJECTION_FAILURES = 3
DEFAULT_EJECTION_WINDOW_SECONDS = 10.0
DEFAULT_EJECTION_HEADER_TIMEOUTS = 2
DEFAULT_EJECTION_HEADER_WINDOW_SECONDS = 600.0
DEFAULT_EJECTION_COOLDOWN_SECONDS = 30.0

EJECTION_FAILURES_ENV = "EXP_GATEWAY_LANE_EJECTION_FAILURES"
EJECTION_WINDOW_ENV = "EXP_GATEWAY_LANE_EJECTION_WINDOW_SECONDS"
EJECTION_HEADER_TIMEOUTS_ENV = "EXP_GATEWAY_LANE_EJECTION_HEADER_TIMEOUTS"
EJECTION_HEADER_WINDOW_ENV = "EXP_GATEWAY_LANE_EJECTION_HEADER_WINDOW_SECONDS"
EJECTION_COOLDOWN_ENV = "EXP_GATEWAY_LANE_EJECTION_COOLDOWN_SECONDS"


def is_header_timeout(failure: GatewayFailure) -> bool:
    """Whether ``failure`` is a header timeout on the lane's OWN allowance.

    The data plane types every header-phase timeout (``header_timeout``) and
    marks the ones the request's shared budget cut as ``budget``: a healthy
    fallback dialed after a hung lead gets only the budget's remainder, and
    must not be struck for running out of someone else's time.
    """
    return failure.header_timeout is GatewayHeaderTimeout.ALLOWANCE


@dataclass(frozen=True)
class LaneEjectionPolicy:
    """When a lane is ejected and for how long.

    Attributes:
        failures: Consecutive dead-lane failures that eject the lane; zero
            disables ejection entirely (header timeouts included).
        window_seconds: The span those failures must fall inside.
        header_timeouts: Consecutive header timeouts that eject the lane;
            zero disables only this streak.
        header_window_seconds: The span those header timeouts must fall inside.
        cooldown_seconds: How long an ejected lane is ordered last before
            one request may probe it.
    """

    failures: int = DEFAULT_EJECTION_FAILURES
    window_seconds: float = DEFAULT_EJECTION_WINDOW_SECONDS
    header_timeouts: int = DEFAULT_EJECTION_HEADER_TIMEOUTS
    header_window_seconds: float = DEFAULT_EJECTION_HEADER_WINDOW_SECONDS
    cooldown_seconds: float = DEFAULT_EJECTION_COOLDOWN_SECONDS

    def __post_init__(self) -> None:
        """Reject a policy that could never eject or never recover.

        Raises:
            ValueError: A negative count or a non-positive duration.
        """
        if self.failures < 0 or self.header_timeouts < 0:
            raise ValueError("lane ejection counts must not be negative")
        if min(self.window_seconds, self.header_window_seconds, self.cooldown_seconds) <= 0:
            raise ValueError("lane ejection windows and cooldown must be positive")

    @classmethod
    def from_environ(cls, environ: Mapping[str, str] | None = None) -> LaneEjectionPolicy:
        """Read the operator overrides, keeping the defaults for unset variables.

        Args:
            environ: The environment to read; ``None`` reads ``os.environ``.

        Returns:
            The policy with every set override applied.

        Raises:
            ValueError: An override does not parse or is out of range.
        """
        mapping = os.environ if environ is None else environ

        def read(name: str, default: float) -> float:
            """One override as a number, or the default when unset."""
            raw = mapping.get(name)
            return default if not raw else float(raw)

        return cls(
            failures=int(read(EJECTION_FAILURES_ENV, DEFAULT_EJECTION_FAILURES)),
            window_seconds=read(EJECTION_WINDOW_ENV, DEFAULT_EJECTION_WINDOW_SECONDS),
            header_timeouts=int(
                read(EJECTION_HEADER_TIMEOUTS_ENV, DEFAULT_EJECTION_HEADER_TIMEOUTS)
            ),
            header_window_seconds=read(
                EJECTION_HEADER_WINDOW_ENV, DEFAULT_EJECTION_HEADER_WINDOW_SECONDS
            ),
            cooldown_seconds=read(EJECTION_COOLDOWN_ENV, DEFAULT_EJECTION_COOLDOWN_SECONDS),
        )


@dataclass
class _LaneState:
    """One lane's streaks and ejection, mutated only under the owner's lock.

    Attributes:
        failures: Monotonic times of the current dead-lane streak.
        header_timeouts: Monotonic times of the current header-timeout streak.
        ejected_until: Monotonic instant the cooldown ends; 0 when not ejected.
        probe_started: Monotonic instant a half-open probe was claimed, or
            ``None`` when no probe is in flight.
    """

    failures: deque[float] = field(default_factory=deque)
    header_timeouts: deque[float] = field(default_factory=deque)
    ejected_until: float = 0.0
    probe_started: float | None = None


def _streak_full(streak: deque[float], threshold: int, window: float, now: float) -> bool:
    """Append ``now`` and report whether the last ``threshold`` entries fit ``window``."""
    streak.append(now)
    while len(streak) > threshold:
        streak.popleft()
    return len(streak) == threshold and now - streak[0] <= window


class LaneEjections:
    """Track dead-lane streaks and ejections per physical lane.

    Not thread-safe on its own: ``DeploymentHealthRegistry`` calls every
    method under its lock.
    """

    def __init__(self, policy: LaneEjectionPolicy) -> None:
        """Bind the ejection policy.

        Args:
            policy: Thresholds and cooldown.
        """
        self._policy = policy
        self._lanes: dict[LaneKey, _LaneState] = {}

    def tracked(self) -> int:
        """How many lanes hold state (a streak or an ejection)."""
        return len(self._lanes)

    def ejected(self, lane: LaneKey, now: float) -> bool:
        """Whether a claim walk should order the lane last right now.

        True inside the cooldown, and after it while another request's
        half-open probe is still in flight (and not yet expired).

        Args:
            lane: The physical lane.
            now: The current monotonic time.

        Returns:
            Whether the lane is ejected for this claim walk.
        """
        state = self._lanes.get(lane)
        if state is None or not state.ejected_until:
            return False
        if now < state.ejected_until:
            return True
        return (
            state.probe_started is not None
            and now - state.probe_started < self._policy.cooldown_seconds
        )

    def claimed(self, lane: LaneKey, now: float) -> None:
        """Note that a request claimed the lane, making it the probe when half-open.

        Two claim walks that read a half-open lane at the same moment may both
        order it normally and both dial it: two probes instead of one. That is
        harmless (the second probe is one more request on a lane that is
        either back, or dead and failing fast, and either outcome settles the
        lane the same way), so the walk does not serialize on it.

        Args:
            lane: The physical lane.
            now: The current monotonic time.
        """
        state = self._lanes.get(lane)
        if state is None or not state.ejected_until or now < state.ejected_until:
            return
        if (
            state.probe_started is None
            or now - state.probe_started >= self._policy.cooldown_seconds
        ):
            state.probe_started = now

    def released(self, lane: LaneKey) -> None:
        """Forget an in-flight probe whose claim never reached dispatch.

        Args:
            lane: The physical lane.
        """
        state = self._lanes.get(lane)
        if state is not None:
            state.probe_started = None

    def succeeded(self, lane: LaneKey) -> None:
        """Close the lane's streaks and ejection: the box answered.

        Args:
            lane: The physical lane.
        """
        self._lanes.pop(lane, None)

    def failed(self, lane: LaneKey, failure: GatewayFailure, now: float) -> None:
        """Apply one settled failure to the lane.

        A dead-lane failure or a header timeout extends its own streak, ejects
        at that streak's threshold, and re-ejects a half-open lane at once.
        Any other failure only frees a tracked lane's probe slot so the next
        request may probe; it never allocates state for an untracked lane.

        Args:
            lane: The physical lane.
            failure: The settled attempt's failure.
            now: The current monotonic time.
        """
        if self._policy.failures == 0:
            return
        dead = failure.failure_class in DEAD_LANE_FAILURES
        hung = self._policy.header_timeouts > 0 and is_header_timeout(failure)
        if not (dead or hung):
            self.released(lane)
            return
        state = self._lanes.setdefault(lane, _LaneState())
        probing = bool(state.ejected_until) and now >= state.ejected_until
        state.probe_started = None
        if probing:
            state.ejected_until = now + self._policy.cooldown_seconds
            return
        full = (
            _streak_full(state.failures, self._policy.failures, self._policy.window_seconds, now)
            if dead
            else _streak_full(
                state.header_timeouts,
                self._policy.header_timeouts,
                self._policy.header_window_seconds,
                now,
            )
        )
        if full:
            state.ejected_until = now + self._policy.cooldown_seconds
            state.failures.clear()
            state.header_timeouts.clear()
