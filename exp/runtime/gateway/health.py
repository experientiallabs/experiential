"""Bounded per-deployment circuit and throttle state for exact-model waterfalls.

The registry also owns the per-lane dead-box ejection (``lane_ejection``): the
same settled outcomes feed it, and claim walks read it to order an ejected
lane last (``order``).
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal

from exp.runtime.gateway.contracts import GatewayFailure, GatewayFailureClass
from exp.runtime.gateway.lane_ejection import LaneEjectionPolicy, LaneEjections, LaneKey

DeploymentHealthKey = tuple[str, str, str]

_HARD_FAILURES = {
    GatewayFailureClass.PROVIDER_AUTHENTICATION,
    GatewayFailureClass.PROVIDER_NOT_FOUND,
    # An unfunded provider account (402) is the same sticky operator-actionable
    # deadness as a bad credential: open the circuit immediately so the dead
    # rung stops being attempted first, and let the half-open probe rediscover
    # it once the operator funds or enables the account.
    GatewayFailureClass.PROVIDER_QUOTA,
}
_OPERATIONAL_FAILURES = {
    GatewayFailureClass.TRANSPORT,
    GatewayFailureClass.TIMEOUT,
    GatewayFailureClass.MALFORMED_RESPONSE,
    GatewayFailureClass.PROVIDER_INTERNAL,
}

# Clamp on a provider-stated Retry-After when it sizes a throttle window. The
# floor keeps a degenerate "0"/"1" from thrashing the rung; the ceiling keeps
# one absurd header from suppressing a lane for days while still letting a
# daily-quota reset (hours) mark the rung dead-for-hours so the waterfall
# skips it for the whole window.
RETRY_AFTER_WINDOW_MINIMUM_SECONDS = 5.0
RETRY_AFTER_WINDOW_MAXIMUM_SECONDS = 6.0 * 3_600.0
# A plan's usage window resets on the provider's own clock, and the longest one
# (the weekly window) can be days away. Suppression follows the stated reset up
# to one week plus a day of slack; the ceiling only guards a garbage reset.
PLAN_WINDOW_MAXIMUM_SECONDS = 8.0 * 24.0 * 3_600.0
# Spacing between paced probes through a hintless throttle window. A window the
# provider did not size (a 429 with no Retry-After) is the engine's guess, so a
# route with nothing else to claim may test it with at most one dispatch per
# interval per physical lane on this worker instead of refusing every caller for
# the whole default window. An explicit wait or a plan reset is never probed.
THROTTLE_PROBE_INTERVAL_SECONDS = 1.0


def health_failure_cause(
    failure_class: GatewayFailureClass,
) -> Literal["transport", "credential", "throttle"] | None:
    """Classify deployment-affecting failures for circuits and scoped recovery.

    Args:
        failure_class: Normalized provider or gateway failure category.

    Returns:
        Transport for operational failures, credential for operator-actionable
        account failures, throttle for explicit backoff, or None when the outcome
        says nothing about deployment health. Customer ownership does not change
        the provider failure's health effect.
    """
    if failure_class == GatewayFailureClass.THROTTLED:
        return "throttle"
    if failure_class in _HARD_FAILURES:
        return "credential"
    if failure_class in _OPERATIONAL_FAILURES:
        return "transport"
    return None


def _lane(key: DeploymentHealthKey) -> LaneKey:
    """The physical lane behind a health key: ``(deployment_id, connection_sha256)``."""
    return (key[1], key[2])


@dataclass
class _DeploymentHealth:
    """Mutable content-free health state protected by the registry lock.

    Attributes:
        consecutive_failures: Operational failures since the last success; a
            credential failure sets it straight to the threshold.
        open_until: Monotonic instant the open circuit admits a half-open probe.
        throttle_until: Monotonic instant the provider-requested backoff ends.
        half_open_probe: Whether the one half-open probe is in flight.
        last_resort_probe: Whether the one last-resort probe is in flight.
        refusal_count: Content refusals observed on this deployment.
        circuit_failure: The account failure (authentication, not found, quota)
            that holds the circuit open, reduced to its class, canonical message
            and ownership so it carries nothing from the request that observed
            it. ``None`` while the circuit is closed or held open by
            operational failures, and never recorded for a tenant origin, whose
            failures stay opaque to callers.
        newer_throttle_until: Monotonic end of the throttle windows observed after
            ``circuit_failure`` was recorded (zero when none); only these, never an
            older window, outrank the recorded account failure.
        stated_throttle_until: Monotonic end of the windows the provider sized
            itself (a parsed ``Retry-After`` or a plan reset). Only the part of
            ``throttle_until`` past this instant is the engine's own default
            guess, which a paced probe may test.
        probe_attempts: Attempts reserved inside a hintless window since it was
            last armed (``bind_throttle_probe``); only THEIR success lifts it,
            never an older request's.
    """

    consecutive_failures: int = 0
    open_until: float = 0.0
    throttle_until: float = 0.0
    half_open_probe: bool = False
    last_resort_probe: bool = False
    refusal_count: int = 0
    circuit_failure: GatewayFailure | None = None
    newer_throttle_until: float = 0.0
    stated_throttle_until: float = 0.0
    probe_attempts: set[str] = field(default_factory=set)


class DeploymentHealthRegistry:
    """Apply isolated circuits and throttle windows without changing logical models."""

    def __init__(
        self,
        *,
        failure_threshold: int = 2,
        open_seconds: float = 30.0,
        throttle_seconds: float = 30.0,
        lane_ejection: LaneEjectionPolicy | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Initialize finite in-process deployment health policy.

        Args:
            failure_threshold: Consecutive operational failures that open one circuit.
            open_seconds: Circuit cooldown before one half-open probe.
            throttle_seconds: Independent suppression window after provider throttling.
            lane_ejection: Per-lane dead-box ejection thresholds; ``None`` reads
                the operator's environment overrides over the defaults.
            clock: Injectable monotonic clock.

        Raises:
            ValueError: A threshold or duration is not positive.
        """
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be at least one")
        if open_seconds <= 0 or throttle_seconds <= 0:
            raise ValueError("health suppression windows must be positive")
        self._failure_threshold = failure_threshold
        self._open_seconds = open_seconds
        self._throttle_seconds = throttle_seconds
        self._clock = clock
        self._states: dict[DeploymentHealthKey, _DeploymentHealth] = {}
        self._lanes = LaneEjections(lane_ejection or LaneEjectionPolicy.from_environ())
        # Probe pacing is per physical lane, so two catalog revisions sharing a
        # lane during a rollout never multiply the probe rate.
        self._probe_paced_at: dict[LaneKey, float] = {}
        self._lock = threading.Lock()

    def order(self, keys: tuple[DeploymentHealthKey, ...], candidates: Sequence[int]) -> list[int]:
        """Return ``candidates`` with every ejected lane moved last, order otherwise kept.

        Args:
            keys: One health key per ordered route deployment.
            candidates: Route indexes in the walk's preference order.

        Returns:
            The same indexes, live lanes first, ejected lanes after them.
        """
        now = self._clock()
        with self._lock:
            ejected = {
                index for index in candidates if self._lanes.ejected(_lane(keys[index]), now)
            }
        return [index for index in candidates if index not in ejected] + [
            index for index in candidates if index in ejected
        ]

    def lane_ejected(self, key: DeploymentHealthKey) -> bool:
        """Whether the lane behind ``key`` is ejected on this worker right now."""
        now = self._clock()
        with self._lock:
            return self._lanes.ejected(_lane(key), now)

    def claim(self, key: DeploymentHealthKey) -> bool:
        """Reserve one eligible route or half-open probe.

        Args:
            key: Catalog, deployment, and connection identity tuple.

        Returns:
            Whether the caller may dispatch this deployment now.
        """
        now = self._clock()
        with self._lock:
            state = self._states.setdefault(key, _DeploymentHealth())
            if state.throttle_until > now:
                return False
            if state.open_until > now:
                return False
            if state.consecutive_failures < self._failure_threshold:
                self._lanes.claimed(_lane(key), now)
                return True
            if state.half_open_probe:
                return False
            state.half_open_probe = True
            self._lanes.claimed(_lane(key), now)
            return True

    def claim_last_resort(self, key: DeploymentHealthKey) -> bool:
        """Reserve one bounded probe through an open circuit when nothing else is eligible.

        A request with no claimable deployment would otherwise fail for the whole
        cooldown even after the provider has recovered. One in-flight probe per
        deployment is allowed through an open circuit so recovery is discovered by
        real traffic instead of a fixed timer. Throttle windows stay authoritative
        because the provider explicitly asked for backoff.

        Args:
            key: Catalog, deployment, and connection identity tuple.

        Returns:
            Whether the caller may dispatch this suppressed deployment now.
        """
        now = self._clock()
        with self._lock:
            state = self._states.setdefault(key, _DeploymentHealth())
            if state.throttle_until > now:
                return False
            if state.last_resort_probe:
                return False
            state.last_resort_probe = True
            self._lanes.claimed(_lane(key), now)
            return True

    def claim_forced(self, key: DeploymentHealthKey) -> bool:
        """Admit one dispatch through an open circuit when a request would otherwise fail.

        Circuit suppression is load shedding, not correctness: refusing the last
        eligible route turns a possibly-down deployment into a certain caller
        failure. When every healthy claim and bounded probe is unavailable, each
        request whose alternatives are exhausted may still dispatch through an
        open circuit, so concurrent failures never strand eligible traffic behind
        the single half-open or last-resort probe. Throttle windows stay
        authoritative because the provider explicitly asked for backoff.

        Args:
            key: Catalog, deployment, and connection identity tuple.

        Returns:
            Whether the caller may dispatch this suppressed deployment now.
        """
        now = self._clock()
        with self._lock:
            state = self._states.setdefault(key, _DeploymentHealth())
            if state.throttle_until > now:
                return False
            self._lanes.claimed(_lane(key), now)
            return True

    def claim_throttle_redial(self, key: DeploymentHealthKey) -> bool:
        """Admit one post-backoff redial of a rung inside its own throttle window.

        The throttle window is the fleet-shared "leave this rung alone"
        signal for OTHER requests; the request that was throttled and has
        already waited the pool's backoff is the one deliberately probing the
        rung back, so its redial passes the window. An open circuit (some
        other failure class marked the rung dead meanwhile) still refuses,
        because a dead rung has no cache worth the wait.

        Args:
            key: Catalog, deployment, and connection identity tuple.

        Returns:
            Whether the redial may dispatch this deployment now.
        """
        now = self._clock()
        with self._lock:
            state = self._states.setdefault(key, _DeploymentHealth())
            return state.open_until <= now

    def claim_throttle_probe(self, key: DeploymentHealthKey) -> bool:
        """Admit one paced dispatch through a hintless throttle window.

        A 429 that names no wait arms the engine's default window, and on a
        route with nothing else to claim that guess refuses every caller for
        the whole window even while the provider is serving. Such a window is tested
        instead: at most one dispatch per ``THROTTLE_PROBE_INTERVAL_SECONDS``
        per physical lane on this worker, so concurrent throttles never stampede
        the provider, and the probe's own success lifts the window
        (:meth:`bind_throttle_probe`, :meth:`succeeded`). A window the provider
        sized itself (Retry-After, a plan reset) is never probed: the provider
        asked for that wait.

        Only a reservation with no failure of its own and every claimable depth
        inside a throttle window reaches this pass (``claim_route_from`` with
        ``throttle_probe=True``), so a route with an unthrottled sibling still
        moves its traffic there.

        Args:
            key: Catalog, deployment, and connection identity tuple.

        Returns:
            Whether the caller may dispatch this throttled deployment now.
        """
        now = self._clock()
        with self._lock:
            state = self._states.setdefault(key, _DeploymentHealth())
            if state.throttle_until <= now or state.stated_throttle_until > now:
                return False
            last = self._probe_paced_at.get(_lane(key))
            if last is not None and now - last < THROTTLE_PROBE_INTERVAL_SECONDS:
                return False
            self._probe_paced_at[_lane(key)] = now
            self._lanes.claimed(_lane(key), now)
            return True

    def bind_throttle_probe(
        self, key: DeploymentHealthKey, attempt_id: str, *, probe: bool = True
    ) -> None:
        """Record a no-failure dial's reserved attempt as a probe of its window.

        Ownership is decided at reservation, by the claimant, from the window
        as it stands then: an attempt reserved while the deployment sits in a
        hintless window can only have passed it as a probe, so its own success
        lifts the window. Nothing is shared between claim and reservation, so
        an unrelated release cannot drop the probe, a probe shed and then
        force-admitted still counts, and a 429 armed before this reservation
        is already behind it. A 429 armed after it clears it (:meth:`failed`).

        Args:
            key: Catalog, deployment, and connection identity tuple.
            attempt_id: The reserved attempt.
            probe: Whether the reservation came from a dial with no failure of
                its own; any other (a throttle redial) is never a probe here.
        """
        if not probe:
            return
        now = self._clock()
        with self._lock:
            state = self._states.get(key)
            if state is None or state.throttle_until <= now or state.stated_throttle_until > now:
                return
            state.probe_attempts.add(attempt_id)

    def dispatch_opened(self, key: DeploymentHealthKey, attempt_id: str | None = None) -> None:
        """Restore admission once one provider dispatch opens successfully.

        An opened stream proves the deployment is reachable again, so waiting for
        the terminal event would refuse concurrent traffic for the full stream
        duration. A later terminal failure re-applies suppression normally.

        Args:
            key: Catalog, deployment, and connection identity tuple.
            attempt_id: The opened attempt; a bound throttle probe lifts its
                hintless window.
        """
        with self._lock:
            state = self._states.setdefault(key, _DeploymentHealth())
            state.consecutive_failures = 0
            state.open_until = 0.0
            state.half_open_probe = False
            state.last_resort_probe = False
            state.circuit_failure = None
            self._lift_probed_throttle(state, attempt_id)
            self._lanes.succeeded(_lane(key))

    def succeeded(self, key: DeploymentHealthKey, attempt_id: str | None = None) -> None:
        """Close one circuit after a successful terminal provider result.

        Args:
            key: Catalog, deployment, and connection identity tuple.
            attempt_id: The settled attempt; a bound throttle probe lifts its
                hintless window.
        """
        with self._lock:
            state = self._states.setdefault(key, _DeploymentHealth())
            state.consecutive_failures = 0
            state.open_until = 0.0
            state.half_open_probe = False
            state.last_resort_probe = False
            state.circuit_failure = None
            self._lift_probed_throttle(state, attempt_id)
            self._lanes.succeeded(_lane(key))

    def failed(
        self, key: DeploymentHealthKey, failure: GatewayFailure, attempt_id: str | None = None
    ) -> None:
        """Apply one normalized provider outcome to circuit or throttle state.

        Args:
            key: Catalog, deployment, and connection identity tuple.
            failure: Sanitized provider failure classification.
            attempt_id: The settled attempt, retired from the bound probes.
        """
        now = self._clock()
        cause = health_failure_cause(failure.failure_class)
        with self._lock:
            state = self._states.setdefault(key, _DeploymentHealth())
            if attempt_id is not None:
                state.probe_attempts.discard(attempt_id)
            state.half_open_probe = False
            state.last_resort_probe = False
            self._lanes.failed(_lane(key), failure, now)
            if cause == "throttle":
                window_end = now + self._throttle_window_seconds(failure)
                state.throttle_until = max(state.throttle_until, window_end)
                state.newer_throttle_until = max(state.newer_throttle_until, window_end)
                # A newer 429 outranks every probe dispatched before it.
                state.probe_attempts.clear()
                if failure.retry_after_seconds is not None:
                    state.stated_throttle_until = max(state.stated_throttle_until, window_end)
                return
            if failure.failure_class == GatewayFailureClass.REFUSAL:
                state.refusal_count += 1
                return
            if cause == "credential":
                state.consecutive_failures = self._failure_threshold
                state.open_until = now + self._open_seconds
                state.circuit_failure = (
                    None
                    if failure.tenant_origin
                    else GatewayFailure(
                        failure_class=failure.failure_class,
                        safe_message=failure.safe_message,
                        customer_owned=failure.customer_owned,
                    )
                )
                state.newer_throttle_until = 0.0
                return
            if cause == "transport":
                state.consecutive_failures += 1
                if state.consecutive_failures >= self._failure_threshold:
                    state.open_until = now + self._open_seconds
                    state.circuit_failure = None

    def exhausted(self, key: DeploymentHealthKey, reset_after_seconds: float) -> None:
        """Suppress one deployment until its provider-stated usage window resets.

        A plan rung reports its rolling usage windows on every response, so a
        window that reaches 100 percent is known BEFORE the next request would
        429: the rung is throttled until the reset the provider stated, however
        many days away (a weekly window), and the waterfall moves on to the next
        plan in the pool. A success on the same response leaves the circuit closed.

        Args:
            key: Catalog, deployment, and connection identity tuple.
            reset_after_seconds: Provider-stated seconds until the window reopens.
        """
        now = self._clock()
        window = min(
            max(float(reset_after_seconds), RETRY_AFTER_WINDOW_MINIMUM_SECONDS),
            PLAN_WINDOW_MAXIMUM_SECONDS,
        )
        with self._lock:
            state = self._states.setdefault(key, _DeploymentHealth())
            state.throttle_until = max(state.throttle_until, now + window)
            state.newer_throttle_until = max(state.newer_throttle_until, now + window)
            state.stated_throttle_until = max(state.stated_throttle_until, now + window)

    def _lift_probed_throttle(self, state: _DeploymentHealth, attempt_id: str | None) -> None:
        """Lift a hintless throttle window once the provider served the probe itself.

        Only the success of a probe attempt bound since the window was armed
        lifts it, so an older request finishing, a route that never probed
        (one with an unthrottled sibling), or a probe shed before dispatch
        keeps the suppression exactly as before, and a window the provider
        sized itself stays until its stated end. Caller holds the registry lock.

        Args:
            state: The deployment's mutable health state.
            attempt_id: The attempt that succeeded, or ``None`` when unknown.
        """
        if attempt_id is None or attempt_id not in state.probe_attempts:
            return
        state.probe_attempts.discard(attempt_id)
        now = self._clock()
        if state.throttle_until <= now or state.stated_throttle_until > now:
            return
        state.throttle_until = state.stated_throttle_until
        state.newer_throttle_until = min(state.newer_throttle_until, state.stated_throttle_until)

    def _throttle_window_seconds(self, failure: GatewayFailure) -> float:
        """Size one throttle window from the provider's own stated wait.

        A throttled failure carrying a parsed ``Retry-After`` sizes the window
        from it, clamped to the bounded range, so a long provider backoff (an
        exhausted daily quota) actually suppresses the rung for the wait the
        provider asked for instead of re-attempting every fixed default. An
        absent or unparseable wait keeps the fixed default window.

        Args:
            failure: Sanitized throttled failure.

        Returns:
            The window length in seconds.
        """
        if failure.retry_after_seconds is None:
            return self._throttle_seconds
        return min(
            max(float(failure.retry_after_seconds), RETRY_AFTER_WINDOW_MINIMUM_SECONDS),
            RETRY_AFTER_WINDOW_MAXIMUM_SECONDS,
        )

    def suppressed(self, key: DeploymentHealthKey) -> bool:
        """Whether one deployment currently sits inside any suppression window.

        A read-only probe (unlike :meth:`claim` it never reserves a half-open
        slot), used by sticky-affinity admission to bypass and clear a binding
        whose rung is throttled or circuit-open right now.

        Args:
            key: Catalog, deployment, and connection identity tuple.

        Returns:
            Whether the deployment is throttle- or circuit-suppressed.
        """
        now = self._clock()
        with self._lock:
            state = self._states.get(key)
            if state is None:
                return False
            return state.throttle_until > now or state.open_until > now

    def throttled_remaining_seconds(
        self, keys: tuple[DeploymentHealthKey, ...], depths: Sequence[int] | None = None
    ) -> float | None:
        """Return the longest remaining throttle window when EVERY claimable key is inside one.

        A route with no claimable deployment and no classified failure of its
        own can only be throttle-suppressed (forced claims admit any
        non-throttled circuit), so this names that condition explicitly: the
        provider asked for backoff, which is caller-facing rate limiting, not
        platform deadness.

        Args:
            keys: One health key per ordered route deployment.
            depths: The route indexes this reservation could claim; ``None``
                judges every deployment. A depth the reservation may not dial
                says nothing about why it was refused.

        Returns:
            The longest remaining window in seconds, or ``None`` when no
            depth is judged or any judged deployment is outside a throttle window.
        """
        judged = range(len(keys)) if depths is None else depths
        if not judged:
            return None
        now = self._clock()
        remaining = 0.0
        with self._lock:
            for depth in judged:
                state = self._states.get(keys[depth])
                if state is None or state.throttle_until <= now:
                    return None
                remaining = max(remaining, state.throttle_until - now)
        return remaining

    def circuit_failure(
        self, keys: tuple[DeploymentHealthKey, ...], depths: Sequence[int]
    ) -> GatewayFailure | None:
        """Return the account failure holding EVERY claimable depth's circuit open.

        A dial that may not force an open circuit (a caller-selected route's
        first dial) ends with no attempt while its rung's provider account is
        refusing. That answer is the account's own failure, the same class and
        message a dispatched attempt on the rung gets, rather than a generic
        outage of every deployment.

        Args:
            keys: One health key per ordered route deployment.
            depths: The route indexes this reservation could claim.

        Returns:
            The first depth's recorded account failure when every depth sits in
            a circuit an account failure opened that refuses a claim right now
            (inside its cooldown, or its half-open probe in flight) and no newer
            throttle window is still running there, else ``None``.
        """
        # ``circuit_failure`` is set only with the circuit at its threshold and
        # cleared whenever the count resets, so it alone marks an account-opened circuit.
        first: GatewayFailure | None = None
        now = self._clock()
        with self._lock:
            for depth in depths:
                state = self._states.get(keys[depth])
                if (
                    state is None
                    or state.circuit_failure is None
                    or (state.open_until <= now and not state.half_open_probe)
                    or state.newer_throttle_until > now
                ):
                    return None
                first = first or state.circuit_failure
        return first

    def release_probe(self, key: DeploymentHealthKey, attempt_id: str | None = None) -> None:
        """Release a claimed half-open probe that never reached provider dispatch.

        Args:
            key: Catalog, deployment, and connection identity tuple.
            attempt_id: The retired attempt, dropped from the bound throttle probes.
        """
        with self._lock:
            state = self._states.get(key)
            if state is not None:
                state.half_open_probe = False
                state.last_resort_probe = False
                if attempt_id is not None:
                    state.probe_attempts.discard(attempt_id)
            self._lanes.released(_lane(key))


def claim_route_from(
    health: DeploymentHealthRegistry,
    keys: tuple[DeploymentHealthKey, ...],
    start: int,
    depths: Sequence[int] | None = None,
    *,
    throttle_probe: bool = False,
) -> int | None:
    """Claim the first healthy later route, a bounded probe, or a forced dispatch.

    A request skipping an exhausted or failed route can still probe a
    suppressed fallback instead of failing
    for the whole circuit cooldown after the provider has recovered. When
    every healthy claim and bounded probe is unavailable, the first
    non-throttled route is dispatched anyway, subject only to the request
    deadline and to throttle windows the provider explicitly requested.
    A lane this worker ejected as a dead box (``lane_ejection``) is walked
    last in every pass, never dropped: a route with nothing else still dials it.

    Args:
        health: Revision-isolated circuit and throttle registry.
        keys: One health key per ordered route deployment.
        start: First route index eligible for this claim.
        depths: The route indexes this dial may claim at all (the
            ``failover_only_on`` eligibility, ``native_fallback_rules``);
            ``None`` admits every index. Indexes below ``start`` are skipped.
        throttle_probe: Whether a dial that finds every candidate inside a
            throttle window may take one paced probe through a hintless one
            (``DeploymentHealthRegistry.claim_throttle_probe``) instead of
            refusing. Only a reservation with no failure of its own asks (a
            first dial, a tool-search round, a reasoning repair); a failover
            pass already holds the provider's answer.

    Returns:
        The claimed route index, or ``None`` when nothing is claimable.
    """
    candidates = health.order(
        keys,
        [index for index in (range(len(keys)) if depths is None else depths) if index >= start],
    )
    claims = (health.claim, health.claim_last_resort, health.claim_forced)
    if throttle_probe:
        claims = (*claims, health.claim_throttle_probe)
    for claim in claims:
        for route_index in candidates:
            if claim(keys[route_index]):
                return route_index
    return None
