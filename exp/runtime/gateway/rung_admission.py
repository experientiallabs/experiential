"""In-process bounded, rate-limited, weighted fair-share admission per rung.

One registry per worker tracks in-flight dispatches on rungs that author a
``GatewayRungDispatchPolicy``: a rung at its per-worker ``concurrency_bound``
sheds new dispatches down the waterfall (spill in seconds, never a queue that
dies at the request deadline), a rung past its sliding-window request or
token rate sheds the same way BEFORE the provider answers 429, and a
``fair_share`` rung additionally bounds each organization's admissions by its
weighted max-min share while the rung is contended. Every decision is
lock-guarded in-memory arithmetic over counters this registry already holds:
no database read, no shared state, no waiting.

Rate calibration is passive-adaptive: a provider throttle clamps the rung's
working request rate to ninety percent of the rate actually observed at that
moment, and the working ceiling then creeps back up by five percent every
recovery interval without a throttle (each creep step IS the probe: a few
more requests go through to see whether they make it further this time). A
learned ceiling that sees no throttle for the expiry horizon is forgotten.
There are never synthetic probe requests, only real traffic let through.

Fairness is deliberately conservative because there is no queue and no
preemption. Capacity below the bound is always borrowable (work-conserving: a
lone organization uses the whole rung), and an organization above its weighted
share is shed only when admitting it would eat capacity currently reserved for
another RECENTLY ACTIVE under-share organization. A shed marks its organization
active, so a flooded-out organization accrues a reservation and converges to
its share as slots free, without ever pausing a running request. When a rung
authors ``cache_priority_alpha``, each organization's effective weight scales
with the rung's congestion and the worker's EWMA of that organization's
settled cached-token fraction, so at the contended margin cache-heavy traffic
is admitted ahead of cold traffic.
"""

from __future__ import annotations

import math
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

# Physical lane identity: the deployment and its credentialed connection.
# Deliberately NOT revision-scoped (unlike health keys): a catalog write does
# not change the box's capacity, so counters must survive alias revisions.
RungLoadKey = tuple[str, str]

RungShedReason = Literal["fair_share_shed", "queue_bound", "rate_limit", "fresh_session_spill"]

# How long a request (admitted or shed) keeps its organization "active" for
# share accounting. Long enough that a flooded-out caller's retry cadence keeps
# its reservation alive; short enough that a departed caller's share is
# borrowable again almost immediately.
ACTIVITY_WINDOW_SECONDS = 10.0

# The sliding rate window. Both authored rates are per minute, so the window
# is fixed rather than configurable.
RATE_WINDOW_SECONDS = 60.0

# Passive-adaptive calibration constants. A throttle clamps the learned
# request ceiling to the observed rate times the clamp factor; every recovery
# interval without a throttle creeps it up by the creep fraction (at least one
# request); a ceiling unthrottled for the expiry horizon is forgotten.
LEARNED_CLAMP_FACTOR = 0.9
LEARNED_CREEP_FRACTION = 0.05
LEARNED_RECOVERY_INTERVAL_SECONDS = 60.0
LEARNED_EXPIRY_SECONDS = 6.0 * 3_600.0

# Cached-fraction EWMA: half-life of roughly ten minutes of activity, with the
# per-sample decay floored at one second of elapsed time so a burst of settles
# still moves the estimate. An organization's estimate is retained for the
# retention horizon after its last sample so a conversational cadence (minutes
# between turns) keeps its cache standing.
EWMA_HALF_LIFE_SECONDS = 600.0
EWMA_MINIMUM_STEP_SECONDS = 1.0
EWMA_RETENTION_SECONDS = 3_600.0


def _effective_weight(
    load: _OrganizationLoad,
    *,
    alpha: float | None,
    congestion: float,
) -> float:
    """Return one organization's admission weight with the cache-priority term.

    ``weight * (1 + alpha * congestion * cached_fraction)``: the boost is zero
    for an organization with no cache signal, zero on an idle rung, and grows
    with both the rung's congestion and how much of the organization's settled
    input the provider served from cache. ``alpha`` off returns the base weight
    so authored fairness without the term is byte-identical to before.

    Args:
        load: The organization's per-rung state, under the registry lock.
        alpha: The rung's authored ``cache_priority_alpha``, if any.
        congestion: The rung's in-flight total over its bound.

    Returns:
        The effective (float) weight used by share arithmetic.
    """
    if alpha is None or alpha <= 0.0:
        return float(load.weight)
    return load.weight * (1.0 + alpha * congestion * load.cached_fraction)


@dataclass(frozen=True)
class RungShed:
    """One refused reservation and the disclosure reason for the bypass."""

    reason: RungShedReason
    learned_requests_per_minute: int | None = None
    """The learned working request ceiling behind a ``rate_limit`` shed.

    Carried so the shed can be logged and counted with the ceiling that caused
    it; the durable disclosure column stays the bare reason code.
    """


@dataclass
class _OrganizationLoad:
    """Per-organization in-flight count, recency, and cache EWMA on one rung."""

    inflight: int = 0
    last_seen: float = 0.0
    weight: int = 1
    # Time-decayed EWMA of the organization's settled cached-token fraction on
    # this rung, and the monotonic time of its last sample (None = no signal;
    # new organizations weigh at their base weight).
    cached_fraction: float = 0.0
    cached_sampled_at: float | None = None


@dataclass
class _RungLoad:
    """Aggregate, per-organization, and rate-window state for one rung."""

    total: int = 0
    organizations: dict[str, _OrganizationLoad] = field(default_factory=dict)
    # Sliding 60s dispatch window: (reserved_at, reserved_tokens) per admitted
    # reservation, with running totals so every decision is O(1) amortized.
    window: deque[tuple[float, int]] = field(default_factory=deque)
    window_requests: int = 0
    window_tokens: int = 0
    # Passive-adaptive learned request ceiling (None = nothing learned).
    learned_rpm: float | None = None
    learned_throttled_at: float = 0.0
    learned_crept_at: float = 0.0


class RungLoadRegistry:
    """Reservation ledger for bounded and fair-share rung admission.

    A successful reservation returns an opaque ticket the caller either binds
    to the durable attempt id (released later by ``release_attempt``) or
    releases directly when the dispatch never happened. Both release paths are
    idempotent so settle, abandon, and the sweep can all safely fire.
    """

    def __init__(
        self,
        *,
        activity_window_seconds: float = ACTIVITY_WINDOW_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Initialize empty counters with an injectable clock for tests.

        Args:
            activity_window_seconds: Recency horizon for share reservations.
            clock: Monotonic clock.

        Raises:
            ValueError: The activity window is not positive.
        """
        if activity_window_seconds <= 0:
            raise ValueError("activity window must be positive")
        self._window = activity_window_seconds
        self._clock = clock
        self._rungs: dict[RungLoadKey, _RungLoad] = {}
        self._tickets: dict[str, tuple[RungLoadKey, str]] = {}
        self._attempts: dict[str, str] = {}
        self._lock = threading.Lock()

    def reserve(
        self,
        key: RungLoadKey,
        *,
        organization_id: str,
        weight: int,
        bound: int | None,
        fair_share: bool,
        requests_per_minute: int | None = None,
        tokens_per_minute: int | None = None,
        cache_priority_alpha: float | None = None,
        reserved_tokens: int = 0,
        warm_session: bool = True,
        fresh_spill_fraction: float | None = None,
        force: bool = False,
    ) -> str | RungShed:
        """Reserve one slot on a policy-bounded rung, or shed with a reason.

        Args:
            key: Physical rung identity.
            organization_id: Authorized organization for share accounting.
            weight: The organization's fair-share weight (>= 1).
            bound: The rung's authored per-worker in-flight cap, or ``None``
                when only rate windows are authored.
            fair_share: Whether contended admission is weighted max-min fair.
            requests_per_minute: Authored per-worker dispatch rate cap.
            tokens_per_minute: Authored per-worker token rate cap.
            cache_priority_alpha: Congestion multiplier for the cache-priority
                fairness term; ``None`` leaves base weights.
            reserved_tokens: Worst-case tokens this dispatch reserves, counted
                against the token window.
            warm_session: Whether the request's affinity fingerprint holds a
                live sticky binding on THIS rung (its provider cache is warm
                here); fresh sessions shed at the early threshold.
            fresh_spill_fraction: Fraction of the bound where fresh sessions
                shed early; ``None`` disables the early threshold.
            force: Admit past every policy limit (the caller proved no other
                rung can serve; policy must never manufacture a failure).

        Returns:
            An opaque ticket on admission, else the shed disclosure.
        """
        now = self._clock()
        with self._lock:
            rung = self._rungs.setdefault(key, _RungLoad())
            organization = rung.organizations.setdefault(organization_id, _OrganizationLoad())
            # A shed still marks demand: the flooded-out organization's share
            # is reserved as slots free, which is what converges to fairness.
            organization.last_seen = now
            organization.weight = weight
            self._prune(key, rung, now)
            self._prune_window(rung, now)
            if not force:
                shed = self._shed_reason(
                    rung,
                    organization,
                    now=now,
                    bound=bound,
                    fair_share=fair_share,
                    requests_per_minute=requests_per_minute,
                    tokens_per_minute=tokens_per_minute,
                    cache_priority_alpha=cache_priority_alpha,
                    reserved_tokens=reserved_tokens,
                    warm_session=warm_session,
                    fresh_spill_fraction=fresh_spill_fraction,
                )
                if shed is not None:
                    return shed
            organization.inflight += 1
            rung.total += 1
            # Every policy reservation feeds the dispatch window, not only
            # rate-authored ones: a bound-only rung that later throttles must
            # clamp its learned ceiling to a REAL observed rate, never to an
            # empty window's floor of one.
            rung.window.append((now, reserved_tokens))
            rung.window_requests += 1
            rung.window_tokens += reserved_tokens
            ticket = f"rung-{uuid.uuid4().hex}"
            self._tickets[ticket] = (key, organization_id)
            return ticket

    def _shed_reason(
        self,
        rung: _RungLoad,
        organization: _OrganizationLoad,
        *,
        now: float,
        bound: int | None,
        fair_share: bool,
        requests_per_minute: int | None,
        tokens_per_minute: int | None,
        cache_priority_alpha: float | None,
        reserved_tokens: int,
        warm_session: bool,
        fresh_spill_fraction: float | None,
    ) -> RungShed | None:
        """Decide one reservation under the registry lock; ``None`` admits.

        The bound is hard: at or beyond it every arrival spills, which is the
        queue-death fix. Fresh sessions (no warm sticky standing on this rung)
        spill earlier, at ``bound * fresh_spill_fraction``, reserving the top
        slice of the bound for sessions whose provider cache lives here. The
        rate check sheds a dispatch the sliding window cannot absorb under the
        working ceiling (the authored rate clamped by the learned one) BEFORE
        the provider answers 429. Below all of those, fairness sheds an
        over-share organization only when the remaining slots are reserved for
        other recently active under-share organizations; otherwise unused
        capacity is borrowable. Shares stay EXACT (a 3:1:1 weighting of a
        bound of 8 guarantees 4.8:1.6:1.6, never a per-share rounding), and
        only the AGGREGATE reservation is floored to whole slots: slots are
        indivisible, so the sub-slot remainder of the summed deficits is
        capacity no organization could occupy right now, and reserving it
        would strand the bound's last slots against sustained demand (three
        equal organizations on a bound of 8 would otherwise freeze at 6).
        With ``cache_priority_alpha`` authored, every share reads EFFECTIVE
        weights ``weight * (1 + alpha * congestion * cached_fraction)``, so the
        boost is zero on an idle rung and strongest exactly at saturation.
        """
        if bound is not None:
            if rung.total >= bound:
                return RungShed("queue_bound")
            if (
                fresh_spill_fraction is not None
                and not warm_session
                and rung.total >= bound * fresh_spill_fraction
            ):
                return RungShed("fresh_session_spill")
        working_rpm = self._working_rpm(rung, requests_per_minute, now)
        if working_rpm is not None and rung.window_requests + 1 > working_rpm:
            learned = None if rung.learned_rpm is None else int(rung.learned_rpm)
            return RungShed("rate_limit", learned_requests_per_minute=learned)
        if tokens_per_minute is not None and rung.window_tokens + reserved_tokens > (
            tokens_per_minute
        ):
            learned = None if rung.learned_rpm is None else int(rung.learned_rpm)
            return RungShed("rate_limit", learned_requests_per_minute=learned)
        if not fair_share or bound is None:
            return None
        congestion = rung.total / bound
        recency_floor = organization.last_seen - self._window
        active = [
            candidate
            for candidate in rung.organizations.values()
            if candidate.inflight > 0 or candidate.last_seen >= recency_floor
        ]
        total_weight = sum(
            _effective_weight(candidate, alpha=cache_priority_alpha, congestion=congestion)
            for candidate in active
        )
        share = (
            bound
            * _effective_weight(organization, alpha=cache_priority_alpha, congestion=congestion)
            / total_weight
        )
        if organization.inflight + 1 <= share:
            return None
        reserved_deficit = sum(
            max(
                0.0,
                bound
                * _effective_weight(candidate, alpha=cache_priority_alpha, congestion=congestion)
                / total_weight
                - candidate.inflight,
            )
            for candidate in active
            if candidate is not organization
        )
        if rung.total + 1 + int(reserved_deficit) > bound:
            return RungShed("fair_share_shed")
        return None

    def _working_rpm(self, rung: _RungLoad, authored: int | None, now: float) -> float | None:
        """Return the rung's working request ceiling, advancing calibration.

        The working ceiling is the authored rate clamped by the learned one.
        Lazily applied here (no timer thread): an expired learned ceiling (no
        throttle for the expiry horizon) is forgotten, and every elapsed
        recovery interval since the last creep raises the learned ceiling by
        the creep fraction (at least one request), capped at the authored rate
        when one is authored and unbounded otherwise, because each creep step
        is exactly "send a few more and see whether they make it further".

        Args:
            rung: The rung's mutable state, under the registry lock.
            authored: The authored per-worker requests-per-minute, if any.
            now: Monotonic decision time.

        Returns:
            The working ceiling, or ``None`` when nothing limits requests.
        """
        learned = rung.learned_rpm
        if learned is not None:
            if now - rung.learned_throttled_at >= LEARNED_EXPIRY_SECONDS:
                learned = None
            else:
                steps = int((now - rung.learned_crept_at) // LEARNED_RECOVERY_INTERVAL_SECONDS)
                for _ in range(steps):
                    learned += max(1.0, float(math.ceil(LEARNED_CREEP_FRACTION * learned)))
                    if authored is not None and learned >= authored:
                        learned = float(authored)
                        break
                if steps:
                    rung.learned_crept_at += steps * LEARNED_RECOVERY_INTERVAL_SECONDS
            rung.learned_rpm = learned
        if learned is None:
            return None if authored is None else float(authored)
        if authored is None:
            return learned
        return min(float(authored), learned)

    def record_throttle(self, key: RungLoadKey) -> None:
        """Clamp one rung's learned request ceiling after a provider throttle.

        The ceiling becomes the dispatch rate actually observed in the sliding
        window at this moment times the clamp factor (floored at one request
        per minute): the provider just proved the observed rate is too high,
        so the worker assumes it throttles there again and lets recovery creep
        re-discover the real headroom.

        Args:
            key: Physical rung identity.
        """
        now = self._clock()
        with self._lock:
            rung = self._rungs.setdefault(key, _RungLoad())
            self._prune_window(rung, now)
            observed = max(1, rung.window_requests)
            rung.learned_rpm = max(1.0, observed * LEARNED_CLAMP_FACTOR)
            rung.learned_throttled_at = now
            rung.learned_crept_at = now

    def record_settle(
        self,
        key: RungLoadKey,
        organization_id: str,
        *,
        cached_tokens: int,
        input_tokens: int,
    ) -> None:
        """Fold one settled attempt's cached-token fraction into the org EWMA.

        Args:
            key: Physical rung identity.
            organization_id: The settling organization.
            cached_tokens: Provider-reported cached input tokens.
            input_tokens: Provider-reported (uncached) input tokens.
        """
        denominator = cached_tokens + input_tokens
        if denominator <= 0:
            return
        sample = cached_tokens / denominator
        now = self._clock()
        with self._lock:
            # A long stream can settle after its organization's request-recency
            # entry was pruned; the sample still counts, so the entry is
            # recreated (inactive for fairness until its next reservation).
            rung = self._rungs.setdefault(key, _RungLoad())
            organization = rung.organizations.setdefault(organization_id, _OrganizationLoad())
            if organization.cached_sampled_at is None:
                organization.cached_fraction = sample
            else:
                elapsed = max(EWMA_MINIMUM_STEP_SECONDS, now - organization.cached_sampled_at)
                retained = 0.5 ** (elapsed / EWMA_HALF_LIFE_SECONDS)
                organization.cached_fraction = organization.cached_fraction * retained + sample * (
                    1.0 - retained
                )
            organization.cached_sampled_at = now

    def learned_ceilings(self) -> dict[str, int]:
        """Return live learned request ceilings keyed by rung, for metrics.

        Keys are ``deployment_id:connection-prefix`` (catalog identifiers,
        content-free); only rungs currently holding a learned ceiling appear,
        so the map stays as small as the set of recently throttled rungs.
        """
        with self._lock:
            return {
                f"{deployment_id}:{connection[:8]}": int(rung.learned_rpm)
                for (deployment_id, connection), rung in self._rungs.items()
                if rung.learned_rpm is not None
            }

    def bind(self, ticket: str, attempt_id: str) -> None:
        """Attach one reservation to its durable attempt for settle release.

        Args:
            ticket: Reservation returned by :meth:`reserve`.
            attempt_id: The durably reserved attempt now holding the slot.
        """
        with self._lock:
            if ticket in self._tickets:
                self._attempts[attempt_id] = ticket

    def release_ticket(self, ticket: str) -> None:
        """Release one reservation that never dispatched; idempotent.

        Args:
            ticket: Reservation returned by :meth:`reserve`.
        """
        with self._lock:
            self._release_locked(ticket)

    def release_attempt(self, attempt_id: str) -> None:
        """Release the reservation held by one settled attempt; idempotent.

        Args:
            attempt_id: The settled or abandoned attempt.
        """
        with self._lock:
            ticket = self._attempts.pop(attempt_id, None)
            if ticket is not None:
                self._release_locked(ticket)

    def inflight(self, key: RungLoadKey, organization_id: str | None = None) -> int:
        """Return one rung's total or per-organization in-flight count.

        Args:
            key: Physical rung identity.
            organization_id: Scope the count to one organization when given.
        """
        with self._lock:
            rung = self._rungs.get(key)
            if rung is None:
                return 0
            if organization_id is None:
                return rung.total
            organization = rung.organizations.get(organization_id)
            return 0 if organization is None else organization.inflight

    def _release_locked(self, ticket: str) -> None:
        """Return one reserved slot to its rung under the registry lock."""
        entry = self._tickets.pop(ticket, None)
        if entry is None:
            return
        key, organization_id = entry
        rung = self._rungs.get(key)
        if rung is None:
            return
        organization = rung.organizations.get(organization_id)
        if organization is not None and organization.inflight > 0:
            organization.inflight -= 1
        if rung.total > 0:
            rung.total -= 1

    def _prune(self, key: RungLoadKey, rung: _RungLoad, now: float) -> None:
        """Drop idle organizations past the activity window, bounding memory.

        An organization holding a live cached-fraction estimate is retained
        past the activity window (its cache standing outlives one request's
        recency), until the EWMA retention horizon passes with no new sample.
        A rung entry itself survives while it holds any organization, window
        entry, or learned ceiling, because the learned ceiling must outlive
        the traffic that taught it.
        """
        stale = [
            organization_id
            for organization_id, load in rung.organizations.items()
            if load.inflight == 0
            and load.last_seen < now - self._window
            and (
                load.cached_sampled_at is None
                or load.cached_sampled_at < now - EWMA_RETENTION_SECONDS
            )
        ]
        for organization_id in stale:
            del rung.organizations[organization_id]
        if (
            rung.total == 0
            and not rung.organizations
            and not rung.window
            and rung.learned_rpm is None
        ):
            self._rungs.pop(key, None)

    def _prune_window(self, rung: _RungLoad, now: float) -> None:
        """Slide one rung's dispatch window forward, keeping totals exact."""
        horizon = now - RATE_WINDOW_SECONDS
        window = rung.window
        while window and window[0][0] <= horizon:
            _reserved_at, tokens = window.popleft()
            rung.window_requests -= 1
            rung.window_tokens -= tokens
