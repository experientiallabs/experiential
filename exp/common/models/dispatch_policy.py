"""Authored per-rung dispatch policy: bounds, rate windows, and affinity.

One nested, fully defaulted model hung off ``GatewayDeploymentMetadata``. It
is additive-defaulted on purpose: an unauthored rung contributes zero identity
bytes under the catalog's exclude-defaults digest, so adding this surface
moves no snapshot digests and needs no schema-version bump; authoring a value
is a real catalog change and produces a new content address.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from exp.common.core.artifacts import ContractModel

FailoverMode = Literal["maximize_availability", "maximize_cache", "maximize_cache_affinity"]
SaturationPolicy = Literal["overflow", "refuse"]
"""How a pool's waterfall orders its rungs and reacts to a failed attempt.

``maximize_availability`` (the default, historical behavior) fails over to the
next rung on any failover-eligible error. ``maximize_cache`` does NOT fail over on
a throttle (429) -- it returns the throttle so the caller retries the warm rung
after backoff, preserving its prompt cache rather than restarting cold on another
provider -- while STILL failing over on operational deadness
(auth/not-found/5xx/transport) and on a stalled lane (a first-byte or
header-phase timeout that never answered), for which there is no warm cache to
preserve. A genuinely retryable timeout (provider 408) redials the warm rung in
both modes. Client errors reject without failover in both modes.

``maximize_cache_affinity`` keeps availability-style failover (a throttle DOES
fail over: the deterministic alternate builds warm cache instead of the caller
waiting out a backoff) but replaces the certified initial rung order with a
per-request weighted rendezvous hash of the request's stable affinity
fingerprint, so every worker independently sends one conversation to the same
rung and, when that rung sheds or dies, to the same alternate. Rung weights
come from each deployment's authored ``GatewayRungDispatchPolicy``.

Widening this literal is deployment-ordered, like every catalog vocabulary
change: a new value may be AUTHORED only after every serving worker runs a
build that parses it, because an older worker rejects the unknown value at
hydration and fails that alias closed (the per-alias fail-safe excludes the
alias; the worker still serves everything else). The hosted platform is the
only author and its release contract pins this order: engine release, then
fleet-wide pin bump, then the catalog opt-in.
"""


class GatewayThrottleRedialPolicy(ContractModel):
    """Authored per-pool backoff-and-redial schedule for provider throttles.

    When a pool authors this policy, a throttle (429 or overload) on a rung
    whose warm cache is worth waiting for is no longer failed over or
    surfaced on its first occurrence: the data plane waits with exponential
    backoff and re-dials the SAME rung, up to ``max_attempts`` redials per
    rung, before the ladder advances to the next certified rung, and a
    throttle surfaces to the caller only once every rung is exhausted. Each
    redial waits ``base_delay_ms * 2**n`` milliseconds (``n`` counting from
    zero, equal-jittered, capped at ``max_delay_ms``), or the provider's own
    ``Retry-After`` when it is longer and still within ``max_delay_ms``. A
    ``Retry-After`` above ``max_delay_ms`` means the rung is out for longer
    than the pool is willing to wait, so the ladder advances instead of
    waiting. A wait never exceeds the rung's first-byte allowance or what
    the request deadline leaves for the redial itself.

    Unauthored (``None`` on the pool) keeps the historical behavior exactly:
    a throttle is failover-only and the pool's ``failover_mode`` and
    ``throttle_cache_threshold`` decide between surfacing and cold failover.
    Additive-defaulted like every dispatch control: an unauthored pool
    contributes zero identity bytes under the exclude-defaults digest.
    """

    max_attempts: int = Field(ge=1, le=6)
    """Redials of the throttled rung after its throttled dispatch, per rung.

    Every redial is a durably reserved attempt row disclosed as
    ``throttle_backoff``; the request's hard total attempt cap still bounds
    the whole ladder, so a deep ladder with many redials per rung may not
    reach its last rung.
    """
    base_delay_ms: int = Field(ge=1, le=60_000)
    """Wait before the first redial, in milliseconds; each later redial doubles it."""
    max_delay_ms: int = Field(ge=1, le=120_000)
    """Ceiling on any single wait, in milliseconds, including a provider ``Retry-After``."""

    @model_validator(mode="after")
    def _require_ordered_delays(self) -> GatewayThrottleRedialPolicy:
        """Reject a ceiling below the first wait, which would make every redial impossible."""
        if self.max_delay_ms < self.base_delay_ms:
            raise ValueError("max_delay_ms must be at least base_delay_ms")
        return self


class GatewayRungDispatchPolicy(ContractModel):
    """Authored per-rung dispatch controls: admission bounds, rates, and affinity.

    Every field defaults to inert so an unauthored rung behaves exactly as
    today: unbounded admission, no rate windows, no fairness accounting,
    rendezvous weight 1, no session stickiness. The bound, rate caps, and
    fairness apply on any pool; the affinity weight, fresh-session threshold,
    and sticky binding are read only under a pool's
    ``maximize_cache_affinity`` policy.
    """

    concurrency_bound: int | None = Field(default=None, ge=1)
    """In-flight dispatch cap for this rung, per gateway worker process.

    Beyond the bound a request spills immediately to the waterfall's next rung
    instead of queueing at the deployment (seconds of spill latency, never a
    deadline death). ``None`` leaves admission unbounded (historical behavior).
    The count is per worker process: the platform authors the per-worker value
    (fleet capacity divided by serving replicas), because enforcement is
    in-memory arithmetic with no shared state on the request path.
    """
    saturation: SaturationPolicy = "overflow"
    """What a shed on this rung does when no other rung admits the request.

    ``overflow`` (the default, the historical behavior) force-admits the
    request past this rung's bound and discloses ``saturated_overflow``: an
    authored policy never manufactures a failure. ``refuse`` answers the
    caller at once with a retryable 429 (``lane_saturated``, the protocol's
    throttle Retry-After)
    instead of dispatching one more request onto a lane already at its
    bound: the choice for a lane whose slow tail must never hold more of a
    worker's admission permits than its bound allows, at the price of a
    manufactured refusal when every rung of the pool is full. The default
    bound a worker applies to rungs that author no ``concurrency_bound``
    (``exp.runtime.gateway.lane_saturation``) always refuses: it exists to
    protect the worker, and overflowing it would protect nothing.
    """
    fair_share: bool = False
    """Whether contended admission on this rung is weighted max-min fair.

    When the rung is at or near its ``concurrency_bound``, each organization's
    admissions are limited to its weighted share of the bound (weights ride
    ``AuthorizationSnapshot.fair_share_weight``), with freed capacity reserved
    for recently active under-share organizations. Work-conserving: a lone
    organization borrows the whole bound. Requires ``concurrency_bound``.
    """
    affinity_weight: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    """Rendezvous weight under ``maximize_cache_affinity`` (``None`` means 1.0).

    Heavier rungs attract proportionally more affinity fingerprints; authoring
    a heavy house rung and a moderate cheap-cached-input rung makes the house
    box the warm home and the cheap rung the stable spill target.
    """
    requests_per_minute: int | None = Field(default=None, ge=1)
    """Sliding-window dispatch rate cap for this rung, per gateway worker process.

    A reservation past the 60-second window's cap sheds sideways to the next
    rung (``rate_limit``) BEFORE the provider answers 429, so the pre-emptive
    spill preserves the request instead of burning a provider attempt. Like
    ``concurrency_bound`` the value is authored per worker process (fleet rate
    divided by serving replicas). Usable without a ``concurrency_bound``. The
    worker additionally learns a lower working ceiling from observed provider
    throttles and re-discovers headroom by letting a little more through over
    time, so the authored value is a cap, never a promise the provider honors
    it.
    """
    tokens_per_minute: int | None = Field(default=None, ge=1)
    """Sliding-window token rate cap for this rung, per gateway worker process.

    Counted from each dispatch's worst-case reserved input plus output tokens
    at reservation time (the same conservative bound the platform's token
    windows count), so a concurrent burst binds instead of leaking past the
    cap. Over-window reservations shed sideways as ``rate_limit`` exactly like
    ``requests_per_minute``. Usable without a ``concurrency_bound``.
    """
    cache_priority_alpha: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    """Congestion-dependent boost for cache-heavy organizations under fairness.

    When set on a ``fair_share`` rung, each organization's effective admission
    weight becomes ``weight * (1 + alpha * congestion * cached_fraction)``
    where ``congestion`` is the rung's in-flight total over its bound and
    ``cached_fraction`` is the worker's EWMA of that organization's settled
    cached-token fraction on this rung. At the contended margin the
    organization whose traffic reuses warm provider cache is admitted ahead of
    an equal-weight organization running cold, exactly when cache is most
    valuable. ``None`` disables the term. Requires ``fair_share``.
    """
    fresh_session_spill_fraction: float | None = Field(default=None, gt=0, lt=1)
    """Fraction of the bound where sessions with no warm standing shed early.

    Under a pool's ``maximize_cache_affinity`` policy, a request whose affinity
    fingerprint holds no live sticky binding on this rung sheds sideways once
    in-flight dispatches reach ``concurrency_bound * fraction``
    (``fresh_session_spill``), reserving the top slice of the bound for warm
    sessions, which shed only at the hard bound. ``None`` disables the early
    threshold. Requires ``concurrency_bound`` AND ``sticky_spill_seconds``:
    warm standing IS a live sticky binding, so without a binding lifetime
    every session would stay fresh forever and the reserved top slice would be
    reachable only through force admission.
    """
    sticky_spill_seconds: int | None = Field(default=None, ge=1)
    """How long one conversation stays bound to the rung that served it.

    Under ``maximize_cache_affinity`` each dispatch records a worker-local
    fingerprint-to-deployment binding with this time-to-live, refreshed on
    every hit; the binding is honored ahead of rendezvous order on subsequent
    requests, so a spilled conversation does not bounce back to the
    higher-ranked rung the moment it stops shedding (its warm cache now lives
    on the spill target). Author roughly the provider's prompt-cache lifetime.
    ``None`` records no binding for dispatches landing on this rung.
    """

    @model_validator(mode="after")
    def _require_coherent_authoring(self) -> GatewayRungDispatchPolicy:
        """Reject values whose prerequisite lever is not authored.

        Fairness and the fresh-session threshold divide a capacity, so both
        need the bound; the cache-priority term scales fairness weights, so it
        needs fairness. Failing closed here keeps an inert combination from
        being authored and silently doing nothing.
        """
        if self.fair_share and self.concurrency_bound is None:
            raise ValueError("fair_share requires a concurrency_bound to share")
        if self.cache_priority_alpha is not None and not self.fair_share:
            raise ValueError("cache_priority_alpha requires fair_share to weight")
        if self.fresh_session_spill_fraction is not None:
            if self.concurrency_bound is None:
                raise ValueError("fresh_session_spill_fraction requires a concurrency_bound")
            if self.sticky_spill_seconds is None:
                raise ValueError(
                    "fresh_session_spill_fraction requires sticky_spill_seconds: warm "
                    "standing is a live sticky binding, so without one every session "
                    "stays fresh and the reserved slice is unreachable"
                )
        return self
