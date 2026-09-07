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
    threshold. Requires ``concurrency_bound``.
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
        if self.fresh_session_spill_fraction is not None and self.concurrency_bound is None:
            raise ValueError("fresh_session_spill_fraction requires a concurrency_bound")
        return self
