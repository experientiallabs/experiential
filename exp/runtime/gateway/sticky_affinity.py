"""Worker-local sticky binding of affinity fingerprints to serving rungs.

Under a pool's ``maximize_cache_affinity`` policy, rendezvous hashing gives a
conversation a stable preferred rung, but a conversation that SPILLS (bound,
rate window, fairness, deadness) builds its warm prompt cache on the spill
target. Bouncing it back to the higher-ranked rung the moment that rung stops
shedding would abandon that cache, so each dispatch records a worker-local
``fingerprint -> deployment`` binding with an authored time-to-live
(``GatewayRungDispatchPolicy.sticky_spill_seconds``), refreshed on every hit
and honored ahead of rendezvous order on subsequent requests. The binding is
deliberately worker-local: the serving edge's pooled keep-alives pin one
client to one worker, so per-worker memory covers the common case, and the
cross-worker imprecision costs at most one cold dispatch on the rendezvous
rung. A binding to a dead or throttled rung is cleared by admission so
stickiness can never pin a conversation to a lane that cannot serve it.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass

from exp.runtime.gateway.contracts import AuthorizationSnapshot
from exp.runtime.gateway.health import DeploymentHealthRegistry
from exp.runtime.gateway.native_execution import deployment_health_key
from exp.runtime.gateway.routing import GatewayRoute

# Bound on remembered conversations per worker. At two cache lines of payload
# per entry this is a few megabytes; the least recently used binding is
# evicted first, which is also the coldest provider cache.
_MAXIMUM_BINDINGS = 65_536

# Hard cap on one binding's total age, as a multiple of its authored lifetime.
# Refresh-on-hit alone would let one transient congestion pin a long-running
# agent session to its (possibly pricier) spill rung forever; past the cap the
# binding lapses even under continuous hits, the conversation returns to its
# rendezvous rung once, and a still-congested rung simply re-spills and
# re-binds it.
STICKY_MAXIMUM_AGE_LIFETIMES = 4.0


@dataclass(frozen=True)
class AffinityPlacement:
    """The affinity facts one admission resolved for dispatch accounting.

    ``fingerprint`` is present only on ``maximize_cache_affinity`` routes (the
    tenant-isolated conversation identity that keyed placement), so dispatch
    reservation can read and refresh the worker-local sticky binding and apply
    the fresh-session spill threshold. ``sticky_preferred`` marks a route
    whose depth 0 was chosen by a live sticky binding rather than rendezvous
    order, for the ``affinity_sticky`` disclosure.
    """

    fingerprint: bytes | None = None
    sticky_preferred: bool = False


def sticky_first_order(
    order: tuple[int, ...],
    route: GatewayRoute,
    *,
    fingerprint: bytes,
    sticky: StickySpillRegistry,
    health: DeploymentHealthRegistry,
    authorization: AuthorizationSnapshot,
) -> tuple[tuple[int, ...], int | None]:
    """Move a live sticky binding's rung to the front of the rendezvous order.

    A binding whose rung left the route is ignored (the binding expires on its
    own); a binding whose rung is suppressed right now is cleared and ignored,
    so a throttled or dead spill target releases the conversation back to
    rendezvous placement. A binding already at the rendezvous front changes
    nothing and is not reported as sticky.

    Args:
        order: Rendezvous permutation of the route's deployment indexes.
        route: Frozen route the permutation indexes into.
        fingerprint: The request's affinity fingerprint.
        sticky: Worker-local conversation-to-rung bindings.
        health: Deployment circuit and throttle registry.
        authorization: Frozen authority, for the health key.

    Returns:
        The (possibly reordered) permutation and the sticky rung's route
        index when a live binding moved or confirmed the front (``None``
        when rendezvous order stands on its own).
    """
    bound = sticky.bound_deployment(fingerprint)
    if bound is None:
        return order, None
    sticky_index = next(
        (
            index
            for index, deployment in enumerate(route.deployments)
            if deployment.deployment_id == bound
        ),
        None,
    )
    if sticky_index is None:
        return order, None
    if health.suppressed(deployment_health_key(authorization, route.deployments[sticky_index])):
        sticky.clear(fingerprint)
        return order, None
    if order[0] == sticky_index:
        return order, None
    return (sticky_index, *(index for index in order if index != sticky_index)), sticky_index


class StickySpillRegistry:
    """Bounded, lock-guarded LRU of affinity fingerprints to serving rungs."""

    def __init__(
        self,
        *,
        maximum_bindings: int = _MAXIMUM_BINDINGS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Initialize an empty registry with an injectable clock for tests.

        Args:
            maximum_bindings: LRU capacity; the oldest binding evicts first.
            clock: Monotonic clock.

        Raises:
            ValueError: The capacity is not positive.
        """
        if maximum_bindings < 1:
            raise ValueError("sticky binding capacity must be positive")
        self._maximum = maximum_bindings
        self._clock = clock
        # fingerprint -> (deployment_id, expires_at, age_deadline). The age
        # deadline is fixed when the binding is (re)created for a deployment
        # and survives refreshes, so continuous hits cannot extend one binding
        # forever.
        self._bindings: OrderedDict[bytes, tuple[str, float, float]] = OrderedDict()
        self._lock = threading.Lock()

    def bind(self, fingerprint: bytes, deployment_id: str, *, ttl_seconds: float) -> None:
        """Record or refresh one fingerprint's serving rung.

        A refresh extends the idle lifetime but never the age deadline; a
        binding to a DIFFERENT rung starts a fresh age (a new cache home).

        Args:
            fingerprint: Tenant-isolated affinity fingerprint.
            deployment_id: The rung that served (or is about to serve) it.
            ttl_seconds: Authored binding lifetime; not positive records nothing.
        """
        if ttl_seconds <= 0:
            return
        now = self._clock()
        expires_at = now + ttl_seconds
        age_deadline = now + STICKY_MAXIMUM_AGE_LIFETIMES * ttl_seconds
        with self._lock:
            entry = self._bindings.get(fingerprint)
            if entry is not None and entry[0] == deployment_id:
                age_deadline = entry[2]
            self._bindings[fingerprint] = (
                deployment_id,
                min(expires_at, age_deadline),
                age_deadline,
            )
            self._bindings.move_to_end(fingerprint)
            while len(self._bindings) > self._maximum:
                self._bindings.popitem(last=False)

    def bound_deployment(self, fingerprint: bytes) -> str | None:
        """Return the fingerprint's live binding, dropping an expired one.

        Args:
            fingerprint: Tenant-isolated affinity fingerprint.

        Returns:
            The bound deployment id, or ``None`` when no live binding exists.
        """
        now = self._clock()
        with self._lock:
            entry = self._bindings.get(fingerprint)
            if entry is None:
                return None
            deployment_id, expires_at, _age_deadline = entry
            if expires_at <= now:
                del self._bindings[fingerprint]
                return None
            return deployment_id

    def clear(self, fingerprint: bytes) -> None:
        """Drop one fingerprint's binding (its rung died or throttled); idempotent.

        Args:
            fingerprint: Tenant-isolated affinity fingerprint.
        """
        with self._lock:
            self._bindings.pop(fingerprint, None)

    def size(self) -> int:
        """Return the number of retained bindings, for metrics."""
        with self._lock:
            return len(self._bindings)
