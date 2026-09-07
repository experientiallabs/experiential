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

# Bound on remembered conversations per worker. At two cache lines of payload
# per entry this is a few megabytes; the least recently used binding is
# evicted first, which is also the coldest provider cache.
_MAXIMUM_BINDINGS = 65_536


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
        self._bindings: OrderedDict[bytes, tuple[str, float]] = OrderedDict()
        self._lock = threading.Lock()

    def bind(self, fingerprint: bytes, deployment_id: str, *, ttl_seconds: float) -> None:
        """Record or refresh one fingerprint's serving rung.

        Args:
            fingerprint: Tenant-isolated affinity fingerprint.
            deployment_id: The rung that served (or is about to serve) it.
            ttl_seconds: Authored binding lifetime; not positive records nothing.
        """
        if ttl_seconds <= 0:
            return
        expires_at = self._clock() + ttl_seconds
        with self._lock:
            self._bindings[fingerprint] = (deployment_id, expires_at)
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
            deployment_id, expires_at = entry
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
