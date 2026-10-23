"""Bounded, thread-safe, content-free verdict cache for classification checks."""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Callable

from pydantic import Field

from exp.common.core.artifacts import ContractModel

CategoryScores = tuple[tuple[str, float], ...]
"""Ordered ``(category, score)`` pairs, each score in ``[0, 1]``."""


class VerdictCacheSettings(ContractModel):
    """Bounds for one adapter's verdict cache.

    Attributes:
        max_entries: Least-recently-used entries kept, default 4,096; zero disables caching.
        ttl_seconds: Seconds an entry stays valid after it is stored, default 300.
    """

    max_entries: int = Field(default=4_096, ge=0, le=1_000_000)
    ttl_seconds: float = Field(default=300.0, gt=0, le=86_400, allow_inf_nan=False)


class VerdictCache:
    """Least-recently-used map from a window digest to its category scores.

    Keys are SHA-256 digests and values are scores, so the cache never holds
    request text. Callers store only definitive answers; a classifier error is
    never cached and the next request asks the model again. Isolation workers
    share one cache, so every operation holds one lock.
    """

    def __init__(self, settings: VerdictCacheSettings, *, monotonic: Callable[[], float]) -> None:
        """Bind the bounds and the expiry clock.

        Args:
            settings: Entry and lifetime bounds.
            monotonic: Process-local clock in seconds.
        """
        self._max_entries = settings.max_entries
        self._ttl_seconds = settings.ttl_seconds
        self._monotonic = monotonic
        self._entries: OrderedDict[bytes, tuple[float, CategoryScores]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: bytes) -> CategoryScores | None:
        """Return fresh scores for ``key`` and mark them recently used.

        Args:
            key: SHA-256 digest of the configuration and window.

        Returns:
            The cached scores, or ``None`` when absent or expired.
        """
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            expires_at, scores = entry
            if expires_at <= self._monotonic():
                del self._entries[key]
                return None
            self._entries.move_to_end(key)
            return scores

    def put(self, key: bytes, scores: CategoryScores) -> None:
        """Store one definitive answer and evict the least recently used overflow.

        Args:
            key: SHA-256 digest of the configuration and window.
            scores: Category scores the decision model produced for that window.
        """
        if self._max_entries == 0:
            return
        with self._lock:
            self._entries[key] = (self._monotonic() + self._ttl_seconds, scores)
            self._entries.move_to_end(key)
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)
