"""Tests for the bounded classification verdict cache."""

from __future__ import annotations

from exp.runtime.gateway.guardrails.classification_cache import VerdictCache, VerdictCacheSettings


class _Clock:
    """Manually advanced monotonic clock."""

    def __init__(self) -> None:
        """Start at a fixed positive time."""
        self.now = 100.0

    def __call__(self) -> float:
        """Return the current fake time."""
        return self.now


def test_entries_expire_after_their_ttl() -> None:
    """A stored answer is served until its lifetime ends, then asked again."""
    clock = _Clock()
    cache = VerdictCache(VerdictCacheSettings(ttl_seconds=10), monotonic=clock)
    cache.put(b"window", (("off_topic", 0.4),))

    clock.now += 9.9
    assert cache.get(b"window") == (("off_topic", 0.4),)
    clock.now += 0.1
    assert cache.get(b"window") is None


def test_least_recently_used_entry_is_evicted_first() -> None:
    """A read refreshes recency, so the untouched entry is the one evicted."""
    cache = VerdictCache(VerdictCacheSettings(max_entries=2), monotonic=_Clock())
    cache.put(b"first", (("off_topic", 0.1),))
    cache.put(b"second", (("off_topic", 0.2),))
    assert cache.get(b"first") is not None

    cache.put(b"third", (("off_topic", 0.3),))

    assert cache.get(b"second") is None
    assert cache.get(b"first") == (("off_topic", 0.1),)
    assert cache.get(b"third") == (("off_topic", 0.3),)


def test_zero_entries_disables_caching() -> None:
    """Operators can turn the cache off without a separate switch."""
    cache = VerdictCache(VerdictCacheSettings(max_entries=0), monotonic=_Clock())
    cache.put(b"window", (("off_topic", 0.4),))

    assert cache.get(b"window") is None
