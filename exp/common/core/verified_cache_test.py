"""A bounded least-recently-used map of verified results."""

import pytest

from exp.common.core.verified_cache import VerifiedCache


def test_entries_are_bounded_and_refreshed_by_use() -> None:
    """The least recently used entry is forgotten first; a lookup refreshes recency."""
    cache: VerifiedCache[str, int] = VerifiedCache(2)
    cache.put("a", 1)
    cache.put("b", 2)
    assert cache.get("a") == 1
    cache.put("c", 3)
    assert (cache.get("a"), cache.get("b"), cache.get("c")) == (1, None, 3)
    cache.clear()
    assert cache.get("a") is None
    with pytest.raises(ValueError, match="at least one"):
        VerifiedCache(0)
