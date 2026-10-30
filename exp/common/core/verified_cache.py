"""Process-local reuse of work already done on verified immutable content.

Verifying an immutable artifact (secret scanning its payloads, decoding and validating its
records) is a pure function of its bytes. Once bytes with a given SHA-256 have passed, they pass
again, so a process may remember the result under that digest. Callers key entries only by a
digest they have just recomputed from the bytes in hand (never by a path, a modification time or
an artifact ID alone), so changed content always misses and is verified in full. Only successes
are remembered: a failing artifact is re-verified and refused every time.
"""

import threading
from collections import OrderedDict
from collections.abc import Hashable


class VerifiedCache[KeyT: Hashable, ValueT]:
    """A bounded, thread-safe least-recently-used map from verified digests to results.

    Attributes:
        maximum: Entries kept; the least recently used one is forgotten first.
    """

    def __init__(self, maximum: int) -> None:
        """Start empty with a fixed bound.

        Args:
            maximum: Entries kept; at least one.
        """
        if maximum < 1:
            raise ValueError("a verified cache keeps at least one entry")
        self.maximum = maximum
        self._entries: OrderedDict[KeyT, ValueT] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: KeyT) -> ValueT | None:
        """The remembered result for ``key``, refreshing its recency, or None."""
        with self._lock:
            if key not in self._entries:
                return None
            self._entries.move_to_end(key)
            return self._entries[key]

    def put(self, key: KeyT, value: ValueT) -> None:
        """Remember one verified result, forgetting the oldest beyond the bound."""
        with self._lock:
            self._entries[key] = value
            self._entries.move_to_end(key)
            while len(self._entries) > self.maximum:
                self._entries.popitem(last=False)

    def clear(self) -> None:
        """Forget every entry."""
        with self._lock:
            self._entries.clear()
