"""Fair selection for bounded asynchronous job recovery."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable


class RecoveryRotation[T]:
    """Rotate retained work by its stable identity across bounded sweeps."""

    def __init__(self, identity: Callable[[T], str]) -> None:
        """Bind the work identity extractor; callers serialize selection."""
        self._identity = identity
        self._order: deque[str] = deque()

    def select(self, items: list[T], limit: int) -> list[T]:
        """Visit each pending identity without retaining payloads or replaying one twice."""
        if limit < 0:
            raise ValueError("recovery selection limit must be nonnegative")
        pending = {self._identity(item): item for item in items}
        if len(pending) != len(items):
            raise ValueError("recovery work identities must be unique")
        self._order = deque(identity for identity in self._order if identity in pending)
        present = set(self._order)
        self._order.extend(identity for identity in pending if identity not in present)
        selected: list[T] = []
        for _ in range(min(limit, len(pending))):
            identity = self._order.popleft()
            selected.append(pending[identity])
            self._order.append(identity)
        return selected
