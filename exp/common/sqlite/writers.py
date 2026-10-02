"""Bounded FIFO admission for local SQLite writers sharing one database."""

from __future__ import annotations

import sqlite3
import threading
import time
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from weakref import WeakValueDictionary


class _WriterQueue:
    """A local queue preventing SQLite busy-handler polling from starving older writers.

    Attributes:
        lock: Protects membership and signaling.
        waiters: Arrival-ordered events, with only the first admitted.
    """

    def __init__(self) -> None:
        """Start with no waiting writers."""
        self.lock = threading.Lock()
        self.waiters: deque[threading.Event] = deque()


_registry_lock = threading.Lock()
_queues: WeakValueDictionary[Path, _WriterQueue] = WeakValueDictionary()


@contextmanager
def writer_turn(path: Path, *, timeout_s: float) -> Iterator[float]:
    """Queue one local writer and share its deadline with SQLite's cross-process wait.

    Args:
        path: Canonical database path identifying the local writer queue.
        timeout_s: Combined wait allowance, including subsequent database admission.

    Yields:
        Remaining seconds available for SQLite to acquire its cross-process write lock.

    Raises:
        sqlite3.OperationalError: The writer cannot enter within its bounded deadline.
    """
    deadline = time.monotonic() + max(0.0, timeout_s)
    with _registry_lock:
        queue = _queues.get(path)
        if queue is None:
            queue = _WriterQueue()
            _queues[path] = queue
    turn = threading.Event()
    with queue.lock:
        queue.waiters.append(turn)
        if len(queue.waiters) == 1:
            turn.set()
    try:
        if not turn.wait(max(0.0, deadline - time.monotonic())):
            error = sqlite3.OperationalError("database is locked: local writer deadline exceeded")
            error.sqlite_errorcode = sqlite3.SQLITE_BUSY
            error.sqlite_errorname = "SQLITE_BUSY"
            raise error
        yield max(0.0, deadline - time.monotonic())
    finally:
        with queue.lock:
            queue.waiters.remove(turn)
            if queue.waiters:
                queue.waiters[0].set()
