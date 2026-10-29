"""Domain-neutral SQLite connections with bounded waits and durable WAL commits."""

from __future__ import annotations

import sqlite3
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


def connect_database(
    path: Path, *, busy_timeout_ms: int = 5_000, enable_wal: bool = True
) -> sqlite3.Connection:
    """Open one configured SQLite connection with mandatory safety pragmas.

    Args:
        path: Database path.
        busy_timeout_ms: Bounded lock wait in milliseconds.
        enable_wal: Whether to assert the supported journal mode after version checks.

    Returns:
        Configured connection with row-name access.
    """
    connection = sqlite3.connect(path, timeout=busy_timeout_ms / 1_000, isolation_level=None)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute(f"PRAGMA busy_timeout = {busy_timeout_ms}")
    if enable_wal:
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = FULL")
    return connection


class _ThreadConnectionCache(threading.local):
    """Per-thread idle SQLite connections keyed by path and busy timeout."""

    def __init__(self) -> None:
        """Start each thread with an empty idle-connection map."""
        self.idle: dict[tuple[str, int], sqlite3.Connection] = {}


_connection_cache = _ThreadConnectionCache()


@contextmanager
def persistent_connection(
    path: Path, *, busy_timeout_ms: int = 5_000
) -> Iterator[sqlite3.Connection]:
    """Yield one reusable per-thread connection for repeated database operations.

    Opening a SQLite connection pays file open, pragma, and WAL setup costs on
    every call, which dominates hot request paths. This checkout keeps one idle
    connection per thread, path, and timeout so sequential operations reuse it,
    while overlapping checkouts on the same thread fall back to a fresh
    connection instead of sharing an in-flight transaction.

    Args:
        path: Database path.
        busy_timeout_ms: Bounded lock wait in milliseconds.

    Yields:
        A configured connection; it returns to the idle cache on clean exit.
    """
    key = (str(path), busy_timeout_ms)
    connection = _connection_cache.idle.pop(key, None)
    if connection is None:
        connection = connect_database(path, busy_timeout_ms=busy_timeout_ms)
    try:
        yield connection
    except BaseException:
        connection.close()
        raise
    if connection.in_transaction:
        connection.close()
        return
    previous = _connection_cache.idle.get(key)
    if previous is not None:
        connection.close()
        return
    _connection_cache.idle[key] = connection


def close_idle_connections() -> int:
    """Close and forget the calling thread's cached idle connections.

    A long-lived worker thread that stops servicing database operations calls
    this before it exits, so the database descriptors held by its
    ``persistent_connection`` cache release with the worker instead of
    lingering for the life of the interpreter.

    Returns:
        Number of connections closed.
    """
    idle = _connection_cache.idle
    closed = len(idle)
    for connection in idle.values():
        connection.close()
    idle.clear()
    return closed


def enable_wal_mode(connection: sqlite3.Connection, *, deadline: float) -> None:
    """Enable WAL with a finite retry when another process initializes the same database.

    SQLite journal-mode changes can return SQLITE_BUSY immediately without invoking
    the busy handler. Recheck the committed mode after releasing that attempt so
    concurrent first writers can converge on WAL within their shared wait allowance.

    Args:
        connection: Open database whose complete schema has already been validated.
        deadline: Absolute monotonic deadline shared with all write admission stages.
    """
    while True:
        try:
            set_busy_deadline(connection, deadline=deadline)
            if connection.execute("PRAGMA journal_mode").fetchone()[0] != "wal":
                set_busy_deadline(connection, deadline=deadline)
                connection.execute("PRAGMA journal_mode=WAL").fetchone()
            return
        except sqlite3.OperationalError as error:
            remaining = deadline - time.monotonic()
            if error.sqlite_errorcode & 0xFF != sqlite3.SQLITE_BUSY or remaining <= 0:
                raise
            time.sleep(min(0.01, remaining))


def set_busy_deadline(connection: sqlite3.Connection, *, deadline: float) -> None:
    """Set the next SQLite lock wait to only the unspent monotonic admission budget."""
    milliseconds = max(0, int((deadline - time.monotonic()) * 1000))
    connection.execute(f"PRAGMA busy_timeout={milliseconds}")
