"""Domain-neutral SQLite connections with bounded waits and durable WAL commits."""

from __future__ import annotations

import os
import sqlite3
import threading
import time
from collections import OrderedDict
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from exp.common.core.host_scope import require_host_scope


def connect_database(
    path: Path,
    *,
    busy_timeout_ms: int = 5_000,
    enable_wal: bool = True,
    create: bool = True,
    factory: type[sqlite3.Connection] = sqlite3.Connection,
) -> sqlite3.Connection:
    """Open one configured SQLite connection with mandatory safety pragmas.

    Args:
        path: Database path.
        busy_timeout_ms: Bounded lock wait in milliseconds.
        enable_wal: Whether to assert the supported journal mode after version checks.
        create: Whether a missing file is created; otherwise opening it fails.
        factory: Connection class, so an owner can keep per-connection state on it.

    Returns:
        Configured connection with row-name access.

    Raises:
        HostScopeError: The database directory belongs to another machine or sits on an
            undeclared network filesystem, where SQLite's locks and WAL index do not work.
    """
    if path.parent.is_dir():  # otherwise SQLite itself refuses to open the database
        require_host_scope(path.parent)
    connection = sqlite3.connect(
        path if create else f"{path.absolute().as_uri()}?mode=rw",
        timeout=busy_timeout_ms / 1_000,
        isolation_level=None,
        factory=factory,
        uri=not create,
    )
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute(f"PRAGMA busy_timeout = {busy_timeout_ms}")
    if enable_wal:
        enable_wal_mode(connection, deadline=time.monotonic() + busy_timeout_ms / 1_000)
        connection.execute(f"PRAGMA busy_timeout = {busy_timeout_ms}")
        connection.execute("PRAGMA synchronous = FULL")
    return connection


MAXIMUM_IDLE_CONNECTIONS = 16
"""Idle connections one thread keeps; the least recently used one beyond this closes."""


type _CacheKey = tuple[str, int, bool, bool, type[sqlite3.Connection]]
"""Path, busy timeout, WAL setup, creation, and connection class of one idle slot."""

type _FileIdentity = tuple[int, int] | None
"""Device and inode of the database file a connection opened, when it could be read."""


class _ThreadConnectionCache(threading.local):
    """Per-thread idle SQLite connections and their file identities, in recency order."""

    def __init__(self) -> None:
        """Start each thread with an empty idle-connection map owned by this process."""
        self.idle: OrderedDict[_CacheKey, tuple[sqlite3.Connection, _FileIdentity]] = OrderedDict()
        self.pid = os.getpid()


_connection_cache = _ThreadConnectionCache()
_inherited_connections: list[sqlite3.Connection] = []
"""Handles a forked child inherited; kept referenced so they are never finalized there."""


def _idle_connections() -> OrderedDict[_CacheKey, tuple[sqlite3.Connection, _FileIdentity]]:
    """Return this thread's idle map, forgetting connections inherited across a fork.

    A forked child must never use or close its parent's SQLite handles, so it starts empty
    and keeps the inherited handles referenced: finalizing one would close it in the child.
    """
    if _connection_cache.pid != os.getpid():
        _inherited_connections.extend(
            connection for connection, _ in _connection_cache.idle.values()
        )
        _connection_cache.idle = OrderedDict()
        _connection_cache.pid = os.getpid()
    return _connection_cache.idle


def _file_identity(path: Path) -> _FileIdentity:
    """Return the database file's device and inode, or None when it cannot be read."""
    try:
        metadata = os.stat(path)
    except OSError:
        return None
    return metadata.st_dev, metadata.st_ino


@contextmanager
def persistent_connection(
    path: Path,
    *,
    busy_timeout_ms: int = 5_000,
    enable_wal: bool = True,
    create: bool = True,
    factory: type[sqlite3.Connection] = sqlite3.Connection,
) -> Iterator[sqlite3.Connection]:
    """Yield one reusable per-thread connection for repeated database operations.

    Opening a SQLite connection pays file open, pragma, and WAL setup costs on
    every call, and closing the last connection to a WAL database checkpoints and
    syncs it, which dominates hot request paths. This checkout keeps one idle
    connection per thread, path, and configuration so sequential operations reuse
    it, while overlapping checkouts on the same thread fall back to a fresh
    connection instead of sharing an in-flight transaction. A thread keeps at most
    ``MAXIMUM_IDLE_CONNECTIONS`` idle connections, closing the least recently used.
    An idle connection whose path now names a different file (replaced or deleted
    and recreated) is closed and replaced, so a checkout never writes to an
    unlinked database.

    Args:
        path: Database path.
        busy_timeout_ms: Bounded lock wait in milliseconds.
        enable_wal: Whether a new connection asserts WAL and full commit sync on open.
        create: Whether a missing file is created; otherwise opening it fails.
        factory: Connection class; owners that keep per-connection state pass a subclass.

    Yields:
        A configured connection; it returns to the idle cache on clean exit.
    """
    key = (str(path), busy_timeout_ms, enable_wal, create, factory)
    idle = _idle_connections()
    cached = idle.pop(key, None)
    identity = _file_identity(path)
    if cached is not None and (cached[1] is None or cached[1] != identity):
        cached[0].close()
        cached = None
    if cached is None:
        connection = connect_database(
            path,
            busy_timeout_ms=busy_timeout_ms,
            enable_wal=enable_wal,
            create=create,
            factory=factory,
        )
        identity = _file_identity(path)
    else:
        connection = cached[0]
    try:
        yield connection
    except BaseException:
        connection.close()
        raise
    if connection.in_transaction or key in idle:
        connection.close()
        return
    idle[key] = (connection, identity)
    while len(idle) > MAXIMUM_IDLE_CONNECTIONS:
        idle.popitem(last=False)[1][0].close()


def close_idle_connections() -> int:
    """Close and forget the calling thread's cached idle connections.

    A long-lived worker thread that stops servicing database operations calls
    this before it exits, so the database descriptors held by its
    ``persistent_connection`` cache release with the worker instead of
    lingering for the life of the interpreter.

    Returns:
        Number of connections closed.
    """
    idle = _idle_connections()
    closed = len(idle)
    for connection, _ in idle.values():
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
