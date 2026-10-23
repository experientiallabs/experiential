"""Short transactional access to project state in the shared content database."""

from __future__ import annotations

import os
import random
import sqlite3
import stat
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path

from exp.common.core.host_scope import require_host_scope
from exp.common.project.sqlite_schema import initialize_project_schema
from exp.common.sqlite.connection import (
    enable_wal_mode,
    persistent_connection,
    set_busy_deadline,
)
from exp.common.sqlite.schema import validate_content_tables
from exp.common.sqlite.writers import writer_turn


def content_database_path(root: Path) -> Path:
    """Return the shared local content database without creating state."""
    return root.resolve() / "gateway" / "traffic.db"


@dataclass
class _Transaction:
    """One thread-owned reentrant project transaction.

    Attributes:
        connection: Open connection owned by the outer transaction.
        write: Whether the transaction can mutate project state.
    """

    connection: sqlite3.Connection
    write: bool


class _Transactions(threading.local):
    """Thread-local transactions; independent workers never share connections."""

    def __init__(self) -> None:
        """Start with no active database transactions."""
        self.active: dict[Path, _Transaction] = {}


_transactions = _Transactions()


def _create_database(path: Path) -> None:
    """Create private content storage and reject symlinked or public database files."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.parent.is_symlink():
        raise ValueError("Content database directory must not be a symlink.")
    require_host_scope(path.parent)
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
    except FileExistsError:
        pass
    else:
        os.close(descriptor)
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode) or stat.S_IMODE(metadata.st_mode) & 0o077:
        raise ValueError("Content database must be a private regular file.")


PROJECT_BUSY_TIMEOUT_S = 5.0
"""Registered SQLite lock wait of one project transaction admission attempt."""

PROJECT_LOCK_ATTEMPTS = 4
"""Admission attempts before a project transaction reports the database as busy."""

PROJECT_LOCK_BACKOFF_S = 0.05
"""Base of the jittered exponential pause between busy admission attempts."""

PROJECT_LOCK_BACKOFF_CAP_S = 1.0
"""Longest jittered pause between busy admission attempts."""


class ProjectDatabaseBusyError(sqlite3.OperationalError):
    """Another writer kept the project database locked through every bounded attempt.

    It keeps SQLite's busy or locked error code, so callers that classify lock
    contention by ``sqlite_errorcode`` handle it like any other busy error.
    """


class _ProjectConnection(sqlite3.Connection):
    """A reusable project database connection that remembers its one-time setup.

    Attributes:
        schema_version: SQLite schema cookie when the content tables were last validated,
            or None before the first validation.
        has_project_schema: Whether that validation found the project domain tables.
        durable: Whether WAL and full commit sync are configured for writes.
        read_only: Whether ``query_only`` currently refuses writes for a read snapshot.
    """

    schema_version: int | None = None
    has_project_schema: bool = False
    durable: bool = False
    read_only: bool = False


def _is_lock_error(error: sqlite3.OperationalError) -> bool:
    """Return whether SQLite refused for a busy or locked database, which a retry can clear."""
    return getattr(error, "sqlite_errorcode", 0) & 0xFF in (
        sqlite3.SQLITE_BUSY,
        sqlite3.SQLITE_LOCKED,
    )


def _busy_error(
    path: Path, error: sqlite3.OperationalError, timeout_s: float
) -> ProjectDatabaseBusyError:
    """Describe exhausted lock attempts while keeping SQLite's error code."""
    busy = ProjectDatabaseBusyError(
        f"Project database {path} stayed locked by another writer through "
        f"{PROJECT_LOCK_ATTEMPTS} attempts of up to {timeout_s:g}s each ({error}); nothing "
        "was committed. Retry once the other Experiential process finishes, or run fewer "
        "concurrent processes against this root."
    )
    busy.sqlite_errorcode = error.sqlite_errorcode
    busy.sqlite_errorname = getattr(error, "sqlite_errorname", "SQLITE_BUSY")
    return busy


def _pause(attempt: int) -> None:
    """Sleep a full-jitter exponential backoff before the next admission attempt."""
    ceiling = min(PROJECT_LOCK_BACKOFF_CAP_S, PROJECT_LOCK_BACKOFF_S * 2**attempt)
    time.sleep(random.uniform(0.0, ceiling))


def _begin(connection: _ProjectConnection, *, write: bool, deadline: float) -> None:
    """Configure a connection once, then open one read snapshot or immediate write.

    The content tables are validated before WAL is first enabled, so unrecognized
    state is rejected before the file changes. Inside the transaction, validation
    repeats only when the schema cookie moved since this connection last validated.

    Args:
        connection: Reusable project connection with no open transaction.
        write: Open an immediate write transaction instead of a read snapshot.
        deadline: Absolute monotonic deadline for every lock wait of this attempt.
    """
    if connection.read_only == write:
        connection.execute(f"PRAGMA query_only={'OFF' if write else 'ON'}")
        connection.read_only = not write
    if write and not connection.durable:
        validate_content_tables(connection)
        enable_wal_mode(connection, deadline=deadline)
        connection.execute("PRAGMA synchronous=FULL")
        connection.durable = True
    set_busy_deadline(connection, deadline=deadline)
    connection.execute("BEGIN IMMEDIATE" if write else "BEGIN")
    try:
        version = connection.execute("PRAGMA schema_version").fetchone()[0]
        if version != connection.schema_version:
            tables = validate_content_tables(connection)
            connection.has_project_schema = "project_store_schema" in tables
            connection.schema_version = version
        if write and not connection.has_project_schema:
            initialize_project_schema(connection)
    except BaseException:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise


def _admit(
    connection: _ProjectConnection,
    path: Path,
    *,
    write: bool,
    deadline: float,
    timeout_s: float,
) -> None:
    """Open the transaction, retrying lock refusals with jittered backoff.

    Only admission is retried: no caller statement has run, so nothing can repeat.

    Args:
        connection: Reusable project connection with no open transaction.
        path: Database path, for the exhausted-attempts error.
        write: Open an immediate write transaction instead of a read snapshot.
        deadline: Deadline of the first attempt, shared with local writer admission.
        timeout_s: Lock wait of each later attempt.

    Raises:
        ProjectDatabaseBusyError: Every attempt found the database locked.
    """
    for attempt in range(PROJECT_LOCK_ATTEMPTS):
        try:
            _begin(connection, write=write, deadline=deadline)
            return
        except sqlite3.OperationalError as error:
            if not _is_lock_error(error) or connection.in_transaction:
                raise
            if attempt + 1 == PROJECT_LOCK_ATTEMPTS:
                raise _busy_error(path, error, timeout_s) from error
        _pause(attempt)
        deadline = time.monotonic() + timeout_s


def _commit(connection: sqlite3.Connection, path: Path, *, timeout_s: float) -> None:
    """Commit, retrying a lock refusal only while SQLite reports the transaction still open.

    A refused COMMIT leaves the transaction active and uncommitted, so retrying the
    COMMIT statement cannot repeat a committed change.

    Args:
        connection: Connection inside the write transaction to commit.
        path: Database path, for the exhausted-attempts error.
        timeout_s: Lock wait of each retried attempt.

    Raises:
        ProjectDatabaseBusyError: Every attempt found the database locked; the open
            transaction is rolled back when its connection closes.
    """
    for attempt in range(PROJECT_LOCK_ATTEMPTS):
        try:
            connection.execute("COMMIT")
            return
        except sqlite3.OperationalError as error:
            if not _is_lock_error(error) or not connection.in_transaction:
                raise
            if attempt + 1 == PROJECT_LOCK_ATTEMPTS:
                raise _busy_error(path, error, timeout_s) from error
        _pause(attempt)
        set_busy_deadline(connection, deadline=time.monotonic() + timeout_s)


def _finish_read(connection: sqlite3.Connection) -> None:
    """End a read snapshot, discarding it exactly as closing a read-only connection would."""
    if connection.in_transaction:
        connection.execute("ROLLBACK")


def _detach_all(connection: sqlite3.Connection) -> None:
    """Detach caller-attached databases so a reused connection starts with only its own."""
    for _, name, _ in connection.execute("PRAGMA database_list").fetchall():
        if name not in ("main", "temp"):
            connection.execute(f'DETACH DATABASE "{name}"')


@contextmanager
def project_connection(
    root: Path, *, write: bool = False, timeout_s: float = PROJECT_BUSY_TIMEOUT_S
) -> Iterator[sqlite3.Connection]:
    """Open a bounded read snapshot or reentrant atomic project mutation.

    No provider call may execute inside a write transaction. Nested operations use
    savepoints so an intercepted error cannot commit a partial nested mutation.
    Each thread reuses one connection through ``persistent_connection``; it is
    configured and validated once and revalidated only when the schema changes. A
    transaction admission that finds the database locked past its wait retries with
    jittered backoff, up to ``PROJECT_LOCK_ATTEMPTS`` attempts, before any caller
    statement runs; a COMMIT refused for a lock is retried while still uncommitted.

    Args:
        root: EXP artifact root owning the shared content database.
        write: Create project tables and acquire a short write transaction.
        timeout_s: SQLite lock wait of one attempt; the first attempt shares it with
            local writer admission.

    Yields:
        The connection for the current transaction.

    Raises:
        ProjectDatabaseBusyError: The database stayed locked through every attempt.
    """
    path = content_database_path(root)
    if path.parent.is_symlink() or path.is_symlink():
        raise ValueError("Content database and its directory must not be symlinks.")
    active = _transactions.active.get(path)
    if active is not None:
        if write and not active.write:
            raise ValueError("A read snapshot cannot be upgraded to a project write transaction.")
        if not write:
            yield active.connection
            return
        active.connection.execute("SAVEPOINT project_nested")
        try:
            yield active.connection
        except BaseException:
            active.connection.execute("ROLLBACK TO project_nested")
            active.connection.execute("RELEASE project_nested")
            raise
        else:
            active.connection.execute("RELEASE project_nested")
        return
    timeout_s = max(0.0, timeout_s)
    deadline = time.monotonic() + timeout_s
    admission = writer_turn(path, timeout_s=timeout_s) if write else nullcontext(timeout_s)
    with admission:
        if write:
            _create_database(path)
        elif path.is_symlink():
            raise ValueError("Content database must not be a symlink.")
        with persistent_connection(
            path,
            busy_timeout_ms=int(PROJECT_BUSY_TIMEOUT_S * 1_000),
            enable_wal=False,
            create=False,
            factory=_ProjectConnection,
        ) as connection:
            if not isinstance(connection, _ProjectConnection):
                raise TypeError("project connections must use the project connection class")
            connection.row_factory = None
            _admit(connection, path, write=write, deadline=deadline, timeout_s=timeout_s)
            _transactions.active[path] = _Transaction(connection, write)
            try:
                yield connection
                if write:
                    _commit(connection, path, timeout_s=timeout_s)
                else:
                    _finish_read(connection)
            finally:
                _transactions.active.pop(path, None)
            _detach_all(connection)


def has_project_schema(root: Path) -> bool:
    """Check for project state without creating a file or schema."""
    path = content_database_path(root)
    if path.is_symlink() or path.parent.is_symlink():
        raise ValueError("Content database and its directory must not be symlinks.")
    if not path.exists():
        return False
    with project_connection(root) as connection:
        return (
            connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='project_store_schema'"
            ).fetchone()
            is not None
        )
