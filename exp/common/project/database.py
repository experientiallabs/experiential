"""Short transactional access to project state in the shared content database."""

from __future__ import annotations

import os
import sqlite3
import stat
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path

from exp.common.project.sqlite_schema import initialize_project_schema
from exp.common.sqlite.connection import enable_wal_mode, set_busy_deadline
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
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
    except FileExistsError:
        pass
    else:
        os.close(descriptor)
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode) or stat.S_IMODE(metadata.st_mode) & 0o077:
        raise ValueError("Content database must be a private regular file.")


@contextmanager
def project_connection(
    root: Path, *, write: bool = False, timeout_s: float = 5.0
) -> Iterator[sqlite3.Connection]:
    """Open a bounded read snapshot or reentrant atomic project mutation.

    No provider call may execute inside a write transaction. Nested operations use
    savepoints so an intercepted error cannot commit a partial nested mutation.

    Args:
        root: EXP artifact root owning the shared content database.
        write: Create project tables and acquire a short write transaction.
        timeout_s: Maximum SQLite lock wait, shared with any outer admission deadline.

    Yields:
        The connection for the current transaction.
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
    deadline = time.monotonic() + max(0.0, timeout_s)
    admission = writer_turn(path, timeout_s=timeout_s) if write else nullcontext(timeout_s)
    with admission as remaining_s:
        if write:
            _create_database(path)
        elif path.is_symlink():
            raise ValueError("Content database must not be a symlink.")
        mode = "rw" if write else "ro"
        connection = sqlite3.connect(
            f"{path.as_uri()}?mode={mode}",
            uri=True,
            timeout=max(0.0, remaining_s),
            isolation_level=None,
        )
        try:
            connection.execute("PRAGMA foreign_keys=ON")
            validate_content_tables(connection)
            if write:
                enable_wal_mode(connection, deadline=deadline)
                connection.execute("PRAGMA synchronous=FULL")
            set_busy_deadline(connection, deadline=deadline)
            connection.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            tables = validate_content_tables(connection)
            if write and "project_store_schema" not in tables:
                initialize_project_schema(connection)
            _transactions.active[path] = _Transaction(connection, write)
            try:
                yield connection
                if write:
                    connection.execute("COMMIT")
            finally:
                _transactions.active.pop(path, None)
        finally:
            connection.close()


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
