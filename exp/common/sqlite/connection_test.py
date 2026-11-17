"""Tests for reusable per-thread SQLite connections: bounded idle cache and fork safety."""

from __future__ import annotations

import os
import sqlite3
import threading
from pathlib import Path

import pytest

from exp.common.sqlite import connection as sqlite_connection
from exp.common.sqlite.connection import (
    MAXIMUM_IDLE_CONNECTIONS,
    close_idle_connections,
    persistent_connection,
)


@pytest.fixture(autouse=True)
def _fresh_cache() -> None:
    """Start and end every test with no idle connections on this thread."""
    close_idle_connections()


def test_idle_connections_per_thread_are_bounded_least_recently_used_first(
    tmp_path: Path,
) -> None:
    """Many databases on one thread keep at most the bound open, closing the oldest."""
    paths = [tmp_path / f"db-{index}.sqlite" for index in range(MAXIMUM_IDLE_CONNECTIONS + 3)]
    first = None
    for path in paths:
        with persistent_connection(path) as connection:
            connection.execute("SELECT 1")
            if first is None:
                first = connection
    assert first is not None
    with pytest.raises(Exception, match="closed"):
        first.execute("SELECT 1")
    assert close_idle_connections() == MAXIMUM_IDLE_CONNECTIONS


def test_a_reused_connection_is_refreshed_to_most_recent(tmp_path: Path) -> None:
    """Touching a connection again protects it from eviction."""
    keep = tmp_path / "keep.sqlite"
    with persistent_connection(keep) as kept:
        pass
    for index in range(MAXIMUM_IDLE_CONNECTIONS - 1):
        with persistent_connection(tmp_path / f"other-{index}.sqlite"):
            pass
    with persistent_connection(keep) as again:
        assert again is kept
    with persistent_connection(tmp_path / "one-more.sqlite"):
        pass
    with persistent_connection(keep) as still:
        assert still is kept


def test_a_forked_child_never_reuses_its_parents_connections(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After a fork the inherited idle map is forgotten instead of shared."""
    database = tmp_path / "parent.sqlite"
    with persistent_connection(database) as parent:
        pass
    child_pid = os.getpid() + 1
    monkeypatch.setattr(sqlite_connection.os, "getpid", lambda: child_pid)
    with persistent_connection(database) as child:
        assert child is not parent
    parent.execute("SELECT 1")
    parent.close()


def test_a_replaced_database_file_gets_a_fresh_connection(tmp_path: Path) -> None:
    """An idle connection to a deleted or replaced file is closed, never reused."""
    database = tmp_path / "replaced.sqlite"
    with persistent_connection(database) as original:
        original.execute("CREATE TABLE marker (value TEXT)")
    for suffix in ("", "-wal", "-shm"):
        database.with_name(database.name + suffix).unlink(missing_ok=True)
    with persistent_connection(database) as replacement:
        assert replacement is not original
        assert replacement.execute("SELECT name FROM sqlite_master").fetchall() == []
    with pytest.raises(Exception, match="closed"):
        original.execute("SELECT 1")
    with persistent_connection(database) as again:
        assert again is replacement


def test_connection_configuration_selects_a_separate_idle_slot(tmp_path: Path) -> None:
    """Checkouts with another class or setup never receive a differently configured handle."""

    class Owned(sqlite3.Connection):
        """Connection class of a hypothetical owner with its own per-connection state."""

    database = tmp_path / "shared.sqlite"
    with persistent_connection(database) as plain:
        pass
    with persistent_connection(database, enable_wal=False, factory=Owned) as owned:
        assert isinstance(owned, Owned)
        assert owned is not plain
    with pytest.raises(sqlite3.OperationalError, match="unable to open"):
        with persistent_connection(tmp_path / "absent.sqlite", create=False):
            pass
    assert not (tmp_path / "absent.sqlite").exists()


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires fork")
@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded:DeprecationWarning")
def test_a_real_fork_child_opens_its_own_connection_and_keeps_the_parents(
    tmp_path: Path,
) -> None:
    """A forked child writes through a new handle and never finalizes the inherited one."""
    database = tmp_path / "forked.sqlite"
    with persistent_connection(database) as parent:
        parent.execute("CREATE TABLE writes (origin TEXT)")
    child = os.fork()
    if child == 0:
        status = 1
        try:
            with persistent_connection(database) as connection:
                connection.execute("INSERT INTO writes VALUES ('child')")
                inherited = parent in sqlite_connection._inherited_connections
                status = 0 if connection is not parent and inherited else 2
        finally:
            os._exit(status)
    _, status = os.waitpid(child, 0)
    assert os.waitstatus_to_exitcode(status) == 0
    with persistent_connection(database) as connection:
        assert connection is parent
        connection.execute("INSERT INTO writes VALUES ('parent')")
        rows = connection.execute("SELECT origin FROM writes ORDER BY origin").fetchall()
    assert [row[0] for row in rows] == ["child", "parent"]


def test_concurrent_first_opens_all_converge_on_wal(tmp_path: Path) -> None:
    """Racing first opens retry the journal-mode switch instead of failing as locked."""
    modes: list[str] = []
    errors: list[sqlite3.Error] = []

    def open_once(database: Path, barrier: threading.Barrier) -> None:
        """Open and write the shared path as soon as every thread is ready."""
        barrier.wait()
        try:
            with persistent_connection(database) as connection:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute("CREATE TABLE IF NOT EXISTS marker (value TEXT)")
                connection.execute("COMMIT")
                modes.append(connection.execute("PRAGMA journal_mode").fetchone()[0])
            close_idle_connections()
        except sqlite3.Error as error:
            errors.append(error)

    for round_index in range(10):
        barrier = threading.Barrier(8)
        database = tmp_path / f"racing-{round_index}.sqlite"
        threads = [threading.Thread(target=open_once, args=(database, barrier)) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    assert errors == []
    assert modes == ["wal"] * 80
