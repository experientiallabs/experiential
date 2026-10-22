"""Tests for reusable per-thread SQLite connections: bounded idle cache and fork safety."""

from __future__ import annotations

import os
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
