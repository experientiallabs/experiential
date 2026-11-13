"""Bound descriptor retention across local ledgers without closing active transactions."""

import sqlite3
from pathlib import Path

import pytest

from exp.common.sqlite import connection


def test_idle_database_cache_evicts_the_least_recently_used_connection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Visiting many workspaces retains only a bounded set of idle database handles."""
    connection.close_idle_connections()
    monkeypatch.setattr(connection, "_MAX_IDLE_CONNECTIONS", 2)
    try:
        with connection.persistent_connection(tmp_path / "first.db") as first:
            assert first.execute("select 1").fetchone()[0] == 1
        with connection.persistent_connection(tmp_path / "second.db") as second:
            pass
        with connection.persistent_connection(tmp_path / "first.db") as reused:
            assert reused is first
        with connection.persistent_connection(tmp_path / "third.db"):
            pass
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            second.execute("select 1")
        assert first.execute("select 1").fetchone()[0] == 1
        assert connection.close_idle_connections() == 2
    finally:
        connection.close_idle_connections()


def test_idle_eviction_never_closes_a_checked_out_transaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A nested database checkout cannot evict the caller's active write fence."""
    connection.close_idle_connections()
    monkeypatch.setattr(connection, "_MAX_IDLE_CONNECTIONS", 1)
    try:
        with connection.persistent_connection(tmp_path / "active.db") as active:
            active.execute("begin immediate")
            for name in ("one", "two", "three"):
                with connection.persistent_connection(tmp_path / f"{name}.db"):
                    pass
            assert active.in_transaction
            active.execute("commit")
        assert connection.close_idle_connections() == 1
    finally:
        connection.close_idle_connections()
