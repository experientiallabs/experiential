"""Tests for guarded SQLite initialization and forward migration."""

from __future__ import annotations

import os
import sqlite3
import stat
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from wmo.runtime.gateway.sqlite.migrations import (
    _MIGRATION_1,
    SCHEMA_VERSION,
    GatewaySchemaError,
    connect_database,
    initialize_database,
)


def test_initial_database_is_private_wal_with_foreign_keys(tmp_path: Path) -> None:
    """Fresh state enables WAL, foreign keys, bounded busy waits, and mode 0600."""
    path = tmp_path / "gateway.db"

    assert initialize_database(path) is None
    connection = connect_database(path, busy_timeout_ms=321)
    try:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert connection.execute("PRAGMA busy_timeout").fetchone()[0] == 321
    finally:
        connection.close()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_forward_migration_creates_consistent_private_backup(tmp_path: Path) -> None:
    """Existing old schemas are backed up before the next forward-only migration."""
    path = tmp_path / "gateway.db"
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(descriptor)
    connection = connect_database(path)
    try:
        connection.execute("BEGIN EXCLUSIVE")
        for statement in _MIGRATION_1:
            connection.execute(statement)
        connection.execute("PRAGMA user_version = 1")
        connection.execute("COMMIT")
    finally:
        connection.close()

    backup = initialize_database(path)

    assert backup is not None
    assert backup.exists()
    assert stat.S_IMODE(backup.stat().st_mode) == 0o600
    backup_connection = sqlite3.connect(backup)
    try:
        assert backup_connection.execute("PRAGMA user_version").fetchone()[0] == 1
        assert (
            backup_connection.execute(
                "SELECT 1 FROM sqlite_master WHERE name = 'gateway_requests'"
            ).fetchone()
            is None
        )
    finally:
        backup_connection.close()


def test_concurrent_initializers_choose_migration_plan_under_exclusive_lock(
    tmp_path: Path,
) -> None:
    """Concurrent initializers re-read schema version after exclusive serialization."""
    path = tmp_path / "gateway.db"
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(descriptor)
    connection = connect_database(path)
    try:
        connection.execute("BEGIN EXCLUSIVE")
        for statement in _MIGRATION_1:
            connection.execute(statement)
        connection.execute("PRAGMA user_version = 1")
        connection.execute("COMMIT")
    finally:
        connection.close()
    barrier = threading.Barrier(2)

    def initialize() -> Path | None:
        """Start one initializer at the same concurrency boundary."""
        barrier.wait(timeout=5)
        return initialize_database(path, busy_timeout_ms=10_000)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = (executor.submit(initialize), executor.submit(initialize))
        results = tuple(future.result(timeout=15) for future in futures)

    assert sum(result is not None for result in results) == 1
    connection = connect_database(path)
    try:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    finally:
        connection.close()


def test_newer_and_marker_only_schemas_refuse_without_deleting_state(tmp_path: Path) -> None:
    """Unknown future versions and missing schema objects fail closed."""
    newer = tmp_path / "newer.db"
    descriptor = os.open(newer, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(descriptor)
    connection = sqlite3.connect(newer)
    connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")
    connection.close()

    with pytest.raises(GatewaySchemaError, match="newer"):
        initialize_database(newer)
    assert newer.exists()
    connection = sqlite3.connect(newer)
    try:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
    finally:
        connection.close()

    marker_only = tmp_path / "marker-only.db"
    descriptor = os.open(marker_only, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(descriptor)
    connection = sqlite3.connect(marker_only)
    connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    connection.close()
    with pytest.raises(GatewaySchemaError, match="marker"):
        initialize_database(marker_only)

    corrupt = tmp_path / "corrupt.db"
    corrupt.write_bytes(b"not a sqlite database")
    corrupt.chmod(0o600)
    with pytest.raises(GatewaySchemaError, match="corrupt"):
        initialize_database(corrupt)
