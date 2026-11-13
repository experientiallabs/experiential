"""Shared SQLite locking, snapshots and externally readable durable writes."""

import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from exp.common.sqlite.native import connect


def test_native_connections_share_locks_and_transaction_snapshots(tmp_path: Path) -> None:
    """Two native connections see commits immediately and never bypass a live writer."""
    path = tmp_path / "db.sqlite"
    writer = connect(path)
    reader = connect(path, busy_timeout_ms=0)
    try:
        writer.executescript("PRAGMA journal_mode=WAL; CREATE TABLE facts(value INTEGER UNIQUE);")
        writer.execute("BEGIN IMMEDIATE")
        writer.execute("INSERT INTO facts VALUES (?)", (42,))
        assert reader.execute("SELECT * FROM facts").fetchall() == []
        with pytest.raises(sqlite3.OperationalError) as busy:
            reader.execute("INSERT INTO facts VALUES (7)")
        assert busy.value.sqlite_errorcode & 255 == sqlite3.SQLITE_BUSY
        writer.commit()
        assert reader.execute("SELECT * FROM facts").fetchall() == [(42,)]
        writer.execute("BEGIN IMMEDIATE")
        writer.execute("INSERT INTO facts VALUES (7)")
        writer.rollback()
        assert reader.execute("SELECT * FROM facts").fetchall() == [(42,)]
        with pytest.raises(sqlite3.IntegrityError):
            writer.execute("INSERT INTO facts VALUES (42)")
    finally:
        writer.close()
        reader.close()


def test_external_sqlite_process_reads_native_commit_with_writer_open(tmp_path: Path) -> None:
    """Independent tools can read and extend a live native WAL database."""
    path = tmp_path / "db.sqlite"
    writer = connect(path)
    try:
        writer.executescript("PRAGMA journal_mode=WAL; CREATE TABLE facts(value INTEGER);")
        writer.execute("INSERT INTO facts VALUES (42)")
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                """
import sqlite3, sys
with sqlite3.connect(sys.argv[1]) as connection:
    assert connection.execute('SELECT value FROM facts').fetchall() == [(42,)]
    connection.execute('INSERT INTO facts VALUES (7)')
""",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert writer.execute("SELECT value FROM facts ORDER BY value").fetchall() == [(7,), (42,)]
    finally:
        writer.close()


def test_native_cursor_rows_readonly_and_backup(tmp_path: Path) -> None:
    """Bound rows and incremental cursors preserve local store and backup contracts."""
    path = tmp_path / "db.sqlite"
    writer = connect(path)
    target = connect(tmp_path / "copy.sqlite")
    try:
        writer.execute("CREATE TABLE facts(value INTEGER PRIMARY KEY, payload BLOB)")
        writer.execute("BEGIN")
        assert (
            writer.executemany(
                "INSERT INTO facts VALUES (:value, :payload)",
                ({"value": i, "payload": b"\x00\xff"} for i in range(100)),
            ).rowcount
            == 100
        )
        writer.commit()
        writer.row_factory = sqlite3.Row
        cursor = writer.execute("SELECT value, payload FROM facts ORDER BY value")
        assert cursor.fetchone()["VALUE"] == 0
        assert len(cursor.fetchmany(10)) == 10
        row = cursor.fetchone()
        assert dict(row) == {"value": 11, "payload": b"\x00\xff"}
        cursor.close()
        with pytest.raises(sqlite3.ProgrammingError):
            cursor.fetchone()
        writer.backup(target)
        assert target.execute("SELECT count(*) FROM facts").fetchone() == (100,)
        readonly = connect(path, mode="ro")
        try:
            with pytest.raises(sqlite3.OperationalError):
                readonly.execute("DELETE FROM facts")
        finally:
            readonly.close()
    finally:
        writer.close()
        target.close()


def test_native_raw_connection_preserves_sqlite_foreign_key_opt_in(tmp_path: Path) -> None:
    """Raw migration readers match sqlite3, while explicitly enabled constraints still reject."""
    with connect(tmp_path / "foreign-keys.db") as connection:
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 0
        connection.execute("PRAGMA foreign_keys=ON")
        connection.executescript(
            "CREATE TABLE parent(id INTEGER PRIMARY KEY); "
            "CREATE TABLE child(parent_id INTEGER REFERENCES parent(id));"
        )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute("INSERT INTO child VALUES (1)")
