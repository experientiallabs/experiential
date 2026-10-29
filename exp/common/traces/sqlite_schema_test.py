"""Traffic-schema ownership is checked before SQLite settings or tables can change."""

import sqlite3
from pathlib import Path

import pytest

from exp.common.traces.sqlite import SQLiteTraceStore
from exp.common.traces.sqlite_schema import TraceStoreError, initialize_schema, validate_schema
from exp.common.traces.sqlite_test import _save
from exp.common.traces.trace_test import _trace


@pytest.mark.parametrize(
    "schema, message",
    [
        ("CREATE TABLE other(payload TEXT)", "Unrecognized"),
        ("CREATE TABLE trace_records(payload TEXT)", "Incomplete"),
    ],
)
def test_unrecognized_database_is_preserved(tmp_path: Path, schema: str, message: str) -> None:
    """Unrelated evidence is not converted or changed into a trace store."""
    path = tmp_path / "existing.db"
    path.touch(mode=0o600)
    with sqlite3.connect(path) as connection:
        connection.execute(schema)
    before = path.read_bytes()
    with pytest.raises(TraceStoreError, match=message):
        _save(SQLiteTraceStore(path), (_trace(),))
    assert path.read_bytes() == before
    assert not Path(f"{path}-wal").exists()


def test_trace_schema_can_join_capture_and_rejects_unknown_version(tmp_path: Path) -> None:
    """Only the complete, supported trace namespace may coexist with captures."""
    with sqlite3.connect(tmp_path / "traffic.db") as connection:
        connection.execute("CREATE TABLE gateway_captures(payload TEXT)")
        connection.execute("INSERT INTO gateway_captures VALUES ('existing capture')")
        initialize_schema(connection)
        validate_schema(connection)
        assert connection.execute("SELECT payload FROM gateway_captures").fetchone() == (
            "existing capture",
        )
        connection.execute("PRAGMA ignore_check_constraints=ON")
        connection.execute("UPDATE trace_store_schema SET version=2")
        with pytest.raises(TraceStoreError, match="Unsupported"):
            validate_schema(connection)


@pytest.mark.parametrize(
    "change",
    [
        "ALTER TABLE trace_records ADD COLUMN injected TEXT",
        "DROP TABLE trace_project_imports; CREATE TABLE trace_project_imports "
        "(sequence INTEGER PRIMARY KEY AUTOINCREMENT, "
        "project_id TEXT NOT NULL, import_id TEXT NOT NULL)",
        "DROP TABLE trace_records; CREATE TABLE trace_records "
        "(record_sha256 TEXT PRIMARY KEY, trace_id TEXT NOT NULL, payload TEXT NOT NULL) STRICT",
    ],
)
def test_complete_names_do_not_accept_incompatible_definitions(tmp_path: Path, change: str) -> None:
    """Matching names/version cannot hide missing constraints or a changed table shape."""
    path = tmp_path / "traffic.db"
    path.touch(mode=0o600)
    with sqlite3.connect(path) as connection:
        initialize_schema(connection)
        connection.executescript(change)
    before = path.read_bytes()
    with pytest.raises(TraceStoreError, match="Incompatible trace table"):
        _save(SQLiteTraceStore(path), (_trace(),))
    assert path.read_bytes() == before
