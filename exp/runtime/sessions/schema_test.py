"""Tests for the session database schema: creation, identity and the hard cutover refusal."""

from __future__ import annotations

import shutil
import sqlite3
import threading
from pathlib import Path

import pytest

from exp.common.sqlite.connection import connect_database
from exp.runtime.sessions.errors import SessionStoreError
from exp.runtime.sessions.schema import (
    SCHEMA_VERSION,
    STATE_CODEC,
    is_session_database,
    open_schema,
)
from exp.runtime.sessions.store import SessionStore

FULL_STATE_SCHEMA = (
    """CREATE TABLE sessions (
        session_id TEXT PRIMARY KEY,
        binding_json TEXT NOT NULL,
        version INTEGER NOT NULL CHECK (version >= 0),
        state_json TEXT NOT NULL,
        draining INTEGER NOT NULL CHECK (draining IN (0, 1))
    ) STRICT""",
    """CREATE TABLE steps (
        session_id TEXT NOT NULL REFERENCES sessions(session_id),
        step_id TEXT NOT NULL,
        request_sha256 TEXT NOT NULL,
        base_version INTEGER NOT NULL,
        status TEXT NOT NULL,
        claim_token TEXT NOT NULL,
        proposal_json TEXT,
        response_json TEXT,
        resulting_version INTEGER,
        PRIMARY KEY (session_id, step_id)
    ) STRICT""",
)
"""The previous full-state layout, which kept only one complete state per session."""


def test_a_full_state_database_is_refused_with_instructions_and_left_untouched(
    tmp_path: Path,
) -> None:
    """A schema 3 store keeps no genesis or effects, so it is refused rather than converted."""
    path = tmp_path / "sessions.sqlite"
    raw = sqlite3.connect(path)
    for statement in FULL_STATE_SCHEMA:
        raw.execute(statement)
    raw.execute("INSERT INTO sessions VALUES ('s', '{}', 4, '{\"business_state\": {}}', 0)")
    raw.execute("PRAGMA user_version = 3")
    raw.commit()
    raw.close()
    preserved = tmp_path / "preserved.sqlite"
    shutil.copyfile(path, preserved)
    with pytest.raises(SessionStoreError, match="uses schema 3") as refusal:
        SessionStore(path)
    assert "fresh session database path" in str(refusal.value)
    assert "not converted" in str(refusal.value)
    assert path.read_bytes() == preserved.read_bytes()


def test_a_new_database_records_its_schema_codec_and_immutable_identity(tmp_path: Path) -> None:
    """Every store gets one identity for its references, and the identity cannot change."""
    path = tmp_path / "sessions.sqlite"
    first = SessionStore(path)
    assert SessionStore(path).store_id == first.store_id
    assert SessionStore(tmp_path / "other.sqlite").store_id != first.store_id
    with sqlite3.connect(path) as raw:
        assert raw.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        assert raw.execute("SELECT codec FROM store_identity").fetchone()[0] == STATE_CODEC
        with pytest.raises(sqlite3.IntegrityError):
            raw.execute("UPDATE store_identity SET store_id = 'forged'")


def test_a_database_with_another_codec_is_refused(tmp_path: Path) -> None:
    """A store whose roots another codec produced is never read as this codec."""
    path = tmp_path / "sessions.sqlite"
    SessionStore(path)
    with sqlite3.connect(path) as raw:
        raw.execute("DROP TRIGGER store_identity_immutable")
        raw.execute("UPDATE store_identity SET codec = 'json-nodes-v0'")
    connection = connect_database(path)
    try:
        with pytest.raises(SessionStoreError, match="store identity"):
            open_schema(connection, path)
    finally:
        connection.close()


def test_concurrent_first_opens_create_one_schema(tmp_path: Path) -> None:
    """Racing creators converge on one schema and one identity."""
    path = tmp_path / "sessions.sqlite"
    identities: list[str] = []
    barrier = threading.Barrier(4)

    def open_store() -> None:
        """Open the shared path once the other threads are ready."""
        barrier.wait()
        identities.append(SessionStore(path).store_id)

    threads = [threading.Thread(target=open_store) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(set(identities)) == 1 and len(identities) == 4


def test_session_databases_are_recognized_read_only_among_other_files(tmp_path: Path) -> None:
    """Ledgers, other schemas and non-SQLite files are not stores, and nothing is created."""
    store = tmp_path / "store.sqlite"
    SessionStore(store)
    ledger = tmp_path / "ledger.sqlite"
    with sqlite3.connect(ledger) as raw:
        raw.execute("CREATE TABLE slots (n INTEGER)")
        raw.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    other = tmp_path / "notes.sqlite"
    other.write_text("not a database")
    assert is_session_database(store)
    assert not is_session_database(ledger)
    assert not is_session_database(other)
    empty = tmp_path / "empty.sqlite"
    sqlite3.connect(empty).close()
    assert not is_session_database(empty)
    assert empty.stat().st_size == 0
