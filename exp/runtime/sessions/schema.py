"""Schema identity of the durable session database, created once and refused when stale.

Schema 4 stores candidate state incrementally: immutable content-addressed nodes, an immutable
per-session event log from genesis, revision-bound heads, retention pins and a resumable
collection cursor. Earlier schemas kept one complete state document per session and recorded no
replayable genesis or effects, so they cannot be converted without inventing history; they are
refused with an explicit instruction instead.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from uuid import uuid4

from exp.runtime.sessions.errors import SessionStoreError

SCHEMA_VERSION = 4
"""Schema identity stored in ``PRAGMA user_version``. Mismatches fail closed."""

STATE_CODEC = "json-nodes-v1"
"""Identity of the node codec and edit interpreter every stored root was produced by."""

_SCHEMA_STATEMENTS = (
    """CREATE TABLE store_identity (
        singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
        store_id TEXT NOT NULL,
        codec TEXT NOT NULL
    ) STRICT""",
    """CREATE TRIGGER store_identity_immutable BEFORE UPDATE ON store_identity
    BEGIN SELECT RAISE(ABORT, 'store identity is immutable'); END""",
    """CREATE TABLE sessions (
        session_id TEXT PRIMARY KEY,
        binding_json TEXT NOT NULL,
        version INTEGER NOT NULL CHECK (version >= 0),
        root BLOB NOT NULL CHECK (length(root) = 32),
        draining INTEGER NOT NULL CHECK (draining IN (0, 1))
    ) STRICT""",
    """CREATE TRIGGER sessions_binding_immutable BEFORE UPDATE OF binding_json ON sessions
    BEGIN SELECT RAISE(ABORT, 'session binding is immutable'); END""",
    """CREATE TABLE state_nodes (
        node_id INTEGER PRIMARY KEY AUTOINCREMENT,
        digest BLOB NOT NULL UNIQUE CHECK (length(digest) = 32),
        template TEXT NOT NULL,
        children BLOB NOT NULL CHECK (length(children) % 32 = 0)
    ) STRICT""",
    """CREATE TABLE state_events (
        session_id TEXT NOT NULL REFERENCES sessions(session_id),
        revision INTEGER NOT NULL CHECK (revision >= 0),
        step_id TEXT,
        effect_json TEXT NOT NULL,
        root BLOB NOT NULL CHECK (length(root) = 32),
        PRIMARY KEY (session_id, revision),
        UNIQUE (session_id, step_id)
    ) STRICT""",
    """CREATE TRIGGER state_events_no_update BEFORE UPDATE ON state_events
    BEGIN SELECT RAISE(ABORT, 'state events are immutable'); END""",
    """CREATE TRIGGER state_events_no_delete BEFORE DELETE ON state_events
    BEGIN SELECT RAISE(ABORT, 'state events are immutable'); END""",
    """CREATE TABLE state_pins (
        owner_id TEXT NOT NULL,
        session_id TEXT NOT NULL REFERENCES sessions(session_id),
        start_revision INTEGER NOT NULL CHECK (start_revision >= 0),
        end_revision INTEGER NOT NULL CHECK (end_revision > start_revision),
        PRIMARY KEY (owner_id, session_id)
    ) STRICT""",
    """CREATE TABLE state_collection (
        singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
        high_water INTEGER NOT NULL,
        cursor INTEGER NOT NULL
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
    """CREATE UNIQUE INDEX steps_one_open_per_session
        ON steps(session_id) WHERE status IN ('claimed', 'provider_started', 'ambiguous')""",
    """CREATE TABLE audit (
        seq INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        step_id TEXT,
        event TEXT NOT NULL,
        detail_sha256 TEXT NOT NULL,
        recorded_at TEXT NOT NULL
    ) STRICT""",
)


def refuse_stale_schema(path: Path) -> None:
    """Refuse an existing database of another schema before anything writes to it.

    The check uses a read-only connection, so a refused file keeps every byte, including its
    journal mode, for the release that wrote it.

    Args:
        path: Database file; a missing file is accepted and created later.

    Raises:
        SessionStoreError: If the file has another nonzero schema version.
    """
    if not path.exists():
        return
    connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    try:
        version = connection.execute("PRAGMA user_version").fetchone()[0]
    finally:
        connection.close()
    if version not in (0, SCHEMA_VERSION):
        raise SessionStoreError(_stale_message(path, version))


def is_session_database(path: Path) -> bool:
    """Whether ``path`` is a session database of this schema, checked read-only.

    Other SQLite files (dispatch ledgers, project metadata) share directories with session
    stores; this lets a caller find the stores without opening, and so initializing, anything.

    Args:
        path: Existing file.

    Returns:
        True for a schema-4 store with its identity table.
    """
    try:
        connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    except sqlite3.Error:
        return False
    try:
        if connection.execute("PRAGMA user_version").fetchone()[0] != SCHEMA_VERSION:
            return False
        tables = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
    except sqlite3.DatabaseError:
        return False
    finally:
        connection.close()
    return {"store_identity", "state_pins", "state_nodes"} <= tables


def _stale_message(path: Path, version: int) -> str:
    """Explain a schema refusal and what to do about it."""
    return (
        f"session database {path} uses schema {version}, but this release stores "
        f"incremental state under schema {SCHEMA_VERSION}. Earlier schemas keep no "
        "replayable genesis or effects, so they are not converted. Preserve the file for "
        "the release that wrote it, and point this run at a fresh session database path."
    )


def open_schema(connection: sqlite3.Connection, path: Path) -> str:
    """Create the schema in an empty database or verify an existing one.

    Args:
        connection: Open connection outside any transaction.
        path: Database file, named in refusals.

    Returns:
        The store's immutable identity, bound into every state reference it issues.

    Raises:
        SessionStoreError: If the database has another schema or codec.
    """
    version = connection.execute("PRAGMA user_version").fetchone()[0]
    if version == 0:
        connection.execute("BEGIN IMMEDIATE")
        try:
            if connection.execute("PRAGMA user_version").fetchone()[0] == 0:
                for statement in _SCHEMA_STATEMENTS:
                    connection.execute(statement)
                connection.execute(
                    "INSERT INTO store_identity (singleton, store_id, codec) VALUES (1, ?, ?)",
                    (uuid4().hex, STATE_CODEC),
                )
                connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        except BaseException:
            connection.execute("ROLLBACK")
            raise
        connection.execute("COMMIT")
        version = SCHEMA_VERSION
    if version != SCHEMA_VERSION:
        raise SessionStoreError(_stale_message(path, version))
    row = connection.execute("SELECT store_id, codec FROM store_identity").fetchone()
    if row is None or row["codec"] != STATE_CODEC:
        raise SessionStoreError(
            f"session database {path} has no {STATE_CODEC} store identity; "
            "use a fresh session database path"
        )
    return str(row["store_id"])
