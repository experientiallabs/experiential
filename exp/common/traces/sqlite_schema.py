"""Owned SQLite tables for canonical trace imports beside native gateway captures."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from exp.common.sqlite.content_tables import TRACE_TABLE_SQL
from exp.common.sqlite.schema import ContentSchemaError, validate_content_tables


class TraceStoreError(ValueError):
    """A trace database cannot be used without changing or losing stored evidence."""


def trace_database_path(root: Path) -> Path:
    """Return the shared content database used by capture and local ingestion."""
    return (root / "gateway" / "traffic.db").resolve()


def validate_schema(connection: sqlite3.Connection) -> None:
    """Reject unrelated or partially initialized databases before any mutation.

    Args:
        connection: Open connection to the proposed content database.

    Raises:
        TraceStoreError: Existing tables or the trace schema version are unsupported.
    """
    try:
        validate_content_tables(connection)
    except ContentSchemaError as exc:
        raise TraceStoreError(str(exc)) from exc


def initialize_schema(connection: sqlite3.Connection) -> None:
    """Create trace tables inside the caller's transaction, leaving capture rows untouched."""
    for statement in TRACE_TABLE_SQL.values():
        connection.execute(statement.replace("CREATE TABLE ", "CREATE TABLE IF NOT EXISTS ", 1))
    connection.execute("INSERT OR IGNORE INTO trace_store_schema VALUES (1)")
