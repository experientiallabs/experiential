"""Explicit domain ownership for the shared local content database."""

from __future__ import annotations

import sqlite3

from exp.common.sqlite.content_tables import PROJECT_TABLE_SQL, TRACE_TABLE_SQL

TRACE_TABLES = frozenset(TRACE_TABLE_SQL)
PROJECT_TABLES = frozenset(PROJECT_TABLE_SQL)
CONTENT_TABLES = TRACE_TABLES | PROJECT_TABLES | {"gateway_captures"}


class ContentSchemaError(ValueError):
    """The database contains unrecognized or incomplete durable domain state."""


def validate_content_tables(connection: sqlite3.Connection) -> frozenset[str]:
    """Reject unrelated tables and incomplete versioned domains before writes.

    Args:
        connection: Open connection to the shared content database.

    Returns:
        Complete recognized table names.
    """
    tables = frozenset(
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
    )
    if tables - CONTENT_TABLES:
        raise ContentSchemaError(
            "Unrecognized traffic database; preserve it and select another --root."
        )
    for domain, definitions in (("trace", TRACE_TABLE_SQL), ("project", PROJECT_TABLE_SQL)):
        owned = frozenset(definitions)
        marker = f"{domain}_store_schema"
        present = tables & owned
        if present and present != owned:
            raise ContentSchemaError(
                f"Incomplete {domain} database schema; preserve it and select another --root."
            )
        for table in present:
            saved_sql = connection.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,)
            ).fetchone()[0]
            if (
                " ".join(saved_sql.split()).casefold()
                != " ".join(definitions[table].split()).casefold()
            ):
                raise ContentSchemaError(
                    f"Incompatible {domain} table definition; "
                    "preserve it and select another --root."
                )
        if present:
            versions = tuple(row[0] for row in connection.execute(f"SELECT version FROM {marker}"))
            if versions != (1,):
                raise ContentSchemaError(
                    f"Unsupported {domain} schema version; use a matching Experiential release."
                )
    return tables
