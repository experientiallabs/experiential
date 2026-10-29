"""Add nullable cache-write evidence without rewriting past accounting."""

import sqlite3

CACHE_WRITE_COLUMNS = (
    "cache_creation_input_rate",
    "cache_creation_1h_input_rate",
    "long_context_cache_creation_input_rate",
    "long_context_cache_creation_1h_input_rate",
    "preferred_cache_creation_input_rate",
    "preferred_cache_creation_1h_input_rate",
    "cache_creation_input_tokens",
    "cache_creation_1h_input_tokens",
)
"""Frozen nano-USD rates and observed token counts; existing rows remain NULL."""

CACHE_WRITE_MIGRATION = tuple(
    f"ALTER TABLE gateway_attempts ADD COLUMN {column} INTEGER "
    f"CHECK ({column} IS NULL OR {column} >= 0)"
    for column in CACHE_WRITE_COLUMNS
)


def migrate_cache_write(connection: sqlite3.Connection) -> None:
    """Unify the two published schema-22 layouts without rewriting attempt values.

    Schema 22 carries either upstream attribution or the complete cache-write
    extension. The missing nullable extension is added inside the caller's
    migration transaction; partial cache-write layouts fail closed.

    Args:
        connection: Ledger connection inside an exclusive migration transaction.

    Raises:
        sqlite3.DatabaseError: The cache-write layout is incomplete or DDL fails.
    """
    columns = {row[1] for row in connection.execute("PRAGMA table_info(gateway_attempts)")}
    existing = columns.intersection(CACHE_WRITE_COLUMNS)
    if existing and existing != set(CACHE_WRITE_COLUMNS):
        raise sqlite3.DatabaseError("gateway cache-write schema is incomplete")
    if not existing:
        for statement in CACHE_WRITE_MIGRATION:
            connection.execute(statement)
    if "upstream_provider" not in columns:
        connection.execute("ALTER TABLE gateway_attempts ADD COLUMN upstream_provider TEXT")
