"""Fresh schema inspection preserves domain isolation without per-table SQL lookups."""

import sqlite3

import pytest

from exp.common.project.sqlite_schema import initialize_project_schema
from exp.common.sqlite.schema import CONTENT_TABLES, ContentSchemaError, validate_content_tables
from exp.common.traces.sqlite_schema import initialize_schema


def test_complete_domains_need_one_catalog_read_and_one_version_read_each() -> None:
    """The shared schema has a bounded query cost independent of its table count."""
    with sqlite3.connect(":memory:") as connection:
        initialize_project_schema(connection)
        initialize_schema(connection)
        connection.execute("CREATE TABLE gateway_captures(payload TEXT)")
        queries: list[str] = []
        connection.set_trace_callback(queries.append)
        assert validate_content_tables(connection) == CONTENT_TABLES
        assert len(queries) == 3


@pytest.mark.parametrize(
    "change, message",
    [
        ("CREATE TABLE unexpected(value TEXT)", "Unrecognized"),
        ("DROP TABLE project_artifacts", "Incomplete project"),
        ("ALTER TABLE project_artifacts ADD COLUMN injected TEXT", "Incompatible project"),
        ("DELETE FROM project_store_schema", "Unsupported project"),
    ],
)
def test_a_previous_valid_read_cannot_hide_later_schema_changes(change: str, message: str) -> None:
    """Every inspection reads current definitions and version rows; there is no trust cache."""
    with sqlite3.connect(":memory:") as connection:
        initialize_project_schema(connection)
        validate_content_tables(connection)
        connection.execute(change)
        with pytest.raises(ContentSchemaError, match=message):
            validate_content_tables(connection)
