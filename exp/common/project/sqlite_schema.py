"""Initialize only the independently versioned project metadata domain."""

import sqlite3

from exp.common.sqlite.content_tables import PROJECT_TABLE_SQL


def initialize_project_schema(connection: sqlite3.Connection) -> None:
    """Create project tables and lookup indexes inside the caller's write transaction."""
    for statement in PROJECT_TABLE_SQL.values():
        connection.execute(statement.replace("CREATE TABLE ", "CREATE TABLE IF NOT EXISTS ", 1))
    connection.execute("INSERT OR IGNORE INTO project_store_schema VALUES (1)")
    connection.execute(
        "CREATE INDEX IF NOT EXISTS project_artifact_type "
        "ON project_artifacts(project_id, artifact_type, artifact_id)"
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS project_artifact_lineage "
        "ON project_artifact_inputs(project_id, input_id, sha256)"
    )
