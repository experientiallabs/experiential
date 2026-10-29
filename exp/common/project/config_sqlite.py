"""Immutable configuration versions and atomic selection of each project's current version."""

import hashlib
from pathlib import Path

from exp.common.project.database import has_project_schema, project_connection
from exp.common.project.paths import ProjectPaths
from exp.common.project.records import verified_payload


def require_current_layout(paths: ProjectPaths) -> None:
    """Preserve unsupported folder metadata instead of silently initializing over it."""
    if (paths.project_directory / "project.toml").exists() or (
        paths.project_directory / "artifacts"
    ).exists():
        raise ValueError(
            "This project uses the unsupported folder layout. Preserve it with its matching "
            "Experiential release; restore a verified project bundle into a fresh root."
        )


def config_exists(paths: ProjectPaths) -> bool:
    """Check committed configuration without creating the database."""
    require_current_layout(paths)
    if not has_project_schema(paths.root):
        return False
    with project_connection(paths.root) as connection:
        return (
            connection.execute(
                "SELECT 1 FROM project_config_heads WHERE project_id=?", (paths.project_id,)
            ).fetchone()
            is not None
        )


def read_config(paths: ProjectPaths, *, sha256: str | None = None) -> bytes:
    """Read the selected configuration or an exact frozen configuration digest."""
    require_current_layout(paths)
    if not has_project_schema(paths.root):
        raise ValueError("Project configuration does not exist; initialize the project first.")
    with project_connection(paths.root) as connection:
        if sha256 is None:
            row = connection.execute(
                "SELECT v.payload, v.sha256 FROM project_config_versions v JOIN "
                "project_config_heads h "
                "ON v.project_id=h.project_id AND v.version=h.version WHERE h.project_id=?",
                (paths.project_id,),
            ).fetchone()
        else:
            row = connection.execute(
                "SELECT payload, sha256 FROM project_config_versions WHERE project_id=? AND "
                "sha256=?",
                (paths.project_id, sha256),
            ).fetchone()
    if row is None:
        raise ValueError("Project configuration version does not exist; use a committed snapshot.")
    return verified_payload(row[0], row[1])


def write_config(paths: ProjectPaths, payload: bytes) -> str:
    """Retain an exact configuration version and atomically advance its project head."""
    require_current_layout(paths)
    digest = hashlib.sha256(payload).hexdigest()
    with project_connection(paths.root, write=True) as connection:
        row = connection.execute(
            "SELECT version, payload FROM project_config_versions WHERE project_id=? AND sha256=?",
            (paths.project_id, digest),
        ).fetchone()
        if row is None:
            version = connection.execute(
                "SELECT COALESCE(MAX(version), 0)+1 FROM project_config_versions WHERE "
                "project_id=?",
                (paths.project_id,),
            ).fetchone()[0]
            connection.execute(
                "INSERT INTO project_config_versions VALUES (?, ?, ?, ?)",
                (paths.project_id, version, digest, payload),
            )
        else:
            version = row[0]
            if row[1] != payload:
                raise ValueError("Project configuration conflicts with its immutable digest.")
        connection.execute(
            "INSERT INTO project_config_heads VALUES (?, ?) ON CONFLICT(project_id) "
            "DO UPDATE SET version=excluded.version",
            (paths.project_id, version),
        )
    return digest


def list_projects(root: Path) -> tuple[str, ...]:
    """List committed project identities without interpreting loose folders as state."""
    if not has_project_schema(root):
        return ()
    with project_connection(root) as connection:
        return tuple(
            row[0]
            for row in connection.execute(
                "SELECT project_id FROM project_config_heads ORDER BY project_id"
            )
        )
