"""Crash-recoverable publication of a verified project into the shared database."""

from __future__ import annotations

import os
import sqlite3

from exp.common.project.artifact_files import (
    _fsync_directory_strict,
    _read_artifact_file_snapshot,
)
from exp.common.project.database import content_database_path, project_connection
from exp.common.project.paths import ProjectPaths
from exp.common.project.records import ProjectRecords

_TABLES = (
    "project_config_versions",
    "project_config_heads",
    "project_artifacts",
    "project_artifact_inputs",
    "project_artifact_files",
    "project_state_records",
    "project_state_events",
)
_NAMESPACE = "bundle-restore"
_PENDING = "pending-publication"


def _recovery(paths: ProjectPaths) -> ProjectRecords:
    """Bind the one durable publication intent for an unselected destination."""
    return ProjectRecords(paths.root, paths.project_id, _NAMESPACE)


def check_restore_destination(paths: ProjectPaths, bundle_sha256: str) -> None:
    """Allow only an absent destination or a recoverable publication of this exact bundle."""
    destination = paths.project_directory
    if destination.is_symlink() or (
        destination.exists()
        and (
            not destination.is_dir()
            or _recovery(paths).read(_PENDING) != bundle_sha256.encode("ascii")
        )
    ):
        raise ValueError(
            "restore destination must be absent or belong to this interrupted bundle: "
            f"{destination}"
        )


def _require_unselected(connection: sqlite3.Connection, paths: ProjectPaths) -> None:
    """Refuse existing project rows, excluding only this restore's durable intent."""
    for table in _TABLES:
        extra = " AND NOT (namespace=? AND record_id=?)" if table == "project_state_records" else ""
        parameters = (paths.project_id, _NAMESPACE, _PENDING) if extra else (paths.project_id,)
        if connection.execute(
            f"SELECT 1 FROM {table} WHERE project_id=?{extra} LIMIT 1", parameters
        ).fetchone():
            raise ValueError("restore destination already contains project state")


def _files(paths: ProjectPaths) -> dict[str, bytes | None]:
    """Snapshot only regular descendants, rejecting symlinks and special files."""
    result: dict[str, bytes | None] = {}
    for entry in paths.project_directory.rglob("*"):
        name = entry.relative_to(paths.project_directory).as_posix()
        if entry.is_symlink():
            raise ValueError("restore publication contains a symlink")
        if entry.is_dir():
            result[name] = None
        elif entry.is_file():
            result[name] = _read_artifact_file_snapshot(
                paths.root, entry.relative_to(paths.root).as_posix()
            )
        else:
            raise ValueError("restore publication contains a non-regular file")
    return result


def publish_restored_project(
    staged: ProjectPaths, destination: ProjectPaths, *, bundle_sha256: str
) -> None:
    """Publish files under a durable intent, then atomically install verified project rows.

    A crash before the final commit leaves the exact bundle digest in SQLite. A retry
    verifies the published bytes against fresh verified staging before installing rows.
    The intent is deleted in the same commit that selects the restored configuration.
    Existing capture, trace and other project rows remain untouched.

    Args:
        staged: Independently verified staging root and project identity.
        destination: Shared destination root and the same project identity.
        bundle_sha256: Exact verified archive digest authorizing recoverable publication.
    """
    if staged.project_id != destination.project_id:
        raise ValueError("restored project identity changed before publication")
    pending = _recovery(destination)
    digest = bundle_sha256.encode("ascii")
    with project_connection(destination.root, write=True) as connection:
        _require_unselected(connection, destination)
        check_restore_destination(destination, bundle_sha256)
        previous = pending.read(_PENDING)
        if previous is not None and previous != digest:
            raise ValueError("restore destination has an interrupted different bundle")
        pending.write(_PENDING, digest)
    with project_connection(destination.root, write=True) as connection:
        _require_unselected(connection, destination)
        check_restore_destination(destination, bundle_sha256)
        connection.execute(
            "ATTACH DATABASE ? AS restored",
            (f"{content_database_path(staged.root).as_uri()}?mode=ro",),
        )
        if destination.project_directory.exists():
            if _files(staged) != _files(destination):
                raise ValueError("interrupted restore publication differs from verified bundle")
        else:
            os.rename(staged.project_directory, destination.project_directory)
        _fsync_directory_strict(destination.projects_directory)
        pending.replace(_PENDING, expected=digest, replacement=None)
        for table in _TABLES:
            connection.execute(
                f"INSERT INTO main.{table} SELECT * FROM restored.{table} WHERE project_id=?",
                (destination.project_id,),
            )
