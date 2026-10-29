"""Immutable project artifact records with exact manifests and external blob references."""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType

from exp.common.core.artifacts import canonical_json_bytes, validate_artifact_file_path
from exp.common.project.artifact_files import _read_artifact_file_snapshot, publish_blob
from exp.common.project.database import has_project_schema, project_connection
from exp.common.project.errors import ArtifactAlreadyExistsError, ArtifactCorruptionError
from exp.common.project.manifests import ArtifactManifest, artifact_input
from exp.common.project.paths import ProjectPaths
from exp.common.project.records import ProjectRecordError, verified_payload

INLINE_LIMIT_BYTES = 1_048_576


@dataclass(frozen=True)
class StoredArtifact:
    """One verified immutable manifest and an exact snapshot of its data.

    Attributes:
        manifest: Canonical manifest whose identity is independent of storage layout.
        payloads: Verified payload bytes keyed by manifest-relative name.
    """

    manifest: ArtifactManifest
    payloads: Mapping[str, bytes]


def write_artifact(
    paths: ProjectPaths, manifest: ArtifactManifest, files: Mapping[str, bytes]
) -> None:
    """Publish external bytes, then atomically commit the artifact and all its metadata."""
    prepared: list[tuple[str, str, int, bytes | None, str | None]] = []
    for entry in manifest.files:
        payload = files[entry.path]
        if len(payload) > INLINE_LIMIT_BYTES or Path(entry.path).suffix == ".html":
            digest = publish_blob(
                paths.root, PurePosixPath("projects") / paths.project_id / "blobs", payload
            )
            blob_path = f"projects/{paths.project_id}/blobs/{digest}"
            prepared.append((entry.path, entry.sha256, entry.size_bytes, None, blob_path))
        else:
            prepared.append((entry.path, entry.sha256, entry.size_bytes, payload, None))
    try:
        with project_connection(paths.root, write=True) as connection:
            if connection.execute(
                "SELECT 1 FROM project_artifacts WHERE project_id=? AND artifact_id=?",
                (paths.project_id, manifest.artifact_id),
            ).fetchone():
                raise ArtifactAlreadyExistsError(
                    f"completed artifact already exists and is immutable: {manifest.artifact_id}"
                )
            connection.execute(
                "INSERT INTO project_artifacts VALUES (?, ?, ?, ?, ?)",
                (
                    paths.project_id,
                    manifest.artifact_id,
                    manifest.artifact_type,
                    canonical_json_bytes(manifest),
                    artifact_input(manifest).sha256,
                ),
            )
            connection.executemany(
                "INSERT INTO project_artifact_inputs VALUES (?, ?, ?, ?, ?)",
                [
                    (paths.project_id, manifest.artifact_id, ordinal, item.artifact_id, item.sha256)
                    for ordinal, item in enumerate(manifest.inputs)
                ],
            )
            connection.executemany(
                "INSERT INTO project_artifact_files VALUES (?, ?, ?, ?, ?, ?, ?)",
                [(paths.project_id, manifest.artifact_id, *row) for row in prepared],
            )
    except sqlite3.IntegrityError as exc:
        raise ArtifactCorruptionError(
            "Artifact metadata could not be committed consistently."
        ) from exc


def read_artifact(paths: ProjectPaths, artifact_id: str) -> StoredArtifact:
    """Read one transactionally consistent manifest and verify every payload and reference."""
    if not has_project_schema(paths.root):
        raise ArtifactCorruptionError(f"completed artifact is missing: {artifact_id}")
    try:
        with project_connection(paths.root) as connection:
            row = connection.execute(
                "SELECT artifact_type, manifest, sha256 FROM project_artifacts WHERE "
                "project_id=? AND artifact_id=?",
                (paths.project_id, artifact_id),
            ).fetchone()
            if row is None:
                raise ArtifactCorruptionError(f"completed artifact is missing: {artifact_id}")
            manifest = ArtifactManifest.model_validate_json(row[1])
            if (
                manifest.artifact_id != artifact_id
                or manifest.artifact_type != row[0]
                or artifact_input(manifest).sha256 != row[2]
                or canonical_json_bytes(manifest) != row[1]
            ):
                raise ArtifactCorruptionError(f"artifact {artifact_id} manifest digest mismatch")
            inputs = connection.execute(
                "SELECT ordinal, input_id, sha256 FROM project_artifact_inputs "
                "WHERE project_id=? AND artifact_id=? ORDER BY ordinal",
                (paths.project_id, artifact_id),
            ).fetchall()
            if inputs != [
                (ordinal, item.artifact_id, item.sha256)
                for ordinal, item in enumerate(manifest.inputs)
            ]:
                raise ArtifactCorruptionError(
                    f"artifact {artifact_id} input references differ from its manifest"
                )
            rows = connection.execute(
                "SELECT path, sha256, size_bytes, payload, blob_path FROM project_artifact_files "
                "WHERE project_id=? AND artifact_id=? ORDER BY path",
                (paths.project_id, artifact_id),
            ).fetchall()
        if [(row[0], row[1], row[2]) for row in rows] != [
            (entry.path, entry.sha256, entry.size_bytes) for entry in manifest.files
        ]:
            raise ArtifactCorruptionError(
                f"artifact {artifact_id} data records differ from its manifest"
            )
        payloads: dict[str, bytes] = {}
        for path, digest, size, inline, blob_path in rows:
            if (inline is None) == (blob_path is None):
                raise ArtifactCorruptionError(
                    f"artifact {artifact_id} has an invalid file reference"
                )
            if blob_path is not None:
                expected = f"projects/{paths.project_id}/blobs/{digest}"
                if blob_path != expected:
                    raise ArtifactCorruptionError(
                        f"artifact {artifact_id} blob reference differs from its digest"
                    )
                validate_artifact_file_path(blob_path)
                payload = _read_artifact_file_snapshot(paths.root, blob_path)
            else:
                payload = bytes(inline)
            try:
                verified_payload(payload, digest)
            except ProjectRecordError as exc:
                raise ArtifactCorruptionError(
                    f"artifact {artifact_id} data file digest mismatch: {path}"
                ) from exc
            if len(payload) != size:
                raise ArtifactCorruptionError(
                    f"artifact {artifact_id} data file size mismatch: {path}"
                )
            payloads[path] = payload
        return StoredArtifact(manifest, MappingProxyType(payloads))
    except (OSError, ValueError, ProjectRecordError, sqlite3.Error) as exc:
        raise ArtifactCorruptionError(f"artifact {artifact_id} cannot be verified: {exc}") from exc


def artifact_ids(paths: ProjectPaths, *, artifact_type: str | None = None) -> tuple[str, ...]:
    """Return committed artifact IDs, optionally selecting an indexed domain type."""
    if not has_project_schema(paths.root):
        return ()
    with project_connection(paths.root) as connection:
        if artifact_type is None:
            rows = connection.execute(
                "SELECT artifact_id FROM project_artifacts WHERE project_id=? ORDER BY artifact_id",
                (paths.project_id,),
            )
        else:
            rows = connection.execute(
                "SELECT artifact_id FROM project_artifacts WHERE project_id=? AND "
                "artifact_type=? ORDER BY artifact_id",
                (paths.project_id, artifact_type),
            )
        return tuple(row[0] for row in rows)


def artifact_exists(paths: ProjectPaths, artifact_id: str) -> bool:
    """Check committed artifact identity without treating filesystem remnants as completion."""
    if not has_project_schema(paths.root):
        return False
    with project_connection(paths.root) as connection:
        return (
            connection.execute(
                "SELECT 1 FROM project_artifacts WHERE project_id=? AND artifact_id=?",
                (paths.project_id, artifact_id),
            ).fetchone()
            is not None
        )
