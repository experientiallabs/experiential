"""Direct SQLite corruption fixtures for tests of evidence verification.

These helpers deliberately bypass immutable-store validation. Production code must
use ArtifactStore instead. Updating a manifest keeps its indexed metadata aligned
so tests can exercise semantic lineage checks beyond byte-integrity validation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from exp.common.project.artifact_files import _read_artifact_file_snapshot
from exp.common.project.database import project_connection
from exp.common.project.paths import ProjectPaths


@dataclass(frozen=True)
class RawArtifact:
    """Unverified artifact identity for adversarial test fixtures.

    Attributes:
        paths: Project identity and database location.
        artifact_id: Artifact whose stored evidence a test intentionally mutates.
    """

    paths: ProjectPaths
    artifact_id: str

    def __truediv__(self, name: str) -> RawArtifactFile:
        """Select a logical file record without accessing an artifact directory."""
        return RawArtifactFile(self, name)


@dataclass(frozen=True)
class RawArtifactFile:
    """Unverified metadata or payload row used only by corruption tests.

    Attributes:
        artifact: Project and artifact identity.
        name: Logical manifest-relative payload name, or manifest.json.
    """

    artifact: RawArtifact
    name: str

    def read_bytes(self) -> bytes:
        """Return raw persisted bytes without checking a digest."""
        paths, artifact_id = self.artifact.paths, self.artifact.artifact_id
        with project_connection(paths.root) as connection:
            if self.name == "manifest.json":
                return connection.execute(
                    "SELECT manifest FROM project_artifacts WHERE project_id=? AND artifact_id=?",
                    (paths.project_id, artifact_id),
                ).fetchone()[0]
            payload, blob_path = connection.execute(
                "SELECT payload, blob_path FROM project_artifact_files "
                "WHERE project_id=? AND artifact_id=? AND path=?",
                (paths.project_id, artifact_id, self.name),
            ).fetchone()
        if blob_path is not None:
            return _read_artifact_file_snapshot(paths.root, blob_path)
        return payload

    def read_text(self, encoding: str = "utf-8") -> str:
        """Decode the raw payload for a test's JSON mutation."""
        return self.read_bytes().decode(encoding)

    def write_text(self, value: str, encoding: str = "utf-8") -> None:
        """Replace raw payload bytes while bypassing immutable-store checks."""
        self.write_bytes(value.encode(encoding))

    def write_bytes(self, payload: bytes) -> None:
        """Forge a payload, or publish a rehashed manifest and its indexed metadata."""
        paths, artifact_id = self.artifact.paths, self.artifact.artifact_id
        with project_connection(paths.root, write=True) as connection:
            if self.name != "manifest.json":
                connection.execute(
                    "INSERT INTO project_artifact_files VALUES (?, ?, ?, ?, ?, ?, NULL) "
                    "ON CONFLICT(project_id, artifact_id, path) DO UPDATE SET "
                    "payload=excluded.payload, blob_path=NULL",
                    (
                        paths.project_id,
                        artifact_id,
                        self.name,
                        hashlib.sha256(payload).hexdigest(),
                        len(payload),
                        payload,
                    ),
                )
                return
            connection.execute(
                "UPDATE project_artifacts SET manifest=?, sha256=? "
                "WHERE project_id=? AND artifact_id=?",
                (payload, hashlib.sha256(payload).hexdigest(), paths.project_id, artifact_id),
            )
            try:
                manifest = json.loads(payload)
            except ValueError:
                return
            if (
                not isinstance(manifest, dict)
                or not {"artifact_type", "inputs", "files"} <= manifest.keys()
            ):
                return
            connection.execute(
                "UPDATE project_artifacts SET artifact_type=? WHERE project_id=? AND artifact_id=?",
                (manifest["artifact_type"], paths.project_id, artifact_id),
            )
            connection.execute(
                "DELETE FROM project_artifact_inputs WHERE project_id=? AND artifact_id=?",
                (paths.project_id, artifact_id),
            )
            connection.executemany(
                "INSERT INTO project_artifact_inputs VALUES (?, ?, ?, ?, ?)",
                [
                    (paths.project_id, artifact_id, i, item["artifact_id"], item["sha256"])
                    for i, item in enumerate(manifest["inputs"])
                ],
            )
            connection.executemany(
                "UPDATE project_artifact_files SET sha256=?, size_bytes=? "
                "WHERE project_id=? AND artifact_id=? AND path=?",
                [
                    (
                        item["sha256"],
                        item["size_bytes"],
                        paths.project_id,
                        artifact_id,
                        item["path"],
                    )
                    for item in manifest["files"]
                ],
            )

    def unlink(self) -> None:
        """Delete a payload row to simulate lost evidence."""
        paths = self.artifact.paths
        with project_connection(paths.root, write=True) as connection:
            connection.execute(
                "DELETE FROM project_artifact_files "
                "WHERE project_id=? AND artifact_id=? AND path=?",
                (paths.project_id, self.artifact.artifact_id, self.name),
            )

    def rename(self, target: RawArtifactFile) -> None:
        """Rename a payload record to test noncanonical logical file names."""
        assert target.artifact == self.artifact
        paths = self.artifact.paths
        with project_connection(paths.root, write=True) as connection:
            connection.execute(
                "UPDATE project_artifact_files SET path=? "
                "WHERE project_id=? AND artifact_id=? AND path=?",
                (target.name, paths.project_id, self.artifact.artifact_id, self.name),
            )
