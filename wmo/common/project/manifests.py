"""Immutable artifact-manifest contracts for the local project store."""

from __future__ import annotations

import hashlib

from pydantic import Field, field_validator

from wmo.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactId,
    ArtifactInput,
    ContractModel,
    Sha256,
    sha256_json,
)
from wmo.common.project.paths import validate_artifact_file_path


class ArtifactFile(ContractModel):
    """A digest-addressed data file owned by one completed immutable artifact."""

    path: str = Field(min_length=1)
    sha256: Sha256
    size_bytes: int = Field(ge=0)

    @field_validator("path")
    @classmethod
    def _require_safe_relative_path(cls, value: str) -> str:
        return validate_artifact_file_path(value).as_posix()


class ArtifactManifest(ArtifactEnvelope):
    """The single immutable manifest for a completed local artifact directory."""

    artifact_id: ArtifactId
    artifact_type: ArtifactId
    files: tuple[ArtifactFile, ...]

    @field_validator("files")
    @classmethod
    def _require_sorted_unique_files(
        cls, value: tuple[ArtifactFile, ...]
    ) -> tuple[ArtifactFile, ...]:
        if not value:
            raise ValueError("completed artifacts need at least one data file")
        paths = tuple(file.path for file in value)
        if len(set(paths)) != len(paths):
            raise ValueError("artifact manifests must not repeat a data-file path")
        if paths != tuple(sorted(paths)):
            raise ValueError("artifact manifest data files must be sorted by path")
        return value


def artifact_input(manifest: ArtifactManifest) -> ArtifactInput:
    """Create the immutable input reference used by a dependent artifact.

    Args:
        manifest: Verified manifest for the completed input artifact.

    Returns:
        The input's stable ID and canonical manifest digest.
    """
    return ArtifactInput(artifact_id=manifest.artifact_id, sha256=sha256_json(manifest))


def file_digest(path: str, payload: bytes) -> ArtifactFile:
    """Build one file digest record from complete staged bytes."""
    return ArtifactFile(
        path=path,
        sha256=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
    )
