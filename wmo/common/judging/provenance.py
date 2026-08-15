"""Verified immutable-artifact helpers shared by common judging services."""

from __future__ import annotations

from collections.abc import Iterable

from pydantic import BaseModel, ValidationError

from wmo.common.core.artifacts import ArtifactId, ArtifactInput, sorted_unique_inputs
from wmo.common.project import (
    ArtifactCorruptionError,
    ProjectStore,
    StoredArtifact,
    artifact_input,
)


class JudgingProvenanceError(ValueError):
    """Raised when immutable evidence cannot be resolved or verified."""


def resolve_artifact(
    store: ProjectStore,
    *,
    artifact_id: ArtifactId,
    expected_artifact_type: str,
    expected_input: ArtifactInput | None = None,
) -> tuple[StoredArtifact, ArtifactInput]:
    """Open one completed artifact and derive its canonical manifest input.

    Args:
        store: Project store that owns the immutable artifact directory.
        artifact_id: Completed artifact directory to verify.
        expected_artifact_type: Required stable manifest artifact type.
        expected_input: Optional caller-provided reference that must equal the manifest digest.

    Returns:
        The fully verified artifact and its manifest-derived input reference.

    Raises:
        JudgingProvenanceError: The artifact is missing, corrupt, wrong-typed, or hash-mismatched.
    """
    try:
        stored = store.artifacts.read(artifact_id)
    except (ArtifactCorruptionError, ValueError) as exc:
        raise JudgingProvenanceError(
            f"required {expected_artifact_type} artifact is unavailable: {artifact_id}"
        ) from exc
    if stored.manifest.artifact_type != expected_artifact_type:
        raise JudgingProvenanceError(
            f"artifact {artifact_id} must be {expected_artifact_type}, not "
            f"{stored.manifest.artifact_type}"
        )
    derived = artifact_input(stored.manifest)
    if expected_input is not None and expected_input != derived:
        raise JudgingProvenanceError(
            f"artifact {artifact_id} manifest digest does not match the supplied evidence"
        )
    return stored, derived


def read_artifact_json[ModelT: BaseModel](
    store: ProjectStore,
    *,
    artifact_id: ArtifactId,
    expected_artifact_type: str,
    relative_path: str,
    model_type: type[ModelT],
    expected_input: ArtifactInput | None = None,
) -> tuple[ModelT, ArtifactInput]:
    """Load one typed data file only after verifying its complete artifact directory.

    Args:
        store: Project store that owns the immutable artifact directory.
        artifact_id: Completed artifact directory to verify.
        expected_artifact_type: Required stable manifest artifact type.
        relative_path: Typed JSON data file required inside the artifact directory.
        model_type: Pydantic contract used to parse the verified JSON data.
        expected_input: Optional caller-provided reference that must match the manifest.

    Returns:
        The parsed immutable record and its canonical manifest input reference.

    Raises:
        JudgingProvenanceError: The artifact or its required typed data file is invalid.
    """
    _stored, derived = resolve_artifact(
        store,
        artifact_id=artifact_id,
        expected_artifact_type=expected_artifact_type,
        expected_input=expected_input,
    )
    try:
        value = model_type.model_validate_json(
            store.artifacts.read_bytes(artifact_id, relative_path)
        )
    except (ArtifactCorruptionError, ValidationError, ValueError) as exc:
        raise JudgingProvenanceError(
            f"artifact {artifact_id} has no valid {relative_path} {expected_artifact_type} record"
        ) from exc
    return value, derived


def sorted_verified_inputs(inputs: Iterable[ArtifactInput]) -> tuple[ArtifactInput, ...]:
    """Deduplicate equal verified inputs and return canonical artifact-ID ordering.

    Args:
        inputs: Manifest-derived immutable input references.

    Returns:
        One input per artifact ID in deterministic artifact-ID order.

    Raises:
        JudgingProvenanceError: One artifact ID is associated with conflicting manifest hashes.
    """
    return sorted_unique_inputs(*inputs, error_type=JudgingProvenanceError)
