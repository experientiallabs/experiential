"""Idempotent exact persistence for deterministic router artifacts."""

from __future__ import annotations

from collections.abc import Mapping

from wmo.common.core.artifacts import ArtifactEnvelope
from wmo.common.project import ArtifactAlreadyExistsError, ArtifactStore
from wmo.common.project.manifests import ArtifactManifest


def write_or_verify_exact(
    store: ArtifactStore,
    *,
    artifact_id: str,
    artifact_type: str,
    envelope: ArtifactEnvelope,
    files: Mapping[str, bytes],
) -> ArtifactManifest:
    """Write a deterministic artifact or verify an existing exact replay.

    Args:
        store: Project-local immutable artifact store.
        artifact_id: Stable content-derived artifact identity.
        artifact_type: Expected immutable artifact type.
        envelope: Exact expected artifact provenance envelope.
        files: Exact expected serialized payloads.

    Returns:
        Newly written or byte-for-byte verified artifact manifest.

    Raises:
        ValueError: An existing artifact has different manifest fields or payload bytes.
        ArtifactStoreError: The existing artifact is corrupt or the write fails.
    """
    try:
        return store.write(
            artifact_id=artifact_id,
            artifact_type=artifact_type,
            envelope=envelope,
            files=files,
        )
    except ArtifactAlreadyExistsError:
        stored = store.read(artifact_id)
    manifest = stored.manifest
    actual = (
        manifest.artifact_type,
        manifest.schema_version,
        manifest.created_at,
        manifest.inputs,
        manifest.code_revision,
        manifest.source,
    )
    expected = (
        artifact_type,
        envelope.schema_version,
        envelope.created_at,
        envelope.inputs,
        envelope.code_revision,
        envelope.source,
    )
    if actual != expected:
        raise ValueError(f"existing artifact {artifact_id} manifest differs from exact replay")
    if tuple(sorted(files)) != tuple(item.path for item in manifest.files):
        raise ValueError(f"existing artifact {artifact_id} file set differs from exact replay")
    for relative_path, expected_payload in files.items():
        if store.read_bytes(artifact_id, relative_path) != expected_payload:
            raise ValueError(f"existing artifact {artifact_id} payload differs from exact replay")
    return manifest
