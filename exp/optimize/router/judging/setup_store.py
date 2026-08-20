"""Immutable persistence and verified reads for finalized manual judge setups."""

from __future__ import annotations

from exp.common.core.artifacts import ArtifactInput, canonical_json_bytes
from exp.common.judging.provenance import JudgingProvenanceError, read_artifact_json
from exp.common.project import ArtifactCorruptionError, ProjectStore, artifact_input
from exp.optimize.router.judging.contracts import ManualJudgeError, ManualJudgeSetupArtifact


def write_setup_artifact(store: ProjectStore, setup: ManualJudgeSetupArtifact) -> ArtifactInput:
    """Persist or verify one finalized manual judge setup.

    Args:
        store: Project-local immutable artifact store.
        setup: Confirmed setup contract.

    Returns:
        Exact manifest pointer for the stored setup.

    Raises:
        ManualJudgeError: An existing artifact conflicts or cannot be verified.
    """
    try:
        _stored, manifest = store.artifacts.write_or_replay(
            artifact_id=setup.setup_id,
            artifact_type="manual-judge-setup",
            envelope=setup,
            envelope_path="setup.json",
            envelope_type=ManualJudgeSetupArtifact,
            files={"setup.json": canonical_json_bytes(setup)},
        )
    except ArtifactCorruptionError as exc:
        raise ManualJudgeError("existing manual judge setup cannot be resumed safely") from exc
    except ValueError as exc:
        raise ManualJudgeError("existing manual judge setup conflicts with confirmation") from exc
    return artifact_input(manifest)


def read_setup_artifact(store: ProjectStore, expected: ArtifactInput) -> ManualJudgeSetupArtifact:
    """Read one setup and require its exact manifest pointer.

    Args:
        store: Project-local immutable artifact store.
        expected: Review-state setup pointer.

    Returns:
        Verified setup contract.

    Raises:
        ManualJudgeError: The artifact is absent, malformed, or changed.
    """
    saved, saved_input = read_setup_with_input(store, expected.artifact_id)
    if saved_input != expected:
        raise ManualJudgeError("manual judge setup manifest differs from review state")
    return saved


def read_setup_with_input(
    store: ProjectStore, artifact_id: str
) -> tuple[ManualJudgeSetupArtifact, ArtifactInput]:
    """Read a setup artifact through the shared provenance verifier.

    Args:
        store: Project-local immutable artifact store.
        artifact_id: Setup artifact identifier.

    Returns:
        Verified setup and canonical manifest input.

    Raises:
        ManualJudgeError: The setup cannot be verified.
    """
    try:
        return read_artifact_json(
            store,
            artifact_id=artifact_id,
            expected_artifact_type="manual-judge-setup",
            relative_path="setup.json",
            model_type=ManualJudgeSetupArtifact,
        )
    except JudgingProvenanceError as exc:
        raise ManualJudgeError("completed manual judge setup is unavailable") from exc
