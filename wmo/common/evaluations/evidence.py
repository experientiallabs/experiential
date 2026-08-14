"""Verified immutable evidence helpers for evaluation construction."""

from __future__ import annotations

from collections.abc import Iterable

from pydantic import BaseModel, model_validator

from wmo.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactId,
    ArtifactInput,
    ContractModel,
    StructuredFailure,
    sha256_json,
    unique_sorted_inputs,
)
from wmo.common.evaluations.dataset import EvaluationProtocol, FidelityReport
from wmo.common.evaluations.plan import EvaluationPlan, FidelityGate, FidelityThresholds
from wmo.common.judging import JudgeCalibration, Judgment
from wmo.common.project import (
    ArtifactCorruptionError,
    ArtifactStore,
    artifact_input,
)
from wmo.common.rollouts import RolloutArtifact


class EvaluationEvidenceError(ValueError):
    """Immutable evaluation evidence is missing, conflicting, or hash-mismatched."""


class EvaluationCellEvidence(ContractModel):
    """Execution and judgment references for one explicit planned evaluation cell.

    A cell without a rollout or failure is explicitly ``not_run``. A failure without a rollout
    represents a pre-rollout failure. Completed and failed rollouts retain their own source run.
    """

    cell_id: ArtifactId
    protocol_id: ArtifactId
    rollout_artifact_id: ArtifactId | None = None
    judgment_artifact_id: ArtifactId | None = None
    source_run_id: str | None = None
    failure: StructuredFailure | None = None

    @model_validator(mode="after")
    def _require_coherent_execution_shape(self) -> EvaluationCellEvidence:
        if self.judgment_artifact_id is not None and self.rollout_artifact_id is None:
            raise ValueError("evaluation judgment evidence requires a rollout artifact")
        if self.failure is not None:
            if self.rollout_artifact_id is not None or self.judgment_artifact_id is not None:
                raise ValueError("pre-rollout failure evidence cannot also name a rollout")
            if self.source_run_id is None:
                raise ValueError("pre-rollout failure evidence requires a source_run_id")
        if self.source_run_id is not None and not self.source_run_id:
            raise ValueError("evaluation evidence source_run_id must be non-empty")
        return self


def evaluation_protocol_digest(protocol: EvaluationProtocol) -> str:
    """Return the non-circular digest used to bind a fidelity report to its protocol.

    Args:
        protocol: Evaluation protocol whose optional report reference must not hash itself.

    Returns:
        SHA-256 digest of the protocol with ``fidelity_report_id`` unset.
    """
    return sha256_json(protocol.model_copy(update={"fidelity_report_id": None}))


def read_evaluation_plan(
    store: ArtifactStore, artifact_id: ArtifactId
) -> tuple[EvaluationPlan, ArtifactInput]:
    """Load a verified evaluation plan and its manifest-derived input.

    Args:
        store: Project-local immutable artifact store.
        artifact_id: Evaluation-plan artifact identity.

    Returns:
        Parsed plan plus the exact digest of its generic artifact manifest.

    Raises:
        EvaluationEvidenceError: The artifact is unavailable, wrong-typed, or invalid.
    """
    value, input_record = _read_json(
        store,
        artifact_id=artifact_id,
        artifact_type="evaluation-plan",
        relative_path="plan.json",
        model_type=EvaluationPlan,
    )
    _require_identity(value.plan_id, artifact_id, "evaluation plan")
    return value, input_record


def read_fidelity_gate(
    store: ArtifactStore, artifact_id: ArtifactId
) -> tuple[FidelityGate, ArtifactInput]:
    """Load a verified fidelity gate and its manifest-derived input.

    Args:
        store: Project-local immutable artifact store.
        artifact_id: Fidelity-gate artifact identity.

    Returns:
        Parsed gate plus the exact digest of its generic artifact manifest.

    Raises:
        EvaluationEvidenceError: The artifact is unavailable, wrong-typed, or invalid.
    """
    value, input_record = _read_json(
        store,
        artifact_id=artifact_id,
        artifact_type="fidelity-gate",
        relative_path="gate.json",
        model_type=FidelityGate,
    )
    _require_identity(value.fidelity_gate_id, artifact_id, "fidelity gate")
    return value, input_record


def read_fidelity_thresholds(
    store: ArtifactStore, artifact_id: ArtifactId
) -> tuple[FidelityThresholds, ArtifactInput]:
    """Load verified reusable fidelity thresholds without approval authority.

    Args:
        store: Project-local immutable artifact store.
        artifact_id: Fidelity-threshold artifact identity.

    Returns:
        Parsed thresholds and their exact manifest-derived input.
    """
    value, input_record = _read_json(
        store,
        artifact_id=artifact_id,
        artifact_type="fidelity-thresholds",
        relative_path="thresholds.json",
        model_type=FidelityThresholds,
    )
    _require_identity(value.fidelity_thresholds_id, artifact_id, "fidelity thresholds")
    return value, input_record


def read_fidelity_report(
    store: ArtifactStore, artifact_id: ArtifactId
) -> tuple[FidelityReport, ArtifactInput]:
    """Load a verified fidelity report and its manifest-derived input.

    Args:
        store: Project-local immutable artifact store.
        artifact_id: Fidelity-report artifact identity.

    Returns:
        Parsed report plus the exact digest of its generic artifact manifest.

    Raises:
        EvaluationEvidenceError: The artifact is unavailable, wrong-typed, or invalid.
    """
    value, input_record = _read_json(
        store,
        artifact_id=artifact_id,
        artifact_type="fidelity-report",
        relative_path="report.json",
        model_type=FidelityReport,
    )
    _require_identity(value.fidelity_report_id, artifact_id, "fidelity report")
    return value, input_record


def read_rollout(
    store: ArtifactStore, artifact_id: ArtifactId
) -> tuple[RolloutArtifact, ArtifactInput]:
    """Load a verified rollout and its manifest-derived input.

    Args:
        store: Project-local immutable artifact store.
        artifact_id: Rollout artifact identity.

    Returns:
        Parsed rollout plus the exact digest of its generic artifact manifest.

    Raises:
        EvaluationEvidenceError: The artifact is unavailable, wrong-typed, or invalid.
    """
    value, input_record = _read_json(
        store,
        artifact_id=artifact_id,
        artifact_type="rollout",
        relative_path="rollout.json",
        model_type=RolloutArtifact,
    )
    _require_identity(value.artifact_id, artifact_id, "rollout")
    _require_identity(value.rollout_id, artifact_id, "rollout")
    return value, input_record


def read_judgment(store: ArtifactStore, artifact_id: ArtifactId) -> tuple[Judgment, ArtifactInput]:
    """Load a verified judgment and its manifest-derived input.

    Args:
        store: Project-local immutable artifact store.
        artifact_id: Judgment artifact identity.

    Returns:
        Parsed judgment plus the exact digest of its generic artifact manifest.

    Raises:
        EvaluationEvidenceError: The artifact is unavailable, wrong-typed, or invalid.
    """
    value, input_record = _read_json(
        store,
        artifact_id=artifact_id,
        artifact_type="judgment",
        relative_path="judgment.json",
        model_type=Judgment,
    )
    _require_identity(value.judgment_id, artifact_id, "judgment")
    return value, input_record


def read_calibration(
    store: ArtifactStore, artifact_id: ArtifactId
) -> tuple[JudgeCalibration, ArtifactInput]:
    """Load a verified judge calibration and its manifest-derived input.

    Args:
        store: Project-local immutable artifact store.
        artifact_id: Judge-calibration artifact identity.

    Returns:
        Parsed calibration plus the exact digest of its generic artifact manifest.

    Raises:
        EvaluationEvidenceError: The artifact is unavailable, wrong-typed, or invalid.
    """
    value, input_record = _read_json(
        store,
        artifact_id=artifact_id,
        artifact_type="judge-calibration",
        relative_path="calibration.json",
        model_type=JudgeCalibration,
    )
    _require_identity(value.calibration_id, artifact_id, "judge calibration")
    return value, input_record


def sorted_evaluation_inputs(inputs: Iterable[ArtifactInput]) -> tuple[ArtifactInput, ...]:
    """Return one verified input per artifact ID in canonical order.

    Args:
        inputs: Manifest-derived artifact inputs.

    Returns:
        Deduplicated inputs ordered by artifact ID.

    Raises:
        EvaluationEvidenceError: One ID appears with conflicting digests.
    """
    return unique_sorted_inputs(
        inputs,
        conflict_error=lambda artifact_id: EvaluationEvidenceError(
            f"artifact input {artifact_id} has conflicting manifest hashes"
        ),
    )


def _read_json[ModelT: BaseModel](
    store: ArtifactStore,
    *,
    artifact_id: ArtifactId,
    artifact_type: str,
    relative_path: str,
    model_type: type[ModelT],
) -> tuple[ModelT, ArtifactInput]:
    """Read one typed JSON record after verifying its complete artifact directory."""
    try:
        stored = store.read(artifact_id)
        if stored.manifest.artifact_type != artifact_type:
            raise EvaluationEvidenceError(
                f"artifact {artifact_id} must be {artifact_type}, not "
                f"{stored.manifest.artifact_type}"
            )
        value = model_type.model_validate_json(store.read_bytes(artifact_id, relative_path))
        if isinstance(value, ArtifactEnvelope):
            envelope_values = (
                value.schema_version,
                value.created_at,
                value.inputs,
                value.code_revision,
                value.source,
            )
            manifest_values = (
                stored.manifest.schema_version,
                stored.manifest.created_at,
                stored.manifest.inputs,
                stored.manifest.code_revision,
                stored.manifest.source,
            )
            if envelope_values != manifest_values:
                raise EvaluationEvidenceError(
                    f"artifact {artifact_id} data envelope differs from its manifest"
                )
    except (ArtifactCorruptionError, ValueError) as exc:
        if isinstance(exc, EvaluationEvidenceError):
            raise
        raise EvaluationEvidenceError(
            f"required {artifact_type} artifact is unavailable or invalid: {artifact_id}"
        ) from exc
    return value, artifact_input(stored.manifest)


def _require_identity(actual: str, expected: str, label: str) -> None:
    """Require a domain record ID to equal its immutable artifact directory ID."""
    if actual != expected:
        raise EvaluationEvidenceError(f"{label} record does not match artifact {expected}")
