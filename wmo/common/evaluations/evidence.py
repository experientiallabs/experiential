"""Verified immutable evidence helpers for evaluation construction."""

from __future__ import annotations

from collections.abc import Iterable

from pydantic import model_validator

from wmo.common.core.artifacts import (
    ArtifactId,
    ArtifactInput,
    ContractModel,
    StructuredFailure,
    sha256_json,
    sorted_unique_inputs,
)
from wmo.common.evaluations.dataset import EvaluationProtocol, FidelityReport
from wmo.common.evaluations.plan import EvaluationPlan
from wmo.common.judging import JudgeCalibration, Judgment
from wmo.common.judging.provenance import read_artifact_json
from wmo.common.project import ArtifactStore
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
    """Return the canonical digest used to bind a fidelity report to its protocol.

    Args:
        protocol: Complete evaluation protocol used for fidelity measurement.

    Returns:
        SHA-256 digest of the complete protocol.
    """
    return sha256_json(protocol)


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
    value, input_record = read_artifact_json(
        store,
        artifact_id=artifact_id,
        expected_artifact_type="evaluation-plan",
        relative_path="plan.json",
        model_type=EvaluationPlan,
        error=EvaluationEvidenceError,
    )
    _require_identity(value.plan_id, artifact_id, "evaluation plan")
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
    value, input_record = read_artifact_json(
        store,
        artifact_id=artifact_id,
        expected_artifact_type="fidelity-report",
        relative_path="report.json",
        model_type=FidelityReport,
        error=EvaluationEvidenceError,
    )
    _require_identity(value.fidelity_report_id, artifact_id, "fidelity report")
    plan, plan_input = read_evaluation_plan(store, value.evaluation_plan_id)
    if value.evaluation_plan_sha256 != plan_input.sha256 or plan_input not in value.inputs:
        raise EvaluationEvidenceError("fidelity report differs from its exact evaluation plan")
    if value.protocol_sha256 != plan.fidelity_protocol_sha256:
        raise EvaluationEvidenceError("fidelity report protocol differs from its evaluation plan")
    overlap_cell_ids = tuple(cell.cell_id for cell in plan.cells if cell.purpose == "fidelity")
    if not overlap_cell_ids or value.overlap_cell_ids != overlap_cell_ids:
        raise EvaluationEvidenceError("fidelity report overlap scope differs from its plan")
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
    value, input_record = read_artifact_json(
        store,
        artifact_id=artifact_id,
        expected_artifact_type="rollout",
        relative_path="rollout.json",
        model_type=RolloutArtifact,
        error=EvaluationEvidenceError,
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
    value, input_record = read_artifact_json(
        store,
        artifact_id=artifact_id,
        expected_artifact_type="judgment",
        relative_path="judgment.json",
        model_type=Judgment,
        error=EvaluationEvidenceError,
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
    value, input_record = read_artifact_json(
        store,
        artifact_id=artifact_id,
        expected_artifact_type="judge-calibration",
        relative_path="calibration.json",
        model_type=JudgeCalibration,
        error=EvaluationEvidenceError,
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
    return sorted_unique_inputs(*inputs, error_type=EvaluationEvidenceError)


def _require_identity(actual: str, expected: str, label: str) -> None:
    """Require a domain record ID to equal its immutable artifact directory ID."""
    if actual != expected:
        raise EvaluationEvidenceError(f"{label} record does not match artifact {expected}")
