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
    sorted_artifact_inputs,
)
from wmo.common.evaluations.dataset import EvaluationProtocol, FidelityReport
from wmo.common.evaluations.plan import EvaluationPlan, FidelityGate, FidelityThresholds
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
    value, input_record = read_artifact_json(
        store,
        artifact_id=artifact_id,
        expected_artifact_type="fidelity-gate",
        relative_path="gate.json",
        model_type=FidelityGate,
        error=EvaluationEvidenceError,
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
    value, input_record = read_artifact_json(
        store,
        artifact_id=artifact_id,
        expected_artifact_type="fidelity-thresholds",
        relative_path="thresholds.json",
        model_type=FidelityThresholds,
        error=EvaluationEvidenceError,
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
    value, input_record = read_artifact_json(
        store,
        artifact_id=artifact_id,
        expected_artifact_type="fidelity-report",
        relative_path="report.json",
        model_type=FidelityReport,
        error=EvaluationEvidenceError,
    )
    _require_identity(value.fidelity_report_id, artifact_id, "fidelity report")
    gate, gate_input = read_fidelity_gate(store, value.gate_id)
    if (
        gate_input.sha256 != value.gate_sha256
        or gate_input not in value.inputs
        or gate.planned_overlaps != value.planned_overlap_count
    ):
        raise EvaluationEvidenceError("fidelity report differs from its exact frozen gate")
    verify_fidelity_report_gate(value, gate)
    return value, input_record


def verify_fidelity_report_gate(report: FidelityReport, gate: FidelityGate) -> None:
    """Verify report status and measurements against its recursively loaded gate.

    Args:
        report: Structurally valid fidelity report.
        gate: Exact immutable gate named by the report.

    Raises:
        EvaluationEvidenceError: Status, denominator, usable count, or MAE violates the gate.
    """
    if gate.planned_overlaps != report.planned_overlap_count:
        raise EvaluationEvidenceError("fidelity report denominator differs from its frozen gate")
    insufficient = report.usable_overlap_count < gate.minimum_usable_overlaps
    excessive_mae = report.score_mae is None or report.score_mae > gate.maximum_score_mae
    if report.status == "approved" and (insufficient or excessive_mae):
        raise EvaluationEvidenceError("approved fidelity report does not satisfy its frozen gate")
    if report.status == "insufficient" and not insufficient:
        raise EvaluationEvidenceError("fidelity report meets the gate's usable evidence minimum")
    if report.status == "rejected" and (insufficient or not excessive_mae):
        raise EvaluationEvidenceError("rejected fidelity report does not exceed its frozen MAE")


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
    try:
        return sorted_artifact_inputs(inputs)
    except ValueError as exc:
        raise EvaluationEvidenceError(str(exc)) from exc


def _require_identity(actual: str, expected: str, label: str) -> None:
    """Require a domain record ID to equal its immutable artifact directory ID."""
    if actual != expected:
        raise EvaluationEvidenceError(f"{label} record does not match artifact {expected}")
