"""Reusable verification of persisted judge calibrations and their report evidence."""

from __future__ import annotations

from contextvars import ContextVar

from exp.common.core.artifacts import ArtifactId, ArtifactInput
from exp.common.judging.calibration import (
    CalibrationError,
    _load_report,
    _require_calibration_report_binding,
    _verify_report_sources,
)
from exp.common.judging.provenance import JudgingProvenanceError, read_artifact_json
from exp.common.judging.rubric import JudgeCalibration
from exp.common.project import ProjectStore

_VERIFICATION_STACK: ContextVar[tuple[ArtifactId, ...]] = ContextVar(
    "judge_calibration_verification_stack",
    default=(),
)


def verify_persisted_calibration(
    store: ProjectStore, calibration_artifact_id: ArtifactId
) -> tuple[JudgeCalibration, ArtifactInput]:
    """Resolve and prove one persisted calibration and its named report provenance.

    Args:
        store: Project store that owns immutable calibration evidence.
        calibration_artifact_id: Completed calibration artifact expected by a consumer.

    Returns:
        The verified calibration and its canonical manifest-derived input.

    Raises:
        CalibrationError: The calibration, report, inputs, or recursively referenced evidence
            cannot be proven canonical.
    """
    stack = _VERIFICATION_STACK.get()
    if calibration_artifact_id in stack:
        raise CalibrationError("judge calibration provenance contains a cycle")
    token = _VERIFICATION_STACK.set((*stack, calibration_artifact_id))
    try:
        return _verify_persisted_calibration(store, calibration_artifact_id)
    finally:
        _VERIFICATION_STACK.reset(token)


def _load_authoritative_persisted_calibration(
    store: ProjectStore, calibration_artifact_id: ArtifactId
) -> tuple[JudgeCalibration, ArtifactInput]:
    """Load one recursively verified calibration eligible for an authoritative operation.

    This is intentionally an internal data-loading helper, not a caller-held authorization
    capability. An LM-facing operation must invoke it for its own ``ProjectStore`` and
    artifact ID immediately before calling the model.

    Args:
        store: Project store that owns immutable calibration evidence.
        calibration_artifact_id: Completed judge-calibration artifact to verify.

    Returns:
        The immutable calibration and its canonical manifest-derived input.

    Raises:
        CalibrationError: The calibration cannot be recursively verified or remains
            insufficient for authoritative use.
    """
    calibration, calibration_input = verify_persisted_calibration(store, calibration_artifact_id)
    if calibration.status == "insufficient":
        raise CalibrationError(
            "insufficient judge calibrations cannot authorize authoritative LM judgments"
        )
    if calibration.status not in {"provisional", "human_calibrated"}:
        raise CalibrationError("judge calibration has an unsupported authorization status")
    return calibration, calibration_input


def _verify_persisted_calibration(
    store: ProjectStore, calibration_artifact_id: ArtifactId
) -> tuple[JudgeCalibration, ArtifactInput]:
    """Perform one nonrecursive persisted-calibration verification pass."""
    try:
        calibration, calibration_input = read_artifact_json(
            store,
            artifact_id=calibration_artifact_id,
            expected_artifact_type="judge-calibration",
            relative_path="calibration.json",
            model_type=JudgeCalibration,
        )
    except JudgingProvenanceError as exc:
        raise CalibrationError("completed judge calibration is unavailable") from exc
    if calibration.calibration_id != calibration_artifact_id:
        raise CalibrationError("stored judge calibration record has the wrong identity")
    report, report_input = _load_report(store, calibration.out_of_fold_report_id)
    if calibration.out_of_fold_report_sha256 != report_input.sha256:
        raise CalibrationError("judge calibration does not name its report manifest digest")
    if report_input not in calibration.inputs:
        raise CalibrationError("judge calibration does not hash its named report manifest")
    _verify_report_sources(store, report)
    _require_calibration_report_binding(store, report, calibration, report_input)
    return calibration, calibration_input
