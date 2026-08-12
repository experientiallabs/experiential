"""Reusable verification of persisted judge calibrations and their report evidence."""

from __future__ import annotations

from contextvars import ContextVar

from wmo.common.core.artifacts import ArtifactId, ArtifactInput
from wmo.common.judging.calibration import (
    CalibrationError,
    _load_report,
    _require_calibration_report_binding,
    _verify_report_sources,
)
from wmo.common.judging.provenance import JudgingProvenanceError, read_artifact_json
from wmo.common.judging.rubric import JudgeCalibration
from wmo.common.project import ProjectStore

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
