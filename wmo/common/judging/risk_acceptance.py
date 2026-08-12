"""Immutable low-sample calibration-risk acceptance provenance."""

from __future__ import annotations

from datetime import datetime

from wmo.common.core.artifacts import ArtifactId, ArtifactInput, stable_id
from wmo.common.judging.calibration_contracts import (
    CalibrationReport,
    InsufficientCalibrationRiskAcceptance,
)
from wmo.common.judging.provenance import (
    JudgingProvenanceError,
    read_artifact_json,
    sorted_verified_inputs,
)
from wmo.common.judging.rubric import JudgeCalibration
from wmo.common.project import ArtifactAlreadyExistsError, ProjectStore, artifact_input


class RiskAcceptanceError(ValueError):
    """Raised when low-sample calibration-risk acceptance evidence is invalid."""


def calibration_inputs(
    report_input: ArtifactInput,
    report_inputs: tuple[ArtifactInput, ...],
    risk_acceptance: ArtifactInput | None,
) -> tuple[ArtifactInput, ...]:
    """Return canonical report inputs with optional acceptance evidence.

    Args:
        report_input: Verified manifest reference for the calibration report.
        report_inputs: Immutable artifacts already named by the calibration report.
        risk_acceptance: Explicit low-sample acceptance artifact, when required.

    Returns:
        De-duplicated inputs in deterministic artifact-ID order.
    """
    sources = (report_input, *report_inputs)
    return sorted_verified_inputs(
        sources if risk_acceptance is None else (*sources, risk_acceptance)
    )


def require_calibration_risk_acceptance(
    store: ProjectStore,
    *,
    report: CalibrationReport,
    report_input: ArtifactInput,
    calibration: JudgeCalibration,
) -> None:
    """Require risk evidence exactly when a human calibration accepts insufficient data.

    Args:
        store: Project store that owns immutable calibration acceptance evidence.
        report: Verified report named by the calibration.
        report_input: Verified manifest reference for ``report``.
        calibration: Candidate calibration whose acceptance provenance is required.

    Raises:
        RiskAcceptanceError: The calibration lacks required acceptance evidence or names it when
            the report did not require that exception.
    """
    if report.status == "insufficient" and calibration.status == "human_calibrated":
        if report.eligible_rollout_count >= report.recommended_label_count:
            raise RiskAcceptanceError(
                "insufficient human calibration risk acceptance is limited to fewer than ten "
                "eligible rollouts"
            )
        if calibration.risk_acceptance is None:
            raise RiskAcceptanceError(
                "human calibration of insufficient labels requires risk acceptance"
            )
        verify_insufficient_calibration_risk_acceptance(
            store,
            acceptance_input=calibration.risk_acceptance,
            report=report,
            report_input=report_input,
            approved_at=calibration.approved_at,
        )
    elif calibration.risk_acceptance is not None:
        raise RiskAcceptanceError("only insufficient human calibration can name risk acceptance")


def write_insufficient_calibration_risk_acceptance(
    store: ProjectStore,
    *,
    report: CalibrationReport,
    report_input: ArtifactInput,
    accepted_at: datetime,
) -> ArtifactInput:
    """Persist one explicit human decision to accept an insufficient calibration report.

    Args:
        store: Project store that owns immutable report and acceptance artifacts.
        report: Exact persisted insufficient report being accepted.
        report_input: Verified manifest reference for ``report``.
        accepted_at: Time the human explicitly accepted the low-sample risk.

    Returns:
        The canonical manifest reference for the persisted acceptance artifact.

    Raises:
        RiskAcceptanceError: The report is not an eligible low-sample report or an existing
            acceptance cannot be proven equivalent.
    """
    acceptance = _acceptance_from_report(report, report_input, accepted_at)
    try:
        manifest = store.artifacts.write_json(
            artifact_id=acceptance.acceptance_id,
            artifact_type="insufficient-calibration-risk-acceptance",
            envelope=acceptance,
            files={"acceptance.json": acceptance},
        )
    except ArtifactAlreadyExistsError:
        stored, stored_input = _read_acceptance(store, acceptance.acceptance_id)
        if stored != acceptance:
            raise RiskAcceptanceError(
                "existing insufficient calibration risk acceptance conflicts with this approval"
            ) from None
        return stored_input
    return artifact_input(manifest)


def verify_insufficient_calibration_risk_acceptance(
    store: ProjectStore,
    *,
    acceptance_input: ArtifactInput,
    report: CalibrationReport,
    report_input: ArtifactInput,
    approved_at: datetime | None,
) -> None:
    """Verify that persisted low-sample risk acceptance is exact and report-bound.

    Args:
        store: Project store that owns immutable acceptance evidence.
        acceptance_input: Manifest reference named by the judge calibration.
        report: Persisted report the calibration claims to use.
        report_input: Verified manifest reference for ``report``.
        approved_at: Approval time recorded by the judge calibration.

    Raises:
        RiskAcceptanceError: The acceptance is missing, corrupt, does not bind the exact report,
            or does not match the calibration approval time.
    """
    if approved_at is None:
        raise RiskAcceptanceError("human calibration risk acceptance requires approved_at")
    acceptance, _acceptance_input = _read_acceptance(
        store,
        acceptance_input.artifact_id,
        expected_input=acceptance_input,
    )
    expected = _acceptance_from_report(report, report_input, approved_at)
    if acceptance != expected:
        raise RiskAcceptanceError(
            "risk acceptance does not match the exact insufficient calibration report"
        )


def _acceptance_from_report(
    report: CalibrationReport,
    report_input: ArtifactInput,
    accepted_at: datetime,
) -> InsufficientCalibrationRiskAcceptance:
    """Build the one canonical acceptance record for an insufficient report."""
    if report.status != "insufficient":
        raise RiskAcceptanceError("risk acceptance requires an insufficient calibration report")
    if report.eligible_rollout_count >= report.recommended_label_count:
        raise RiskAcceptanceError("risk acceptance requires fewer than ten eligible rollouts")
    if report_input.artifact_id != report.report_id:
        raise RiskAcceptanceError("risk acceptance report manifest has the wrong artifact identity")
    acceptance_id = stable_id(
        "insufficient-calibration-risk-acceptance",
        {
            "report": report_input.model_dump(mode="json"),
            "eligible_label_count": report.eligible_label_count,
            "eligible_rollout_count": report.eligible_rollout_count,
            "recommended_label_count": report.recommended_label_count,
            "accepted_at": accepted_at.isoformat(),
            "code_revision": report.code_revision,
        },
    )
    return InsufficientCalibrationRiskAcceptance(
        schema_version=1,
        created_at=accepted_at,
        inputs=(report_input,),
        code_revision=report.code_revision,
        acceptance_id=acceptance_id,
        report=report_input,
        eligible_label_count=report.eligible_label_count,
        eligible_rollout_count=report.eligible_rollout_count,
        accepted_at=accepted_at,
    )


def _read_acceptance(
    store: ProjectStore,
    acceptance_id: ArtifactId,
    *,
    expected_input: ArtifactInput | None = None,
) -> tuple[InsufficientCalibrationRiskAcceptance, ArtifactInput]:
    """Read one immutable acceptance record with its verified manifest reference."""
    try:
        acceptance, acceptance_input = read_artifact_json(
            store,
            artifact_id=acceptance_id,
            expected_artifact_type="insufficient-calibration-risk-acceptance",
            relative_path="acceptance.json",
            model_type=InsufficientCalibrationRiskAcceptance,
            expected_input=expected_input,
        )
    except JudgingProvenanceError as exc:
        raise RiskAcceptanceError("risk acceptance artifact is unavailable or corrupt") from exc
    if acceptance.acceptance_id != acceptance_id:
        raise RiskAcceptanceError("risk acceptance record has the wrong artifact identity")
    return acceptance, acceptance_input
