"""Canonical assembly of report-bound judge calibration artifacts."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from wmo.common.core.artifacts import ArtifactInput, stable_id
from wmo.common.judging.calibration_contracts import CalibrationReport
from wmo.common.judging.risk_acceptance import calibration_inputs
from wmo.common.judging.rubric import JudgeCalibration


def calibration_from_report(
    report: CalibrationReport,
    *,
    report_input: ArtifactInput,
    status: Literal["provisional", "insufficient", "human_calibrated"],
    approved_at: datetime | None,
    risk_acceptance: ArtifactInput | None = None,
) -> JudgeCalibration:
    """Build a calibration whose inputs include every frozen report identity.

    Args:
        report: Verified report whose score maps and identifiers are frozen into the result.
        report_input: Canonical manifest reference for ``report``.
        status: Lifecycle state of the resulting calibration.
        approved_at: Human approval time for a final calibration, when applicable.
        risk_acceptance: Explicit immutable provenance for accepted low-sample risk.

    Returns:
        A deterministic calibration derived only from the report and optional acceptance input.
    """
    inputs = calibration_inputs(report_input, report.inputs, risk_acceptance)
    return JudgeCalibration(
        schema_version=1,
        created_at=report.created_at if approved_at is None else approved_at,
        inputs=inputs,
        code_revision=report.code_revision,
        calibration_id=stable_id(
            "judge-calibration",
            {
                "report": report_input.model_dump(mode="json"),
                "inputs": [item.model_dump(mode="json") for item in inputs],
                "status": status,
                "approved_at": approved_at.isoformat() if approved_at is not None else None,
            },
        ),
        rubric_id=report.rubric_id,
        judge_model=report.judge_model,
        judge_prompt_id=report.judge_prompt_id,
        judge_prompt_sha256=report.judge_prompt_sha256,
        label_set_id=report.label_set_id,
        calibration_lineage_ids=report.eligible_lineage_ids,
        excluded_router_held_out_lineage_ids=report.router_lineages.held_out_lineage_ids,
        validation_method="grouped_k_fold",
        out_of_fold_report_id=report.report_id,
        out_of_fold_report_sha256=report_input.sha256,
        score_maps=report.score_maps,
        label_count=report.eligible_label_count,
        status=status,
        approved_at=approved_at,
        risk_acceptance=risk_acceptance,
    )
