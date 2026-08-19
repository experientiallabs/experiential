"""Tests for sparse judged evaluation rows and fidelity-report contracts."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from wmo.common.core.artifacts import FailureCode, StructuredFailure
from wmo.common.evaluations import (
    EvaluationDataset,
    EvaluationDatasetManifest,
    EvaluationProtocol,
    EvaluationRow,
    FidelityFailure,
    FidelityPair,
    FidelityReport,
)
from wmo.common.models import BillingSource, ModelSnapshot, RoutedCandidateSnapshot

_DIGEST = "a" * 64


def _pairs(
    cell_ids: tuple[str, ...], usable: int, *, absolute_error: float = 0.0
) -> tuple[FidelityPair, ...]:
    """Create complete overlap pair fixtures with explicit failures."""
    failure = StructuredFailure(code=FailureCode.TIMEOUT, message="overlap failed")
    return tuple(
        FidelityPair(
            fidelity_cell_id=cell_id,
            observed_cell_id=f"observed-{cell_id}",
            observed_rollout_id=f"observed-rollout-{index}",
            simulated_rollout_id=f"simulated-rollout-{index}" if index < usable else None,
            observed_score=0.8 if index < usable else None,
            simulated_score=0.8 - absolute_error if index < usable else None,
            absolute_error=absolute_error if index < usable else None,
            status="usable" if index < usable else "failed",
            error=None if index < usable else failure,
        )
        for index, cell_id in enumerate(cell_ids)
    )


def _row() -> EvaluationRow:
    return EvaluationRow(
        cell_id="cell-1",
        task_id="task-1",
        candidate_alias="candidate-economy",
        repeat=0,
        protocol_id="protocol-1",
        source_run_id="run-1",
        purpose="fit",
        status="completed",
        rollout_id="rollout-1",
        judgment_id="judgment-1",
        score=0.8,
    )


def _manifest() -> EvaluationDatasetManifest:
    created_at = datetime(2026, 8, 11, tzinfo=UTC)
    candidate = RoutedCandidateSnapshot(
        alias="candidate-economy",
        model=ModelSnapshot(
            billing_source=BillingSource.CUSTOMER_MANAGED,
            provider="openai",
            model_id="gpt-5.4-mini",
            capabilities_sha256=_DIGEST,
            connection_sha256=_DIGEST,
        ),
    )
    protocol = EvaluationProtocol(
        protocol_id="protocol-1",
        evidence_source="world_model",
        agent_id="customer-agent",
        simulator_id="world-model-v1",
        world_model=ModelSnapshot(
            billing_source=BillingSource.CUSTOMER_MANAGED,
            provider="openai",
            model_id="gpt-5.4-mini",
            capabilities_sha256=_DIGEST,
            connection_sha256=_DIGEST,
        ),
        simulator_prompt_id="world-model-prompt-v1",
        rubric_id="rubric-1",
        judge_calibration_id="calibration-1",
        pricing_snapshot_id="pricing-1",
    )
    return EvaluationDatasetManifest(
        schema_version=1,
        created_at=created_at,
        code_revision="e7aad17",
        evaluation_id="evaluation-1",
        evaluation_plan_id="plan-1",
        evaluation_plan_sha256=_DIGEST,
        task_set_id="task-set-1",
        fit_task_ids=("task-1",),
        held_out_task_ids=("task-2",),
        candidate_snapshots=(candidate,),
        protocols=(protocol,),
        rows_path="rows.jsonl",
        rows_sha256=_DIGEST,
    )


def _fidelity_report(
    *,
    overlap_cell_ids: tuple[str, ...],
    usable_overlap_count: int,
    failed_overlap_count: int,
    failures: tuple[FidelityFailure, ...] = (),
    score_mae: float | None = None,
) -> FidelityReport:
    return FidelityReport(
        schema_version=2,
        created_at=datetime(2026, 8, 11, tzinfo=UTC),
        code_revision="e7aad17",
        fidelity_report_id="fidelity-report-1",
        evaluation_plan_id="plan-1",
        evaluation_plan_sha256=_DIGEST,
        protocol_sha256=_DIGEST,
        overlap_cell_ids=overlap_cell_ids,
        planned_overlap_count=len(overlap_cell_ids),
        usable_overlap_count=usable_overlap_count,
        failed_overlap_count=failed_overlap_count,
        score_mae=score_mae,
        failures=failures,
        pairs=_pairs(
            overlap_cell_ids,
            usable_overlap_count,
            absolute_error=score_mae or 0.0,
        ),
    )


def test_rows_keep_failures_and_reject_duplicate_or_missing_evidence() -> None:
    """Failed and not-run cells stay explicit while completed evidence cannot disappear."""
    failed = EvaluationRow(
        cell_id="cell-failed-1",
        task_id="task-1",
        candidate_alias="candidate-economy",
        repeat=0,
        protocol_id="protocol-1",
        source_run_id="run-1",
        purpose="fit",
        status="failed",
        error=StructuredFailure(
            code=FailureCode.TIMEOUT,
            message="candidate timed out",
            retryable=True,
        ),
    )
    assert failed.rollout_id is None
    with pytest.raises(ValidationError, match="require a rollout_id"):
        EvaluationRow(
            cell_id="cell-completed-1",
            task_id="task-1",
            candidate_alias="candidate-economy",
            repeat=0,
            protocol_id="protocol-1",
            purpose="fit",
            status="completed",
        )
    with pytest.raises(ValidationError, match="repeat a cell ID"):
        EvaluationDataset(manifest=_manifest(), rows=(_row(), _row()))
    with pytest.raises(ValidationError, match="started evaluation rows require a source_run_id"):
        EvaluationRow.model_validate({**_row().model_dump(), "source_run_id": None})


def test_fidelity_report_schema_contains_measurements_without_decisions() -> None:
    """The report contract exposes measurements without threshold or approval fields."""
    report = _fidelity_report(
        overlap_cell_ids=("cell-fidelity-0",),
        usable_overlap_count=1,
        failed_overlap_count=0,
        score_mae=0.5,
    )
    payload = report.model_dump(mode="json")

    assert FidelityReport.model_validate(payload).model_dump(mode="json") == payload
    assert report.score_mae == 0.5
    assert not {
        "status",
        "approved_at",
        "gate_id",
        "gate_sha256",
        "minimum_usable_overlap_count",
        "maximum_allowed_score_mae",
    }.intersection(payload)


def test_fidelity_reports_account_for_each_unique_planned_overlap() -> None:
    """Reports preserve every measured overlap and reject inconsistent accounting."""
    full_overlap_ids = tuple(f"cell-fidelity-{index}" for index in range(10))
    failures = (
        FidelityFailure(
            cell_id="cell-fidelity-8",
            failure=StructuredFailure(code=FailureCode.TIMEOUT, message="simulator timed out"),
        ),
        FidelityFailure(
            cell_id="cell-fidelity-9",
            failure=StructuredFailure(code=FailureCode.PROVIDER, message="provider failed"),
        ),
    )
    _fidelity_report(
        overlap_cell_ids=full_overlap_ids,
        usable_overlap_count=8,
        failed_overlap_count=2,
        failures=failures,
        score_mae=0.08,
    )

    with pytest.raises(ValidationError, match="must match the planned overlap count"):
        _fidelity_report(
            overlap_cell_ids=full_overlap_ids,
            usable_overlap_count=8,
            failed_overlap_count=0,
            score_mae=0.08,
        )
    with pytest.raises(ValidationError, match="must not repeat overlap cells"):
        _fidelity_report(
            overlap_cell_ids=("cell-fidelity-0", "cell-fidelity-0"),
            usable_overlap_count=2,
            failed_overlap_count=0,
        )
