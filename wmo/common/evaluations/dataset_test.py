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
    FidelityReport,
)
from wmo.common.models import ModelSnapshot, RoutedCandidateSnapshot

_DIGEST = "a" * 64


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
            provider="openai",
            model_id="gpt-5.4-mini",
            capabilities_sha256=_DIGEST,
        ),
    )
    protocol = EvaluationProtocol(
        protocol_id="protocol-1",
        evidence_source="world_model",
        agent_id="customer-agent",
        simulator_id="world-model-v1",
        world_model=ModelSnapshot(
            provider="openai",
            model_id="gpt-5.4-mini",
            capabilities_sha256=_DIGEST,
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
        task_set_id="task-set-1",
        fit_task_ids=("task-1",),
        held_out_task_ids=("task-2",),
        candidate_snapshots=(candidate,),
        protocols=(protocol,),
        rows_path="rows.jsonl",
        rows_sha256=_DIGEST,
    )


def test_dataset_and_fidelity_report_round_trip() -> None:
    """Sparse rows retain explicit protocols, missing-state semantics, and fidelity provenance."""
    manifest = _manifest()
    dataset = EvaluationDataset(manifest=manifest, rows=(_row(),))
    report = FidelityReport(
        schema_version=1,
        created_at=datetime(2026, 8, 11, tzinfo=UTC),
        code_revision="e7aad17",
        fidelity_report_id="fidelity-report-1",
        protocol_sha256=_DIGEST,
        overlap_cell_ids=tuple(f"cell-fidelity-{index}" for index in range(10)),
        planned_overlap_count=10,
        usable_overlap_count=8,
        failed_overlap_count=2,
        score_mae=0.08,
        failures=(
            FidelityFailure(
                cell_id="cell-fidelity-8",
                failure=StructuredFailure(code=FailureCode.TIMEOUT, message="simulator timed out"),
            ),
            FidelityFailure(
                cell_id="cell-fidelity-9",
                failure=StructuredFailure(code=FailureCode.PROVIDER, message="provider failed"),
            ),
        ),
        gate_id="fidelity-gate-v1",
        gate_sha256=_DIGEST,
        status="approved",
        approved_at=datetime(2026, 8, 11, tzinfo=UTC),
    )

    assert EvaluationDataset.model_validate_json(dataset.model_dump_json()) == dataset
    assert FidelityReport.model_validate_json(report.model_dump_json()) == report


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
    with pytest.raises(ValidationError, match="frozen 8-pair"):
        FidelityReport(
            schema_version=1,
            created_at=datetime(2026, 8, 11, tzinfo=UTC),
            code_revision="e7aad17",
            fidelity_report_id="fidelity-report-2",
            protocol_sha256=_DIGEST,
            overlap_cell_ids=tuple(f"cell-fidelity-{index}" for index in range(10)),
            planned_overlap_count=10,
            usable_overlap_count=7,
            failed_overlap_count=0,
            score_mae=0.08,
            gate_id="fidelity-gate-v1",
            gate_sha256=_DIGEST,
            status="approved",
            approved_at=datetime(2026, 8, 11, tzinfo=UTC),
        )
