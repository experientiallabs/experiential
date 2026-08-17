"""Tests for explicit sparse router and fidelity measurement plans."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from wmo.common.evaluations import EvaluationCell, EvaluationPlan
from wmo.common.models import ModelSnapshot, RoutedCandidateSnapshot

_DIGEST = "a" * 64


def _candidate() -> RoutedCandidateSnapshot:
    return RoutedCandidateSnapshot(
        alias="candidate-economy",
        model=ModelSnapshot(
            provider="openai",
            model_id="gpt-5.4-mini",
            capabilities_sha256=_DIGEST,
            connection_sha256=_DIGEST,
        ),
    )


def test_evaluation_cells_reject_implicit_or_inconsistent_evidence() -> None:
    """A plan cannot hide a missing observed rollout or misuse fidelity comparisons."""
    with pytest.raises(ValidationError, match="require observed_rollout_id"):
        EvaluationCell(
            cell_id="cell-observed-1",
            task_id="task-1",
            candidate_alias="candidate-economy",
            repeat=0,
            purpose="fit",
            execution="observed",
        )
    with pytest.raises(ValidationError, match="fidelity cells"):
        EvaluationCell(
            cell_id="cell-fidelity-1",
            task_id="task-1",
            candidate_alias="candidate-economy",
            repeat=0,
            purpose="fidelity",
            execution="simulate",
        )
    observed = EvaluationCell(
        cell_id="cell-observed-1",
        task_id="task-1",
        candidate_alias="candidate-economy",
        repeat=0,
        purpose="fit",
        execution="observed",
        observed_rollout_id="rollout-observed-1",
    )
    mismatched_fidelity = EvaluationCell(
        cell_id="cell-fidelity-1",
        task_id="task-2",
        candidate_alias="candidate-economy",
        repeat=0,
        purpose="fidelity",
        execution="simulate",
        comparison_observed_cell_id=observed.cell_id,
    )
    with pytest.raises(ValidationError, match="preserve the compared"):
        EvaluationPlan(
            schema_version=4,
            created_at=datetime(2026, 8, 11, tzinfo=UTC),
            code_revision="e7aad17",
            plan_id="plan-v1",
            task_set_id="task-set-v1",
            candidate_snapshots=(_candidate(),),
            pricing_snapshot_id="pricing-v1",
            pricing_snapshot_sha256=_DIGEST,
            fidelity_protocol_sha256=_DIGEST,
            cells=(observed, mismatched_fidelity),
        )
    with pytest.raises(ValidationError, match="outside the plan snapshots"):
        EvaluationPlan(
            schema_version=2,
            created_at=datetime(2026, 8, 11, tzinfo=UTC),
            code_revision="e7aad17",
            plan_id="plan-v1",
            task_set_id="task-set-v1",
            candidate_snapshots=(_candidate(),),
            pricing_snapshot_id="pricing-v1",
            pricing_snapshot_sha256=_DIGEST,
            cells=(
                EvaluationCell(
                    cell_id="cell-unknown-candidate",
                    task_id="task-1",
                    candidate_alias="candidate-unknown",
                    repeat=0,
                    purpose="fit",
                    execution="simulate",
                ),
            ),
        )


def test_fidelity_cells_and_protocol_identity_are_one_measurement_scope() -> None:
    """Fidelity cells require protocol provenance without a threshold or decision artifact."""
    observed = EvaluationCell(
        cell_id="cell-observed-1",
        task_id="task-1",
        candidate_alias="candidate-economy",
        repeat=0,
        purpose="fit",
        execution="observed",
        observed_rollout_id="rollout-observed-1",
    )
    fidelity = EvaluationCell(
        cell_id="cell-fidelity-1",
        task_id="task-1",
        candidate_alias="candidate-economy",
        repeat=0,
        purpose="fidelity",
        execution="simulate",
        comparison_observed_cell_id=observed.cell_id,
    )
    with pytest.raises(ValidationError, match="protocol identity"):
        EvaluationPlan(
            schema_version=4,
            created_at=datetime(2026, 8, 11, tzinfo=UTC),
            code_revision="e7aad17",
            plan_id="plan-v1",
            task_set_id="task-set-v1",
            candidate_snapshots=(_candidate(),),
            pricing_snapshot_id="pricing-v1",
            pricing_snapshot_sha256=_DIGEST,
            cells=(observed, fidelity),
        )
    with pytest.raises(ValidationError, match="protocol identity"):
        EvaluationPlan(
            schema_version=4,
            created_at=datetime(2026, 8, 11, tzinfo=UTC),
            code_revision="e7aad17",
            plan_id="plan-v1",
            task_set_id="task-set-v1",
            candidate_snapshots=(_candidate(),),
            pricing_snapshot_id="pricing-v1",
            pricing_snapshot_sha256=_DIGEST,
            fidelity_protocol_sha256=_DIGEST,
            cells=(observed,),
        )
