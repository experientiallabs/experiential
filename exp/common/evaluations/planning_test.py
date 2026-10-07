"""Evaluation plan construction over an explicit task subset."""

from pathlib import Path

import pytest

from exp.common.evaluations import EvaluationEvidenceError
from exp.common.evaluations.build_test import _TIME, _planned_fixture
from exp.common.evaluations.planning import build_evaluation_plan


def test_plan_task_subset_limits_cells_and_rejects_unknown_tasks(tmp_path: Path) -> None:
    """A task subset plans only those tasks and must name distinct tasks of the set."""
    store, plan, _ = _planned_fixture(tmp_path)
    subset = ("task-fit-03", "task-held-out")
    planned = build_evaluation_plan(
        store,
        task_set_id=plan.task_set_id,
        candidate_snapshots=plan.candidate_snapshots,
        pricing_snapshot_id=plan.pricing_snapshot_id,
        observed_cells=(),
        task_ids=subset,
        created_at=_TIME,
        code_revision="test-revision",
    )
    assert {cell.task_id for cell in planned.cells} == set(subset)
    assert len(planned.cells) == 4
    before = store.list_ids()
    for task_ids in ((), ("task-fit-03", "task-fit-03"), ("missing",)):
        with pytest.raises(EvaluationEvidenceError, match="planned task subset"):
            build_evaluation_plan(
                store,
                task_set_id=plan.task_set_id,
                candidate_snapshots=plan.candidate_snapshots,
                pricing_snapshot_id=plan.pricing_snapshot_id,
                observed_cells=(),
                task_ids=task_ids,
                created_at=_TIME,
                code_revision="test-revision",
            )
    assert store.list_ids() == before
