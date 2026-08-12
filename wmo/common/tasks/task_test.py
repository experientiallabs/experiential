"""Tests for canonical representative task contracts."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from wmo.common.tasks import TaskCase, TaskSet, ToolSchema

_DIGEST = "a" * 64


def _task() -> TaskCase:
    return TaskCase(
        task_id="refund-12",
        lineage_group_id="refund-lineage",
        partition="fit",
        instruction="Help the customer request a refund.",
        initial_context={"customer_tier": "standard"},
        tools=(
            ToolSchema(
                name="lookup_order",
                description="Look up an order by ID.",
                input_schema={"type": "object"},
            ),
        ),
        workload_weight=3.5,
        source_trace_ids=("0123456789abcdef0123456789abcdef",),
    )


def test_task_and_task_set_round_trip() -> None:
    """Representative task provenance and task-set hashes round trip without conversion."""
    task = _task()
    task_set = TaskSet(
        schema_version=1,
        created_at=datetime(2026, 8, 11, tzinfo=UTC),
        code_revision="e7aad17",
        task_set_id="task-set-v1",
        task_ids=(task.task_id,),
        tasks_path="tasks.jsonl",
        tasks_sha256=_DIGEST,
        coverage_path="coverage.json",
        coverage_sha256=_DIGEST,
    )

    assert TaskCase.model_validate_json(task.model_dump_json()) == task
    assert TaskSet.model_validate_json(task_set.model_dump_json()) == task_set


def test_task_rejects_nonfinite_weight_and_duplicate_source_trace_ids() -> None:
    """Weighting remains finite and source trace provenance remains unambiguous."""
    with pytest.raises(ValidationError):
        TaskCase(
            task_id="refund-12",
            lineage_group_id="refund-lineage",
            partition="fit",
            instruction="Help the customer request a refund.",
            workload_weight=float("nan"),
            source_trace_ids=("trace-1",),
        )
    with pytest.raises(ValidationError, match="duplicates"):
        TaskCase(
            task_id="refund-12",
            lineage_group_id="refund-lineage",
            partition="fit",
            instruction="Help the customer request a refund.",
            workload_weight=1,
            source_trace_ids=("trace-1", "trace-1"),
        )
    with pytest.raises(ValidationError, match="Field required"):
        TaskCase.model_validate(
            {
                "task_id": "refund-12",
                "lineage_group_id": "refund-lineage",
                "partition": "fit",
                "instruction": "Help the customer request a refund.",
                "workload_weight": 1,
            }
        )
    with pytest.raises(ValidationError, match="coverage path and digest"):
        TaskSet(
            schema_version=1,
            created_at=datetime(2026, 8, 11, tzinfo=UTC),
            code_revision="e7aad17",
            task_set_id="task-set-v1",
            task_ids=("refund-12",),
            tasks_path="tasks.jsonl",
            tasks_sha256=_DIGEST,
            coverage_path="coverage.json",
        )
