"""Tests for canonical representative task contracts."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from exp.common.tasks import TaskCase, TaskSet

_DIGEST = "a" * 64


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
    with pytest.raises(ValidationError, match="at least one source trace"):
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


def test_task_origin_governs_source_traces_and_reference_answers() -> None:
    """Mined tasks name source traces; authored tasks have none and alone carry a reference."""
    shared = {
        "task_id": "refund-12",
        "lineage_group_id": "refund-lineage",
        "partition": "fit",
        "instruction": "Help the customer request a refund.",
        "workload_weight": 1,
    }
    with pytest.raises(ValidationError, match="at least one source trace"):
        TaskCase.model_validate(shared)
    with pytest.raises(ValidationError, match="only an authored task"):
        TaskCase.model_validate(
            {**shared, "source_trace_ids": ["trace-1"], "reference_answer": "Refunded."}
        )
    with pytest.raises(ValidationError, match="no source traces"):
        TaskCase.model_validate({**shared, "origin": "authored", "source_trace_ids": ["trace-1"]})
    authored = TaskCase.model_validate(
        {**shared, "origin": "authored", "reference_answer": "Refunded."}
    )
    assert authored.model_dump(mode="json")["reference_answer"] == "Refunded."
    assert TaskCase.model_validate_json(authored.model_dump_json()) == authored
    mined = TaskCase.model_validate({**shared, "source_trace_ids": ["trace-1"]})
    # Mined records keep their serialized bytes, so existing task-set digests are unchanged.
    assert set(mined.model_dump(mode="json")) == {*shared, "initial_context", "tools"} | {
        "source_trace_ids"
    }
