"""Tests for durable teacher replay resume behavior."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from replay_admitted_teacher_trajectories import (
    AdmittedTrace,
    load_completed_results,
)


def trace() -> AdmittedTrace:
    """Build one trace without actions for resume metadata tests."""
    return AdmittedTrace(
        source_row_index=3,
        source_row_sha256="source-hash",
        task_id="task-3",
        rollout_id="rollout-3",
        first_user_content="task",
        actions=(),
    )


def write_result(root: Path, value: dict[str, object]) -> None:
    """Write one synthetic episode result."""
    path = root / "episodes/task-3/replay_result.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(value))


def test_only_finished_matching_results_are_resumed(tmp_path: Path) -> None:
    value = {
        "task_id": "task-3",
        "source_row_sha256": "source-hash",
        "finished_at": 123.0,
    }
    write_result(tmp_path, value)
    assert load_completed_results(tmp_path, [trace()]) == {"task-3": value}


def test_partial_result_is_retried(tmp_path: Path) -> None:
    write_result(
        tmp_path,
        {
            "task_id": "task-3",
            "source_row_sha256": "source-hash",
            "finished_at": None,
        },
    )
    assert load_completed_results(tmp_path, [trace()]) == {}


def test_source_mismatch_fails_closed(tmp_path: Path) -> None:
    write_result(
        tmp_path,
        {
            "task_id": "task-3",
            "source_row_sha256": "wrong",
            "finished_at": 123.0,
        },
    )
    with pytest.raises(ValueError, match="source hash mismatch"):
        load_completed_results(tmp_path, [trace()])
