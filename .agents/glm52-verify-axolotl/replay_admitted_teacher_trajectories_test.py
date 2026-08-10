"""Tests for durable teacher replay resume behavior."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from replay_admitted_teacher_trajectories import (
    AdmittedTrace,
    load_admitted_traces,
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


def replay_record(*, selected_for_replay: bool, selected_for_sft: bool) -> dict[str, object]:
    """Return one replay-loader record with a single bash action."""
    messages = [
        {"role": "user", "content": "task instruction"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "function": {"name": "bash", "arguments": '{"command":"pwd"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "/workdir\n"},
    ]
    return {
        "source_row_index": 3,
        "source_row_sha256": "hash",
        "source": {
            "task_id": "task-3",
            "rollout_id": "rollout-3",
            "message_log_json": json.dumps(messages),
        },
        "admission": {
            "selected_for_replay": selected_for_replay,
            "selected_for_sft": selected_for_sft,
        },
    }


@pytest.mark.parametrize(
    ("selected_for_replay", "selected_for_sft"), [(True, False), (False, True)]
)
def test_loader_accepts_replay_only_and_legacy_sft_records(
    tmp_path: Path, selected_for_replay: bool, selected_for_sft: bool
) -> None:
    path = tmp_path / "audit.jsonl"
    path.write_text(
        json.dumps(
            replay_record(
                selected_for_replay=selected_for_replay,
                selected_for_sft=selected_for_sft,
            )
        )
        + "\n"
    )
    traces = load_admitted_traces(path)
    assert len(traces) == 1
    assert traces[0].actions[0].command == "pwd"


def test_loader_rejects_rows_selected_for_neither_stage(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    path.write_text(
        json.dumps(replay_record(selected_for_replay=False, selected_for_sft=False))
        + "\n"
    )
    with pytest.raises(ValueError, match="not selected for replay"):
        load_admitted_traces(path)
