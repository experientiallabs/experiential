"""Tests for durable teacher replay resume behavior."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
from replay_admitted_teacher_trajectories import (
    AdmittedTrace,
    BashAction,
    extract_bash_actions,
    load_admitted_traces,
    load_completed_results,
    matches_recorded_exit_timeout,
    parse_recorded_exit_code,
    recorded_output_body,
    replay_trace,
    validate_code_commit,
    validate_positive_integer,
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
        "status": "scored",
        "reward": 0.0,
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


@pytest.mark.parametrize(
    "status", ["replay_action_timeout", "episode_error", "infrastructure_setup_failed"]
)
def test_finished_unscored_result_is_retried(tmp_path: Path, status: str) -> None:
    write_result(
        tmp_path,
        {
            "task_id": "task-3",
            "source_row_sha256": "source-hash",
            "finished_at": 123.0,
            "status": status,
            "reward": None,
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


def test_optional_bash_description_does_not_change_replayed_command() -> None:
    """Accept the observed auxiliary description while replaying only command."""
    record = replay_record(selected_for_replay=True, selected_for_sft=False)
    messages = json.loads(record["source"]["message_log_json"])
    messages[1]["tool_calls"][0]["function"]["arguments"] = {
        "command": "pwd",
        "description": "Show the current directory",
    }
    actions = extract_bash_actions(messages)
    assert len(actions) == 1
    assert actions[0].command == "pwd"


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        ("ok\n\n(exit_code=0)", 0),
        ("(no output)\n\n(exit_code=-1)\n", -1),
        ("plain output without marker", None),
    ],
)
def test_recorded_exit_code_parser(output: str, expected: int | None) -> None:
    assert parse_recorded_exit_code(output) == expected


def test_recorded_timeout_metadata_is_extracted() -> None:
    record = replay_record(selected_for_replay=True, selected_for_sft=False)
    messages = json.loads(record["source"]["message_log_json"])
    messages[2]["content"] = "(no output)\n\n(exit_code=-1)"
    actions = extract_bash_actions(messages)
    assert actions[0].recorded_exit_code == -1


def test_recorded_output_body_removes_only_terminal_exit_marker() -> None:
    assert recorded_output_body("Terminated\n\n(exit_code=143)") == "Terminated"
    assert recorded_output_body("prefix (exit_code=143) suffix") == (
        "prefix (exit_code=143) suffix"
    )


def test_recorded_exit_timeout_match_requires_exit_and_output() -> None:
    action = BashAction(
        2,
        0,
        "call-1",
        "start services",
        "Terminated\n\n(exit_code=143)",
        143,
    )
    assert matches_recorded_exit_timeout(
        action, replay_output="Terminated\n", exit_code=143, timed_out=True
    )
    assert not matches_recorded_exit_timeout(
        action, replay_output="different\n", exit_code=143, timed_out=True
    )
    assert not matches_recorded_exit_timeout(
        action, replay_output="Terminated\n", exit_code=137, timed_out=True
    )
    assert not matches_recorded_exit_timeout(
        action, replay_output="Terminated\n", exit_code=143, timed_out=False
    )


@dataclass
class FakeReport:
    ok: bool = True
    shell_restarts: int = 1


@dataclass
class FakeCheck:
    passed: bool = True
    reward: float = 1.0


class FakeSandbox:
    def __init__(self) -> None:
        self.report = FakeReport()
        self.commands: list[str] = []

    async def check_initial_state(self) -> FakeCheck:
        return FakeCheck()

    async def shell(self, command: str, *, timeout_s: int) -> SimpleNamespace:
        assert timeout_s == 120
        self.commands.append(command)
        if command == "expected timeout":
            return SimpleNamespace(output="", exit_code=-1, timed_out=True)
        if command == "expected exit timeout":
            return SimpleNamespace(output="Terminated\n", exit_code=143, timed_out=True)
        if command == "wrong output exit timeout":
            return SimpleNamespace(output="different\n", exit_code=143, timed_out=True)
        if command == "wrong code exit timeout":
            return SimpleNamespace(output="Terminated\n", exit_code=137, timed_out=True)
        return SimpleNamespace(output="ok", exit_code=0, timed_out=False)

    async def verify(self) -> FakeCheck:
        return FakeCheck()


class FakeSession:
    def __init__(self, sandbox: FakeSandbox) -> None:
        self.sandbox = sandbox

    async def __aenter__(self) -> FakeSandbox:
        return self.sandbox

    async def __aexit__(self, *_args: object) -> None:
        return None


class FakePool:
    def __init__(self, sandbox: FakeSandbox) -> None:
        self.sandbox = sandbox

    def session(self, _task: object) -> FakeSession:
        return FakeSession(self.sandbox)


def test_replay_continues_after_timeout_recorded_by_teacher(tmp_path: Path) -> None:
    sandbox = FakeSandbox()
    result = asyncio.run(
        replay_trace(
            trace=AdmittedTrace(
                source_row_index=3,
                source_row_sha256="source-hash",
                task_id="task-3",
                rollout_id="rollout-3",
                first_user_content="task instruction",
                actions=(
                    BashAction(2, 0, "call-1", "expected timeout", "", -1),
                    BashAction(4, 0, "call-2", "continue", "ok", 0),
                ),
            ),
            task=SimpleNamespace(instruction="task instruction"),
            pool=FakePool(sandbox),
            out_root=tmp_path,
            bash_timeout_s=120,
        )
    )
    assert sandbox.commands == ["expected timeout", "continue"]
    assert result["status"] == "scored"
    assert result["reward"] == 1.0
    assert result["matched_recorded_timeout_actions"] == [0]


def test_replay_continues_after_exact_recorded_exit_transport_timeout(
    tmp_path: Path,
) -> None:
    sandbox = FakeSandbox()
    result = asyncio.run(
        replay_trace(
            trace=AdmittedTrace(
                source_row_index=3,
                source_row_sha256="source-hash",
                task_id="task-3",
                rollout_id="rollout-3",
                first_user_content="task instruction",
                actions=(
                    BashAction(
                        2,
                        0,
                        "call-1",
                        "expected exit timeout",
                        "Terminated\n\n(exit_code=143)",
                        143,
                    ),
                    BashAction(4, 0, "call-2", "continue", "ok", 0),
                ),
            ),
            task=SimpleNamespace(instruction="task instruction"),
            pool=FakePool(sandbox),
            out_root=tmp_path,
            bash_timeout_s=120,
        )
    )
    assert sandbox.commands == ["expected exit timeout", "continue"]
    assert result["status"] == "scored"
    assert result["reward"] == 1.0
    assert result["matched_recorded_exit_timeout_actions"] == [0]


@pytest.mark.parametrize(
    "command", ["wrong output exit timeout", "wrong code exit timeout"]
)
def test_replay_fails_closed_on_recorded_exit_timeout_mismatch(
    tmp_path: Path, command: str
) -> None:
    sandbox = FakeSandbox()
    result = asyncio.run(
        replay_trace(
            trace=AdmittedTrace(
                source_row_index=3,
                source_row_sha256="source-hash",
                task_id="task-3",
                rollout_id="rollout-3",
                first_user_content="task instruction",
                actions=(
                    BashAction(
                        2,
                        0,
                        "call-1",
                        command,
                        "Terminated\n\n(exit_code=143)",
                        143,
                    ),
                    BashAction(4, 0, "call-2", "continue", "ok", 0),
                ),
            ),
            task=SimpleNamespace(instruction="task instruction"),
            pool=FakePool(sandbox),
            out_root=tmp_path,
            bash_timeout_s=120,
        )
    )
    assert sandbox.commands == [command]
    assert result["status"] == "replay_action_timeout"
    assert result.get("reward") is None


@pytest.mark.parametrize("description", [None, 3, {"text": "bad"}])
def test_nonstring_optional_bash_description_fails_closed(description: object) -> None:
    record = replay_record(selected_for_replay=True, selected_for_sft=False)
    messages = json.loads(record["source"]["message_log_json"])
    messages[1]["tool_calls"][0]["function"]["arguments"] = {
        "command": "pwd",
        "description": description,
    }
    with pytest.raises(ValueError, match="description is not a string"):
        extract_bash_actions(messages)


def test_code_commit_validation_is_exact_and_backward_compatible() -> None:
    validate_code_commit(None)
    validate_code_commit("a" * 40)
    with pytest.raises(ValueError, match="full lowercase Git SHA"):
        validate_code_commit("A" * 40)
    with pytest.raises(ValueError, match="full lowercase Git SHA"):
        validate_code_commit("a" * 39)


@pytest.mark.parametrize("value", [1, 120, 1200])
def test_positive_runtime_values_are_accepted(value: int) -> None:
    validate_positive_integer(value, name="verify timeout")


@pytest.mark.parametrize("value", [0, -1])
def test_nonpositive_runtime_values_fail_closed(value: int) -> None:
    with pytest.raises(ValueError, match="verify timeout must be positive"):
        validate_positive_integer(value, name="verify timeout")
