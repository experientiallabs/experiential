"""Tests for joining harbor trial rewards with recorded token spans."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from wmh.distill.tokens import (
    TrialRecord,
    assemble_trial_records,
    load_trial_spans,
    read_trial_stop_reason,
)
from wmh.harness.scoring import ScoreCell
from wmh.providers.tinker import TokenRecorder, TokenSpan


def _span(call_index: int) -> TokenSpan:
    return TokenSpan(
        call_index=call_index,
        prompt_token_ids=[1, 2, call_index],
        sampled_token_ids=[65, 66],
        sampled_logprobs=[-0.5, -1.5],
    )


def _cell(task_id: str, attempt: int, *, reward: float, artifact_dir: Path) -> ScoreCell:
    return ScoreCell(
        task_id=task_id,
        attempt=attempt,
        reward=reward,
        passed=reward == 1.0,
        artifact_dir=str(artifact_dir),
    )


def _write_trace(trial_dir: Path, payload: str) -> None:
    agent_dir = trial_dir / "agent"
    agent_dir.mkdir(parents=True, exist_ok=True)
    (agent_dir / "wmh-run.json").write_text(payload, encoding="utf-8")


def test_load_trial_spans_round_trips_the_recorder_sink_format(tmp_path: Path) -> None:
    """The reader is coupled to TokenRecorder's as-built sink: write through it."""
    recorder = TokenRecorder(jsonl_path=tmp_path / "task-a__x1.jsonl")
    recorder.record(_span(0))
    recorder.record(_span(1))

    spans = load_trial_spans(tmp_path, "task-a__x1")

    assert spans == recorder.spans()
    assert [span.call_index for span in spans] == [0, 1]


def test_load_trial_spans_missing_sink_is_empty_not_an_error(tmp_path: Path) -> None:
    assert load_trial_spans(tmp_path, "task-a__never-ran") == []


def test_load_trial_spans_tolerates_blank_lines(tmp_path: Path) -> None:
    sink = tmp_path / "t.jsonl"
    sink.write_text(_span(0).model_dump_json() + "\n\n", encoding="utf-8")
    assert len(load_trial_spans(tmp_path, "t")) == 1


def test_load_trial_spans_corrupt_line_is_actionable(tmp_path: Path) -> None:
    sink = tmp_path / "t.jsonl"
    sink.write_text(_span(0).model_dump_json() + '\n{"call_index": "nope"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"line 2 of .*t\.jsonl"):
        load_trial_spans(tmp_path, "t")
    with pytest.raises(ValueError, match="delete this step's token sink directory"):
        load_trial_spans(tmp_path, "t")


def test_load_trial_spans_rejects_a_sink_appended_by_two_recorders(tmp_path: Path) -> None:
    """A call_index reset means two episodes shared one sink; spans can no longer be
    attributed to the reported trial, so the reader refuses instead of guessing."""
    sink = tmp_path / "t.jsonl"
    lines = [_span(0).model_dump_json(), _span(1).model_dump_json(), _span(0).model_dump_json()]
    sink.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"call_index sequence \[0, 1, 0\]"):
        load_trial_spans(tmp_path, "t")


def test_load_trial_spans_rejects_a_gap_in_the_sequence(tmp_path: Path) -> None:
    sink = tmp_path / "t.jsonl"
    lines = [_span(0).model_dump_json(), _span(2).model_dump_json()]
    sink.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected 0..1"):
        load_trial_spans(tmp_path, "t")


def test_read_trial_stop_reason_full_and_partial_traces(tmp_path: Path) -> None:
    full = tmp_path / "task-a__x1"
    _write_trace(
        full,
        json.dumps({"task_id": "t", "steps": [], "stop_reason": "submitted", "turns": 1}),
    )
    partial = tmp_path / "task-a__x2"
    _write_trace(
        partial,
        json.dumps({"stop_reason": "cancelled-by-harbor-timeout", "partial": True}),
    )
    assert read_trial_stop_reason(full) == "submitted"
    assert read_trial_stop_reason(partial) == "cancelled-by-harbor-timeout"


def test_read_trial_stop_reason_falls_back_to_the_trial_root(tmp_path: Path) -> None:
    trial = tmp_path / "task-a__x3"
    trial.mkdir()
    (trial / "wmh-run.json").write_text(json.dumps({"stop_reason": "max_turns"}), encoding="utf-8")
    assert read_trial_stop_reason(trial) == "max_turns"


def test_read_trial_stop_reason_missing_or_unreadable_is_none(tmp_path: Path) -> None:
    missing = tmp_path / "task-a__gone"
    assert read_trial_stop_reason(missing) is None

    unreadable = tmp_path / "task-a__bad"
    _write_trace(unreadable, "{not json")
    assert read_trial_stop_reason(unreadable) is None

    non_string = tmp_path / "task-a__odd"
    _write_trace(non_string, json.dumps({"stop_reason": 7}))
    assert read_trial_stop_reason(non_string) is None


def test_assemble_joins_spans_and_stop_reasons_by_trial_name(tmp_path: Path) -> None:
    sink_dir = tmp_path / "tokens"
    sink_dir.mkdir()
    trials_dir = tmp_path / "job"

    solved = trials_dir / "task-a__s1"
    _write_trace(solved, json.dumps({"stop_reason": "submitted"}))
    recorder = TokenRecorder(jsonl_path=sink_dir / "task-a__s1.jsonl")
    recorder.record(_span(0))
    recorder.record(_span(1))

    # This trial died before its first completion: no sink file, no trace.
    dead = trials_dir / "task-b__d1"
    dead.mkdir(parents=True)

    cells = [
        _cell("task-a", 1, reward=1.0, artifact_dir=solved),
        _cell("task-b", 1, reward=0.0, artifact_dir=dead),
    ]

    records = assemble_trial_records(cells, sink_dir)

    assert [record.trial_name for record in records] == ["task-a__s1", "task-b__d1"]
    assert records[0].task_id == "task-a"
    assert records[0].attempt == 1
    assert records[0].reward == 1.0
    assert records[0].passed is True
    assert records[0].spans == recorder.spans()
    assert records[0].stop_reason == "submitted"
    assert records[0].artifact_dir == str(solved)
    # The span-less trial is recorded, not dropped: its reward is real signal.
    assert records[1].spans == []
    assert records[1].stop_reason is None
    assert records[1].passed is False


def test_assemble_accepts_an_injected_stop_reason_reader(tmp_path: Path) -> None:
    seen: list[Path] = []

    def reader(artifact_dir: Path) -> str | None:
        seen.append(artifact_dir)
        return "custom"

    trial = tmp_path / "job" / "task-a__s1"
    records = assemble_trial_records(
        [_cell("task-a", 1, reward=0.0, artifact_dir=trial)],
        tmp_path / "tokens",
        read_stop_reason=reader,
    )
    assert records[0].stop_reason == "custom"
    assert seen == [trial]


def test_assemble_rejects_cells_without_an_artifact_dir(tmp_path: Path) -> None:
    cell = ScoreCell(task_id="task-a", attempt=1, reward=0.0, passed=False, artifact_dir="")
    with pytest.raises(ValueError, match="carries no artifact dir"):
        assemble_trial_records([cell], tmp_path)


def test_trial_record_validation() -> None:
    record = TrialRecord(
        task_id="task-a",
        attempt=1,
        trial_name="task-a__s1",
        reward=1.0,
        passed=True,
        artifact_dir="/tmp/job/task-a__s1",
    )
    assert record.spans == []
    assert record.stop_reason is None
    with pytest.raises(ValidationError):
        TrialRecord(
            task_id="task-a",
            attempt=0,
            trial_name="task-a__s1",
            reward=1.0,
            passed=True,
            artifact_dir="x",
        )
    with pytest.raises(ValidationError):
        TrialRecord(
            task_id="task-a",
            attempt=1,
            trial_name="task-a__s1",
            reward=1.5,
            passed=True,
            artifact_dir="x",
        )
