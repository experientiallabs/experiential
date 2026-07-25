"""Tests for joining harbor trial rewards with recorded token spans."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from llm_waterfall.types import (
    ChatFunctionCall,
    ChatFunctionDefinition,
    ChatMessage,
    ChatTool,
    ChatToolCall,
)
from pydantic import ValidationError

from wmh.distill.rendering import ParsedAssistantMessage
from wmh.distill.tokens import (
    TrialRecord,
    assemble_trial_records,
    load_trial_spans,
    read_trial_stop_reason,
    reconstruct_conversation,
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


_BASH_TOOL = ChatTool(
    function=ChatFunctionDefinition(
        name="bash", description="run bash", parameters={"type": "object"}
    )
)


class _ScriptedParser:
    """A `SampledTurnParser` that parses sampled ids from a canned table.

    Keeps the replay tests independent of any real renderer: the mapping from
    token ids to an assistant turn is exactly what a renderer supplies.
    """

    def __init__(self, table: dict[tuple[int, ...], ParsedAssistantMessage]) -> None:
        self._table = table

    def parse_response(self, sampled_ids: list[int]) -> ParsedAssistantMessage:
        parsed = self._table.get(tuple(sampled_ids))
        assert parsed is not None, f"no scripted parse for sampled ids {sampled_ids}"
        return parsed


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


def _tool_call() -> ChatToolCall:
    return ChatToolCall(
        id="call_0", function=ChatFunctionCall(name="bash", arguments='{"cmd": "ls"}')
    )


def _tool_use_episode() -> tuple[list[TokenSpan], _ScriptedParser]:
    """Two spans of one tool-using episode, shaped exactly as the provider records them."""
    spans = [
        TokenSpan(
            call_index=0,
            prompt_token_ids=[1, 2, 3],
            sampled_token_ids=[10, 11],
            sampled_logprobs=[-0.1, -0.2],
            delta_messages=[
                ChatMessage(role="system", content="be terse"),
                ChatMessage(role="user", content="list files"),
            ],
            tools=[_BASH_TOOL],
        ),
        TokenSpan(
            call_index=1,
            prompt_token_ids=[1, 2, 3, 10, 11, 4],
            sampled_token_ids=[12, 13],
            sampled_logprobs=[-0.3, -0.4],
            # Only the NEW message: the caller's echo of the assistant turn is
            # the previous span's sampled ids, never repeated here.
            delta_messages=[
                ChatMessage.model_validate(
                    {"role": "tool", "content": "a.txt b.txt", "tool_call_id": "call_0"}
                )
            ],
            tools=[_BASH_TOOL],
        ),
    ]
    parser = _ScriptedParser(
        {
            (10, 11): ParsedAssistantMessage(text="on it", tool_calls=[_tool_call()], stopped=True),
            (12, 13): ParsedAssistantMessage(text="a.txt b.txt is all", stopped=True),
        }
    )
    return spans, parser


def test_load_trial_spans_round_trips_the_canonical_messages_and_tools(tmp_path: Path) -> None:
    """The teacher-facing fields must survive the recorder's jsonl sink verbatim."""
    spans, _ = _tool_use_episode()
    recorder = TokenRecorder(jsonl_path=tmp_path / "task-a__x1.jsonl")
    for span in spans:
        recorder.record(span)

    loaded = load_trial_spans(tmp_path, "task-a__x1")

    assert loaded == spans
    assert loaded[0].delta_messages is not None
    assert [message.role for message in loaded[0].delta_messages] == ["system", "user"]
    assert loaded[1].delta_messages is not None
    assert loaded[1].delta_messages[0].tool_call_id == "call_0"
    assert loaded[1].tools is not None
    assert loaded[1].tools[0].function.name == "bash"


def test_load_trial_spans_still_reads_a_sink_without_the_canonical_messages(
    tmp_path: Path,
) -> None:
    """Real sinks on disk carry only the four original keys; they MUST still load."""
    old_format = [
        {
            "call_index": 0,
            "prompt_token_ids": [1, 2],
            "sampled_token_ids": [65, 66],
            "sampled_logprobs": [-0.5, -1.5],
        },
        {
            "call_index": 1,
            "prompt_token_ids": [1, 2, 65, 66, 7],
            "sampled_token_ids": [67],
            "sampled_logprobs": [-0.25],
        },
    ]
    sink = tmp_path / "t.jsonl"
    sink.write_text("".join(json.dumps(item) + "\n" for item in old_format), encoding="utf-8")

    spans = load_trial_spans(tmp_path, "t")

    assert [span.call_index for span in spans] == [0, 1]
    assert spans[0].sampled_logprobs == [-0.5, -1.5]
    assert all(span.delta_messages is None for span in spans)
    assert all(span.tools is None for span in spans)


def test_reconstruct_conversation_replays_a_multi_turn_tool_use_episode() -> None:
    spans, parser = _tool_use_episode()

    replay = reconstruct_conversation(spans, parser)

    assert replay is not None
    assert [message.role for message in replay.messages] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert replay.messages[1].content == "list files"
    first_assistant = replay.messages[2]
    assert first_assistant.content == "on it"
    assert first_assistant.tool_calls is not None
    assert first_assistant.tool_calls[0].function.name == "bash"
    assert first_assistant.tool_calls[0].function.arguments == '{"cmd": "ls"}'
    assert replay.messages[3].tool_call_id == "call_0"
    assert replay.messages[4].content == "a.txt b.txt is all"
    assert replay.messages[4].tool_calls is None
    # The planner pairs each sampled span with the message its tokens produced.
    assert replay.assistant_index_by_span == {0: 2, 1: 4}
    assert replay.tools == [_BASH_TOOL]


def test_reconstruct_conversation_returns_none_for_an_old_sink(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """An old sink has no canonical messages: degrade honestly, never guess one."""
    sink = tmp_path / "t.jsonl"
    sink.write_text(
        json.dumps(
            {
                "call_index": 0,
                "prompt_token_ids": [1, 2],
                "sampled_token_ids": [10, 11],
                "sampled_logprobs": [-0.5, -1.5],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    spans = load_trial_spans(tmp_path, "t")
    _, parser = _tool_use_episode()

    with caplog.at_level("WARNING", logger="wmh.distill.tokens"):
        assert reconstruct_conversation(spans, parser) is None

    assert any("carries no delta_messages" in record.message for record in caplog.records)
    assert any("Re-run the rollout step" in record.message for record in caplog.records)


def test_reconstruct_conversation_returns_none_without_spans() -> None:
    _, parser = _tool_use_episode()
    assert reconstruct_conversation([], parser) is None


def test_reconstruct_conversation_returns_none_when_spans_disagree_about_tools(
    caplog: pytest.LogCaptureFixture,
) -> None:
    spans, parser = _tool_use_episode()
    spans[1].tools = None

    with caplog.at_level("WARNING", logger="wmh.distill.tokens"):
        assert reconstruct_conversation(spans, parser) is None

    assert any("different tool schemas" in record.message for record in caplog.records)


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
