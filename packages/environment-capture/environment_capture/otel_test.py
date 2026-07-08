"""Tests for the OTel GenAI JSONL emitter."""

from __future__ import annotations

import json
from pathlib import Path

from environment_capture.otel import Span, trace_id_for, trajectory_to_spans, write_spans_jsonl
from environment_capture.trajectory import StepRecord, Task, ToolCall, Trajectory


def _trajectory() -> Trajectory:
    return Trajectory(
        task=Task(
            task_id="fb-train-0", prompt="What is 3M's FY2018 capex?", data={"stratum": "easy"}
        ),
        steps=[
            StepRecord(
                action=ToolCall(name="bash", arguments={"command": "ls docs"}),
                output="a.txt\nb.txt",
                is_error=False,
            ),
            StepRecord(
                action=ToolCall(name="bash", arguments={"command": "cat missing.txt"}),
                output="cat: missing.txt: No such file or directory",
                is_error=True,
            ),
        ],
        final_answer="$1577.00",
        reward=1.0,
        model="gpt-5.4",
        split="train",
        metadata={"passed": True},
    )


def test_trace_id_is_deterministic_and_distinct() -> None:
    assert trace_id_for("financebench", "fb-train-0") == trace_id_for("financebench", "fb-train-0")
    assert trace_id_for("financebench", "fb-train-0") != trace_id_for("financebench", "fb-train-1")
    assert len(trace_id_for("financebench", "fb-train-0")) == 32


def test_trajectory_to_spans_emits_action_observation_pairs() -> None:
    spans = trajectory_to_spans(_trajectory(), benchmark="financebench")
    assert len(spans) == 4  # 2 steps x (action span + tool span)

    action0, tool0, action1, tool1 = spans
    trace_id = trace_id_for("financebench", "fb-train-0")
    assert all(s["traceId"] == trace_id for s in spans)
    assert len({s["spanId"] for s in spans}) == 4
    # Spans are ordered by start time so ingestion reconstructs the step order.
    starts = [s["startTimeUnixNano"] for s in spans]
    assert starts == sorted(starts)

    def attrs(span: Span) -> dict[str, str]:
        raw = span["attributes"]
        assert isinstance(raw, list)
        out: dict[str, str] = {}
        for attribute in raw:
            assert isinstance(attribute, dict)
            value = attribute["value"]
            assert isinstance(value, dict)
            key, string_value = attribute["key"], value["stringValue"]
            assert isinstance(key, str) and isinstance(string_value, str)
            out[key] = string_value
        return out

    a0 = attrs(action0)
    assert a0["gen_ai.operation.name"] == "chat"
    assert a0["gen_ai.tool.name"] == "bash"
    assert json.loads(a0["gen_ai.tool.call.arguments"]) == {"command": "ls docs"}
    # Task prompt + trace metadata ride only on the first action span.
    assert a0["gen_ai.prompt"] == "What is 3M's FY2018 capex?"
    metadata = json.loads(a0["wmh.trace.metadata"])
    assert metadata["benchmark"] == "financebench"
    assert metadata["task_id"] == "fb-train-0"
    assert metadata["model"] == "gpt-5.4"
    assert metadata["reward"] == 1.0
    assert metadata["passed"] is True
    a1 = attrs(action1)
    assert "gen_ai.prompt" not in a1
    assert "wmh.trace.metadata" not in a1

    t0 = attrs(tool0)
    assert t0["gen_ai.operation.name"] == "execute_tool"
    assert t0["gen_ai.tool.message"] == "a.txt\nb.txt"
    assert tool0["status"] == {"code": "STATUS_CODE_OK"}
    assert tool1["status"] == {"code": "STATUS_CODE_ERROR"}


def test_write_spans_jsonl_round_trips(tmp_path: Path) -> None:
    spans = trajectory_to_spans(_trajectory(), benchmark="financebench")
    out = tmp_path / "traces.otel.jsonl"
    n = write_spans_jsonl(spans, out)
    assert n == 4
    lines = out.read_text().splitlines()
    assert [json.loads(line)["spanId"] for line in lines] == [s["spanId"] for s in spans]
    # Append mode extends rather than truncates.
    n2 = write_spans_jsonl(spans, out, append=True)
    assert n2 == 4
    assert len(out.read_text().splitlines()) == 8
