"""Round-trip test: emitted spans must parse through wmh's real OTel GenAI ingest adapter.

This is the one test in the package allowed to import wmh — it pins the wire format against its
actual consumer. If environment-capture is ever extracted to its own repo, this test moves to the
wmh side of the boundary.
"""

from __future__ import annotations

import json
from pathlib import Path

from environment_capture.otel import trajectory_to_spans, write_spans_jsonl
from environment_capture.trajectory import StepRecord, Task, ToolCall, Trajectory
from wmh.ingest.otel_genai import OtelGenAIAdapter


def test_emitted_spans_ingest_as_the_same_steps(tmp_path: Path) -> None:
    trajectory = Trajectory(
        task=Task(task_id="fb-train-0", prompt="What is 3M's FY2018 capex?", data={}),
        steps=[
            StepRecord(
                action=ToolCall(name="bash", arguments={"command": "ls docs"}),
                output="a.txt",
                is_error=False,
            ),
            StepRecord(
                action=ToolCall(name="bash", arguments={"command": "cat gone"}),
                output="cat: gone: No such file or directory",
                is_error=True,
            ),
        ],
        final_answer="$1577.00",
        reward=1.0,
        model="gpt-5.4",
        split="train",
    )
    path = tmp_path / "traces.otel.jsonl"
    write_spans_jsonl(trajectory_to_spans(trajectory, benchmark="financebench"), path)

    traces = OtelGenAIAdapter().from_file(str(path))
    assert len(traces) == 1
    trace = traces[0]
    assert trace.metadata["benchmark"] == "financebench"
    assert trace.metadata["task_id"] == "fb-train-0"
    assert trace.metadata["reward"] == 1.0
    assert len(trace.steps) == 2

    first, second = trace.steps
    assert first.task == "What is 3M's FY2018 capex?"
    assert first.action.name == "bash"
    arguments = first.action.arguments
    if isinstance(arguments, str):  # ingest may keep arguments as the raw JSON string
        arguments = json.loads(arguments)
    assert arguments["command"] == "ls docs"
    assert first.observation.content == "a.txt"
    assert first.observation.is_error is False
    assert second.observation.is_error is True
