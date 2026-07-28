"""Round-trip test: environment-capture's emitted spans parse through the real ingest adapter.

`environment_capture.otel` writes OTel GenAI span JSONL; `wmo.ingest.otel_genai` reads it. There
is no longer any code dependency between the two — `wmo` vendors the bundle-fetch core it used
to import (`wmo/hub.py`) and the wheel carries no `environment-capture` requirement — but the
WIRE format is still a shared contract, and drift in it fails silently on users: the producer
keeps writing, the consumer keeps parsing, and the fields just stop lining up.

That is why this file lives in `contracts/` and not in either tree. It imports BOTH packages, so
it cannot sit under `wmo/` (which must import nothing from `packages/`) or under
`packages/environment-capture/` (members never import `wmo`). It belongs to neither party; it
belongs to the boundary. See AGENTS.md § Monorepo.
"""

from __future__ import annotations

import json
from pathlib import Path

from environment_capture.otel import trajectory_to_spans, write_spans_jsonl
from environment_capture.trajectory import StepRecord, Task, ToolCall, Trajectory

from wmo.ingest.otel_genai import OtelGenAIAdapter


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
