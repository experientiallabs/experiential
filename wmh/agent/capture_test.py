"""Tests that captured runs round-trip through the real otel-genai adapter into Trace/Step form."""

from __future__ import annotations

from pathlib import Path

from wmh.agent.capture import run_to_trace, write_otel_traces
from wmh.agent.runtime import RunResult, StopReason
from wmh.core.types import Action, ActionKind, Observation, Step
from wmh.ingest import get_adapter


def _sample_result() -> RunResult:
    steps = [
        Step(
            action=Action(kind=ActionKind.TOOL_CALL, name="bash", arguments={"command": "ls"}),
            observation=Observation(content="file.txt", is_error=False),
            task="list the files",
        ),
        Step(
            action=Action(
                kind=ActionKind.TOOL_CALL, name="submit", arguments={"answer": "file.txt"}
            ),
            observation=Observation(content="file.txt"),
            task="list the files",
        ),
    ]
    return RunResult(
        task_id="task1",
        harness="base",
        steps=steps,
        stop_reason=StopReason.SUBMITTED,
        answer="file.txt",
        turns=2,
    )


def test_captured_trace_reingests_through_otel_adapter(tmp_path: Path) -> None:
    trace = run_to_trace(_sample_result(), gold=["found file.txt"])
    path = write_otel_traces([trace], tmp_path / "run.otel.jsonl")

    reloaded = get_adapter("otel-genai").from_file(str(path))
    assert len(reloaded) == 1
    got = reloaded[0]
    # The gold assertions we stamped survive the round-trip in trace metadata.
    assert got.metadata.get("gold") == ["found file.txt"]
    assert got.metadata.get("harness") == "base"
    # The bash action + its observation come back as a step.
    names = [s.action.name for s in got.steps]
    assert "bash" in names
    bash_step = next(s for s in got.steps if s.action.name == "bash")
    assert "file.txt" in bash_step.observation.content


def test_write_otel_traces_is_deterministic(tmp_path: Path) -> None:
    trace = run_to_trace(_sample_result(), gold=["g"])
    a = write_otel_traces([trace], tmp_path / "a.jsonl").read_text()
    b = write_otel_traces([trace], tmp_path / "b.jsonl").read_text()
    assert a == b  # no wall-clock/RNG: same run -> byte-identical output
