"""Round-trip tests: Trace -> otel_writer JSONL -> otel-genai adapter -> Trace (lossless)."""

from __future__ import annotations

from pathlib import Path

from wmh.core.types import Action, ActionKind, EnvState, Observation, Step, Trace
from wmh.ingest.otel_genai import OtelGenAIAdapter
from wmh.ingest.otel_writer import trace_to_spans, write_traces_jsonl


def _trace() -> Trace:
    return Trace(
        trace_id="a" * 32,
        source="test",
        metadata={"benchmark": "demo", "gold": {"answer": "42"}},
        steps=[
            Step(
                action=Action(kind=ActionKind.TOOL_CALL, name="get_user", arguments={"id": "u1"}),
                observation=Observation(content="found u1", is_error=False),
                state_before=EnvState(structured={"cart": ["a"]}, scratchpad="logged in"),
                task="look up u1",
            ),
            Step(
                action=Action(kind=ActionKind.TOOL_CALL, name="rm", arguments={"p": "/x"}),
                observation=Observation(content="permission denied", is_error=True),
                task="look up u1",
            ),
            Step(
                action=Action(kind=ActionKind.MESSAGE, content="done"),
                observation=Observation(content=""),
                task="look up u1",
            ),
        ],
    )


def test_roundtrip_preserves_steps_state_and_metadata(tmp_path: Path) -> None:
    out = tmp_path / "rt.otel.jsonl"
    n_spans = write_traces_jsonl([_trace()], out)
    # 2 paired steps (action+obs) + 1 trailing message (action only) = 5 spans.
    assert n_spans == 5

    reloaded = OtelGenAIAdapter().from_file(str(out))
    assert len(reloaded) == 1
    rt = reloaded[0]
    assert rt.trace_id == "a" * 32
    assert rt.metadata == {"benchmark": "demo", "gold": {"answer": "42"}}
    assert len(rt.steps) == 3

    s0 = rt.steps[0]
    assert s0.action.name == "get_user"
    assert s0.action.arguments == {"id": "u1"}
    assert s0.observation.content == "found u1"
    assert s0.observation.is_error is False
    assert s0.state_before.structured == {"cart": ["a"]}
    assert s0.state_before.scratchpad == "logged in"
    assert s0.task == "look up u1"

    s1 = rt.steps[1]
    assert s1.action.name == "rm"
    assert s1.observation.is_error is True

    s2 = rt.steps[2]
    assert s2.action.kind == ActionKind.MESSAGE
    assert s2.action.content == "done"
    assert s2.observation.content == ""


def test_writer_is_deterministic() -> None:
    # No wall-clock: the same trace serializes byte-identically across calls.
    assert trace_to_spans(_trace()) == trace_to_spans(_trace())
