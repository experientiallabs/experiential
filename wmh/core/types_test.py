"""Tests for the core data types."""

from __future__ import annotations

from wmh.core.types import (
    Action,
    ActionKind,
    EnvState,
    HarnessContext,
    Observation,
    Session,
    Step,
    ToolDefinition,
    Trace,
)


def test_types_instantiate() -> None:
    action = Action(kind=ActionKind.TOOL_CALL, name="cd", arguments={"path": "/tmp"})
    obs = Observation(content="", is_error=False)
    step = Step(action=action, observation=obs, state_before=EnvState(), task="poke around")
    trace = Trace(trace_id="t1", steps=[step], source="file:demo.jsonl")
    session = Session(id="s1", task="poke around")
    assert trace.steps[0].action.name == "cd"
    assert session.history == []
    assert trace.harness is None  # bare traces carry no harness context


def test_harness_context_holds_system_prompt_and_tools() -> None:
    harness = HarnessContext(
        system_prompt="You are a coding agent.",
        tools=[
            ToolDefinition(
                name="bash",
                description="Run a shell command",
                parameters={"type": "object", "properties": {"command": {"type": "string"}}},
            )
        ],
    )
    trace = Trace(trace_id="t1", harness=harness)
    assert trace.harness is not None
    assert trace.harness.tools[0].name == "bash"
    assert not HarnessContext()  # empty context is falsy, so callers can gate on it


def test_harness_context_truthiness() -> None:
    assert HarnessContext(system_prompt="x")
    assert HarnessContext(tools=[ToolDefinition(name="bash")])
    assert not HarnessContext(system_prompt="", tools=[])
