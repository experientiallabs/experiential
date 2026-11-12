"""Tests for numbering trajectory points by actor turn."""

from __future__ import annotations

import pytest

from exp.common.core.artifacts import JsonObject, JsonValue
from exp.optimize.evaluation.variance.build_source import build_sources
from exp.simulation.mining.traceml_turns import trajectory_by_turns, workflow_step_turns


def _receipt(*calls: tuple[str, str], finish: str = "completed") -> JsonObject:
    """One saved actor call receipt making ``(name, call_id)`` calls in one response."""
    tool_calls: list[JsonValue] = [
        {"call_id": call_id, "name": name, "arguments": {}} for name, call_id in calls
    ]
    output: JsonObject = {"content": None if calls else "done", "tool_calls": tool_calls}
    return {"response": {"output": output, "finish_reason": finish}}


def _called(name: str, call_id: str) -> JsonObject:
    """One saved step for the call."""
    return {"action": {"call_id": call_id, "name": name, "arguments": {}}, "status": "valid"}


def test_steps_of_one_assistant_turn_share_its_point() -> None:
    """Parallel calls of one response are decided at the same point, the turn's index."""
    steps = [_called("read", "a"), _called("grep", "b"), _called("edit", "c"), _called("test", "d")]
    calls = [
        _receipt(("read", "a"), ("grep", "b")),
        _receipt(("edit", "c"), finish="length"),
        _receipt(("edit", "c"), ("test", "d")),
        _receipt(),
    ]
    assert workflow_step_turns(steps, calls) == (0, 0, 2, 2)
    trajectory = trajectory_by_turns("t", "kimi-k3", steps, turns=(0, 0, 2, 2), final_answer="done")
    assert [step.boundary for step in trajectory.steps] == [0, 0, 2, 2, 3]
    assert workflow_step_turns(steps[:3], calls) == (0, 0, 2)


def test_steps_that_disagree_with_the_actor_calls_are_refused() -> None:
    """A step that is not the next call, or a step left over, has no known fork point."""
    calls = [_receipt(("read", "a"))]
    with pytest.raises(ValueError, match="disagree"):
        workflow_step_turns([_called("read", "z")], calls)
    with pytest.raises(ValueError, match="match no recorded"):
        workflow_step_turns([_called("read", "a"), _called("edit", "b")], calls)
    with pytest.raises(ValueError, match="turns"):
        trajectory_by_turns("t", "kimi-k3", [_called("read", "a")] * 2, turns=(0,))
    with pytest.raises(ValueError, match="in order"):
        trajectory_by_turns("t", "kimi-k3", [_called("read", "a")] * 2, turns=(1, 0))


def test_turn_numbering_stays_outside_the_variance_build_identity() -> None:
    """Editing this module must never re-key saved variance builds."""
    assert "exp.simulation.mining.traceml_turns" not in build_sources()
