"""Tests for scenario extraction from traces."""

from __future__ import annotations

from wmo.common.core.types import Action, ActionKind, Observation, Step, Trace
from wmo.simulation.scenarios.spec import Scenario, scenarios_from_traces


def _trace(trace_id: str, task: str | None) -> Trace:
    step = Step(
        action=Action(kind=ActionKind.TOOL_CALL, name="t", arguments={}),
        observation=Observation(content="ok"),
        task=task,
    )
    return Trace(trace_id=trace_id, steps=[step])


def test_extracts_unique_tasks_with_provenance() -> None:
    traces = [
        _trace("a", "book a flight"),
        _trace("b", "cancel order 7"),
        _trace("c", "book a flight"),  # duplicate task -> same scenario, extra provenance
    ]
    scenarios = scenarios_from_traces(traces)
    assert scenarios == [
        Scenario(task="book a flight", provenance=["a", "c"]),
        Scenario(task="cancel order 7", provenance=["b"]),
    ]


def test_skips_traces_without_a_task() -> None:
    traces = [_trace("a", None), _trace("b", "   "), _trace("c", "real task")]
    scenarios = scenarios_from_traces(traces)
    assert [s.task for s in scenarios] == ["real task"]


def test_empty_input_gives_empty_output() -> None:
    assert scenarios_from_traces([]) == []


def test_tools_hint_from_traces_summarizes_tool_surface() -> None:
    from wmo.common.core.types import Action, ActionKind, EnvState, Observation, Step, Trace
    from wmo.simulation.scenarios.spec import tools_hint_from_traces

    steps = [
        Step(
            state_before=EnvState(),
            action=Action(kind=ActionKind.TOOL_CALL, name="run_sql", arguments={"query": "q"}),
            observation=Observation(content="ok"),
        ),
        Step(
            state_before=EnvState(),
            action=Action(
                kind=ActionKind.TOOL_CALL, name="run_sql", arguments={"query": "q", "db": "x"}
            ),
            observation=Observation(content="ok"),
        ),
        Step(
            state_before=EnvState(),
            action=Action(kind=ActionKind.MESSAGE, content="hello"),
            observation=Observation(content="ok"),
        ),
    ]
    traces = [Trace(trace_id="t1", steps=steps)]
    hint = tools_hint_from_traces(traces)
    assert "run_sql" in hint
    assert "db" in hint and "query" in hint
    assert "hello" not in hint  # messages are not tools


def test_tools_hint_empty_for_toolless_traces() -> None:
    from wmo.common.core.types import Action, ActionKind, EnvState, Observation, Step, Trace
    from wmo.simulation.scenarios.spec import tools_hint_from_traces

    steps = [
        Step(
            state_before=EnvState(),
            action=Action(kind=ActionKind.MESSAGE, content="hi"),
            observation=Observation(content="ok"),
        )
    ]
    assert tools_hint_from_traces([Trace(trace_id="t", steps=steps)]) == ""
