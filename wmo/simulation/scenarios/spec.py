"""Scenarios: the task prompts an agent trains and is evaluated on, derived from traces.

v1 scenario creation is deliberately minimal (the prompts already recorded in the corpus's traces
ARE the scenarios): `scenarios_from_traces` extracts one `Scenario` per unique task from the given
traces. Callers control leakage by choosing which traces to pass — extract training scenarios from
the train split and held-out scenarios from the test split, and the two can never overlap because
the whole-trace split already separated them.

Principled scenario *generation* (coverage, difficulty calibration, counterfactuals) is a later
layer that will produce the same `Scenario` type.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from wmo.core.types import ActionKind, Trace


class Scenario(BaseModel):
    """One task an agent can attempt against an environment."""

    task: str  # the prompt handed to the agent (and to Env.reset)
    provenance: list[str] = Field(default_factory=list)  # trace_ids this scenario came from


def scenarios_from_traces(traces: list[Trace]) -> list[Scenario]:
    """Extract the unique task prompts from `traces`, in first-seen order.

    Traces without a task (or with a whitespace-only one) are skipped: a scenario is exactly "a
    prompt we can hand to an agent", and an empty prompt isn't one. Duplicate tasks collapse into
    a single scenario whose `provenance` lists every contributing trace.
    """
    by_task: dict[str, Scenario] = {}
    for trace in traces:
        task = _trace_task(trace)
        if task is None:
            continue
        scenario = by_task.get(task)
        if scenario is None:
            by_task[task] = Scenario(task=task, provenance=[trace.trace_id])
        elif trace.trace_id not in scenario.provenance:
            scenario.provenance.append(trace.trace_id)
    return list(by_task.values())


def _trace_task(trace: Trace) -> str | None:
    """The trace's task prompt: first non-empty per-step task (steps carry it in this corpus)."""
    for step in trace.steps:
        if step.task and step.task.strip():
            return step.task.strip()
    return None


def tools_hint_from_traces(traces: list[Trace]) -> str:
    """Summarize the corpus's observed tool surface for the rollout agent's system prompt.

    One line per tool: name + the union of argument keys seen in the traces (the D28
    tools.json derivation, inlined). Empty string for tool-less corpora. This is the
    trace-derived stopgap for the scenario tool-surface contract: agents told what exists
    stop being scored on their willingness to hallucinate affordances.
    """
    args_by_tool: dict[str, set[str]] = {}
    for trace in traces:
        for step in trace.steps:
            action = step.action
            if action.kind is not ActionKind.TOOL_CALL or not action.name:
                continue
            args_by_tool.setdefault(action.name, set()).update(action.arguments)
    if not args_by_tool:
        return ""
    lines = [f"- {name}({', '.join(sorted(keys))})" for name, keys in sorted(args_by_tool.items())]
    return "AVAILABLE TOOLS (observed in this environment):\n" + "\n".join(lines)
