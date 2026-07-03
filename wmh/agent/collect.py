"""Trace collection: run the agent for real in E2B sandboxes and capture the traces.

This is the *producer* side of the build-agent path. For each task we spin up a fresh E2B sandbox
(seeded with the task's `setup`), run the fixed harness against it, and capture the resulting run as
a `Trace` with the task's gold stamped in metadata. The collected traces are written in the
`otel-genai` JSONL shape so `wmh build` ingests them directly — the world model is literally built
from the agent's own behavior, and can then be used to evaluate/evolve that same agent closed-loop.
"""

from __future__ import annotations

from collections.abc import Callable

from wmh.agent.capture import run_to_trace
from wmh.agent.environment import E2BEnvironment
from wmh.agent.runtime import AgentRuntime, RunResult
from wmh.agent.skills import SkillLibrary
from wmh.agent.spec import HarnessSpec
from wmh.agent.tasks import TaskSpec
from wmh.core.types import Trace

CollectProgressFn = Callable[[str, RunResult], None]  # (task_id, result) after each run


def collect_traces(
    spec: HarnessSpec,
    tasks: list[TaskSpec],
    agent_provider: object,  # Provider; typed loosely so the SDK import stays optional
    *,
    library: SkillLibrary | None = None,
    api_key: str | None = None,
    template: str | None = None,
    timeout: int = 300,
    on_progress: CollectProgressFn | None = None,
) -> list[Trace]:
    """Run `spec` on each task in a real E2B sandbox; return the captured traces (gold-stamped).

    The skill `library`, if on-disk, grows as the agent saves skills across tasks — so a collection
    sweep both produces training traces and bootstraps the skill library later variants seed from.
    """
    from wmh.providers.base import Provider

    assert isinstance(agent_provider, Provider)
    traces: list[Trace] = []
    for task in tasks:
        env = E2BEnvironment(api_key=api_key, template=template, timeout=timeout, setup=task.setup)
        try:
            runtime = AgentRuntime(spec, agent_provider, library=library)
            result = runtime.run(task.task_id, task.instruction, env)
        finally:
            env.close()
        traces.append(run_to_trace(result, gold=task.gold))
        if on_progress is not None:
            on_progress(task.task_id, result)
    return traces
