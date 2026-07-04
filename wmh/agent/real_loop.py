"""Real-environment evaluation: score a harness variant by running it for real in E2B.

The mirror of `evaluate_closed_loop`, but each task runs in a fresh E2B sandbox (a real shell)
instead of the world model. It reuses the exact same scoring core (`evaluate_with_env`) — same
runtime, same k-pass protocol, same `GoldJudge` — so the only thing that differs from the simulated
path is the environment. That is what makes the two `ClosedLoopReport`s directly comparable, which
is the whole point of the sim-real agreement check (`wmh.agent.agreement`): is the sim faithful?

Real evaluation is expensive (a live sandbox per task per pass) and needs the `e2b` extra +
`$E2B_API_KEY`; it exists to *validate* the cheap simulated loop, not to replace it.
"""

from __future__ import annotations

from wmh.agent.closed_loop import ClosedLoopReport, evaluate_with_env
from wmh.agent.environment import E2BEnvironment
from wmh.agent.gold import GoldJudge
from wmh.agent.skills import SkillLibrary
from wmh.agent.spec import HarnessSpec
from wmh.agent.tasks import TaskSpec
from wmh.providers.base import Provider


def evaluate_real(
    spec: HarnessSpec,
    tasks: list[TaskSpec],
    agent_provider: Provider,
    judge: GoldJudge,
    *,
    library: SkillLibrary | None = None,
    k: int = 3,
    api_key: str | None = None,
    template: str | None = None,
    timeout: int = 300,
) -> ClosedLoopReport:
    """Score `spec` on `tasks` by running each task for real in a fresh E2B sandbox, k passes each.

    Each task's `setup` seeds its sandbox before the agent runs. Returns the same `ClosedLoopReport`
    shape as the simulated path so the two are directly comparable.
    """

    def make_env(task: TaskSpec) -> E2BEnvironment:
        return E2BEnvironment(api_key=api_key, template=template, timeout=timeout, setup=task.setup)

    return evaluate_with_env(spec, tasks, make_env, agent_provider, judge, library=library, k=k)
