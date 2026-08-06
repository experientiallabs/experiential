"""`run_episode`: the one agent-vs-environment rollout loop.

Every workstream that "runs an agent against an environment for N steps and scores the result"
uses this loop, so episode records are comparable across the world model and real backends.
"""

from __future__ import annotations

import traceback
from enum import StrEnum
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

from wmo.core.types import Action, ActionKind, EnvState, Step
from wmo.runtime.environment import Env

# An agent signals it is finished by returning a MESSAGE action whose content equals this.
DONE_SIGNAL = "<DONE>"


@runtime_checkable
class Agent(Protocol):
    """Anything that maps the episode so far to the next action.

    `act` sees the task, the env's live state view (see `Env.reset`), and the full history of
    steps taken this episode. Return an `Action` to continue, or a MESSAGE action whose content
    is `DONE_SIGNAL` to stop.
    """

    def act(self, task: str | None, state: EnvState, history: list[Step]) -> Action: ...


class StopReason(StrEnum):
    """Why a rollout ended."""

    AGENT_DONE = "agent_done"  # the agent returned DONE_SIGNAL
    MAX_STEPS = "max_steps"  # the step budget ran out
    ENV_ERROR = "env_error"  # env.step raised; episode recorded up to the failure
    AGENT_ERROR = "agent_error"  # agent.act raised; episode recorded up to the failure


class EpisodeResult(BaseModel):
    """One completed rollout: what happened, why it stopped."""

    task: str | None = None
    steps: list[Step] = Field(default_factory=list)
    stop_reason: StopReason
    error: str | None = None  # set on ENV_ERROR/AGENT_ERROR; names the failure
    error_traceback: str | None = None  # full traceback of that failure, for debugging batches


def run_episode(
    env: Env,
    agent: Agent,
    task: str | None = None,
    *,
    seed_state: EnvState | None = None,
    max_steps: int = 20,
) -> EpisodeResult:
    """Roll one episode of `agent` against `env`, bounded by `max_steps`.

    The env's `reset`/`close` bracket the episode; each turn the agent proposes an action from the
    accumulated history and the env answers with an observation. Each recorded step's
    `state_before` is a deep copy of the env's state at that moment (the live state object keeps
    mutating; see `Env.reset`). An env exception is recorded (not raised) so batch runs survive a
    flaky backend; callers inspect `stop_reason`/`error`.
    """
    if max_steps < 1:
        raise ValueError(f"max_steps must be >= 1, got {max_steps}")
    try:
        state = env.reset(task=task, seed_state=seed_state)
        history: list[Step] = []
        for _ in range(max_steps):
            try:
                action = agent.act(task, state, history)
            except Exception as exc:  # noqa: BLE001 - batch runs must survive one bad episode
                return EpisodeResult(
                    task=task,
                    steps=history,
                    stop_reason=StopReason.AGENT_ERROR,
                    error=f"{type(exc).__name__}: {exc} (in agent.act)",
                    error_traceback="".join(traceback.format_exception(exc)),
                )
            if action.kind is ActionKind.MESSAGE and action.content == DONE_SIGNAL:
                return EpisodeResult(task=task, steps=history, stop_reason=StopReason.AGENT_DONE)
            state_before = state.model_copy(deep=True)
            try:
                observation = env.step(action)
            except Exception as exc:  # noqa: BLE001 - batch runs must survive one bad episode
                return EpisodeResult(
                    task=task,
                    steps=history,
                    stop_reason=StopReason.ENV_ERROR,
                    error=(
                        f"{type(exc).__name__}: {exc} "
                        f"(while executing {action.kind.value} {action.name or action.content!r})"
                    ),
                    error_traceback="".join(traceback.format_exception(exc)),
                )
            history.append(
                Step(action=action, observation=observation, state_before=state_before, task=task)
            )
        return EpisodeResult(task=task, steps=history, stop_reason=StopReason.MAX_STEPS)
    finally:
        try:
            env.close()
        except Exception:  # noqa: BLE001, S110 - a teardown failure must not mask the result
            pass
