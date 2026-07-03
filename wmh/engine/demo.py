"""`wmh demo`: replay a real recorded scenario against the world model, open loop.

A randomly sampled trace supplies the task and the agent's recorded actions; the world model
predicts each observation while the session is teacher-forced with the recorded one
(`step_open_loop`), so every prediction is conditioned on the real trajectory. The result shows
predicted vs. actual side by side — the harness working end-to-end without needing a live agent.
"""

from __future__ import annotations

from pydantic import BaseModel

from wmh.core.types import Action, Observation, Trace
from wmh.engine.world_model import WorldModel


class DemoStep(BaseModel):
    """One replayed step: the recorded action with the predicted and recorded observations."""

    action: Action
    predicted: Observation
    actual: Observation

    @property
    def exact_match(self) -> bool:
        return self.predicted.content.strip() == self.actual.content.strip()


class DemoReplay(BaseModel):
    """The rendered scenario replay for `wmh demo`."""

    trace_id: str
    task: str | None
    steps: list[DemoStep]
    first_env_prompt: str  # the exact prompt the world model saw for the first step


def run_demo(world_model: WorldModel, trace: Trace, max_steps: int = 5) -> DemoReplay:
    """Replay up to `max_steps` of `trace` open-loop and collect predicted vs. actual."""
    if not trace.steps:
        raise ValueError(f"trace {trace.trace_id!r} has no steps to replay")
    task = trace.steps[0].task
    session = world_model.new_session(task=task)
    first_env_prompt = world_model.render_step_prompt(session.id, trace.steps[0].action)

    steps: list[DemoStep] = []
    for step in trace.steps[:max_steps]:
        predicted = world_model.step_open_loop(session.id, step.action, step.observation)
        steps.append(DemoStep(action=step.action, predicted=predicted, actual=step.observation))
    return DemoReplay(
        trace_id=trace.trace_id, task=task, steps=steps, first_env_prompt=first_env_prompt
    )
