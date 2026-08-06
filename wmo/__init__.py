"""World Model Optimizer — a frontier LLM acts as your agent's environment.

Public API:
    from wmo import WorldModel
    wm = WorldModel.load(".wmo", provider=...)
    session = wm.new_session(task="browse the shop")
    obs = wm.step(session.id, action)
"""

from wmo.core.types import (
    Action,
    ActionKind,
    EnvState,
    Observation,
    Session,
    Step,
    Trace,
)
from wmo.engine.world_model import WorldModel
from wmo.env.base import WorldModelEnv
from wmo.runtime import DONE_SIGNAL, Agent, Env, EpisodeResult, StopReason, run_episode

__all__ = [
    "WorldModel",
    "Action",
    "ActionKind",
    "Observation",
    "EnvState",
    "Session",
    "Step",
    "Trace",
    "Agent",
    "DONE_SIGNAL",
    "Env",
    "EpisodeResult",
    "StopReason",
    "WorldModelEnv",
    "run_episode",
]
