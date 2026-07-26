"""The environment seam: one interface an agent loop steps against.

`Env` is the contract; `WorldModelEnv` backs it with a world model, and examples back it with
their real environments — so the same agent loop (see `wmo.env.episode.run_episode`) runs
byte-identical against either side.
"""

from wmo.env.base import Env, WorldModelEnv
from wmo.env.episode import DONE_SIGNAL, Agent, EpisodeResult, StopReason, run_episode
from wmo.env.scenarios import Scenario, scenarios_from_traces

__all__ = [
    "DONE_SIGNAL",
    "Agent",
    "Env",
    "EpisodeResult",
    "Scenario",
    "StopReason",
    "WorldModelEnv",
    "run_episode",
    "scenarios_from_traces",
]
