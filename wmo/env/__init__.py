"""The environment seam: one interface an agent loop steps against.

`Env` is the contract; `WorldModelEnv` backs it with a world model, and examples back it with
their real environments, so the same agent loop (see `wmo.runtime.episode.run_episode`) runs
byte-identical against either side.
"""

from wmo.env.base import WorldModelEnv
from wmo.env.scenarios import Scenario, scenarios_from_traces
from wmo.runtime import DONE_SIGNAL, Agent, Env, EpisodeResult, StopReason, run_episode

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
