"""The world-model implementation: prompt assembly, build pipeline, demo, and play.

Evaluation of a built world model (open-loop replay fidelity + closed-loop task success) lives in
`wmo.simulation.evaluation`."""

from wmo.simulation.model.build import (
    DEFAULT_TRAIN_SPLIT,
    build,
    ingest,
    split_holdout,
    split_traces,
    split_traces_3way,
)
from wmo.simulation.model.demo import DemoReplay, DemoStep, run_demo
from wmo.simulation.model.loader import load_world_model
from wmo.simulation.model.play import PlayTurn, parse_action, play_turn
from wmo.simulation.model.reporting import BuildReporter, NullReporter
from wmo.simulation.model.world_model import WorldModel

__all__ = [
    "DEFAULT_TRAIN_SPLIT",
    "build",
    "ingest",
    "split_holdout",
    "split_traces",
    "split_traces_3way",
    "DemoReplay",
    "DemoStep",
    "run_demo",
    "load_world_model",
    "PlayTurn",
    "parse_action",
    "play_turn",
    "BuildReporter",
    "NullReporter",
    "WorldModel",
]
