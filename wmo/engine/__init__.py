"""The world-model engine: prompt assembly, the WorldModel, the build pipeline, demo, play.

Evaluation of a built world model (open-loop replay fidelity + closed-loop task success) lives in
`wmo.evals`."""

from wmo.engine.build import (
    DEFAULT_TRAIN_SPLIT,
    build,
    ingest,
    split_holdout,
    split_traces,
    split_traces_3way,
)
from wmo.engine.demo import DemoReplay, DemoStep, run_demo
from wmo.engine.loader import load_world_model
from wmo.engine.play import PlayTurn, parse_action, play_turn
from wmo.engine.reporting import BuildReporter, NullReporter
from wmo.engine.world_model import WorldModel

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
