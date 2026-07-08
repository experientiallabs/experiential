"""The world-model engine: prompt assembly, the WorldModel, the build pipeline, demo, play.

Evaluation of a built world model (open-loop replay fidelity + closed-loop task success) lives in
`wmh.evals`."""

from wmh.engine.build import build, ingest, split_traces, split_traces_3way
from wmh.engine.demo import DemoReplay, DemoStep, run_demo
from wmh.engine.loader import load_world_model
from wmh.engine.play import PlayTurn, parse_action, play_turn
from wmh.engine.reporting import BuildReporter, NullReporter
from wmh.engine.world_model import WorldModel

__all__ = [
    "build",
    "ingest",
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
