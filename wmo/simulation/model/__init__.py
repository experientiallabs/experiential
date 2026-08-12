"""The world-model implementation: prompt assembly and build pipeline.

Evaluation of a built world model (open-loop replay fidelity + closed-loop task success) lives in
`wmo.simulation.evaluation`."""

from wmo.common.observability.reporting import BuildReporter, NullReporter
from wmo.simulation.model.build import (
    DEFAULT_TRAIN_SPLIT,
    build,
    ingest,
    split_holdout,
    split_traces,
    split_traces_3way,
)
from wmo.simulation.model.loader import load_world_model
from wmo.simulation.model.world_model import WorldModel

__all__ = [
    "DEFAULT_TRAIN_SPLIT",
    "build",
    "ingest",
    "split_holdout",
    "split_traces",
    "split_traces_3way",
    "load_world_model",
    "BuildReporter",
    "NullReporter",
    "WorldModel",
]
