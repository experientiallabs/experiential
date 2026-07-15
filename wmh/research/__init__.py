"""GEPA optimization-research harness.

A small, empirical surface for trying optimization directions and recording results — the seed of a
training-research surface as the harness grows from prompt optimization toward heavier methods. It
wraps the existing pipeline (`GEPAOptimizer`, `predict_observation`, the judge) rather than forking
it, so an experiment measures *what the harness actually does*.

Two layers:

- `ablation` — the framework: a `Condition` (a named set of knobs), an `Ablation` protocol
  (enumerate conditions + run one condition at one seed -> a scalar metric), and `run_ablation`,
  which sweeps every condition across multiple seeds and aggregates mean + std. Adding a new
  experiment = writing one `Ablation`.
- `pipeline` — the reusable build/eval primitives every ablation leans on: `optimize_prompt` (run
  GEPA at a chosen seed) and `score_prompt` (replay-score held-out fidelity via the canonical
  `wmh.engine.replay`, leak-free).

Concrete experiments:

- `seed_stability` — how reproducible is GEPA's evolved prompt across seeds.
- `trace_scaling` — how reconstruction fidelity scales with the number of training traces (the trace
  scaling law), against a fixed held-out test set.
- `concurrency_scaling` — how batch wall-clock scales with concurrency, for the world model vs. the
  real sandbox (the time differential T_real/T_world).

The train-vs-eval temperature sweep is parked because the shipped providers reject sampling params.
"""

from wmh.research.ablation import (
    Ablation,
    AblationReport,
    Condition,
    ConditionReport,
    SeedScore,
    aggregate,
    run_ablation,
)
from wmh.research.concurrency_scaling import (
    ConcurrencyPoint,
    ConcurrencyScalingReport,
    ConcurrencyTrial,
    RealBatch,
    Side,
    WorldBatch,
    run_concurrency_scaling,
)
from wmh.research.pipeline import optimize_prompt, score_prompt
from wmh.research.scaling_split import CorpusSplit, partition_corpus, subsample_train
from wmh.research.seed_stability import SeedStabilityAblation
from wmh.research.trace_scaling import BASE, GEPA, MODES, TraceScalingAblation

__all__ = [
    "Ablation",
    "AblationReport",
    "Condition",
    "ConditionReport",
    "SeedScore",
    "SeedStabilityAblation",
    "TraceScalingAblation",
    "CorpusSplit",
    "BASE",
    "GEPA",
    "MODES",
    "aggregate",
    "optimize_prompt",
    "partition_corpus",
    "run_ablation",
    "score_prompt",
    "subsample_train",
    "ConcurrencyPoint",
    "ConcurrencyScalingReport",
    "ConcurrencyTrial",
    "RealBatch",
    "Side",
    "WorldBatch",
    "run_concurrency_scaling",
]
