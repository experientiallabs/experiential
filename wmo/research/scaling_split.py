"""Fixed-test, growing-train split for the trace-scaling-law experiment.

The scaling law asks "how does world-model fidelity grow as we feed GEPA + retrieval *more* training
traces?" To read that curve cleanly the evaluation target must not move: we hold a **fixed** test
set (and a fixed valid set GEPA selects on) and vary only how many TRAIN traces the run sees.

Two deterministic carves, both stable as the corpus grows so a checked-in run reproduces:

1. `partition_corpus` hashes each `trace_id` into [0, 1) (the same blake2b trick as
   `wmo.simulation.model.build.split_traces`) and slices fixed `test` and `valid` bands off the
   top; everything else is the train *pool*. A trace's band depends only on its id, so adding more
   traces never moves an existing trace between test, valid, and pool; the test set is identical at
   every trace count.
2. `subsample_train` takes `n` traces from that pool in a seed-shuffled order. The order is fixed
   for a given seed, so the n=10 sample is a prefix of the n=20 sample (nested): each step up the
   curve *adds* traces rather than resampling, isolating the effect of corpus size.

Counts are capped at what the pool holds, so the same code runs on a small corpus or the committed
~1000-trace tau2 one without change — it just stops the curve where the data runs out.
"""

from __future__ import annotations

import hashlib
import random

from pydantic import BaseModel, Field

from wmo.core.types import Trace


class CorpusSplit(BaseModel):
    """A corpus partitioned into a fixed test set, a fixed valid set, and a growable train pool."""

    train_pool: list[Trace] = Field(default_factory=list)
    valid: list[Trace] = Field(default_factory=list)
    test: list[Trace] = Field(default_factory=list)


def _fraction(trace_id: str) -> float:
    """Map a trace id to a stable point in [0, 1) (same scheme as `split_traces`)."""
    digest = hashlib.blake2b(trace_id.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") / 2**64


def partition_corpus(
    traces: list[Trace], *, test_frac: float = 0.2, valid_frac: float = 0.15
) -> CorpusSplit:
    """Carve fixed `test`/`valid` bands off `traces` by trace-id hash; the rest is the train pool.

    The bands are `[0, test_frac)` -> test and `[test_frac, test_frac + valid_frac)` -> valid, so
    membership is a pure function of `trace_id`: growing the corpus only ever enlarges the train
    pool, never reshuffles test or valid. Raises if the fractions don't leave room for a train pool.
    """
    if test_frac <= 0 or valid_frac <= 0:
        raise ValueError("test_frac and valid_frac must be > 0")
    if test_frac + valid_frac >= 1.0:
        raise ValueError(f"test_frac + valid_frac must be < 1 (got {test_frac + valid_frac})")

    split = CorpusSplit()
    valid_edge = test_frac + valid_frac
    for trace in traces:
        f = _fraction(trace.trace_id)
        if f < test_frac:
            split.test.append(trace)
        elif f < valid_edge:
            split.valid.append(trace)
        else:
            split.train_pool.append(trace)
    return split


def subsample_train(train_pool: list[Trace], n: int, *, seed: int) -> list[Trace]:
    """Take `n` traces from `train_pool` in a seed-fixed shuffled order (nested as `n` grows).

    The shuffle is keyed only by `seed`, so for a fixed seed the n=10 sample is a prefix of the
    n=20 sample — each step up the scaling curve adds traces to the previous step rather than
    drawing a fresh set. `n` is capped at the pool size, so over-large counts use the whole pool.
    """
    ordered = sorted(train_pool, key=lambda t: t.trace_id)  # stable base order, input-independent
    random.Random(seed).shuffle(ordered)
    return ordered[: max(0, min(n, len(ordered)))]
