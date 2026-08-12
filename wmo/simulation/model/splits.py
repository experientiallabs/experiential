"""Deterministic legacy replay splits retained independently of trace ingestion."""

from __future__ import annotations

import hashlib

from wmo.common.core.types import Trace

DEFAULT_TRAIN_SPLIT = 0.8


def split_traces(traces: list[Trace], train_split: float) -> tuple[list[Trace], list[Trace]]:
    """Split legacy replay traces by a stable trace-ID hash.

    Args:
        traces: Existing in-memory replay traces.
        train_split: Fraction assigned to the first partition.

    Returns:
        Deterministically partitioned training and held-out trace lists.
    """
    train: list[Trace] = []
    held_out: list[Trace] = []
    for trace in traces:
        (train if _trace_fraction(trace) < train_split else held_out).append(trace)
    return train, held_out


def split_traces_3way(
    traces: list[Trace], train_frac: float, val_frac: float
) -> tuple[list[Trace], list[Trace], list[Trace]]:
    """Split legacy replay traces into deterministic train, validation, and test bands.

    Args:
        traces: Existing in-memory replay traces.
        train_frac: Fraction assigned to training.
        val_frac: Fraction assigned to validation after training.

    Returns:
        Deterministically partitioned train, validation, and test trace lists.

    Raises:
        ValueError: The requested bands cannot leave a non-empty test interval.
    """
    if train_frac <= 0 or val_frac <= 0 or train_frac + val_frac >= 1:
        raise ValueError(
            "need train_frac>0, val_frac>0, train_frac+val_frac<1; "
            f"got train_frac={train_frac}, val_frac={val_frac}"
        )
    train: list[Trace] = []
    validation: list[Trace] = []
    test: list[Trace] = []
    validation_cut = train_frac + val_frac
    for trace in traces:
        fraction = _trace_fraction(trace)
        if fraction < train_frac:
            train.append(trace)
        elif fraction < validation_cut:
            validation.append(trace)
        else:
            test.append(trace)
    return train, validation, test


def split_holdout(
    traces: list[Trace], train_frac: float, val_frac: float | None = None
) -> tuple[list[Trace], list[Trace], bool]:
    """Return train and held-out replay traces with the historic tiny-corpus fallback.

    Args:
        traces: Existing in-memory replay traces.
        train_frac: Fraction assigned to training.
        val_frac: Optional validation fraction before the test band.

    Returns:
        Training traces, held-out traces, and whether a tiny corpus required full reuse.
    """
    if val_frac:
        train, _validation, held_out = split_traces_3way(traces, train_frac, val_frac)
    else:
        train, held_out = split_traces(traces, train_frac)
    if not held_out:
        return traces, traces, True
    return train, held_out, False


def _trace_fraction(trace: Trace) -> float:
    """Map one legacy trace ID to a stable fractional split position."""
    digest = hashlib.blake2b(trace.trace_id.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") / 2**64
