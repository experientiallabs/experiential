"""Measure how many decisions to keep in flight against one SystemOne endpoint.

Decision servers batch concurrent requests, so throughput climbs with concurrency until the
server saturates, then latency grows and errors start. ``sweep_concurrency`` replays sample
windows at each caller-chosen in-flight level for a fixed time and picks the level with the most
completed windows per second among levels whose error rate stays within a threshold.
"""

from __future__ import annotations

import asyncio
import itertools
import math
import time
from collections.abc import Callable, Sequence

from pydantic import Field

from exp.common.core.artifacts import ContractModel
from exp.common.tags.windows import TagWindow
from exp.runtime.models.providers.systemone import (
    SystemOneClient,
    SystemOneError,
    SystemOneRequestError,
)
from exp.runtime.tagging.tagger import Tagger


class LevelMeasurement(ContractModel):
    """Throughput, latency and errors measured at one in-flight level.

    Attributes:
        in_flight: Window decisions kept in flight.
        completed: Windows decided successfully.
        errors: Window decisions that failed.
        elapsed_seconds: Time from the first request until the last one finished.
        windows_per_second: Completed windows per elapsed second.
        input_tokens_per_second: Endpoint-reported input tokens of completed windows per
            elapsed second.
        p50_latency_seconds: Median latency of completed windows, or None without any.
        p95_latency_seconds: 95th percentile latency of completed windows, or None without any.
        first_error: Message of the first failure, or None without errors.
    """

    in_flight: int = Field(ge=1)
    completed: int = Field(ge=0)
    errors: int = Field(ge=0)
    elapsed_seconds: float = Field(ge=0)
    windows_per_second: float = Field(ge=0)
    input_tokens_per_second: float = Field(ge=0)
    p50_latency_seconds: float | None
    p95_latency_seconds: float | None
    first_error: str | None

    @property
    def error_rate(self) -> float:
        """Return failed decisions over all finished decisions; 1.0 when none finished."""
        finished = self.completed + self.errors
        return self.errors / finished if finished else 1.0


class ConcurrencySweep(ContractModel):
    """The measured levels and the one to use.

    Attributes:
        best_in_flight: Level with the most windows per second among levels within the error
            threshold; the lower level wins a tie.
        max_error_rate: Error-rate threshold a level had to meet.
        levels: Measurements in the order the levels were given.
    """

    best_in_flight: int = Field(ge=1)
    max_error_rate: float = Field(ge=0, le=1)
    levels: tuple[LevelMeasurement, ...] = Field(min_length=1)


class ConcurrencySweepError(RuntimeError):
    """No in-flight level kept its error rate within the threshold."""


def _percentile(ordered: Sequence[float], fraction: float) -> float | None:
    """Return the nearest-rank percentile of ascending values, or None for no values."""
    return ordered[max(math.ceil(fraction * len(ordered)) - 1, 0)] if ordered else None


async def _measure(
    tagger: Tagger, windows: Sequence[TagWindow], seconds: float, clock: Callable[[], float]
) -> LevelMeasurement:
    """Keep ``tagger.max_in_flight`` decisions in flight for ``seconds``, then drain.

    Args:
        tagger: Tagger bound to the level under test.
        windows: Sample windows, replayed in a cycle.
        seconds: How long to keep starting new decisions.
        clock: Monotonic clock in seconds.

    Returns:
        The level's measurement.

    Raises:
        SystemOneRequestError: A sample window breaks the decision contract.
    """
    latencies: list[float] = []
    tokens = 0
    errors: list[str] = []  # bounded by the level's duration
    cursor = itertools.cycle(windows)
    start = clock()

    async def worker() -> None:
        """Start one decision after another until the level's time is up."""
        nonlocal tokens
        while clock() - start < seconds:
            began = clock()
            try:
                decision = await tagger.decide_window(next(cursor))
            except SystemOneError as error:
                if isinstance(error, SystemOneRequestError):
                    raise
                errors.append(str(error))
                continue
            latencies.append(clock() - began)
            tokens += decision.usage.input_tokens

    workers = [asyncio.create_task(worker()) for _ in range(tagger.max_in_flight)]
    try:
        await asyncio.gather(*workers)
    finally:
        for task in workers:
            task.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
    elapsed = clock() - start
    latencies.sort()
    return LevelMeasurement(
        in_flight=tagger.max_in_flight,
        completed=len(latencies),
        errors=len(errors),
        elapsed_seconds=elapsed,
        windows_per_second=len(latencies) / elapsed if elapsed > 0 else 0.0,
        input_tokens_per_second=tokens / elapsed if elapsed > 0 else 0.0,
        p50_latency_seconds=_percentile(latencies, 0.50),
        p95_latency_seconds=_percentile(latencies, 0.95),
        first_error=errors[0] if errors else None,
    )


async def sweep_concurrency(
    tagger_or_client: Tagger | SystemOneClient,
    sample_windows: Sequence[TagWindow],
    levels: Sequence[int],
    seconds_per_level: float,
    *,
    max_error_rate: float = 0.0,
    clock: Callable[[], float] = time.perf_counter,
) -> ConcurrencySweep:
    """Measure each in-flight level on sample windows and pick the fastest healthy one.

    Each level keeps exactly that many decisions in flight, starting a new one as each
    finishes, for ``seconds_per_level``; decisions still running then finish and count. Run it
    on windows like the ones you will tag (``Tagger.windows`` of a few exchanges) so request
    sizes match.

    Args:
        tagger_or_client: A tagger, whose client and bank are used, or a client, which is asked
            the default bank. Its client's ``max_concurrency`` must allow the largest level.
        sample_windows: Windows to replay in a cycle; at least one.
        levels: Distinct in-flight levels to measure, in the order to measure them.
        seconds_per_level: How long each level keeps starting decisions.
        max_error_rate: Highest error rate a level may have and still be picked.
        clock: Monotonic clock in seconds, injectable for tests.

    Returns:
        Every level's measurement and the level to use.

    Raises:
        ValueError: The inputs are empty, repeated, out of range, or exceed the client's
            ``max_concurrency``.
        SystemOneRequestError: A sample window breaks the decision contract.
        ConcurrencySweepError: No level stayed within ``max_error_rate``.
    """
    if not sample_windows:
        raise ValueError("sweep_concurrency needs at least one sample window")
    if not levels or len(set(levels)) != len(levels):
        raise ValueError("sweep_concurrency needs one or more distinct in-flight levels")
    if not math.isfinite(seconds_per_level) or seconds_per_level <= 0:
        raise ValueError("seconds_per_level must be a finite positive number of seconds")
    if not 0.0 <= max_error_rate <= 1.0:
        raise ValueError("max_error_rate must be between 0 and 1")
    if isinstance(tagger_or_client, Tagger):
        base = tagger_or_client
        taggers = [
            Tagger(base.client, bank=base.bank, budget=base.budget, max_in_flight=level)
            for level in levels
        ]
    else:
        taggers = [Tagger(tagger_or_client, max_in_flight=level) for level in levels]
    measured = []
    for tagger in taggers:
        measured.append(await _measure(tagger, sample_windows, seconds_per_level, clock))
    healthy = [m for m in measured if m.completed and m.error_rate <= max_error_rate]
    if not healthy:
        detail = "; ".join(
            f"{m.in_flight} in flight: {m.errors} errors of {m.completed + m.errors}"
            for m in measured
        )
        first = next((m.first_error for m in measured if m.first_error), "no decision finished")
        raise ConcurrencySweepError(
            f"no in-flight level kept its error rate at or below {max_error_rate:g} ({detail}). "
            f"First error: {first}. Fix the endpoint or try lower levels"
        )
    best = max(healthy, key=lambda m: (m.windows_per_second, -m.in_flight))
    return ConcurrencySweep(
        best_in_flight=best.in_flight, max_error_rate=max_error_rate, levels=tuple(measured)
    )


__all__ = ["ConcurrencySweep", "ConcurrencySweepError", "LevelMeasurement", "sweep_concurrency"]
