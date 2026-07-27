"""Run tracking: aggregate tokens → cost + wall-clock across the harness lifecycle.

A `RunTracker` collects `UsageEvent`s (one per LLM call, tagged by phase) and rolls them up into
`UsageTotals` (tokens, USD, calls) plus a wall-clock duration measured off an injectable `Clock`.
`RunRecord` is the persisted artifact (`.wmo/runs/<run_id>.json`).

The tracker is provider-agnostic: it records `(model, TokenUsage)` and prices via
`wmo.tracking.pricing`. It's fed at the provider boundary by `MeteredProvider` (so GEPA, the judge,
and the world model are all captured without touching the optimizer), and directly by the world
model's serve `step`.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from enum import StrEnum

from pydantic import BaseModel, Field

from wmo.providers.base import TokenUsage
from wmo.tracking.clock import Clock, SystemClock
from wmo.tracking.pricing import cost_usd


class Phase(StrEnum):
    """Lifecycle phase a usage event is attributed to."""

    BUILD = "build"  # world-model build (overall)
    GEPA = "gepa"  # GEPA rollouts + reflection during optimization
    JUDGE = "judge"  # LLM-judge scoring
    SERVE = "serve"  # live world-model step calls
    EMBED = "embed"  # embedding calls (phi)
    OTHER = "other"


class UsageEvent(BaseModel):
    """One metered LLM call."""

    phase: Phase
    model: str
    usage: TokenUsage = Field(default_factory=TokenUsage)
    cost_usd: float = 0.0


class UsageTotals(BaseModel):
    """Rolled-up usage: tokens (with the cache subsets), cost, and call count."""

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    # Cache subsets of input_tokens (same subset contract as TokenUsage), kept so persisted
    # RunRecords can reconstruct cache behavior; old records load with these defaulted to 0.
    cached_input_tokens: int = 0
    cache_write_input_tokens: int = 0
    cost_usd: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def merged(self, other: UsageTotals) -> UsageTotals:
        """The two totals added together, as a new value; neither input is mutated."""
        return UsageTotals(
            calls=self.calls + other.calls,
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cached_input_tokens=self.cached_input_tokens + other.cached_input_tokens,
            cache_write_input_tokens=(
                self.cache_write_input_tokens + other.cache_write_input_tokens
            ),
            cost_usd=self.cost_usd + other.cost_usd,
        )

    def _add(self, event: UsageEvent) -> None:
        self.calls += 1
        self.input_tokens += event.usage.input_tokens
        self.output_tokens += event.usage.output_tokens
        self.cached_input_tokens += event.usage.cached_input_tokens
        self.cache_write_input_tokens += event.usage.cache_write_input_tokens
        self.cost_usd += event.cost_usd


class RunRecord(BaseModel):
    """Persisted summary of one run (build or serve session)."""

    run_id: str
    kind: str  # "build" | "serve" | ... (free-form label for the run)
    duration_seconds: float = 0.0
    total: UsageTotals = Field(default_factory=UsageTotals)
    by_phase: dict[Phase, UsageTotals] = Field(default_factory=dict)


class RunTracker:
    """Accumulates usage events and the wall-clock of a run.

    Duration is measured between `start()` and `stop()` off the injected `Clock`, so tests can pass
    a `FakeClock` and assert exact seconds. `record` is the single entry point for a metered call.
    """

    def __init__(self, run_id: str, kind: str, clock: Clock | None = None) -> None:
        self._run_id = run_id
        self._kind = kind
        self._clock = clock or SystemClock()
        self._events: list[UsageEvent] = []
        self._lock = threading.Lock()
        self._started_at: float | None = None
        self._elapsed: float = 0.0

    def start(self) -> None:
        self._started_at = self._clock.monotonic()

    def stop(self) -> None:
        if self._started_at is not None:
            self._elapsed = self._clock.monotonic() - self._started_at
            self._started_at = None

    @contextmanager
    def timed(self) -> Iterator[RunTracker]:
        """Time a run: `with tracker.timed(): ...` brackets start()/stop() even on error."""
        self.start()
        try:
            yield self
        finally:
            self.stop()

    def record(self, phase: Phase, model: str, usage: TokenUsage) -> UsageEvent:
        """Record one metered LLM call, pricing it via the model pricing table.

        Thread-safe: GEPA evaluates batches concurrently, so metered calls land in parallel.
        """
        event = UsageEvent(phase=phase, model=model, usage=usage, cost_usd=cost_usd(model, usage))
        with self._lock:
            self._events.append(event)
        return event

    @property
    def events(self) -> list[UsageEvent]:
        return list(self._events)

    def totals(self) -> UsageTotals:
        total = UsageTotals()
        for event in self._events:
            total._add(event)
        return total

    def by_phase(self) -> dict[Phase, UsageTotals]:
        buckets: dict[Phase, UsageTotals] = defaultdict(UsageTotals)
        for event in self._events:
            buckets[event.phase]._add(event)
        return dict(buckets)

    def duration_seconds(self) -> float:
        """Elapsed seconds; live (since start) if still running, else the frozen final span."""
        if self._started_at is not None:
            return self._clock.monotonic() - self._started_at
        return self._elapsed

    def record_summary(self) -> RunRecord:
        return RunRecord(
            run_id=self._run_id,
            kind=self._kind,
            duration_seconds=self.duration_seconds(),
            total=self.totals(),
            by_phase=self.by_phase(),
        )


def merge_run_records(records: Sequence[RunRecord], *, run_id: str, kind: str) -> RunRecord:
    """Roll many run records into one, summing totals, phases, and durations.

    For a batch of short-lived runs that together form one logical activity: a routing sweep opens
    one metered world-model session per episode, and what an operator wants persisted is the
    sweep, not a thousand sessions. Phase buckets are preserved rather than flattened, so the
    serve half and the judge half of that cost stay separable afterwards.

    Durations are summed, which for sequentially run records is the activity's own wall clock and
    for concurrent ones is total occupancy. An empty sequence gives an empty record, which is a
    truthful "nothing was metered" rather than an error.
    """
    total = UsageTotals()
    by_phase: dict[Phase, UsageTotals] = {}
    duration = 0.0
    for record in records:
        total = total.merged(record.total)
        duration += record.duration_seconds
        for phase, bucket in record.by_phase.items():
            by_phase[phase] = by_phase.get(phase, UsageTotals()).merged(bucket)
    return RunRecord(
        run_id=run_id, kind=kind, duration_seconds=duration, total=total, by_phase=by_phase
    )
