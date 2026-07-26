"""Tests for RunTracker aggregation + deterministic timing via a fake clock."""

from __future__ import annotations

import pytest

from wmo.providers.base import TokenUsage
from wmo.tracking.tracker import Phase, RunTracker


class FakeClock:
    """Scripted monotonic clock: each `monotonic()` returns the next value in `ticks`."""

    def __init__(self, ticks: list[float]) -> None:
        self._ticks = ticks
        self._i = 0

    def monotonic(self) -> float:
        value = self._ticks[self._i]
        self._i += 1
        return value


def test_totals_sum_tokens_and_cost_across_events() -> None:
    tracker = RunTracker(run_id="r", kind="build")
    tracker.record(Phase.GEPA, "claude-opus-4-8", TokenUsage(input_tokens=1000, output_tokens=200))
    tracker.record(Phase.JUDGE, "claude-opus-4-8", TokenUsage(input_tokens=500, output_tokens=100))

    total = tracker.totals()
    assert total.calls == 2
    assert total.input_tokens == 1500
    assert total.output_tokens == 300
    assert total.total_tokens == 1800
    # 1500*5/1e6 + 300*25/1e6 = 0.0075 + 0.0075 = 0.015 (float division → approx)
    assert total.cost_usd == pytest.approx(0.015)


def test_by_phase_buckets_events() -> None:
    tracker = RunTracker(run_id="r", kind="build")
    tracker.record(Phase.GEPA, "claude-opus-4-8", TokenUsage(input_tokens=1000, output_tokens=0))
    tracker.record(Phase.GEPA, "claude-opus-4-8", TokenUsage(input_tokens=1000, output_tokens=0))
    tracker.record(Phase.JUDGE, "claude-opus-4-8", TokenUsage(input_tokens=400, output_tokens=0))

    by_phase = tracker.by_phase()
    assert by_phase[Phase.GEPA].calls == 2
    assert by_phase[Phase.GEPA].input_tokens == 2000
    assert by_phase[Phase.JUDGE].calls == 1


def test_duration_is_measured_off_injected_clock() -> None:
    clock = FakeClock([100.0, 105.5])  # start, stop
    tracker = RunTracker(run_id="r", kind="build", clock=clock)
    tracker.start()
    tracker.stop()
    assert tracker.duration_seconds() == 5.5


def test_timed_contextmanager_brackets_start_stop() -> None:
    clock = FakeClock([10.0, 13.0])
    tracker = RunTracker(run_id="r", kind="serve", clock=clock)
    with tracker.timed():
        pass
    assert tracker.record_summary().duration_seconds == 3.0


def test_duration_is_live_while_running() -> None:
    clock = FakeClock([0.0, 2.0])  # start, then a live read
    tracker = RunTracker(run_id="r", kind="serve", clock=clock)
    tracker.start()
    assert tracker.duration_seconds() == 2.0  # not yet stopped → measured live


def test_record_summary_carries_id_kind_and_breakdown() -> None:
    clock = FakeClock([0.0, 1.0])
    tracker = RunTracker(run_id="abc", kind="build", clock=clock)
    with tracker.timed():
        tracker.record(Phase.GEPA, "claude-opus-4-8", TokenUsage(input_tokens=10, output_tokens=2))
    record = tracker.record_summary()
    assert record.run_id == "abc"
    assert record.kind == "build"
    assert record.duration_seconds == 1.0
    assert record.total.calls == 1
    assert Phase.GEPA in record.by_phase
