"""Tests for RunTracker aggregation + deterministic timing via a fake clock."""

from __future__ import annotations

import pytest

from wmo.common.observability.tracker import (
    Phase,
    RunRecord,
    RunTracker,
    UsageTotals,
    merge_run_records,
)
from wmo.common.providers.base import TokenUsage


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


def test_totals_carry_cache_subsets() -> None:
    tracker = RunTracker(run_id="r", kind="serve")
    tracker.record(
        Phase.SERVE,
        "claude-opus-4-8",
        TokenUsage(input_tokens=1000, cached_input_tokens=600, cache_write_input_tokens=100),
    )
    tracker.record(
        Phase.SERVE,
        "claude-opus-4-8",
        TokenUsage(input_tokens=500, cached_input_tokens=200),
    )

    total = tracker.totals()
    assert total.cached_input_tokens == 800
    assert total.cache_write_input_tokens == 100


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


def _totals(cost: float, calls: int = 1) -> UsageTotals:
    return UsageTotals(
        calls=calls,
        input_tokens=100 * calls,
        output_tokens=10 * calls,
        cached_input_tokens=5 * calls,
        cache_write_input_tokens=2 * calls,
        cost_usd=cost,
    )


def test_merged_totals_add_every_column_without_mutating_either_side() -> None:
    left, right = _totals(1.0), _totals(0.5, calls=3)
    combined = left.merged(right)
    assert combined.calls == 4
    assert combined.input_tokens == 400 and combined.output_tokens == 40
    assert combined.cached_input_tokens == 20 and combined.cache_write_input_tokens == 8
    assert combined.cost_usd == pytest.approx(1.5)
    assert left.calls == 1 and right.calls == 3  # values, not accumulators


def test_merging_run_records_sums_totals_and_keeps_the_phases_apart() -> None:
    # A sweep opens one metered session per episode; what an operator wants persisted is the
    # sweep. Phase buckets survive the roll-up, so the serve half and the judge half of that
    # cost stay separable afterwards.
    records = [
        RunRecord(
            run_id=f"s{index}",
            kind="serve",
            duration_seconds=0.5,
            total=_totals(0.03).merged(_totals(0.01)),
            by_phase={Phase.SERVE: _totals(0.03), Phase.JUDGE: _totals(0.01)},
        )
        for index in range(4)
    ]
    merged = merge_run_records(records, run_id="sweep-1", kind="sweep")
    assert merged.run_id == "sweep-1" and merged.kind == "sweep"
    assert merged.duration_seconds == pytest.approx(2.0)
    assert merged.total.cost_usd == pytest.approx(0.16)
    assert merged.by_phase[Phase.SERVE].cost_usd == pytest.approx(0.12)
    assert merged.by_phase[Phase.JUDGE].cost_usd == pytest.approx(0.04)
    assert merged.by_phase[Phase.SERVE].calls == 4


def test_merging_nothing_is_an_empty_record_rather_than_an_error() -> None:
    # "Nothing was metered" is a truthful answer a caller can print; raising would make the
    # caller invent a number or swallow the case.
    merged = merge_run_records([], run_id="sweep-0", kind="sweep")
    assert merged.total.cost_usd == 0.0 and merged.total.calls == 0
    assert merged.by_phase == {}


def test_records_with_disjoint_phases_keep_both() -> None:
    merged = merge_run_records(
        [
            RunRecord(run_id="a", kind="serve", by_phase={Phase.SERVE: _totals(1.0)}),
            RunRecord(run_id="b", kind="serve", by_phase={Phase.EMBED: _totals(2.0)}),
        ],
        run_id="sweep-2",
        kind="sweep",
    )
    assert set(merged.by_phase) == {Phase.SERVE, Phase.EMBED}
