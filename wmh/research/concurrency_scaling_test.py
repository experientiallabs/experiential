"""Tests for the concurrency-scaling experiment driver (no network/Docker; fake runners)."""

from __future__ import annotations

import pytest

from wmh.research.concurrency_scaling import (
    RealBatch,
    RealRunner,
    Side,
    WorldBatch,
    WorldRunner,
    run_concurrency_scaling,
)


def _world_runner(per_scenario: float, n: int) -> WorldRunner:
    """Fake world runner: wall = ideal parallel time (work spread over `level` workers)."""

    def run(level: int) -> WorldBatch:
        work = per_scenario * n
        return WorldBatch(
            wall_seconds=work / level,  # perfect scaling, for a clean speedup curve
            work_seconds=work,
            ok=n,
            total=n,
            tokens=100 * n,
            cost_usd=0.01 * n,
            fidelity=1.0,
        )

    return run


def _real_runner(per_sandbox: float, n: int, *, cap: int) -> RealRunner:
    """Fake real runner: only `cap` sandboxes truly overlap (sandbox standup contends)."""

    def run(level: int) -> RealBatch:
        effective = min(level, cap)
        work = per_sandbox * n
        return RealBatch(wall_seconds=work / effective, work_seconds=work, ok=n, total=n)

    return run


def test_speedup_baseline_is_first_level() -> None:
    report = run_concurrency_scaling(
        _world_runner(1.0, 8),
        None,
        levels=[1, 2, 4, 8],
        scenarios=8,
        side=Side.WORLD,
    )
    by_level = {p.level: p for p in report.points}
    assert by_level[1].speedup == pytest.approx(1.0)
    assert by_level[1].efficiency == pytest.approx(1.0)
    assert by_level[2].speedup == pytest.approx(2.0)
    assert by_level[4].speedup == pytest.approx(4.0)
    assert by_level[8].speedup == pytest.approx(8.0)
    # Perfect scaling keeps efficiency at 1.0 across the board.
    assert all(p.efficiency == pytest.approx(1.0) for p in report.points)


def test_efficiency_relative_to_baseline_when_first_level_not_one() -> None:
    # Baseline is the FIRST level (2, not 1); efficiency divides by the concurrency RATIO to it,
    # so the baseline reports 100% and W=8 (4x the baseline concurrency, 4x speedup) also 100%.
    report = run_concurrency_scaling(
        _world_runner(1.0, 8), None, levels=[2, 4, 8], scenarios=8, side=Side.WORLD
    )
    by_level = {p.level: p for p in report.points}
    assert by_level[2].speedup == pytest.approx(1.0)
    assert by_level[2].efficiency == pytest.approx(1.0)  # baseline vs itself, not 1/2
    assert by_level[4].speedup == pytest.approx(2.0)
    assert by_level[4].efficiency == pytest.approx(1.0)  # 2x speedup at 2x concurrency
    assert by_level[8].speedup == pytest.approx(4.0)
    assert by_level[8].efficiency == pytest.approx(1.0)


def test_concurrency_lowers_wall_clock() -> None:
    report = run_concurrency_scaling(
        _world_runner(2.0, 4), None, levels=[1, 4], scenarios=4, side=Side.WORLD
    )
    walls = {p.level: p.world_wall_mean for p in report.points}
    assert walls[4] < walls[1]
    assert walls[1] == pytest.approx(8.0)  # 2.0 * 4 scenarios, one worker
    assert walls[4] == pytest.approx(2.0)  # spread over 4 workers


def test_both_sides_compute_differential() -> None:
    report = run_concurrency_scaling(
        _world_runner(1.0, 8),
        _real_runner(3.0, 8, cap=2),  # real side caps at 2 concurrent sandboxes
        levels=[1, 8],
        scenarios=8,
        side=Side.BOTH,
    )
    point = {p.level: p for p in report.points}[8]
    # world wall = 8/8 = 1.0; real wall = 24/min(8,2) = 12.0; differential = 12.0.
    assert point.world_wall_mean == pytest.approx(1.0)
    assert point.real_wall_mean == pytest.approx(12.0)
    assert point.differential == pytest.approx(12.0)


def test_side_world_skips_real_runner() -> None:
    calls: list[int] = []

    def real(level: int) -> RealBatch:
        calls.append(level)
        return RealBatch(wall_seconds=1.0, ok=1, total=1)

    report = run_concurrency_scaling(
        _world_runner(1.0, 2), real, levels=[1, 2], scenarios=2, side=Side.WORLD
    )
    assert calls == []  # real runner never invoked under side=world
    assert all(p.real_wall_mean == 0.0 for p in report.points)


def test_trials_produce_error_bars() -> None:
    walls = iter([1.0, 3.0])  # two trials at the single level -> mean 2.0, nonzero std

    def world(level: int) -> WorldBatch:
        return WorldBatch(wall_seconds=next(walls), ok=1, total=1, fidelity=1.0)

    report = run_concurrency_scaling(
        world, None, levels=[1], scenarios=1, trials=2, side=Side.WORLD
    )
    point = report.points[0]
    assert point.trials == 2
    assert point.world_wall_mean == pytest.approx(2.0)
    assert point.world_wall_std == pytest.approx(1.0)


def test_invalid_inputs_raise() -> None:
    with pytest.raises(ValueError, match="trials"):
        run_concurrency_scaling(_world_runner(1.0, 1), None, levels=[1], scenarios=1, trials=0)
    with pytest.raises(ValueError, match="scenarios"):
        run_concurrency_scaling(_world_runner(1.0, 1), None, levels=[1], scenarios=0)
    with pytest.raises(ValueError, match="level"):
        run_concurrency_scaling(_world_runner(1.0, 1), None, levels=[0], scenarios=1)


def test_on_point_called_per_level() -> None:
    seen: list[int] = []
    run_concurrency_scaling(
        _world_runner(1.0, 2),
        None,
        levels=[1, 2, 4],
        scenarios=2,
        side=Side.WORLD,
        on_point=lambda p: seen.append(p.level),
    )
    assert seen == [1, 2, 4]


def test_best_speedup() -> None:
    report = run_concurrency_scaling(
        _world_runner(1.0, 8), None, levels=[1, 2, 8], scenarios=8, side=Side.WORLD
    )
    best = report.best_speedup()
    assert best is not None
    assert best.level == 8
