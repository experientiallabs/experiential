"""Tests for run-record persistence under .wmo/runs/."""

from __future__ import annotations

from pathlib import Path

from wmo.common.observability.store import load_runs, save_run
from wmo.common.observability.tracker import Phase, RunRecord, UsageTotals


def _record(run_id: str) -> RunRecord:
    return RunRecord(
        run_id=run_id,
        kind="build",
        duration_seconds=2.5,
        total=UsageTotals(calls=2, input_tokens=300, output_tokens=50, cost_usd=0.01),
        by_phase={Phase.GEPA: UsageTotals(calls=2, input_tokens=300, output_tokens=50)},
    )


def test_save_then_load_round_trips(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    save_run(_record("abc"), runs)

    loaded = load_runs(runs)
    assert len(loaded) == 1
    assert loaded[0].run_id == "abc"
    assert loaded[0].total.input_tokens == 300
    assert loaded[0].by_phase[Phase.GEPA].calls == 2


def test_load_missing_dir_returns_empty(tmp_path: Path) -> None:
    assert load_runs(tmp_path / "does-not-exist") == []


def test_save_writes_one_file_per_run(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    save_run(_record("r1"), runs)
    save_run(_record("r2"), runs)
    assert {p.stem for p in runs.glob("*.json")} == {"r1", "r2"}
    assert len(load_runs(runs)) == 2
