"""Tests for the concurrency-scaling plot loader + renderer (needs the viz extra, in dev)."""

from __future__ import annotations

from pathlib import Path

import pytest

from wmh.research.concurrency_plot import (
    _load_points,
    render_cost,
    render_report,
    render_speedup,
)
from wmh.research.concurrency_scaling import (
    ConcurrencyPoint,
    ConcurrencyScalingReport,
    Side,
)


def _write_report(path: Path, *, both: bool) -> str:
    points = [
        ConcurrencyPoint(
            level=lvl,
            trials=1,
            world_wall_mean=float(32 // lvl),
            world_wall_std=0.0,
            real_wall_mean=float(50 // lvl) if both else 0.0,
            speedup=float(lvl),
            efficiency=1.0,
            differential=(50 / 32) if both else 0.0,
        )
        for lvl in (1, 2, 4)
    ]
    report = ConcurrencyScalingReport(
        benchmark="tau-bench",
        side=Side.BOTH if both else Side.WORLD,
        scenarios=8,
        levels=[1, 2, 4],
        points=points,
    )
    path.write_text(report.model_dump_json(), encoding="utf-8")
    return str(path)


def test_load_points_both_sides(tmp_path: Path) -> None:
    df = _load_points(_write_report(tmp_path / "r.json", both=True))
    assert set(df["side"]) == {"world model", "real sandbox"}
    assert sorted(df[df["side"] == "world model"]["level"]) == [1, 2, 4]


def test_load_points_world_only_has_no_real_rows(tmp_path: Path) -> None:
    df = _load_points(_write_report(tmp_path / "r.json", both=False))
    assert set(df["side"]) == {"world model"}


def test_load_points_bad_json_raises_valueerror(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text('{"not": "a report"', encoding="utf-8")  # truncated / invalid
    with pytest.raises(ValueError, match="not a valid concurrency-scaling report"):
        _load_points(str(bad))


def test_render_report_writes_image(tmp_path: Path) -> None:
    report = _write_report(tmp_path / "r.json", both=True)
    out = tmp_path / "fig.png"
    written = render_report(report, str(out), title="test")
    assert written == str(out)
    assert out.exists() and out.stat().st_size > 0


def test_render_speedup_and_cost_write_images_for_multiple_reports(tmp_path: Path) -> None:
    paths = [
        _write_report(tmp_path / "tau.json", both=True),
        _write_report(tmp_path / "swe.json", both=True),
    ]
    speedup = tmp_path / "speedup.png"
    cost = tmp_path / "cost.png"
    assert render_speedup(paths, str(speedup)) == str(speedup)
    assert render_cost(paths, str(cost)) == str(cost)
    assert speedup.exists() and speedup.stat().st_size > 0
    assert cost.exists() and cost.stat().st_size > 0


def test_render_speedup_and_cost_single_report_ok(tmp_path: Path) -> None:
    # One report is a degenerate overlay but must still render (no crash on a single series).
    paths = [_write_report(tmp_path / "r.json", both=True)]
    assert Path(render_speedup(paths, str(tmp_path / "s.png"))).exists()
    assert Path(render_cost(paths, str(tmp_path / "c.png"))).exists()


def test_render_speedup_rejects_empty_list(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no reports"):
        render_speedup([], str(tmp_path / "s.png"))


def test_render_cost_rejects_empty_list(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no reports"):
        render_cost([], str(tmp_path / "c.png"))


def test_render_speedup_bad_json_raises_valueerror(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text('{"nope":', encoding="utf-8")
    with pytest.raises(ValueError, match="not a valid concurrency-scaling report"):
        render_speedup([str(bad)], str(tmp_path / "s.png"))


def test_render_speedup_rejects_report_with_no_points(tmp_path: Path) -> None:
    empty = ConcurrencyScalingReport(
        benchmark="x", side=Side.BOTH, scenarios=8, levels=[1], points=[]
    )
    p = tmp_path / "empty.json"
    p.write_text(empty.model_dump_json(), encoding="utf-8")
    with pytest.raises(ValueError, match="no timed points"):
        render_speedup([str(p)], str(tmp_path / "s.png"))
