"""Test the grid plot renders a PNG (skipped when the viz extra isn't installed)."""

from __future__ import annotations

import pytest

from wmh.engine.grid import GridCell, GridResult
from wmh.engine.grid_plot import plot_grid

pytest.importorskip("matplotlib")  # viz extra; skip cleanly when absent


def _cell(label: str, cond: str, fid: float, cost: float | None) -> GridCell:
    return GridCell(
        model_label=label,
        provider="bedrock",
        model="us.anthropic.claude-opus-4-8",
        condition=cond,
        condition_label="wmh/rag" if cond == "base_rag" else "base",
        fidelity=fid,
        error_flag_acc=0.9,
        n_steps=50,
        cost_usd=cost,
    )


def test_plot_grid_writes_png(tmp_path) -> None:  # noqa: ANN001 - fixture
    result = GridResult(
        suite="tiny",
        judge_model="us.anthropic.claude-opus-4-8",
        judge_provider="bedrock",
        train_split=0.7,
        top_k=5,
        seed=0,
        sample_turns="all",
        total_test_steps=100,
        cells=[
            _cell("Opus 4.8", "base_rag", 0.78, 67.37),
            _cell("Haiku 4.5", "base_rag", 0.73, 10.49),
            _cell("Qwen", "base", 0.71, None),  # unpriced -> no cost label, must not crash
        ],
    )
    out = plot_grid(result, tmp_path / "grid.png", dataset_label="tiny-ds", n_test_traces=8)
    assert out.exists()
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"  # valid PNG magic


def test_plot_grid_empty_raises(tmp_path) -> None:  # noqa: ANN001 - fixture
    result = GridResult(
        suite="t",
        judge_model="m",
        judge_provider="bedrock",
        train_split=0.7,
        top_k=5,
        seed=0,
        sample_turns="all",
    )
    with pytest.raises(ValueError, match="no cells"):
        plot_grid(result, tmp_path / "x.png", dataset_label="d", n_test_traces=0)
