"""Tests for the RouterBench adapter (matrix loading, splits, baselines)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from wmh.optimize.outcomes import OutcomeMatrix
from wmh.research.routerbench import (
    best_single_model,
    load_routerbench,
    oracle,
    random_baseline,
    split_scenario_ids,
)

_MODELS = ["model-a", "model-b"]


def _frame() -> pd.DataFrame:
    # 4 prompts over 2 eval names; model-a is strong on math, model-b on prose, and
    # model-b is always cheaper.
    return pd.DataFrame(
        {
            "sample_id": ["m.0", "m.1", "p.0", "p.1"],
            "prompt": ["2+2?", "3*3?", "write a haiku", "write a poem"],
            "eval_name": ["math", "math", "prose", "prose"],
            "model-a": [1.0, 1.0, 0.0, 0.5],
            "model-b": [0.0, 0.5, 1.0, 1.0],
            "model-a|total_cost": [0.02, 0.02, 0.02, 0.02],
            "model-b|total_cost": [0.001, 0.001, 0.001, 0.001],
            "model-a|model_response": ["4", "9", "no", "meh"],
            "model-b|model_response": ["5", "9?", "haiku", "poem"],
        }
    )


def _pickle(tmp_path: Path) -> Path:
    path = tmp_path / "routerbench_0shot.pkl"
    _frame().to_pickle(path)
    return path


def test_load_maps_rewards_costs_and_ids(tmp_path: Path) -> None:
    matrix = load_routerbench(_pickle(tmp_path), models=_MODELS)
    assert matrix.model_names() == _MODELS
    assert len(matrix.outcomes) == 8  # 4 prompts x 2 models
    assert matrix.scenario_ids() == ["math:m.0", "math:m.1", "prose:p.0", "prose:p.1"]
    row = next(o for o in matrix.outcomes if o.scenario_id == "math:m.0" and o.model == "model-a")
    assert row.reward == 1.0
    assert row.cost_usd == 0.02
    assert row.task == "2+2?"
    assert row.replies == []  # responses skipped by default (memory)


def test_load_filters_benchmarks_and_samples(tmp_path: Path) -> None:
    matrix = load_routerbench(_pickle(tmp_path), models=_MODELS, benchmarks=["math"])
    assert matrix.scenario_ids() == ["math:m.0", "math:m.1"]
    sampled = load_routerbench(_pickle(tmp_path), models=_MODELS, sample=2, seed=0)
    assert len(sampled.scenario_ids()) == 2
    resampled = load_routerbench(_pickle(tmp_path), models=_MODELS, sample=2, seed=0)
    assert sampled.scenario_ids() == resampled.scenario_ids()  # deterministic


def test_split_is_stratified_and_deterministic(tmp_path: Path) -> None:
    matrix = load_routerbench(_pickle(tmp_path), models=_MODELS)
    fit, test = split_scenario_ids(matrix, train_fraction=0.5, seed=0)
    assert sorted(fit + test) == sorted(matrix.scenario_ids())
    # Stratified: each eval_name contributes to both sides.
    assert any(s.startswith("math:") for s in fit)
    assert any(s.startswith("math:") for s in test)
    fit2, test2 = split_scenario_ids(matrix, train_fraction=0.5, seed=0)
    assert (fit, test) == (fit2, test2)


def test_baselines() -> None:
    frame_matrix = _loaded()
    # Best single chosen on fit ids, evaluated on eval ids.
    ids = frame_matrix.scenario_ids()
    name, accuracy, cost = best_single_model(frame_matrix, fit_ids=ids, eval_ids=ids)
    # Both models mean 0.625 on the fit ids; the tie breaks toward the CHEAPER model.
    assert name == "model-b"
    assert accuracy == pytest.approx(0.625)
    assert cost == pytest.approx(0.001)
    oracle_accuracy, oracle_cost = oracle(frame_matrix, ids)
    assert oracle_accuracy == pytest.approx(1.0)
    # Oracle tie-break prefers the cheaper model: prose rows tie at 1.0 for model-b only,
    # math rows are model-a's; cost = mean(0.02, 0.02, 0.001, 0.001).
    assert oracle_cost == pytest.approx(0.0105)
    rand_accuracy, rand_cost = random_baseline(frame_matrix, ids)
    assert rand_accuracy == pytest.approx((0.625 + 0.625) / 2)
    assert rand_cost == pytest.approx((0.02 + 0.001) / 2)


def _loaded() -> OutcomeMatrix:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "rb.pkl"
        _frame().to_pickle(path)
        return load_routerbench(path, models=_MODELS)
