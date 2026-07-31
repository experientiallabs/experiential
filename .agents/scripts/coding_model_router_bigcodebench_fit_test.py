"""Tests for the promoted external BigCodeBench router fitter."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from scipy import sparse


def _module() -> ModuleType:
    path = Path(__file__).with_name("coding_model_router_bigcodebench_fit.py")
    spec = importlib.util.spec_from_file_location("bigcodebench_fit", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


module = _module()


def _write_object(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_loader_refuses_scores_when_oracle_did_not_pass(tmp_path: Path) -> None:
    _write_object(
        tmp_path / "oracle-report.json",
        {
            "protocol": {"target_outcomes_used": False},
            "passed": False,
        },
    )
    with pytest.raises(ValueError, match="router fitting is forbidden"):
        module.load_fit_data(tmp_path)


def test_grouped_folds_have_zero_family_overlap() -> None:
    groups = [f"family-{index // 2}" for index in range(20)]
    folds = module.grouped_folds(groups)
    assert len(folds) == 5
    assert sorted(np.concatenate([test for _, test in folds]).tolist()) == list(range(20))
    for train, test in folds:
        assert {groups[index] for index in train}.isdisjoint({groups[index] for index in test})


def test_ordinal_predictions_are_bounded_and_monotone() -> None:
    train_features = sparse.csr_matrix(np.asarray([[0.0], [0.2], [0.8], [1.0]], dtype=np.float64))
    test_features = sparse.csr_matrix(np.asarray([[0.1], [0.9]], dtype=np.float64))
    rewards = np.asarray(
        [
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [0.2, 0.3, 0.4, 0.5, 0.6],
            [0.4, 0.5, 0.6, 0.7, 0.8],
            [0.5, 0.6, 0.7, 0.8, 0.9],
        ],
        dtype=np.float64,
    )
    predicted = module.ordinal_ridge_predictions(
        train_features,
        test_features,
        rewards,
        alpha=1.0,
    )
    assert predicted.shape == (2, len(module.ARMS))
    assert np.all((0.0 <= predicted) & (predicted <= 1.0))
    assert np.all(np.diff(predicted, axis=1) >= 0.0)


def test_matched_task_blind_control_preserves_arm_mix() -> None:
    rewards = np.zeros((4, len(module.ARMS)), dtype=np.float64)
    rewards[:2, 0] = 1.0
    rewards[2:, 1] = 1.0
    costs = np.tile(np.arange(1.0, 6.0), (4, 1))
    choices = np.asarray([0, 0, 1, 1], dtype=np.int64)
    value = module.evaluate_choices(rewards, costs, choices)
    assert value.reward == 1.0
    assert value.matched_blind_reward == 0.5
    assert value.cost_usd == 1.5
    assert value.matched_blind_cost_usd == 1.5
    assert value.arm_counts == {
        "luna-low": 2,
        "luna-medium": 2,
        "luna-high": 0,
        "luna-xhigh": 0,
        "luna-max": 0,
    }


def test_doubly_robust_dense_targets_equal_observed_arm_means() -> None:
    rewards = np.zeros((3, len(module.ARMS), module.ATTEMPTS), dtype=np.float64)
    rewards[0, 0, :] = 1.0
    rewards[1, 2, :3] = 1.0
    rewards[2, 4, 0] = 1.0
    direct = np.full((3, len(module.ARMS)), 0.37, dtype=np.float64)
    pseudo = module.doubly_robust_pseudo_values(rewards, direct)
    assert np.allclose(pseudo, rewards.mean(axis=2))


def test_empirical_bayes_uses_loo_and_unseen_global_fallback() -> None:
    rewards = np.zeros((3, len(module.ARMS), module.ATTEMPTS), dtype=np.float64)
    rewards[0, 0, :] = 1.0
    rewards[1, 0, :] = 1.0
    rewards[2, 1, :] = 1.0
    train_base, test_base = module.empirical_bayes_family_predictions(
        ["shared", "shared", "solo"],
        ["shared", "unseen"],
        rewards,
        prior_strength=5.0,
    )
    global_mean = rewards.mean(axis=(0, 2))
    expected_shared = (np.asarray([10.0, 0.0, 0.0, 0.0, 0.0]) + 5.0 * global_mean) / 15.0
    assert np.allclose(train_base[0], (rewards[1].sum(axis=1) + 5.0 * global_mean) / 10.0)
    assert np.allclose(train_base[2], global_mean)
    assert np.allclose(test_base[0], expected_shared)
    assert np.allclose(test_base[1], global_mean)


def test_empirical_bayes_residual_predictions_are_monotone() -> None:
    train_features = sparse.csr_matrix(np.eye(4, dtype=np.float64))
    test_features = sparse.csr_matrix(np.eye(4, dtype=np.float64)[:2])
    rewards = np.zeros((4, len(module.ARMS), module.ATTEMPTS), dtype=np.float64)
    rewards[0, 0:2, :] = 1.0
    rewards[1, 0:3, :] = 1.0
    rewards[2, 0:4, :] = 1.0
    rewards[3, :, :] = 1.0
    predicted = module.empirical_bayes_ridge_predictions(
        train_features,
        test_features,
        ["a", "a", "b", "b"],
        ["a", "new"],
        rewards,
        prior_strength=5.0,
        alpha=1.0,
    )
    assert predicted.shape == (2, len(module.ARMS))
    assert np.all((0.0 <= predicted) & (predicted <= 1.0))
    assert np.all(np.diff(predicted, axis=1) >= 0.0)
