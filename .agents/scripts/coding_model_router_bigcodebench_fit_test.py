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


def test_seeded_outer_splits_are_distinct_and_grouped() -> None:
    groups = [f"family-{index // 2}" for index in range(40)]
    splits = module.outer_splits(groups)
    assert [split.seed for split in splits] == list(module.OUTER_SEEDS)
    assert len({tuple(split.test_indices.tolist()) for split in splits}) == 5
    for split in splits:
        assert sorted([*split.train_indices, *split.test_indices]) == list(range(40))
        assert 6 <= len(split.test_indices) <= 10
        assert {groups[index] for index in split.train_indices}.isdisjoint(
            {groups[index] for index in split.test_indices}
        )


def _knn_data() -> object:
    texts = [
        *(f"sql query select join table {index}" for index in range(10)),
        *(f"async python coroutine await task {index}" for index in range(10)),
    ]
    rewards = np.zeros((20, len(module.ARMS), module.ATTEMPTS), dtype=np.float64)
    rewards[:10, 0, :] = 1.0
    rewards[10:, 4, :] = 1.0
    costs = np.broadcast_to(
        np.asarray([0.001, 0.002, 0.003, 0.004, 0.005])[None, :, None],
        rewards.shape,
    ).copy()
    return module.FitData(
        task_ids=[f"task-{index}" for index in range(20)],
        groups=[f"family-{index // 2}" for index in range(20)],
        texts=texts,
        is_hard=np.zeros(20, dtype=np.bool_),
        rewards=rewards,
        costs=costs,
    )


def test_outcome_matrix_preserves_every_attempt() -> None:
    data = _knn_data()
    matrix = module.outcome_matrix(data)
    assert [entry.name for entry in matrix.pool] == list(module.ARMS)
    assert len(matrix.outcomes) == 20 * len(module.ARMS) * module.ATTEMPTS
    assert {outcome.model for outcome in matrix.outcomes} == set(module.ARMS)


def test_native_knn_replay_matches_tensor_value(tmp_path: Path) -> None:
    data = _knn_data()
    replay = module.fit_native_knn_replay(
        data,
        np.asarray([*range(8), *range(10, 18)]),
        np.asarray([8, 9, 18, 19]),
        bank_path=tmp_path / "native-knn.bank.npz",
        dim=512,
        guard_arm="luna-max",
        rag_num=8,
        rag_thres=0.9,
        z=0.0,
        min_pairs=3,
        se_floor=False,
        floor_q=0.0,
        pick_lam=0.0,
        guard_mode="symmetric",
    )
    assert replay.bank_path.exists()
    assert replay.policy.kind == "knn"
    assert replay.policy.guard_model == "luna-max"
    assert replay.choices.shape == (4,)
    assert replay.value.reward == 1.0


def test_native_knn_refuses_overlapping_split(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="overlap"):
        module.fit_native_knn_replay(
            _knn_data(),
            np.asarray([0, 1]),
            np.asarray([1, 2]),
            bank_path=tmp_path / "unused.npz",
            dim=512,
            guard_arm="luna-max",
            rag_num=4,
            rag_thres=0.9,
            z=0.0,
            min_pairs=3,
            se_floor=False,
            floor_q=0.0,
            pick_lam=0.0,
            guard_mode="symmetric",
        )


def test_fit_selected_static_uses_fit_only_quality_then_cost() -> None:
    data = _knn_data()
    data.rewards[:, 3:, :] = 1.0
    selected = module.fit_selected_static(data, np.arange(20))
    assert selected.name == "luna-xhigh"
    assert selected.reward == 1.0
    assert selected.cost_usd == pytest.approx(0.004)


def test_fit_candidate_selects_cheapest_quality_feasible_point() -> None:
    candidates = [
        module.CandidateMetric("best", 1.0, 1.0, 0.2, 100, 0),
        module.CandidateMetric("cheap", 0.95, 0.4, 0.3, 80, 1),
        module.CandidateMetric("too-low", 0.94, 0.1, 0.1, 20, 2),
    ]
    assert module.select_fit_candidate(candidates, baseline_reward=1.0).name == "cheap"


def test_fit_candidate_fallback_maximizes_quality() -> None:
    candidates = [
        module.CandidateMetric("less", 0.7, 0.1, 0.1, 10, 0),
        module.CandidateMetric("more", 0.8, 0.2, 0.2, 20, 1),
    ]
    assert module.select_fit_candidate(candidates, baseline_reward=1.0).name == "more"


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
