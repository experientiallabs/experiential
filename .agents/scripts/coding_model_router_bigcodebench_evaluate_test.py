"""Tests for leakage-safe BigCodeBench outer-heldout replay."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest


def _load(name: str) -> ModuleType:
    path = Path(__file__).with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


fit = _load("coding_model_router_bigcodebench_fit")
select = _load("coding_model_router_bigcodebench_select")
module = _load("coding_model_router_bigcodebench_evaluate")


def _data() -> object:
    tasks = 30
    rewards = np.zeros((tasks, len(fit.ARMS), fit.ATTEMPTS), dtype=np.float64)
    rewards[:15, 0, :] = 1.0
    rewards[15:, 4, :] = 1.0
    costs = np.broadcast_to(
        np.asarray([0.001, 0.002, 0.003, 0.004, 0.005])[None, :, None],
        rewards.shape,
    ).copy()
    return fit.FitData(
        task_ids=[f"task-{index}" for index in range(tasks)],
        groups=[f"group-{index // 2}" for index in range(tasks)],
        texts=[
            f"sql query {index}" if index < 15 else f"async await {index}" for index in range(tasks)
        ],
        is_hard=np.zeros(tasks, dtype=np.bool_),
        rewards=rewards,
        costs=costs,
    )


def _split() -> tuple[np.ndarray, np.ndarray]:
    return np.arange(24, dtype=np.int64), np.arange(24, 30, dtype=np.int64)


def test_non_knn_routes_do_not_depend_on_heldout_rewards(tmp_path: Path) -> None:
    data = _data()
    train, heldout = _split()
    spec = select.CandidateSpec("ordinal", "ridge", 512, 0, alpha=1.0)
    initial = module.replay_outer_heldout(
        data,
        train,
        heldout,
        spec,
        seed=7,
        work_dir=tmp_path / "initial",
    )
    changed_rewards = data.rewards.copy()
    changed_rewards[heldout] = 1.0 - changed_rewards[heldout]
    changed = fit.FitData(
        task_ids=data.task_ids,
        groups=data.groups,
        texts=data.texts,
        is_hard=data.is_hard,
        rewards=changed_rewards,
        costs=data.costs,
    )
    replay = module.replay_outer_heldout(
        changed,
        train,
        heldout,
        spec,
        seed=7,
        work_dir=tmp_path / "changed",
    )
    assert np.array_equal(initial.choices, replay.choices)
    assert initial.value.reward != replay.value.reward


def test_knn_routes_do_not_depend_on_heldout_rewards(tmp_path: Path) -> None:
    data = _data()
    train, heldout = _split()
    spec = select.KnnCandidateSpec(512, 8, 0.9, 0.0, 3, 0)
    initial = module.replay_outer_heldout(
        data,
        train,
        heldout,
        spec,
        seed=7,
        work_dir=tmp_path / "initial",
    )
    changed_rewards = data.rewards.copy()
    changed_rewards[heldout] = 1.0 - changed_rewards[heldout]
    changed = fit.FitData(
        task_ids=data.task_ids,
        groups=data.groups,
        texts=data.texts,
        is_hard=data.is_hard,
        rewards=changed_rewards,
        costs=data.costs,
    )
    replay = module.replay_outer_heldout(
        changed,
        train,
        heldout,
        spec,
        seed=7,
        work_dir=tmp_path / "changed",
    )
    assert np.array_equal(initial.choices, replay.choices)


def test_outer_replay_rejects_group_overlap(tmp_path: Path) -> None:
    data = _data()
    spec = select.CandidateSpec("ordinal", "ridge", 512, 0, alpha=1.0)
    with pytest.raises(ValueError, match="group crossed"):
        module.replay_outer_heldout(
            data,
            np.arange(23),
            np.arange(23, 30),
            spec,
            seed=7,
            work_dir=tmp_path,
        )


def test_every_frozen_base_candidate_round_trips_from_lock() -> None:
    candidates = [*select.candidate_grid(), *select.knn_candidate_grid()]
    for candidate in candidates:
        config_json, _ = fit.canonical_candidate_config(candidate.config())
        rebuilt = module.candidate_spec_from_lock(
            "knn" if isinstance(candidate, select.KnnCandidateSpec) else candidate.family,
            config_json,
            name=candidate.name,
            order=candidate.order,
        )
        assert rebuilt == candidate


def test_economic_knn_candidate_round_trips_from_lock() -> None:
    candidate = select.KnnCandidateSpec(
        2_048,
        32,
        0.95,
        1.0,
        16,
        1_027,
        guard_model="luna-low",
        guard_mode="asymmetric",
        pick_lam=0.03,
    )
    config_json, _ = fit.canonical_candidate_config(candidate.config())
    rebuilt = module.candidate_spec_from_lock(
        "knn",
        config_json,
        name=candidate.name,
        order=candidate.order,
    )
    assert rebuilt == candidate
