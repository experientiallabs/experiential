"""Tests for the Codeforces nested grouped router fitter."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np


def _module() -> ModuleType:
    path = Path(__file__).with_name("coding_model_router_codeforces_fit.py")
    spec = importlib.util.spec_from_file_location("codeforces_fit", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


module = _module()


def test_choose_uses_threshold_and_cheapest_tie() -> None:
    predictions = np.asarray(
        [
            [0.02, 0.01, 0.0, -0.1, -0.2],
            [0.01, 0.01, 0.0, -0.1, -0.2],
        ]
    )
    costs = np.asarray([1.0, 2.0, 4.0, 8.0, 16.0])
    choices = module._choose(predictions, costs, threshold=0.01)
    assert choices.tolist() == [0, 0]
    assert module._choose(predictions, costs, threshold=0.03).tolist() == [2, 2]


def test_value_builds_exact_matched_blind_control() -> None:
    rewards = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
        ]
    )
    data = module.Data(
        task_ids=["a", "b"],
        groups=["one", "two"],
        texts=["a", "b"],
        structural=np.zeros((2, 1)),
        rewards=rewards,
        costs=np.ones_like(rewards),
    )
    value = module._value(data, np.arange(2), np.asarray([0, 1]))
    assert value["reward"] == 1.0
    assert value["matched_blind_reward"] == 0.5
    assert value["advantage"] == 0.5


def test_spearman_handles_ties_and_order() -> None:
    assert module._spearman(np.asarray([1.0, 2.0, 3.0]), np.asarray([4.0, 5.0, 6.0])) == 1.0
    assert module._spearman(np.zeros(3), np.arange(3)) == 0.0


def test_bootstrap_respects_whole_groups() -> None:
    router = np.asarray([1.0, 1.0, 0.0, 0.0])
    blind = np.asarray([0.0, 0.0, 0.0, 0.0])
    interval = module._bootstrap(
        ["a", "a", "b", "b"],
        router,
        blind,
        seed=7,
    )
    assert interval[0] == 0.0
    assert interval[2] == 1.0
