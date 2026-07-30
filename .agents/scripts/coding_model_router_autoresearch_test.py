"""Tests for the external-only DeepSWE autoresearch runner."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np


def _module() -> ModuleType:
    path = Path(__file__).with_name("coding_model_router_autoresearch.py")
    spec = importlib.util.spec_from_file_location("coding_model_router_autoresearch", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_operating_point_uses_strong_only_where_score_is_high() -> None:
    module = _module()
    scores = np.asarray([0.9, 0.8, 0.2, 0.1])
    weak = np.asarray([0.0, 0.0, 1.0, 1.0])
    strong = np.asarray([1.0, 1.0, 1.0, 1.0])
    point = module._operating_point(
        scores,
        weak,
        strong,
        ["a", "a", "b", "b"],
        0.95,
    )
    assert point["strong_traffic"] == 0.5
    assert point["mean_retention"] == 1.0


def test_three_level_route_is_monotone_in_predicted_uplift() -> None:
    module = _module()
    decisions = module._route_indices(
        np.asarray([-0.1, 0.3, 0.8]),
        [0.2, 0.6],
        3,
    )
    assert decisions.tolist() == [0, 1, 2]


def test_combine_deduplicates_exact_normalized_text() -> None:
    module = _module()
    source = module.SourceData(
        name="first",
        task_ids=["a", "b"],
        groups=["r1", "r2"],
        texts=["same task", "unique"],
        weak=np.asarray([0.0, 0.0]),
        strong=np.asarray([1.0, 1.0]),
        weak_attempts=np.ones(2),
        strong_attempts=np.ones(2),
    )
    duplicate = module.SourceData(
        name="second",
        task_ids=["c"],
        groups=["r3"],
        texts=["same   task"],
        weak=np.asarray([0.0]),
        strong=np.asarray([1.0]),
        weak_attempts=np.ones(1),
        strong_attempts=np.ones(1),
    )
    combined = module._combine([source, duplicate])
    assert combined.task_ids == ["a", "b"]
