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


def test_empirical_bayes_rate_preserves_source_mean_and_arm_order() -> None:
    module = _module()
    weak_mean = 0.075
    strong_mean = 0.112
    attempts = 2
    weak_rates = np.asarray(
        [
            module._empirical_bayes_rate(0.0, attempts, weak_mean),
            module._empirical_bayes_rate(1.0, attempts, weak_mean),
        ]
    )
    strong_rates = np.asarray(
        [
            module._empirical_bayes_rate(0.0, attempts, strong_mean),
            module._empirical_bayes_rate(1.0, attempts, strong_mean),
        ]
    )
    assert np.isclose(weak_rates.mean(), (0.5 + 4.0 * weak_mean) / 6.0)
    assert np.isclose(strong_rates.mean(), (0.5 + 4.0 * strong_mean) / 6.0)
    assert strong_rates.mean() > weak_rates.mean()


def test_calibrated_irt_probability_matches_arm_mean_and_peaks_uplift_midrange() -> None:
    module = _module()
    easiness = np.linspace(-4.0, 4.0, 401)
    weak = module._calibrated_irt_probability(easiness, 0.60)
    strong = module._calibrated_irt_probability(easiness, 0.72)
    uplift = strong - weak
    assert np.isclose(weak.mean(), 0.60)
    assert np.isclose(strong.mean(), 0.72)
    assert int(np.argmax(uplift)) not in (0, len(uplift) - 1)
    assert uplift[len(uplift) // 2] > uplift[0]
    assert uplift[len(uplift) // 2] > uplift[-1]
