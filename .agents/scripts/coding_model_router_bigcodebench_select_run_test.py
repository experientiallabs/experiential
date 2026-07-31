"""Tests for durable remote BigCodeBench seed-selection reports."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np


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
module = _load("coding_model_router_bigcodebench_select_run")


def _validation(
    spec: object,
    *,
    reward: float,
    cost: float,
    baseline_reward: float = 0.8,
) -> object:
    value = fit.PolicyValue(reward, cost, reward - 0.01, cost, {fit.ARMS[0]: 1})
    baseline = fit.PolicyValue(
        baseline_reward,
        1.0,
        baseline_reward,
        1.0,
        {fit.ARMS[-1]: 1},
    )
    return select.CandidateValidation(
        spec=spec,
        value=value,
        baseline=baseline,
        metric=fit.CandidateMetric(spec.name, reward, cost, 0.0, 0, spec.order),
    )


def test_family_winner_uses_the_frozen_quality_floor() -> None:
    non_knn = _validation(
        select.CandidateSpec("ordinal", "ridge", 512, 0, alpha=1.0),
        reward=0.77,
        cost=0.6,
    )
    knn = _validation(
        select.KnnCandidateSpec(512, 8, 0.9, 0.5, 8, 576),
        reward=0.76,
        cost=0.4,
    )
    assert module.select_family_winner(non_knn, knn) is knn


def test_candidate_record_canonicalizes_config() -> None:
    result = _validation(
        select.KnnCandidateSpec(512, 8, 0.9, 0.5, 8, 576),
        reward=0.76,
        cost=0.4,
    )
    record = module.candidate_record(result)
    assert record.family == "knn"
    assert record.name == result.spec.name
    assert len(record.config_sha256) == 64
    assert record.fit_reward == 0.76


def test_seed_report_rejects_duplicate_candidate_names() -> None:
    result = _validation(
        select.KnnCandidateSpec(512, 8, 0.9, 0.5, 8, 576),
        reward=0.76,
        cost=0.4,
    )
    record = module.candidate_record(result)
    values = [record.model_copy(update={"name": f"candidate-{index}"}) for index in range(1_028)]
    values[-1] = values[0]
    with np.testing.assert_raises(ValueError):
        module.SeedFitReport(
            seed=0,
            code_commit="a" * 40,
            tasks_sha256="b" * 64,
            scores_sha256="c" * 64,
            outcomes_sha256="d" * 64,
            oracle_report_sha256="e" * 64,
            fit_tasks=240,
            heldout_tasks=60,
            fit_ids_sha256="f" * 64,
            heldout_ids_sha256="0" * 64,
            baseline_arm=fit.ARMS[-1],
            baseline_fit_reward=0.8,
            baseline_fit_cost_usd=1.0,
            candidates=values,
            selected_name=values[0].name,
        )
