"""Tests for broad SWE-smith artifact auditing and consensus selection."""

from __future__ import annotations

import dataclasses
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import joblib
import numpy as np


def _module() -> ModuleType:
    scripts = Path(__file__).parent
    sys.path.insert(0, str(scripts))
    path = scripts / "coding_model_router_swe_smith_postfit.py"
    spec = importlib.util.spec_from_file_location("coding_model_router_swe_smith_postfit", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_numeric_artifact_round_trip_has_stable_routes(tmp_path: Path) -> None:
    module = _module()
    candidate = next(
        spec
        for spec in module.selection._candidate_specs()
        if spec.name == "hash512-ridge-heads-a1"
    )
    winner = {
        "name": candidate.name,
        "family": "numeric",
        "config": dataclasses.asdict(candidate),
        "primary": {"threshold": 0.0},
    }
    texts = [f"fix function {index} with regression test" for index in range(12)]
    weak = np.asarray([float(index % 3 == 0) for index in range(12)])
    strong = np.asarray([float(index % 2 == 0) for index in range(12)])
    first_path = tmp_path / "first.joblib"
    second_path = tmp_path / "second.joblib"
    module.selection._fit_artifact(winner, texts, weak, strong, first_path)
    module.selection._fit_artifact(winner, texts, weak, strong, second_path)
    first = module._artifact_routes(joblib.load(first_path), texts)
    second = module._artifact_routes(joblib.load(second_path), texts)
    assert first.tolist() == second.tolist()
    assert module._route_digest(first) == module._route_digest(second)


def test_knn_artifact_routes_unseen_prompts_without_network(tmp_path: Path) -> None:
    module = _module()
    config = module.selection.KnnConfig(512, 8, 0.9, 0.0, 8)
    winner = {
        "name": config.name,
        "family": "knn",
        "config": dataclasses.asdict(config),
        "primary": {"threshold": 0.0},
    }
    texts = [f"repair parser edge case {index}" for index in range(16)]
    weak = np.zeros(16)
    strong = np.ones(16)
    artifact = tmp_path / "knn.joblib"
    module.selection._fit_artifact(winner, texts, weak, strong, artifact)
    routes = module._artifact_routes(joblib.load(artifact), ["repair parser edge case 99"])
    assert routes.tolist() == [True]


def test_candidate_summary_requires_every_seed_to_retain_quality() -> None:
    module = _module()
    rows = [
        {
            "family": "numeric",
            "config": {"value": seed},
            "primary": {
                "retention": retention,
                "strong_traffic": 0.5,
                "router_reward": 0.4,
            },
        }
        for seed, retention in enumerate((0.96, 0.95, 0.97, 0.94, 0.98))
    ]
    summary = module._candidate_summary("candidate", rows)
    assert summary["minimum_retention"] == 0.94
    assert summary["fit_quality_feasible"] is False
    assert len(module._canonical_order()) == 455


def test_outcome_blind_controls_have_frozen_route_counts() -> None:
    module = _module()
    task_ids = [f"task-{index}" for index in range(21)]
    first = module._hashed_exact_count(task_ids, 7, "matched")
    second = module._hashed_exact_count(task_ids, 7, "matched")
    assert first.tolist() == second.tolist()
    assert int(np.sum(first)) == 7
    uniform = module._hashed_uniform(task_ids, "uniform")
    assert uniform.tolist() == module._hashed_uniform(task_ids, "uniform").tolist()


def test_repository_bootstrap_is_seed_balanced_and_deterministic() -> None:
    module = _module()
    rows = [
        {
            "seed": seed,
            "repo": f"repo-{repo}",
            "router_reward": 1.0,
            "strong_reward": 1.0,
            "weak_reward": 0.0,
            "task_blind_reward": 0.0,
            "shuffled_reward": 0.0,
            "random_reward": 0.0,
        }
        for seed in range(5)
        for repo in range(3)
    ]
    first = module._bootstrap_sample(rows, np.random.default_rng(20260731))
    second = module._bootstrap_sample(rows, np.random.default_rng(20260731))
    assert first == second
    assert len(first) == len(rows)
    assert {row["seed"] for row in first} == set(range(5))
