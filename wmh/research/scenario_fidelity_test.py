"""Tests for predictive-fidelity math (rank correlations, report assembly, score matrix)."""

from __future__ import annotations

import numpy as np
import pytest

from wmh.engine.world_model import WorldModel
from wmh.env.episode import Agent
from wmh.providers.base import Completion, Message, ProviderConfig, ProviderKind
from wmh.research.scenario_fidelity import (
    ScoreMatrix,
    fidelity_report,
    kendall,
    random_subsets,
    score_matrix,
    spearman,
)
from wmh.scenarios.synthesis import EvalScenario
from wmh.scenarios.verification import CHECKLIST_SYSTEM, ChecklistJudge
from wmh.scenarios.verification_test import EmptyRetriever, OneShotAgent


class RoutedProvider:
    """Answers checklist-judge prompts with a full pass, everything else with an observation."""

    def __init__(self) -> None:
        self.config = ProviderConfig(kind=ProviderKind.ANTHROPIC, model="m")

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 8192,
    ) -> Completion:
        if system == CHECKLIST_SYSTEM:
            return Completion(text='{"passed": [true], "success": true, "critique": "ok"}')
        return Completion(text='{"output": "ok", "is_error": false}')

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]

    def verify(self):  # noqa: ANN201
        raise NotImplementedError


def test_score_matrix_means_cells_and_never_enriches_the_index() -> None:
    provider = RoutedProvider()
    retriever = EmptyRetriever()
    world_model = WorldModel(provider, retriever, telemetry_root="/tmp/wmh-test-telemetry")
    pool = [EvalScenario(scenario_id=f"s{i}", task="t", checklist=["c"]) for i in range(2)]
    agents: dict[str, Agent] = {"a1": OneShotAgent(), "a2": OneShotAgent()}
    matrix = score_matrix(
        world_model, agents, pool, ChecklistJudge(provider), passes=2, max_steps=3, workers=2
    )
    assert matrix.passes == 2
    assert set(matrix.scores) == {"a1", "a2"}
    assert all(matrix.scores[a][s.scenario_id] == 1.0 for a in agents for s in pool)
    # No cell's rollout may become another cell's retrieval context.
    assert retriever.added == []


def test_spearman_perfect_and_reversed() -> None:
    a = np.asarray([1.0, 2.0, 3.0, 4.0])
    assert spearman(a, a * 10) == pytest.approx(1.0)
    assert spearman(a, -a) == pytest.approx(-1.0)


def test_spearman_constant_input_is_zero() -> None:
    assert spearman(np.asarray([1.0, 1.0, 1.0]), np.asarray([1.0, 2.0, 3.0])) == 0.0


def test_kendall_perfect_reversed_and_ties() -> None:
    a = np.asarray([1.0, 2.0, 3.0, 4.0])
    assert kendall(a, a * 2) == pytest.approx(1.0)
    assert kendall(a, -a) == pytest.approx(-1.0)
    assert kendall(np.asarray([1.0, 1.0]), np.asarray([1.0, 2.0])) == 0.0


def _matrix() -> ScoreMatrix:
    return ScoreMatrix(
        passes=3,
        scores={
            "agent-strong": {"s1": 0.9, "s2": 0.8, "s3": 0.7, "s4": 0.6},
            "agent-mid": {"s1": 0.6, "s2": 0.5, "s3": 0.4, "s4": 0.3},
            "agent-weak": {"s1": 0.3, "s2": 0.2, "s3": 0.1, "s4": 0.0},
        },
    )


def test_fidelity_report_full_pool_subset_is_exact() -> None:
    pool = ["s1", "s2", "s3", "s4"]
    report = fidelity_report(_matrix(), pool, {"all": {sid: 1.0 for sid in pool}})
    method = report.methods[0]
    assert method.mae == pytest.approx(0.0)
    assert method.spearman == pytest.approx(1.0)
    assert method.kendall == pytest.approx(1.0)


def test_fidelity_report_weighted_subset() -> None:
    pool = ["s1", "s2", "s3", "s4"]
    report = fidelity_report(_matrix(), pool, {"ours": {"s1": 0.5, "s4": 0.5}})
    method = report.methods[0]
    # Weighted mean of s1/s4 equals the full-pool mean for this matrix (symmetric spread).
    assert method.mae == pytest.approx(0.0)
    assert method.predicted["agent-strong"] == pytest.approx(0.75)


def test_fidelity_report_rejects_ids_outside_pool_and_zero_weight() -> None:
    with pytest.raises(ValueError, match="outside the pool"):
        fidelity_report(_matrix(), ["s1", "s2"], {"bad": {"s9": 1.0}})
    with pytest.raises(ValueError, match="positive weight"):
        fidelity_report(_matrix(), ["s1"], {"bad": {"s1": 0.0}})


def test_random_subsets_are_deterministic_and_sized() -> None:
    pool = [f"s{i}" for i in range(10)]
    first = random_subsets(pool, 3, seeds=(0, 1))
    second = random_subsets(pool, 3, seeds=(0, 1))
    assert first == second
    assert len(first) == 2
    assert all(len(subset) == 3 for subset in first.values())
