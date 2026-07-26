"""Tests for the closed-loop outcome matrix types (the routing optimizer's training data)."""

from __future__ import annotations

from pathlib import Path

import pytest

from wmo.optimize.outcomes import OutcomeMatrix, ScenarioOutcome
from wmo.providers.base import ProviderKind, TokenUsage
from wmo.providers.pool import PoolEntry


def _outcome(
    scenario_id: str, model: str, *, reward: float = 0.5, episode: int = 0
) -> ScenarioOutcome:
    return ScenarioOutcome(
        scenario_id=scenario_id,
        task="do the thing",
        model=model,
        episode=episode,
        reward=reward,
        success=reward >= 0.5,
        critique="ok",
        steps=3,
        stop_reason="agent_done",
        usage=TokenUsage(input_tokens=100, output_tokens=50),
        cost_usd=0.01,
        call_seconds=[0.2, 0.3, 0.25],
        replies=["{}", "{}", '{"done": true}'],
    )


def _matrix() -> OutcomeMatrix:
    entries = [
        PoolEntry(name="fable-5", kind=ProviderKind.ANTHROPIC, model="claude-fable-5"),
        PoolEntry(name="haiku-4-5", kind=ProviderKind.ANTHROPIC, model="claude-haiku-4-5"),
    ]
    outcomes = [
        _outcome("s1", "fable-5", reward=0.9),
        _outcome("s2", "fable-5", reward=0.7),
        _outcome("s1", "haiku-4-5", reward=0.4),
        _outcome("s2", "haiku-4-5", reward=0.8),
    ]
    return OutcomeMatrix(pool=entries, outcomes=outcomes)


def test_matrix_accessors() -> None:
    matrix = _matrix()
    assert matrix.model_names() == ["fable-5", "haiku-4-5"]
    assert matrix.scenario_ids() == ["s1", "s2"]
    assert matrix.mean_reward("fable-5") == pytest.approx(0.8)
    assert matrix.mean_reward("haiku-4-5") == pytest.approx(0.6)
    assert [o.model for o in matrix.for_scenario("s1")] == ["fable-5", "haiku-4-5"]


def test_matrix_round_trips_through_json(tmp_path: Path) -> None:
    matrix = _matrix()
    path = tmp_path / "outcomes.json"
    matrix.save(path)
    loaded = OutcomeMatrix.load(path)
    assert loaded == matrix


def test_mean_reward_unknown_model_errors() -> None:
    with pytest.raises(KeyError, match="fable-5"):
        _matrix().mean_reward("nope")


def test_outcomes_must_name_pool_models() -> None:
    # A ghost model used to reach the fitter and die on a bare KeyError; the matrix names it.
    matrix = _matrix()
    with pytest.raises(ValueError, match="ghost-model"):
        OutcomeMatrix(
            pool=matrix.pool,
            outcomes=[*matrix.outcomes, _outcome("s1", "ghost-model")],
        )
