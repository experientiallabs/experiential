"""Tests for the closed-loop pool evaluation (candidate models rolled against an Env)."""

from __future__ import annotations

from typing import cast

import pytest

from wmh.core.types import Action, EnvState, Observation
from wmh.env.closed_loop import evaluate_pool
from wmh.env.scenarios import Scenario
from wmh.optimize.reward import EpisodeScore
from wmh.providers.base import (
    Completion,
    Message,
    ProviderKind,
    TokenUsage,
    VerifyResult,
)
from wmh.providers.pool import ModelPool, PoolEntry


class _FakeEnv:
    """Scripted Env: every action gets one observation; scoring is canned."""

    def __init__(self, score: EpisodeScore | None) -> None:
        self.last_score = score

    def reset(self, task: str | None = None, seed_state: EnvState | None = None) -> EnvState:
        return EnvState()

    def step(self, action: Action) -> Observation:
        return Observation(content="ok")

    def close(self) -> None:
        return None


class _ScriptedProvider:
    """Provider whose completions are a fixed script (one tool call, then done)."""

    def __init__(self, entry: PoolEntry) -> None:
        self.config = entry.provider_config()
        self._script = [
            '{"tool": "ls", "arguments": {}}',
            '{"done": true, "summary": "finished"}',
        ]
        self._i = 0

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 8192,
    ) -> Completion:
        text = self._script[min(self._i, len(self._script) - 1)]
        self._i += 1
        return Completion(text=text, usage=TokenUsage(input_tokens=10, output_tokens=5))

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    def verify(self) -> VerifyResult:
        raise NotImplementedError


def _pool() -> ModelPool:
    return ModelPool(
        models=[
            PoolEntry(
                name="candidate-a",
                kind=ProviderKind.OPENAI,
                model="custom-a",
                input_per_mtok=1.0,
                output_per_mtok=2.0,
            ),
            PoolEntry(
                name="candidate-b",
                kind=ProviderKind.OPENAI,
                model="custom-b",
                input_per_mtok=10.0,
                output_per_mtok=20.0,
            ),
        ]
    )


_SCENARIOS = [
    Scenario(task="list the files", provenance=["trace-1"]),
    Scenario(task="delete the files", provenance=["trace-2"]),
]


def test_evaluate_pool_builds_full_matrix() -> None:
    matrix = evaluate_pool(
        lambda: _FakeEnv(EpisodeScore(reward=0.8, success=True, critique="fine")),
        _pool(),
        _SCENARIOS,
        provider_factory=_ScriptedProvider,
        max_steps=5,
    )

    assert matrix.model_names() == ["candidate-a", "candidate-b"]
    assert matrix.scenario_ids() == ["trace-1", "trace-2"]
    assert len(matrix.outcomes) == 4  # 2 models x 2 scenarios x 1 episode
    outcome = matrix.for_scenario("trace-1")[0]
    assert outcome.model == "candidate-a"
    assert outcome.reward == pytest.approx(0.8)
    assert outcome.success is True
    assert outcome.critique == "fine"
    assert outcome.stop_reason == "agent_done"
    # Two policy calls (tool call + done), scripted usage 10in/5out each.
    assert outcome.usage == TokenUsage(input_tokens=20, output_tokens=10)
    assert len(outcome.call_seconds) == 2
    assert len(outcome.replies) == 2
    # Cost prices the POOL ENTRY's override, not the built-in table.
    assert outcome.cost_usd == pytest.approx((20 * 1.0 + 10 * 2.0) / 1_000_000)
    expensive = matrix.for_scenario("trace-1")[1]
    assert expensive.cost_usd == pytest.approx((20 * 10.0 + 10 * 20.0) / 1_000_000)


def test_evaluate_pool_repeats_episodes() -> None:
    matrix = evaluate_pool(
        lambda: _FakeEnv(EpisodeScore(reward=0.5, success=True, critique="")),
        _pool(),
        _SCENARIOS[:1],
        provider_factory=_ScriptedProvider,
        episodes_per_scenario=3,
    )
    episodes = [o.episode for o in matrix.outcomes if o.model == "candidate-a"]
    assert episodes == [0, 1, 2]


def test_evaluate_pool_requires_a_scoring_env() -> None:
    with pytest.raises(ValueError, match="score"):
        evaluate_pool(
            lambda: _FakeEnv(None),
            _pool(),
            _SCENARIOS[:1],
            provider_factory=_ScriptedProvider,
        )


def test_scenario_ids_fall_back_to_task_hash() -> None:
    matrix = evaluate_pool(
        lambda: _FakeEnv(EpisodeScore(reward=0.1, success=False, critique="")),
        _pool(),
        [Scenario(task="no provenance here")],
        provider_factory=_ScriptedProvider,
    )
    (scenario_id,) = matrix.scenario_ids()
    assert scenario_id  # deterministic, non-empty
    rerun = evaluate_pool(
        lambda: _FakeEnv(EpisodeScore(reward=0.1, success=False, critique="")),
        _pool(),
        [Scenario(task="no provenance here")],
        provider_factory=_ScriptedProvider,
    )
    assert rerun.scenario_ids() == [scenario_id]


def test_wrong_typed_score_yields_unscored_row_with_reason() -> None:
    # A last_score that isn't an EpisodeScore must not silently become an
    # unscored-with-no-error row: unscored rows always say why (outcomes contract).
    matrix = evaluate_pool(
        lambda: _FakeEnv(cast("EpisodeScore", {"reward": 1.0})),
        _pool(),
        _SCENARIOS[:1],
        provider_factory=_ScriptedProvider,
    )
    outcome = matrix.outcomes[0]
    assert outcome.reward is None
    assert outcome.error is not None and "not EpisodeScore" in outcome.error
