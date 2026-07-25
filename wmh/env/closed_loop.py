"""Closed-loop evaluation: roll every pool candidate over a scenario set, collect the matrix.

Each (candidate model, scenario, episode) cell runs one `run_episode` with the candidate driving
`LLMAgent` against the env, then reads the env's episode score (VERIFY). The result is the
`OutcomeMatrix` the routing optimizer fits on and the improvement report cites.

Measurement notes:
- Latency is measured per POLICY CALL (the candidate's own completions), not per episode: episode
  wall time is dominated by the world model's simulation latency, which production traffic never
  pays, so quoting it would flatter nobody honestly.
- Cost is the candidate side only, priced by its own pool entry; the env's serve/judge cost is
  metered separately by the world model (D12 cost split).
- Every raw candidate reply is stored (`ScenarioOutcome.replies`): that is the future
  distillation feed. Providers do not yet surface separated thinking blocks; when they do, the
  capture point is `_TimedProvider.complete`.
"""

from __future__ import annotations

import hashlib
import logging
import time
from typing import TYPE_CHECKING

from wmh.env.base import Env
from wmh.env.episode import run_episode
from wmh.env.llm_agent import LLMAgent
from wmh.env.scenarios import Scenario
from wmh.optimize.outcomes import OutcomeMatrix, ScenarioOutcome
from wmh.optimize.reward import EpisodeScore
from wmh.providers.base import (
    DEFAULT_MAX_TOKENS,
    Completion,
    Message,
    Provider,
    ProviderConfig,
    TokenUsage,
    VerifyResult,
)
from wmh.providers.pool import ModelPool, PoolEntry, pool_provider

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


class _TimedProvider:
    """Wraps the candidate's provider to record per-call seconds, usage, and raw replies."""

    def __init__(self, provider: Provider) -> None:
        self._provider = provider
        self.call_seconds: list[float] = []
        self.replies: list[str] = []
        self.usage = TokenUsage()

    @property
    def config(self) -> ProviderConfig:
        return self._provider.config

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> Completion:
        started = time.monotonic()
        completion = self._provider.complete(
            system, messages, temperature=temperature, max_tokens=max_tokens
        )
        self.call_seconds.append(time.monotonic() - started)
        self.replies.append(completion.text)
        self.usage = TokenUsage(
            input_tokens=self.usage.input_tokens + completion.usage.input_tokens,
            cached_input_tokens=self.usage.cached_input_tokens
            + completion.usage.cached_input_tokens,
            output_tokens=self.usage.output_tokens + completion.usage.output_tokens,
        )
        return completion

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._provider.embed(texts)

    def verify(self) -> VerifyResult:
        return self._provider.verify()


def scenario_id(scenario: Scenario) -> str:
    """Stable id for a scenario: its first provenance trace id, else a hash of the task.

    Provisional until wm-create's generate contract ships first-class scenario ids
    (DECISIONS.md 2026-07-23); both forms are deterministic across runs.
    """
    if scenario.provenance:
        return scenario.provenance[0]
    return hashlib.sha256(scenario.task.encode("utf-8")).hexdigest()[:12]


def evaluate_pool(
    env_factory: Callable[[], Env],
    pool: ModelPool,
    scenarios: list[Scenario],
    *,
    episodes_per_scenario: int = 1,
    max_steps: int = 20,
    agent_temperature: float = 0.0,
    tools_hint: str | None = None,
    provider_factory: Callable[[PoolEntry], Provider] = pool_provider,
    on_outcome: Callable[[ScenarioOutcome], None] | None = None,
) -> OutcomeMatrix:
    """Run every pool candidate over `scenarios`, one fresh env per episode.

    The env must score episodes on close (`WorldModelEnv(..., score_on_close=True)`): a matrix
    without verified rewards is not evidence. Episodes that error before scoring are recorded
    unscored (`reward=None`, `error` set) rather than defaulted to 0. `on_outcome` fires after
    each cell for progress display.
    """
    outcomes: list[ScenarioOutcome] = []
    for entry in pool.models:
        for scenario in scenarios:
            sid = scenario_id(scenario)
            for episode in range(episodes_per_scenario):
                timed = _TimedProvider(provider_factory(entry))
                agent = LLMAgent(timed, temperature=agent_temperature, tools_hint=tools_hint)
                env = env_factory()
                result = run_episode(env, agent, scenario.task, max_steps=max_steps)
                score = getattr(env, "last_score", None)
                error = result.error
                if score is None and error is None:
                    raise ValueError(
                        "env produced no episode score; evaluate_pool needs a scoring env "
                        "(e.g. WorldModelEnv(world_model, score_on_close=True))"
                    )
                if score is not None and not isinstance(score, EpisodeScore):
                    # An unscored row must always say WHY (the outcomes contract); a wrong-typed
                    # score silently becoming reward=None/error=None would violate it.
                    error = error or (
                        f"env last_score is {type(score).__name__}, not EpisodeScore; "
                        "episode left unscored"
                    )
                    score = None
                outcome = ScenarioOutcome(
                    scenario_id=sid,
                    task=scenario.task,
                    model=entry.name,
                    episode=episode,
                    reward=score.reward if score else None,
                    success=score.success if score else False,
                    critique=score.critique if score else "",
                    steps=len(result.steps),
                    stop_reason=str(result.stop_reason),
                    usage=timed.usage,
                    cost_usd=entry.cost_usd(timed.usage),
                    call_seconds=timed.call_seconds,
                    replies=timed.replies,
                    error=error,
                )
                outcomes.append(outcome)
                if on_outcome is not None:
                    on_outcome(outcome)
                logger.info(
                    "closed-loop %s on %s ep%d: reward=%s cost=$%.5f steps=%d",
                    entry.name,
                    sid,
                    episode,
                    "unscored" if outcome.reward is None else f"{outcome.reward:.2f}",
                    outcome.cost_usd,
                    outcome.steps,
                )
    return OutcomeMatrix(pool=pool.models, outcomes=outcomes)
