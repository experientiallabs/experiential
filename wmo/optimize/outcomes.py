"""Closed-loop outcome matrix: per (scenario x candidate model x episode) eval records.

This is the routing optimizer's training data and the improvement report's evidence base, the
RouterBench-style precomputed matrix: run the pool over the scenario set once, then compare any
number of policy variants offline on identical data. Produced by `wmo.env.closed_loop`
(kept import-free of `wmo.env`/`wmo.engine` here so optimizers can consume it without cycles).
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, model_validator

from wmo.providers.base import TokenUsage
from wmo.providers.pool import PoolEntry


class ScenarioOutcome(BaseModel):
    """One episode of one candidate model on one scenario.

    `reward` is None only for unscored episodes (`error` says why); consumers fitting policies
    or reporting numbers must skip unscored rows, never default them to 0 (an infrastructure
    failure is not a judge verdict).
    """

    scenario_id: str
    task: str
    model: str  # pool entry name (the stable handle policy artifacts key on)
    episode: int = 0
    reward: float | None = None
    success: bool = False
    critique: str = ""
    steps: int = 0
    stop_reason: str = ""
    usage: TokenUsage = TokenUsage()
    cost_usd: float = 0.0  # candidate-side cost, priced by ITS pool entry
    call_seconds: list[float] = []  # wall seconds per policy call (env time excluded)
    # Raw completion texts per call: the future distillation feed (stored, never used by v1
    # fitting). Reasoning models that emit thought before the JSON action keep it here.
    replies: list[str] = []
    error: str | None = None

    @property
    def scored(self) -> bool:
        return self.reward is not None


class OutcomeMatrix(BaseModel):
    """The full pool x scenario outcome grid, plus the pool it was measured on.

    Carrying the pool snapshot makes the matrix self-describing: a policy fitted from it can
    record exactly which candidates (and at what prices) its assignments were chosen over.
    """

    pool: list[PoolEntry]
    outcomes: list[ScenarioOutcome]

    @model_validator(mode="after")
    def _outcomes_name_pool_models(self) -> OutcomeMatrix:
        """Every outcome must name a pool entry, or the matrix is not self-describing.

        Consumers index the pool by outcome model (`fit_rank_policy`'s `pool_order`, the report's
        per-candidate table). A row naming a model the pool never heard of used to surface as a
        bare `KeyError` deep inside a fitter; caught here it names the offender instead.
        """
        names = {entry.name for entry in self.pool}
        ghosts = sorted({o.model for o in self.outcomes if o.model not in names})
        if ghosts:
            raise ValueError(
                f"outcomes name models missing from the pool: {ghosts[:5]}; "
                f"pool models are {sorted(names)}"
            )
        return self

    def model_names(self) -> list[str]:
        return [entry.name for entry in self.pool]

    def scenario_ids(self) -> list[str]:
        seen: dict[str, None] = {}
        for outcome in self.outcomes:
            seen.setdefault(outcome.scenario_id, None)
        return list(seen)

    def for_scenario(self, scenario_id: str) -> list[ScenarioOutcome]:
        return [o for o in self.outcomes if o.scenario_id == scenario_id]

    def mean_reward(self, model: str) -> float:
        """Mean reward of `model` over its scored episodes."""
        if model not in self.model_names():
            raise KeyError(f"no pool model named '{model}'; available: {self.model_names()}")
        rewards = [o.reward for o in self.outcomes if o.model == model and o.reward is not None]
        if not rewards:
            raise ValueError(f"pool model '{model}' has no scored episodes in this matrix")
        return sum(rewards) / len(rewards)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> OutcomeMatrix:
        return cls.model_validate_json(path.read_text(encoding="utf-8"))
