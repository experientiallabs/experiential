"""Closed-loop outcome matrix: per (scenario x candidate model x episode) eval records.

This is the routing optimizer's training data and the improvement report's evidence base, the
RouterBench-style precomputed matrix: run the pool over the scenario set once, then compare any
number of policy variants offline on identical data. Produced by `wmo.env.closed_loop`
(kept import-free of `wmo.env`/`wmo.engine` here so optimizers can consume it without cycles).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from pydantic import BaseModel, model_validator

from wmo.providers.base import TokenUsage
from wmo.providers.pool import PoolEntry

# Provenance carries a digest of the matrix, not just its path: a corpus is routinely rebuilt in
# place under the same filename, and a fit is identified by the data it saw. 16 hex characters
# is 64 bits, far past collision risk for the handful of matrices one artifact directory sees.
MATRIX_DIGEST_CHARS = 16


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


def load_matrix_with_digest(matrix_file: Path) -> tuple[OutcomeMatrix, str]:
    """The matrix and its `<path> sha256=<digest>` provenance, from ONE read of the file.

    The digest is what makes a policy's `fitted_from` an identity rather than a label. `tune`
    compares it against the as-fitted snapshot beside a policy, and two fits of the same path with
    different contents (or the same contents at two paths) have to come out different for that
    check to protect anything.

    Both come out of the same bytes on purpose. Digesting a SECOND read would let a corpus
    rebuilt in place between the two describe the fit as having seen bytes it never saw: the
    policy would be fitted from the old matrix and stamped with the new one's digest, so the
    next fit of that new matrix would match its provenance and `tune` would accept the
    superseded snapshot, the exact failure the digest exists to catch, reintroduced by
    reading twice.
    """
    payload = matrix_file.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    return (
        OutcomeMatrix.model_validate_json(payload),
        f"{matrix_file} sha256={digest[:MATRIX_DIGEST_CHARS]}",
    )
