"""Project role assignments and their explicit reasoning-effort contracts."""

from __future__ import annotations

from pydantic import Field, field_validator, model_validator

from exp.common.core.artifacts import ContractModel
from exp.common.models.model import ReasoningEffort


class ModelRoles(ContractModel):
    """Project roles that select stable aliases without revealing credentials.

    Each completion role may carry its own reasoning-effort choice, so one alias can use
    different efforts as world model, judge, or router candidate. An absent role effort means
    the alias's catalog capability pin applies unchanged.
    """

    candidates: tuple[str, ...] = ()
    incumbent: str | None = None
    world_model: str | None = None
    judge: str | None = None
    rubric_proposer: str | None = None
    embedder: str | None = None
    teacher: str | None = None
    world_model_reasoning_effort: ReasoningEffort | None = None
    judge_reasoning_effort: ReasoningEffort | None = None
    candidate_reasoning_efforts: dict[str, ReasoningEffort] = Field(default_factory=dict)

    @field_validator("candidates")
    @classmethod
    def _require_unique_candidates(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Reject repeated aliases in the ordered candidate set."""
        if len(set(value)) != len(value):
            raise ValueError("candidate aliases must not repeat")
        return value

    @model_validator(mode="after")
    def _require_role_bound_reasoning_efforts(self) -> ModelRoles:
        """Require every role-specific effort to name a currently assigned role alias.

        Returns:
            The validated roles.

        Raises:
            ValueError: An effort is declared for an unassigned role or unknown candidate.
        """
        if self.world_model_reasoning_effort is not None and self.world_model is None:
            raise ValueError("world_model_reasoning_effort requires an assigned world_model")
        if self.judge_reasoning_effort is not None and self.judge is None:
            raise ValueError("judge_reasoning_effort requires an assigned judge")
        unknown = sorted(set(self.candidate_reasoning_efforts).difference(self.candidates))
        if unknown:
            raise ValueError(
                "candidate_reasoning_efforts name unassigned candidates: " + ", ".join(unknown)
            )
        return self
