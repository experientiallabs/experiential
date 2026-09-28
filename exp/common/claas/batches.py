"""Explicit feedback groups bound to one immutable student policy."""

from __future__ import annotations

from pydantic import Field, model_validator

from exp.common.claas.contracts import Identifier
from exp.common.claas.learning import FeedbackSubmission
from exp.common.core.artifacts import ContractModel


class TrainingSubmission(ContractModel):
    """Select all responses and complete feedback for exactly one optimizer update.

    Attributes:
        batch_id: Caller-owned retry identity, unique within the learning run.
        expected_policy_revision: Exact policy that generated every selected response.
        feedback: Ordered, unique response IDs with their complete learning signals.
            Reward attribution and centering belong to the caller.
    """

    batch_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    expected_policy_revision: Identifier
    feedback: tuple[FeedbackSubmission, ...] = Field(min_length=1, max_length=4096)

    @model_validator(mode="after")
    def _unique_responses(self) -> TrainingSubmission:
        """Reject repeated responses before any queue or feedback mutation."""
        identities = tuple(item.response_id for item in self.feedback)
        if len(set(identities)) != len(identities):
            raise ValueError("training submission must not repeat response IDs")
        return self
