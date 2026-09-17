"""Feedback submissions to an independently hosted learner service."""

from __future__ import annotations

from pydantic import Field, model_validator

from exp.common.claas.contracts import Identifier
from exp.common.core.artifacts import ContractModel


class FeedbackSubmission(ContractModel):
    """Explicit feedback for the standard response ID returned by the learner."""

    response_id: Identifier
    reward: float | None = Field(default=None, strict=True, ge=-1, le=1, allow_inf_nan=False)
    success: bool | None = Field(default=None, strict=True)
    text: str | None = Field(default=None, min_length=1, max_length=65_536)

    @model_validator(mode="after")
    def _validate_feedback(self) -> FeedbackSubmission:
        """Keep absent signals absent and reject contradictions or blank labels."""
        if not self.response_id.strip() or len(self.response_id.encode()) > 512:
            raise ValueError("response_id must contain 1 to 512 UTF-8 bytes")
        if self.reward is None and self.success is None and self.text is None:
            raise ValueError("feedback requires reward, success, or text")
        if self.text is not None and (not self.text.strip() or len(self.text.encode()) > 65_536):
            raise ValueError("feedback text must contain 1 to 65536 UTF-8 bytes")
        if (
            self.reward is not None
            and self.success is not None
            and self.reward != float(self.success)
        ):
            raise ValueError(
                "combined feedback requires reward=1 for success or reward=0 for failure"
            )
        return self

    @property
    def training_reward(self) -> float | None:
        """Convert explicit binary feedback to zero or one without scoring text implicitly."""
        return (
            self.reward
            if self.reward is not None
            else None
            if self.success is None
            else float(self.success)
        )
