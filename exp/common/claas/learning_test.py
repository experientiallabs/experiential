"""Sparse and complementary learner feedback validation."""

import pytest
from pydantic import ValidationError

from exp.common.claas.learning import FeedbackSubmission


def test_text_feedback_does_not_imply_a_scalar() -> None:
    """A text-only signal stays eligible for text feedback without invented numeric labels."""
    assert FeedbackSubmission(response_id="response", text="Use the tool").training_reward is None
    assert FeedbackSubmission(response_id="response", success=False).training_reward == 0.0
    assert FeedbackSubmission(response_id="response", reward=-1.0).training_reward == -1.0


@pytest.mark.parametrize(
    "payload", [{}, {"text": " "}, {"reward": 0.5, "success": True}, {"reward": True}]
)
def test_feedback_rejects_missing_or_contradictory_signals(payload: dict[str, object]) -> None:
    """The service must never guess the intended signal or silently coerce a boolean reward."""
    with pytest.raises(ValidationError):
        FeedbackSubmission.model_validate({"response_id": "response", **payload})
