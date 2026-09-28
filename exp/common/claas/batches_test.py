"""Explicit batch identities and response-selection validation."""

import pytest

from exp.common.claas.batches import TrainingSubmission
from exp.common.claas.learning import FeedbackSubmission


def test_training_submission_requires_unique_responses() -> None:
    """One action cannot silently acquire extra weight through a duplicate selection."""
    feedback = FeedbackSubmission(response_id="response", reward=-0.5)
    with pytest.raises(ValueError, match="repeat response"):
        TrainingSubmission(
            batch_id="cohort-1", expected_policy_revision="policy-0", feedback=(feedback, feedback)
        )


@pytest.mark.parametrize("identity", ["", "../other", "with/slash", "with space"])
def test_training_submission_identity_is_safe_in_status_routes(identity: str) -> None:
    """Caller retry identities cannot change the batch-status route's path."""
    with pytest.raises(ValueError, match="batch_id"):
        TrainingSubmission(
            batch_id=identity,
            expected_policy_revision="policy-0",
            feedback=(FeedbackSubmission(response_id="response", reward=0.5),),
        )
