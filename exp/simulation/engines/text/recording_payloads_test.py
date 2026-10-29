"""Regression coverage for merging a resumed visible transcript without duplicating its suffix."""

from datetime import UTC, datetime

from exp.common.models import ModelMessage, ModelRequest
from exp.common.rollouts import RolloutEventKind, RolloutSpan
from exp.simulation.engines.text.recording_payloads import (
    bounded_candidate_request,
    delivered_world_span,
)


def test_resumed_request_keeps_original_task_and_one_complete_transcript() -> None:
    """A restarted agent's matching action suffix is replaced by the complete retained history."""
    task = ModelMessage(role="user", content="question")
    prior = ModelMessage(role="assistant", content="first answer")
    followup = ModelMessage(role="user", content="clarify")
    answer = ModelMessage(role="assistant", content="clarification")
    request = ModelRequest(messages=(task, answer), maximum_output_tokens=500)
    result = bounded_candidate_request(
        request, visible_transcript=(prior, followup, answer), maximum_output_tokens=1000
    )
    assert result.messages == (task, prior, followup, answer)
    assert result.maximum_output_tokens == 500


def test_delivered_observations_obey_the_same_field_redaction_as_diagnostic_payloads() -> None:
    """Adding visible world messages cannot bypass the project's redaction policy."""
    now = datetime(2026, 9, 25, tzinfo=UTC)
    span = RolloutSpan(
        span_id="world-1",
        kind=RolloutEventKind.SIMULATOR_WORLD_MODEL_CALL,
        started_at=now,
        ended_at=now,
        payload={},
    )
    result = delivered_world_span(
        span, (ModelMessage(role="user", content="private value"),), frozenset({"content"})
    )
    assert "private value" not in result.model_dump_json()
    assert "visible_messages" in result.payload
