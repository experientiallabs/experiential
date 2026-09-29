"""Regression coverage for merging a resumed visible transcript without duplicating its suffix."""

from datetime import UTC, datetime

import pytest

from exp.common.models import (
    AssistantAction,
    BillingSource,
    CompletionCostReservation,
    ModelCapabilities,
    ModelMessage,
    ModelRequest,
    ModelSnapshot,
)
from exp.common.rollouts import RolloutEventKind, RolloutSpan
from exp.simulation.engines.text.prompt import (
    TextWorldModelProtocolError,
    parse_world_model_transition,
    retry_world_model_request,
)
from exp.simulation.engines.text.recording_payloads import (
    bounded_candidate_request,
    delivered_world_span,
    world_retry_request,
)
from exp.simulation.engines.text.tokens import Utf8UpperBoundTokenCounter


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


@pytest.mark.parametrize("limited_by", ["context", "reservation"])
@pytest.mark.parametrize("fitting", ["detailed", "generic", "original"])
def test_schema_correction_preserves_generic_guidance_at_both_input_boundaries(
    limited_by: str, fitting: str
) -> None:
    """Extra diagnostics cannot displace a generic correction that fits the same ceiling."""
    action = AssistantAction(content="answer")
    request = ModelRequest(
        messages=(ModelMessage(role="user", content="unchanged full original evidence"),),
        maximum_output_tokens=500,
    )
    with pytest.raises(TextWorldModelProtocolError) as raised:
        parse_world_model_transition(AssistantAction(content='{"message":"hi","extra":true}'))
    reason = raised.value
    detailed = retry_world_model_request(request, action, str(reason))
    generic = retry_world_model_request(request, action, reason.generic_reason)
    counter = Utf8UpperBoundTokenCounter()
    assert counter.count(request) < counter.count(generic) < counter.count(detailed)
    expected = {"detailed": detailed, "generic": generic, "original": request}[fitting]
    ceiling = counter.count(expected)
    reservation = CompletionCostReservation(
        model=ModelSnapshot(
            billing_source=BillingSource.CUSTOMER_MANAGED,
            provider="test",
            model_id="world",
            capabilities_sha256="a" * 64,
            connection_sha256="b" * 64,
        ),
        input_usd_per_million_tokens=1,
        output_usd_per_million_tokens=1,
        cached_input_usd_per_million_tokens=1,
        cache_write_usd_per_million_tokens=1,
        maximum_attempts=3,
        maximum_input_tokens=ceiling,
        maximum_output_tokens=500,
        estimated_maximum_call_cost_usd=3 * (ceiling + 500) / 1_000_000,
    )
    actual = world_retry_request(
        request,
        action,
        str(reason),
        generic_reason=reason.generic_reason,
        capabilities=ModelCapabilities(
            context_window_tokens=ceiling + 500 if limited_by == "context" else 100_000
        ),
        reservation=reservation if limited_by == "reservation" else None,
        token_counter=counter,
    )
    assert actual == expected
    assert actual.messages[: len(request.messages)] == request.messages
    assert actual.maximum_output_tokens == 500


def test_non_schema_protocol_reason_does_not_gain_different_generic_feedback() -> None:
    """Native-tool or action validation errors keep their existing correction reason."""
    with pytest.raises(TextWorldModelProtocolError) as raised:
        parse_world_model_transition(AssistantAction(content="not JSON"))
    assert raised.value.generic_reason == str(raised.value)
