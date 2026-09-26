"""Canonical visible transcripts and redacted model-call evidence for text rollouts."""

from datetime import datetime

from pydantic import JsonValue

from exp.common.models import (
    AssistantAction,
    CompletionCostReservation,
    ModelCapabilities,
    ModelMessage,
    ModelRequest,
    ModelResponse,
)
from exp.common.rollouts import RolloutEventKind, RolloutSpan
from exp.simulation.engines.text.prompt import retry_world_model_request
from exp.simulation.engines.text.redaction import redact_json
from exp.simulation.engines.text.tokens import TokenCounter


def world_retry_request(
    request: ModelRequest,
    action: AssistantAction,
    reason: str,
    *,
    capabilities: ModelCapabilities,
    reservation: CompletionCostReservation | None,
    token_counter: TokenCounter,
) -> ModelRequest:
    """Add format feedback only when it fits the original context and input reservation.

    With no room for feedback, the next simulator attempt uses the admitted original request.
    No evidence is removed and the original output allowance remains available.
    """
    corrected = retry_world_model_request(request, action, reason)
    context = capabilities.context_window_tokens
    output = request.maximum_output_tokens
    if context is None or output is None:
        return request
    ceiling = context - output
    if reservation is not None:
        ceiling = min(ceiling, reservation.maximum_input_tokens)
    return corrected if 0 <= token_counter.count(corrected) <= ceiling else request


def bounded_candidate_request(
    request: ModelRequest,
    *,
    visible_transcript: tuple[ModelMessage, ...],
    maximum_output_tokens: int,
) -> ModelRequest:
    """Inject the visible transcript and enforce a caller-visible output budget."""
    requested_budget = request.maximum_output_tokens
    return request.model_copy(
        update={
            "messages": _messages_with_visible_transcript(request.messages, visible_transcript),
            "maximum_output_tokens": min(
                requested_budget or maximum_output_tokens, maximum_output_tokens
            ),
            "tool_choice": request.tool_choice,
        }
    )


def _messages_with_visible_transcript(
    messages: tuple[ModelMessage, ...],
    visible_transcript: tuple[ModelMessage, ...],
) -> tuple[ModelMessage, ...]:
    """Merge retained turns before the matching suffix owned by a restarted agent."""
    if not visible_transcript:
        return messages
    for overlap in range(min(len(messages), len(visible_transcript)), 0, -1):
        suffix = visible_transcript[-overlap:]
        # Match a complete action boundary, never a coincidentally identical task/user prompt.
        if suffix[0].role == "assistant" and messages[-overlap:] == suffix:
            return (*messages[:-overlap], *visible_transcript)
    return (*messages, *visible_transcript)


def model_span(
    *,
    span_id: str,
    kind: RolloutEventKind,
    started_at: datetime,
    ended_at: datetime,
    request: ModelRequest,
    response: ModelResponse,
    redacted_field_names: frozenset[str],
) -> RolloutSpan:
    """Build one redacted model-call span from canonical request and visible response fields."""
    payload_value: JsonValue = {
        "request": request.model_dump(mode="json", exclude_none=True),
        "response": {
            "output": response.output.model_dump(mode="json", exclude_none=True),
            "finish_reason": response.finish_reason.value,
        },
    }
    payload = redact_json(payload_value, redacted_field_names)
    if not isinstance(payload, dict):  # pragma: no cover - fixed object input remains an object
        raise TypeError("model span payload must remain a JSON object")
    return RolloutSpan(
        span_id=span_id,
        kind=kind,
        started_at=started_at,
        ended_at=ended_at,
        payload=payload,
        model=response.model,
        usage=response.economics.usage,
    )


def delivered_world_span(
    span: RolloutSpan,
    messages: tuple[ModelMessage, ...],
    redacted_field_names: frozenset[str],
) -> RolloutSpan:
    """Attach only accepted, redacted observations for the judge's visible projection."""
    payload = redact_json(
        {
            **span.payload,
            "visible_messages": [
                message.model_dump(mode="json", exclude_none=True) for message in messages
            ],
        },
        redacted_field_names,
    )
    assert isinstance(payload, dict)
    return span.model_copy(update={"payload": payload})
