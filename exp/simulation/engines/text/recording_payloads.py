"""Canonical visible transcripts and redacted model-call evidence for text rollouts."""

from dataclasses import dataclass
from datetime import datetime

from pydantic import JsonValue

from exp.common.models import (
    AssistantAction,
    CompletionCostReservation,
    ModelCapabilities,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    OperationEconomics,
    reconcile_completion_economics,
)
from exp.common.rollouts import RolloutEventKind, RolloutSpan
from exp.runtime.models.providers.errors import ProviderPricingUnavailableError
from exp.simulation.engines.text.packing import pack_world_model_request
from exp.simulation.engines.text.prompt import TextWorldModelTransition, retry_world_model_request
from exp.simulation.engines.text.redaction import redact_json
from exp.simulation.engines.text.tokens import TokenCounter


@dataclass(frozen=True)
class RecordedTextCalls:
    """Immutable recorded calls, visible transitions, and separated operation economics.

    Attributes:
        candidate_spans: Retained candidate-call evidence in dispatch order.
        world_model_spans: Retained world-call evidence in dispatch order.
        candidate_economics: Combined candidate-only usage and cost.
        world_model_economics: Combined world-only usage and cost, including invalid replies.
        retrieval_economics: Combined retained retrieval accounting.
        transitions: Accepted world transitions delivered to the candidate.
        retrieved_transition_ids: Ordered grounding IDs actually dispatched per world response.
    """

    candidate_spans: tuple[RolloutSpan, ...]
    world_model_spans: tuple[RolloutSpan, ...]
    candidate_economics: OperationEconomics
    world_model_economics: OperationEconomics
    retrieval_economics: OperationEconomics
    transitions: tuple[TextWorldModelTransition, ...]
    retrieved_transition_ids: tuple[tuple[str, ...], ...]


def priced_response(
    response: ModelResponse, reservation: CompletionCostReservation | None
) -> tuple[ModelResponse, ValueError | None]:
    """Reconcile a response, retaining paid evidence before a full-schedule pricing error.

    The recorder must append the returned response and span before raising the returned
    error. Unknown valuation never inherits a provider's unrelated dollar-cost field.
    """
    if reservation is None:
        return response, None
    try:
        economics = reconcile_completion_economics(reservation, response.economics)
    except ValueError as error:
        if reservation.token_prices is None and not isinstance(
            error, ProviderPricingUnavailableError
        ):
            raise
        return response.model_copy(
            update={"economics": response.economics.model_copy(update={"cost_usd": None})}
        ), error
    return response.model_copy(update={"economics": economics}), None


def world_retry_request(
    request: ModelRequest,
    action: AssistantAction,
    reason: str,
    *,
    capabilities: ModelCapabilities,
    reservation: CompletionCostReservation | None,
    token_counter: TokenCounter,
) -> ModelRequest:
    """Add format feedback when required content fits the original input reservation.

    With no room for feedback, the next simulator attempt uses the admitted original request.
    Optional examples may be omitted whole; required evidence and output allowance are retained.
    """
    corrected = retry_world_model_request(request, action, reason)
    context = capabilities.context_window_tokens
    output = request.maximum_output_tokens
    if context is None or output is None:
        return request
    ceiling = context - output
    if reservation is not None:
        ceiling = min(ceiling, reservation.maximum_input_tokens)
    corrected, _ = pack_world_model_request(
        corrected, maximum_input_tokens=ceiling, token_counter=token_counter
    )
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
