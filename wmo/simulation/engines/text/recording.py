"""Visible-turn recording, context preflight, and text-only model-call boundaries."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Protocol, cast, runtime_checkable

from pydantic import JsonValue

from wmo.common.core.artifacts import (
    FailureAttribution,
    FailureCode,
    StructuredFailure,
)
from wmo.common.models import (
    AssistantAction,
    EmbeddingCostReservation,
    ModelCapabilities,
    ModelFinishReason,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    NumericMeasurement,
    OperationEconomics,
    Usage,
)
from wmo.common.rollouts import RolloutEventKind, RolloutSpan, StopReason
from wmo.common.tasks import TaskCase
from wmo.runtime.models import ResolvedModel
from wmo.simulation.engines.text.prompt import (
    TextWorldModelProtocolError,
    TextWorldModelTransition,
    text_prompt_sha256,
)
from wmo.simulation.engines.text.redaction import redact_json
from wmo.simulation.retrieval import RAGAction, RAGQuery

if TYPE_CHECKING:
    from wmo.simulation.world_model import GroundedWorldModel


class TextSimulationError(RuntimeError):
    """A text simulator boundary failed with an artifact-safe terminal classification."""

    def __init__(self, stop_reason: StopReason, failure: StructuredFailure) -> None:
        super().__init__(failure.message)
        self.stop_reason = stop_reason
        self.failure = failure


@runtime_checkable
class TokenCounter(Protocol):
    """Counts the full serialized request before a model client can send it."""

    def count(self, request: ModelRequest) -> int:
        """Return a conservative number of context tokens required by one request.

        Args:
            request: Complete provider-neutral request before provider conversion.

        Returns:
            A nonnegative count that includes all visible request content.
        """
        ...


class Utf8UpperBoundTokenCounter:
    """Provider-neutral byte upper bound used when no exact tokenizer is supplied."""

    def count(self, request: ModelRequest) -> int:
        """Count UTF-8 request bytes plus per-message framing as a conservative token bound.

        Args:
            request: Complete provider-neutral request to preflight.

        Returns:
            A conservative nonnegative bound that never silently shortens request content.
        """
        rendered = request.model_dump_json(exclude_none=False)
        return len(rendered.encode("utf-8")) + 4 * len(request.messages)


@dataclass(frozen=True)
class RecordedTextCalls:
    """Immutable recorded calls, visible world transitions, and separated operation economics."""

    candidate_spans: tuple[RolloutSpan, ...]
    world_model_spans: tuple[RolloutSpan, ...]
    candidate_economics: OperationEconomics
    world_model_economics: OperationEconomics
    retrieval_economics: OperationEconomics
    transitions: tuple[TextWorldModelTransition, ...]
    retrieved_transition_ids: tuple[tuple[str, ...], ...]


class RecordingCandidateClient:
    """Injects text-world turns after candidate calls while retaining only visible evidence."""

    def __init__(
        self,
        *,
        task: TaskCase,
        candidate: ResolvedModel,
        world_model: ResolvedModel,
        grounded_world_model: GroundedWorldModel,
        query_embedding: EmbeddingCostReservation,
        maximum_cost_usd: float,
        maximum_steps: int,
        maximum_output_tokens: int,
        redacted_field_names: frozenset[str],
        clock: Callable[[], datetime],
        token_counter: TokenCounter,
    ) -> None:
        """Bind one task, two independent model clients, and strict text-mode boundaries.

        Args:
            task: Canonical no-tools task currently being simulated.
            candidate: Candidate model injected into the customer agent.
            world_model: Model that simulates the next visible text turn.
            grounded_world_model: Artifact-bound executor over the exact fit-only index.
            query_embedding: Exact query-embedding identity, price, and retry reservation.
            maximum_cost_usd: Spend remaining for candidate, retrieval, and world-model calls.
            maximum_steps: Maximum candidate model turns allowed in this episode.
            maximum_output_tokens: Per-call output budget used without silent truncation.
            redacted_field_names: Project fields redacted before events persist.
            clock: Time source used to order emitted spans deterministically in tests.
            token_counter: Full-request counter used before every provider call.
        """
        self._task = task
        self._candidate = candidate
        self._world_model = world_model
        self._grounded_world_model = grounded_world_model
        self._query_embedding = query_embedding
        self._maximum_cost_usd = maximum_cost_usd
        self._maximum_steps = maximum_steps
        self._maximum_output_tokens = maximum_output_tokens
        self._redacted_field_names = redacted_field_names
        self._clock = clock
        self._token_counter = token_counter
        self._candidate_spans: list[RolloutSpan] = []
        self._world_model_spans: list[RolloutSpan] = []
        self._candidate_responses: list[ModelResponse] = []
        self._world_model_responses: list[ModelResponse] = []
        self._transitions: list[TextWorldModelTransition] = []
        self._retrieved_transition_ids: list[tuple[str, ...]] = []
        self._retrieval_economics: list[OperationEconomics] = []
        self._visible_transcript: tuple[ModelMessage, ...] = ()
        self._terminal = False
        self._failure: TextSimulationError | None = None
        self._provider_dispatch_unknown_spend = False

    @property
    def terminal_error(self) -> TextSimulationError | None:
        """Return the first text-boundary failure observed during the agent episode."""
        return self._failure

    @property
    def last_candidate_action(self) -> AssistantAction | None:
        """Return the final visible candidate output recorded before the episode stopped.

        Returns:
            The last recorded visible action, or ``None`` before the first candidate call.
        """
        if not self._candidate_responses:
            return None
        return self._candidate_responses[-1].output

    @property
    def world_model_terminal(self) -> bool:
        """Return whether the world model has explicitly ended the visible scenario."""
        return self._terminal

    @property
    def candidate_turn_count(self) -> int:
        """Return the number of paid candidate turns recorded in this simulation cell."""
        return len(self._candidate_responses)

    @property
    def visible_transcript(self) -> tuple[ModelMessage, ...]:
        """Return only assistant-visible candidate and simulated-user transcript turns."""
        return self._visible_transcript

    def terminal_limit_error(self) -> TextSimulationError | None:
        """Return a no-call maximum-step failure when a nonterminal world turn exhausted v1.

        Returns:
            A structured terminal error after the final permitted nonterminal world transition,
            or ``None`` when another candidate turn remains or the world model ended the scenario.
        """
        if self._terminal or len(self._candidate_responses) < self._maximum_steps:
            return None
        return _text_failure(
            StopReason.MAXIMUM_STEPS,
            FailureCode.BUDGET,
            f"text simulation reached its maximum of {self._maximum_steps} candidate turns",
            phase="candidate_turn_limit",
        )

    @property
    def recorded(self) -> RecordedTextCalls:
        """Return immutable call, retrieval, transition, and economics evidence.

        Returns:
            Snapshot of every completed candidate, retrieval, and world-model operation.
        """
        return RecordedTextCalls(
            candidate_spans=tuple(self._candidate_spans),
            world_model_spans=tuple(self._world_model_spans),
            candidate_economics=combine_economics(
                tuple(response.economics for response in self._candidate_responses)
            ),
            world_model_economics=combine_economics(
                tuple(response.economics for response in self._world_model_responses)
            ),
            retrieval_economics=combine_economics(tuple(self._retrieval_economics)),
            transitions=tuple(self._transitions),
            retrieved_transition_ids=tuple(self._retrieved_transition_ids),
        )

    def complete(self, request: ModelRequest) -> ModelResponse:
        """Complete one candidate turn, then model one visible text transition.

        Args:
            request: Candidate-visible request emitted by the customer agent adapter.

        Returns:
            Candidate response after the corresponding world-model turn has been recorded.

        Raises:
            TextSimulationError: The request uses tools, overflows context, reaches a terminal
                limit, or either model returns an unsupported or truncated response.
        """
        try:
            return self._complete(request)
        except TextSimulationError as exc:
            self._failure = self._failure or exc
            raise
        except Exception as exc:  # provider exceptions become durable episode evidence
            details: dict[str, JsonValue] = {"phase": "candidate_or_world_model"}
            if self._provider_dispatch_unknown_spend:
                details["provider_dispatch_unknown_spend"] = True
            failure = StructuredFailure(
                code=FailureCode.PROVIDER,
                message=f"text simulation provider call failed with {type(exc).__name__}",
                exception_type=type(exc).__name__,
                attribution=FailureAttribution.MODEL,
                details=details,
            )
            text_error = TextSimulationError(StopReason.FAILURE, failure)
            self._failure = self._failure or text_error
            raise text_error from exc

    def _complete(self, request: ModelRequest) -> ModelResponse:
        """Run one candidate, retrieval, and grounded world-model sequence.

        Args:
            request: Candidate-visible request emitted by the agent runtime.

        Returns:
            Validated candidate response after its grounded world transition is recorded.

        Raises:
            TextSimulationError: A boundary, budget, context, identity, or response check fails.
            Exception: The active candidate, embedder, or world-model provider fails.
        """
        if self._terminal:
            raise _text_failure(
                StopReason.COMPLETED,
                FailureCode.UNSUPPORTED,
                "candidate requested another turn after the text world model reached a terminal "
                "state",
                phase="candidate_after_terminal",
            )
        if len(self._candidate_responses) >= self._maximum_steps:
            raise _text_failure(
                StopReason.MAXIMUM_STEPS,
                FailureCode.BUDGET,
                f"text simulation reached its maximum of {self._maximum_steps} candidate turns",
                phase="candidate_turn_limit",
            )
        _require_text_only_candidate_request(request)
        candidate_request = _bounded_candidate_request(
            request,
            visible_transcript=self._visible_transcript,
            maximum_output_tokens=self._maximum_output_tokens,
        )
        _preflight_context(
            self._candidate.alias,
            self._candidate.capabilities,
            candidate_request,
            self._token_counter,
        )
        candidate_started_at = _timestamp(self._clock)
        self._provider_dispatch_unknown_spend = True
        candidate_response = self._candidate.client.complete(candidate_request)
        self._provider_dispatch_unknown_spend = False
        candidate_ended_at = _timestamp(self._clock, not_before=candidate_started_at)
        self._candidate_responses.append(candidate_response)
        self._candidate_spans.append(
            _model_span(
                span_id=f"candidate-{len(self._candidate_responses)}",
                kind=RolloutEventKind.AGENT_MODEL_CALL,
                started_at=candidate_started_at,
                ended_at=candidate_ended_at,
                request=candidate_request,
                response=candidate_response,
                redacted_field_names=self._redacted_field_names,
            )
        )
        _require_response_identity(candidate_response, self._candidate, role="candidate")
        _require_complete_response(candidate_response, role="candidate")
        _require_text_only_action(candidate_response.output, role="candidate")
        candidate_content = candidate_response.output.content
        if candidate_content is None:  # pragma: no cover - text-only validation guarantees text
            raise TypeError("text-only candidate response omitted visible content")
        rag_query = RAGQuery(
            task=self._task.instruction,
            initial_context=self._task.initial_context,
            action=RAGAction(kind="message", content=candidate_content),
            excluded_lineage_ids=(self._task.lineage_group_id,),
            top_k=self._grounded_world_model.artifact.top_k,
        )
        query_economics = self._grounded_world_model.retriever.estimate_query_economics(
            rag_query,
            self._query_embedding,
        )
        _require_query_budget(
            candidate_responses=tuple(self._candidate_responses),
            world_model_responses=tuple(self._world_model_responses),
            prior_retrieval=tuple(self._retrieval_economics),
            query=query_economics,
            maximum_cost_usd=self._maximum_cost_usd,
        )
        self._retrieval_economics.append(query_economics)
        self._provider_dispatch_unknown_spend = True
        prepared = self._grounded_world_model.prepare_turn(
            task=self._task,
            visible_messages=candidate_request.messages,
            candidate_response=candidate_response.output,
            excluded_lineage_ids=(self._task.lineage_group_id,),
            maximum_output_tokens=self._maximum_output_tokens,
        )
        self._provider_dispatch_unknown_spend = False
        _preflight_context(
            self._world_model.alias,
            self._world_model.capabilities,
            prepared.request,
            self._token_counter,
        )
        world_started_at = _timestamp(self._clock, not_before=candidate_ended_at)
        self._provider_dispatch_unknown_spend = True
        dispatched = self._grounded_world_model.complete_turn(prepared)
        self._provider_dispatch_unknown_spend = False
        world_request = dispatched.request
        world_response = dispatched.response
        world_ended_at = _timestamp(self._clock, not_before=world_started_at)
        self._retrieved_transition_ids.append(
            tuple(match.transition.transition_id for match in dispatched.matches)
        )
        self._world_model_responses.append(world_response)
        self._world_model_spans.append(
            _model_span(
                span_id=f"world-model-{len(self._world_model_responses)}",
                kind=RolloutEventKind.SIMULATOR_WORLD_MODEL_CALL,
                started_at=world_started_at,
                ended_at=world_ended_at,
                request=world_request,
                response=world_response,
                redacted_field_names=self._redacted_field_names,
            )
        )
        _require_response_identity(world_response, self._world_model, role="world model")
        _require_complete_response(world_response, role="world model")
        try:
            transition = self._grounded_world_model.parse_turn(dispatched).transition
        except TextWorldModelProtocolError as exc:
            raise _text_failure(
                StopReason.FAILURE,
                FailureCode.PROVIDER,
                str(exc),
                phase="world_model_protocol",
                exception_type=type(exc).__name__,
            ) from exc
        self._transitions.append(transition)
        self._visible_transcript = (
            *self._visible_transcript,
            ModelMessage(role="assistant", assistant_action=candidate_response.output),
            transition.visible_message,
        )
        self._terminal = transition.terminal
        return candidate_response


def combine_economics(records: Sequence[OperationEconomics]) -> OperationEconomics:
    """Aggregate same-role calls without representing a partial total as complete.

    Args:
        records: Economics for all calls made by one named role in one episode.

    Returns:
        One economics value. Usage is summed when present. Cost and latency are summed only when
        every call exposed that measurement, preserving a clear unknown rather than a partial sum.
    """
    if not records:
        return OperationEconomics()
    usages = [record.usage for record in records]
    usage = None
    if any(item is not None for item in usages):
        usage = _combine_usage(tuple(item for item in usages if item is not None))
    return OperationEconomics(
        usage=usage,
        cost_usd=_combine_measurement(tuple(record.cost_usd for record in records)),
        latency_seconds=_combine_measurement(tuple(record.latency_seconds for record in records)),
    )


def _require_query_budget(
    *,
    candidate_responses: Sequence[ModelResponse],
    world_model_responses: Sequence[ModelResponse],
    prior_retrieval: Sequence[OperationEconomics],
    query: OperationEconomics,
    maximum_cost_usd: float,
) -> None:
    """Refuse an embedding call unless known spend and its reservation fit.

    Args:
        candidate_responses: Completed candidate calls in this cell.
        world_model_responses: Completed world-model calls in this cell.
        prior_retrieval: Estimated query-embedding costs already dispatched in this cell.
        query: Conservative economics for the pending exact query.
        maximum_cost_usd: Remaining provider-spend ceiling assigned to this cell.

    Raises:
        TextSimulationError: Prior spend is unknown or the pending query would exceed the ceiling.
    """
    costs = [
        *(response.economics.cost_usd for response in candidate_responses),
        *(response.economics.cost_usd for response in world_model_responses),
        *(economics.cost_usd for economics in prior_retrieval),
        query.cost_usd,
    ]
    if any(cost is None for cost in costs):
        raise _text_failure(
            StopReason.MAXIMUM_COST,
            FailureCode.BUDGET,
            "query embedding is blocked because prior provider spend is unknown",
            phase="query_embedding_budget",
        )
    total = math.fsum(cast(NumericMeasurement, cost).value for cost in costs)
    if total > maximum_cost_usd:
        raise _text_failure(
            StopReason.MAXIMUM_COST,
            FailureCode.BUDGET,
            "query embedding reservation exceeds the remaining simulation spend ceiling",
            phase="query_embedding_budget",
        )


def text_prompt_digest() -> str:
    """Return the immutable prompt digest retained alongside a world-model simulator snapshot."""
    return text_prompt_sha256()


def _bounded_candidate_request(
    request: ModelRequest,
    *,
    visible_transcript: tuple[ModelMessage, ...],
    maximum_output_tokens: int,
) -> ModelRequest:
    """Inject the visible transcript and enforce a caller-visible output budget."""
    requested_budget = request.maximum_output_tokens
    if requested_budget is not None and requested_budget > maximum_output_tokens:
        raise _text_failure(
            StopReason.FAILURE,
            FailureCode.VALIDATION,
            "candidate requested more output tokens than the frozen text simulation budget",
            phase="candidate_output_budget",
        )
    return request.model_copy(
        update={
            "messages": _messages_with_visible_transcript(request.messages, visible_transcript),
            "maximum_output_tokens": requested_budget or maximum_output_tokens,
            "tool_choice": "none",
        }
    )


def _messages_with_visible_transcript(
    messages: tuple[ModelMessage, ...],
    visible_transcript: tuple[ModelMessage, ...],
) -> tuple[ModelMessage, ...]:
    """Append retained visible turns unless a stateful agent already supplied that suffix."""
    if not visible_transcript:
        return messages
    if len(messages) >= len(visible_transcript) and messages[-len(visible_transcript) :] == (
        visible_transcript
    ):
        return messages
    return (*messages, *visible_transcript)


def _require_text_only_candidate_request(request: ModelRequest) -> None:
    """Reject candidate tool configuration before it can reach a provider."""
    if request.tools or (request.tool_choice is not None and request.tool_choice != "none"):
        raise _text_failure(
            StopReason.FAILURE,
            FailureCode.UNSUPPORTED,
            "text simulation accepts only tool-free candidate requests; use sandbox mode for tools",
            phase="candidate_tools",
        )


def _require_text_only_action(action: AssistantAction, *, role: str) -> None:
    """Reject tool-call outputs that cannot be simulated in the v1 text engine."""
    if action.tool_calls or action.content is None:
        raise _text_failure(
            StopReason.FAILURE,
            FailureCode.UNSUPPORTED,
            f"{role} emitted tool calls or no visible text; use sandbox mode for tools",
            phase=f"{role.replace(' ', '_')}_tools",
        )


def _require_response_identity(
    response: ModelResponse,
    resolved: ResolvedModel,
    *,
    role: str,
) -> None:
    """Reject a provider response that was not served by the pinned resolved identity.

    A provider may expose a separately configured served model ID, for example when a cataloged
    routing alias resolves to one pinned dated model. That exception remains explicit and retains
    the configured provider, revision, capability, and connection digests.
    """
    expected = resolved.snapshot
    if response.model == expected:
        return
    served_model_id = resolved.served_model_id
    if (
        served_model_id is not None
        and response.model.model_id == served_model_id
        and response.model.provider == expected.provider
        and response.model.revision == expected.revision
        and response.model.capabilities_sha256 == expected.capabilities_sha256
        and response.model.connection_sha256 == expected.connection_sha256
    ):
        return
    raise _text_failure(
        StopReason.FAILURE,
        FailureCode.VALIDATION,
        f"{role} response identity does not match its pinned resolved model",
        phase=f"{role.replace(' ', '_')}_identity",
    )


def _require_complete_response(response: ModelResponse, *, role: str) -> None:
    """Turn an explicit provider length stop into a durable failed simulation cell."""
    if response.finish_reason == ModelFinishReason.LENGTH:
        raise _text_failure(
            StopReason.LENGTH,
            FailureCode.PROVIDER,
            f"{role} response reached its maximum output budget; WMO did not truncate it",
            phase=f"{role.replace(' ', '_')}_length",
        )


def _preflight_context(
    alias: str,
    capabilities: ModelCapabilities,
    request: ModelRequest,
    token_counter: TokenCounter,
) -> None:
    """Fail closed when full request plus output budget cannot fit a known context window."""
    budget = request.maximum_output_tokens
    if budget is None:  # pragma: no cover - bounded callers always provide an explicit budget
        raise _text_failure(
            StopReason.FAILURE,
            FailureCode.VALIDATION,
            f"model alias {alias!r} has no explicit output budget",
            phase="output_budget",
        )
    if capabilities.maximum_output_tokens is None:
        raise _text_failure(
            StopReason.FAILURE,
            FailureCode.UNSUPPORTED,
            f"model alias {alias!r} does not report an output limit for safe text simulation",
            phase="model_capabilities",
        )
    if capabilities.maximum_output_tokens < budget:
        raise _text_failure(
            StopReason.FAILURE,
            FailureCode.UNSUPPORTED,
            f"model alias {alias!r} cannot supply the requested {budget} output tokens",
            phase="model_capabilities",
        )
    if capabilities.context_window_tokens is None:
        raise _text_failure(
            StopReason.CONTEXT_OVERFLOW,
            FailureCode.CONTEXT_OVERFLOW,
            f"model alias {alias!r} does not report a context window for safe text simulation",
            phase="context_preflight",
        )
    input_tokens = token_counter.count(request)
    if input_tokens < 0:
        raise _text_failure(
            StopReason.FAILURE,
            FailureCode.VALIDATION,
            "text simulation token counter returned a negative request size",
            phase="context_preflight",
        )
    required = input_tokens + budget
    if required > capabilities.context_window_tokens:
        raise _text_failure(
            StopReason.CONTEXT_OVERFLOW,
            FailureCode.CONTEXT_OVERFLOW,
            f"model alias {alias!r} context window {capabilities.context_window_tokens} cannot "
            f"fit {input_tokens} input plus {budget} output tokens",
            phase="context_preflight",
        )


def _model_span(
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


def _combine_usage(records: Sequence[Usage]) -> Usage:
    """Return a typed usage sum after callers excluded absent usage values."""
    if not records:  # pragma: no cover - callers guard this before reaching the helper
        raise ValueError("at least one usage record is required")
    cached = tuple(item.cached_input_tokens for item in records)
    cached_input_tokens: int | None
    if all(item is not None for item in cached):
        cached_input_tokens = sum(cast(int, item) for item in cached)
    else:
        cached_input_tokens = None
    return Usage(
        input_tokens=sum(item.input_tokens for item in records),
        output_tokens=sum(item.output_tokens for item in records),
        cached_input_tokens=cached_input_tokens,
    )


def _combine_measurement(
    records: Sequence[NumericMeasurement | None],
) -> NumericMeasurement | None:
    """Sum a measurement only when every call provided a compatible finite value."""
    if any(record is None for record in records):
        return None
    measurements = tuple(cast(NumericMeasurement, record) for record in records)
    values = tuple(item.value for item in measurements)
    if not all(math.isfinite(value) for value in values):  # pragma: no cover - contract rejects it
        return None
    all_observed = all(item.provenance == "observed" for item in measurements)
    provenance = "observed" if all_observed else "estimated"
    return NumericMeasurement(value=sum(values), provenance=provenance)


def _text_failure(
    stop_reason: StopReason,
    code: FailureCode,
    message: str,
    *,
    phase: str,
    exception_type: str | None = None,
) -> TextSimulationError:
    """Build one non-secret structured simulator failure with a stable phase label."""
    return TextSimulationError(
        stop_reason,
        StructuredFailure(
            code=code,
            message=message,
            exception_type=exception_type,
            attribution=FailureAttribution.MODEL,
            details={"phase": phase},
        ),
    )


def _timestamp(clock: Callable[[], datetime], *, not_before: datetime | None = None) -> datetime:
    """Read a timezone-aware timestamp, never ordering a span backwards in a deterministic fake."""
    timestamp = clock()
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("text simulation clock must return timezone-aware datetimes")
    if not_before is not None and timestamp < not_before:
        return not_before
    return timestamp
