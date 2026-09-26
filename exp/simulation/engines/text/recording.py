"""Visible-turn recording, context preflight, and simulated model-call boundaries."""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime
from threading import Lock
from typing import TYPE_CHECKING, cast

from pydantic import JsonValue

from exp.common.core.artifacts import (
    FailureAttribution,
    FailureCode,
    JsonObject,
    StructuredFailure,
    redact_secret_json,
)
from exp.common.models import (
    AssistantAction,
    CompletionCostReservation,
    EmbeddingCostReservation,
    ModelCapabilities,
    ModelFinishReason,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelSnapshot,
    NumericMeasurement,
    OperationEconomics,
    ToolCall,
    combine_economics,
    completion_request_cost_usd,
    reconcile_completion_economics,
    verify_completion_reservation,
)
from exp.common.rollouts import (
    UNKNOWN_DISPATCH_RESERVED_COST_KEY,
    RolloutEventKind,
    RolloutSpan,
    StopReason,
)
from exp.common.rollouts.checkpoint import TextRolloutCheckpoint
from exp.common.tasks import TaskCase
from exp.runtime.environments import Observation
from exp.runtime.models import ResolvedModel
from exp.runtime.models.providers.errors import ProviderRefusalError, ProviderRetryableResponseError
from exp.runtime.models.providers.transport import classify_retry
from exp.simulation.engines.clock import timestamp
from exp.simulation.engines.text.environment import SimulatedToolUseError
from exp.simulation.engines.text.grounding import estimate_retrieval_economics
from exp.simulation.engines.text.prompt import (
    SimulatedToolResult,
    TextWorldModelProtocolError,
    TextWorldModelTransition,
    candidate_rag_actions,
    parse_world_model_transition,
    text_prompt_sha256,
)
from exp.simulation.engines.text.redaction import redact_json
from exp.simulation.engines.text.tokens import TokenCounter, bound_unpublished_output
from exp.simulation.retrieval import RAGQuery
from exp.simulation.retrieval.retriever import RAGQueryInputLimitError

if TYPE_CHECKING:
    from exp.simulation.world_model import GroundedWorldModel

logger = logging.getLogger(__name__)


class TextSimulationError(RuntimeError):
    """A text simulator boundary failed with an artifact-safe terminal classification."""

    def __init__(self, stop_reason: StopReason, failure: StructuredFailure) -> None:
        super().__init__(failure.message)
        self.stop_reason = stop_reason
        self.failure = failure


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
        candidate_request: CompletionCostReservation | None,
        world_model_request: CompletionCostReservation | None,
        completion_maximum_attempts: int,
        maximum_cost_usd: float,
        stop_on_overspend: bool,
        maximum_steps: int,
        maximum_rollout_output_tokens: int = 1_000_000,
        maximum_output_tokens: int,
        world_model_json_object_output: bool = False,
        redacted_field_names: frozenset[str],
        clock: Callable[[], datetime],
        token_counter: TokenCounter,
    ) -> None:
        """Bind one task, two independent model clients, and strict simulation boundaries.

        Args:
            task: Canonical task currently being simulated.
            candidate: Candidate model injected into the customer agent.
            world_model: Model that simulates the next visible text turn.
            grounded_world_model: Artifact-bound executor over the exact fit-only index.
            query_embedding: Exact query-embedding identity, price, and retry reservation.
            candidate_request: Frozen candidate request ceiling for finite-cost execution.
            world_model_request: Frozen world-model request ceiling for finite-cost execution.
            completion_maximum_attempts: Active provider request-attempt ceiling.
            maximum_cost_usd: Spend remaining for candidate, retrieval, and world-model calls.
            stop_on_overspend: When true, reconciled spend reaching the ceiling blocks the
                next dispatch; by default the authorized episode warns once and continues.
            maximum_steps: Maximum candidate model turns allowed in this episode.
            maximum_output_tokens: Per-call output budget used without silent truncation.
            world_model_json_object_output: Frozen provider JSON mode for simulation responses.
            redacted_field_names: Project fields redacted before events persist.
            clock: Time source used to order emitted spans deterministically in tests.
            token_counter: Full-request counter used before every provider call.
        """
        self._task = task
        self._candidate = candidate
        self._world_model = world_model
        self._grounded_world_model = grounded_world_model
        self._query_embedding = query_embedding
        self._candidate_request = candidate_request
        self._world_model_request = world_model_request
        self._completion_maximum_attempts = completion_maximum_attempts
        self._maximum_cost_usd = maximum_cost_usd
        self._stop_on_overspend = stop_on_overspend
        self._maximum_steps = maximum_steps
        self._maximum_rollout_output_tokens = maximum_rollout_output_tokens
        self._maximum_output_tokens = maximum_output_tokens
        self._world_model_json_object_output = world_model_json_object_output
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
        self._environment_state: JsonObject = {}
        self._pending_tools: dict[str, tuple[ToolCall, SimulatedToolResult]] = {}
        self._tool_lock = Lock()
        self._failure: TextSimulationError | None = None
        self._provider_dispatch_unknown_spend = False
        self._unknown_dispatch_reserved_cost_usd: float | None = None
        self._overspend_warned = False

    def observe_tool(self, action: ToolCall) -> Observation:
        """Consume exactly one generated result for the pending candidate invocation.

        Args:
            action: Exact call emitted by the candidate, including its arguments and ID.

        Returns:
            Simulated content and error status, without external tool execution.

        Raises:
            SimulatedToolUseError: The call is unknown, modified, or already consumed.
        """
        with self._tool_lock:
            pending = self._pending_tools.get(action.call_id)
            if pending is None or pending[0] != action:
                raise SimulatedToolUseError(
                    "tool call does not match an unconsumed simulated result"
                )
            del self._pending_tools[action.call_id]
        result = pending[1]
        return Observation(content=result.content, is_error=result.is_error)

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

    @property
    def turn_limit_reached(self) -> bool:
        """Return whether a nonterminal scenario exhausted the pinned candidate turn ceiling.

        Returns:
            True after the final permitted nonterminal world transition, False when another
            candidate turn remains or the world model ended the scenario.
        """
        return not self._terminal and len(self._candidate_responses) >= self._maximum_steps

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
                tuple(response.economics for response in self._candidate_responses),
                require_complete_usage=False,
            ),
            world_model_economics=combine_economics(
                tuple(response.economics for response in self._world_model_responses),
                require_complete_usage=False,
            ),
            retrieval_economics=combine_economics(
                tuple(self._retrieval_economics),
                require_complete_usage=False,
            ),
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
            TextSimulationError: The request overflows context, reaches a terminal
                limit, or either model returns an unsupported or truncated response.
        """
        try:
            return self._complete(request)
        except TextSimulationError as exc:
            self._failure = self._failure or exc
            raise
        except Exception as exc:  # noqa: BLE001 - provider exceptions become durable episode evidence
            classification = classify_retry(exc)
            details: dict[str, JsonValue] = {
                "phase": "candidate_or_world_model",
                "retry_classification": classification.reason,
            }
            if self._provider_dispatch_unknown_spend:
                details["provider_dispatch_unknown_spend"] = True
                reserved = self._unknown_dispatch_reserved_cost_usd
                if reserved is not None:
                    details[UNKNOWN_DISPATCH_RESERVED_COST_KEY] = reserved
            failure = StructuredFailure(
                code=FailureCode.PROVIDER,
                message=f"text simulation provider call failed with {type(exc).__name__}",
                retryable=classification.retryable
                or isinstance(exc, (ProviderRefusalError, ProviderRetryableResponseError)),
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
        if self._pending_tools:
            raise _text_failure(
                StopReason.FAILURE,
                FailureCode.VALIDATION,
                "consume every simulated tool result before requesting another candidate turn",
                phase="pending_tool_results",
            )
        if request.tools != self._task.tools:
            raise _text_failure(
                StopReason.FAILURE,
                FailureCode.VALIDATION,
                "candidate request must preserve the task's declared tool schemas",
                phase="candidate_tools",
            )
        candidate_request = _bounded_candidate_request(
            request,
            visible_transcript=self._visible_transcript,
            maximum_output_tokens=min(
                self._maximum_output_tokens,
                self._remaining_output_tokens(),
                self._candidate.capabilities.maximum_output_tokens or self._maximum_output_tokens,
            ),
        )
        candidate_request = bound_unpublished_output(
            candidate_request, self._candidate.capabilities, self._token_counter
        )
        _preflight_context(
            self._candidate.alias,
            self._candidate.capabilities,
            candidate_request,
            self._token_counter,
        )
        if self._candidate_request is not None:
            _verify_completion_budget_binding(
                self._candidate_request,
                model=self._candidate.snapshot,
                capabilities=self._candidate.capabilities,
                maximum_attempts=self._completion_maximum_attempts,
                role="candidate",
            )
            _require_completion_request_bounds(
                role="candidate",
                reservation=self._candidate_request,
                request=candidate_request,
                token_counter=self._token_counter,
            )
        self._check_spend_ceiling(role="candidate")
        candidate_started_at = timestamp(self._clock)
        candidate_response = self._dispatch_provider(
            lambda: self._candidate.client.complete(candidate_request),
            reserved_cost_usd=(
                self._candidate_request.estimated_maximum_call_cost_usd
                if self._candidate_request is not None
                else None
            ),
        )
        if self._candidate_request is not None:
            candidate_response = candidate_response.model_copy(
                update={
                    "economics": reconcile_completion_economics(
                        self._candidate_request,
                        candidate_response.economics,
                    )
                }
            )
        candidate_ended_at = timestamp(self._clock, not_before=candidate_started_at)
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
        self._clear_unknown_dispatch()
        _require_complete_response(candidate_response, role="candidate")
        calls = candidate_response.output.tool_calls
        names = {tool.name for tool in self._task.tools}
        if any(call.name not in names for call in calls) or len({c.call_id for c in calls}) != len(
            calls
        ):
            raise _text_failure(
                StopReason.FAILURE,
                FailureCode.VALIDATION,
                "candidate tool calls require declared tool names and unique call IDs",
                phase="candidate_tools",
            )
        queries = tuple(
            RAGQuery(
                task=self._task.instruction,
                initial_context=self._task.initial_context,
                action=action,
                excluded_lineage_ids=(self._task.lineage_group_id,),
                top_k=self._grounded_world_model.artifact.top_k,
            )
            for action in candidate_rag_actions(candidate_response.output)
        )
        try:
            query_economics = estimate_retrieval_economics(
                queries, self._grounded_world_model.retriever, self._query_embedding
            )
        except RAGQueryInputLimitError as exc:
            raise _text_failure(
                StopReason.MAXIMUM_COST,
                FailureCode.BUDGET,
                "grounding query exceeds its reserved input-token ceiling",
                phase="query_embedding_budget",
            ) from exc
        self._check_spend_ceiling(role="query embedding")
        self._retrieval_economics.append(query_economics)
        prepared = self._dispatch_provider(
            lambda: self._grounded_world_model.prepare_turn(
                task=self._task,
                visible_messages=candidate_request.messages,
                candidate_response=candidate_response.output,
                excluded_lineage_ids=(self._task.lineage_group_id,),
                state=self._environment_state,
                maximum_output_tokens=min(
                    self._maximum_output_tokens,
                    self._world_model.capabilities.maximum_output_tokens
                    or self._maximum_output_tokens,
                ),
            ),
            # The retained retrieval estimate above already covers this dispatch's worst case
            # in every reconciliation path, so the window's incremental reservation is zero.
            reserved_cost_usd=0.0,
        )
        self._clear_unknown_dispatch()
        prepared = replace(
            prepared,
            request=bound_unpublished_output(
                prepared.request.model_copy(
                    update={"json_object_output": self._world_model_json_object_output}
                ),
                self._world_model.capabilities,
                self._token_counter,
            ),
        )
        _preflight_context(
            self._world_model.alias,
            self._world_model.capabilities,
            prepared.request,
            self._token_counter,
        )
        if self._world_model_request is not None:
            _verify_completion_budget_binding(
                self._world_model_request,
                model=self._world_model.snapshot,
                capabilities=self._world_model.capabilities,
                maximum_attempts=self._completion_maximum_attempts,
                role="world model",
            )
            _require_completion_request_bounds(
                role="world model",
                reservation=self._world_model_request,
                request=prepared.request,
                token_counter=self._token_counter,
            )
        self._check_spend_ceiling(role="world model")
        world_started_at = timestamp(self._clock, not_before=candidate_ended_at)
        dispatched = self._dispatch_provider(
            lambda: self._grounded_world_model.complete_turn(prepared),
            reserved_cost_usd=(
                self._world_model_request.estimated_maximum_call_cost_usd
                if self._world_model_request is not None
                else None
            ),
        )
        world_request = dispatched.request
        world_response = dispatched.response
        if self._world_model_request is not None:
            world_response = world_response.model_copy(
                update={
                    "economics": reconcile_completion_economics(
                        self._world_model_request,
                        world_response.economics,
                    )
                }
            )
            dispatched = replace(dispatched, response=world_response)
        world_ended_at = timestamp(self._clock, not_before=world_started_at)
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
        self._clear_unknown_dispatch()
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
                retryable=True,
            ) from exc
        self._transitions.append(transition)
        self._visible_transcript = (
            *self._visible_transcript,
            ModelMessage(role="assistant", assistant_action=candidate_response.output),
            *transition.visible_messages,
        )
        self._environment_state = transition.state
        self._pending_tools = {
            call.call_id: (call, result)
            for call, result in zip(calls, transition.tool_results, strict=True)
        }
        self._terminal = transition.terminal
        return candidate_response

    def _remaining_output_tokens(self) -> int:
        """Admit generated tokens, including reasoning, without treating missing usage as zero."""
        usages = [response.economics.usage for response in self._candidate_responses]
        if any(usage is None for usage in usages):
            raise _text_failure(
                StopReason.MAXIMUM_OUTPUT_TOKENS,
                FailureCode.BUDGET,
                "candidate usage is missing; cannot safely admit more output tokens",
                phase="candidate_token_budget",
            )
        used = sum(usage.output_tokens for usage in usages if usage is not None)
        remaining = self._maximum_rollout_output_tokens - used
        if remaining <= 0:
            raise _text_failure(
                StopReason.MAXIMUM_OUTPUT_TOKENS,
                FailureCode.BUDGET,
                "rollout output-token budget exhausted; increase it to continue",
                phase="candidate_token_budget",
            )
        return remaining

    def checkpoint(self) -> TextRolloutCheckpoint | None:
        """Return safe resumable state only at a complete, unredacted world-turn boundary."""
        if self._pending_tools or not (
            len(self._candidate_responses)
            == len(self._world_model_responses)
            == len(self._transitions)
        ):
            return None
        checkpoint = TextRolloutCheckpoint(
            visible_transcript=self._visible_transcript,
            candidate_responses=tuple(self._candidate_responses),
            world_model_responses=tuple(self._world_model_responses),
            retrieval_economics=tuple(self._retrieval_economics),
        )
        raw = checkpoint.model_dump(mode="json")
        safe, _ = redact_secret_json(redact_json(raw, self._redacted_field_names))
        return checkpoint if safe == raw else None

    def restore(self, checkpoint: TextRolloutCheckpoint, spans: tuple[RolloutSpan, ...]) -> None:
        """Restore a validated built-in chat prefix without dispatching any prior call."""
        self._visible_transcript = checkpoint.visible_transcript
        self._candidate_responses = list(checkpoint.candidate_responses)
        self._world_model_responses = list(checkpoint.world_model_responses)
        self._transitions = [
            parse_world_model_transition(item.output) for item in checkpoint.world_model_responses
        ]
        self._environment_state = self._transitions[-1].state if self._transitions else {}
        self._retrieval_economics = list(checkpoint.retrieval_economics)
        self._candidate_spans = [s for s in spans if s.kind == RolloutEventKind.AGENT_MODEL_CALL]
        self._world_model_spans = [
            s for s in spans if s.kind == RolloutEventKind.SIMULATOR_WORLD_MODEL_CALL
        ]

    def _check_spend_ceiling(self, *, role: str) -> None:
        """Apply the episode's overspend policy before one paid dispatch.

        Reconciled actual spend is compared against the cell ceiling: in stop mode unknown
        prior spend or a reached ceiling fails the episode closed before dispatch, and by
        default the authorized episode logs one warning and continues.

        Args:
            role: Candidate, query embedding, or world-model label for safe diagnostics.

        Raises:
            TextSimulationError: Stop mode found unknown prior spend or a reached ceiling.
        """
        phase = f"{role.replace(' ', '_')}_budget"
        costs = [
            *(response.economics.cost_usd for response in self._candidate_responses),
            *(response.economics.cost_usd for response in self._world_model_responses),
            *(economics.cost_usd for economics in self._retrieval_economics),
        ]
        if any(cost is None for cost in costs):
            if self._stop_on_overspend:
                raise _text_failure(
                    StopReason.MAXIMUM_COST,
                    FailureCode.BUDGET,
                    f"{role} call is blocked because prior provider spend is unknown",
                    phase=phase,
                )
            if not self._overspend_warned:
                logger.warning(
                    "prior provider spend is unknown before the %s call; continuing because "
                    "the run is already authorized",
                    role,
                )
                self._overspend_warned = True
            return
        total = math.fsum(cast(NumericMeasurement, cost).value for cost in costs)
        if total < self._maximum_cost_usd:
            return
        if self._stop_on_overspend:
            raise _text_failure(
                StopReason.MAXIMUM_COST,
                FailureCode.BUDGET,
                f"reconciled provider spend reached the simulation ceiling before the {role} call",
                phase=phase,
            )
        if not self._overspend_warned:
            logger.warning(
                "reconciled provider spend $%.4f reached the simulation ceiling $%.4f before "
                "the %s call; continuing because the run is already authorized",
                total,
                self._maximum_cost_usd,
                role,
            )
            self._overspend_warned = True

    def _dispatch_provider[ResultT](
        self,
        operation: Callable[[], ResultT],
        *,
        reserved_cost_usd: float | None,
    ) -> ResultT:
        """Run one provider dispatch inside an explicit unknown-spend accounting window.

        The executing client owns bounded transport retries inside the same retry-inclusive
        reservation that admitted this dispatch, so this boundary never multiplies attempts.
        While the dispatch is in flight its worst-case reservation is retained so a failure
        that leaves spend unknown persists an exact conservative charge with its evidence
        instead of an unpriceable hole.

        Args:
            operation: One provider dispatch whose spend is ambiguous until it returns.
            reserved_cost_usd: Retry-inclusive worst-case charge admitted for this dispatch.

        Returns:
            The successful dispatch result.

        Raises:
            Exception: The active provider client failed after its own bounded retries.
        """
        self._provider_dispatch_unknown_spend = True
        self._unknown_dispatch_reserved_cost_usd = reserved_cost_usd
        return operation()

    def _clear_unknown_dispatch(self) -> None:
        """Mark the most recent provider dispatch as fully priced and recorded."""
        self._provider_dispatch_unknown_spend = False
        self._unknown_dispatch_reserved_cost_usd = None


def _require_completion_request_bounds(
    *,
    role: str,
    reservation: CompletionCostReservation,
    request: ModelRequest,
    token_counter: TokenCounter,
) -> None:
    """Refuse one completion whose pending request breaks a hard reservation bound.

    The pending request must fit the model's real context-derived ceilings. The priced
    pending cost is a planning value only and never gates dispatch on its own.

    Args:
        role: Candidate or world-model label for safe diagnostics.
        reservation: Exact active retry-bound request reservation.
        request: Complete provider-neutral pending request.
        token_counter: Conservative full serialized request counter.

    Raises:
        TextSimulationError: The request lacks an output ceiling or exceeds a hard bound.
    """
    output_tokens = request.maximum_output_tokens
    if output_tokens is None:
        raise _text_failure(
            StopReason.MAXIMUM_COST,
            FailureCode.BUDGET,
            f"{role} request lacks a reserved output ceiling",
            phase=f"{role.replace(' ', '_')}_budget",
        )
    try:
        completion_request_cost_usd(
            reservation,
            input_tokens=token_counter.count(request),
            output_tokens=output_tokens,
        )
    except ValueError as exc:
        raise _text_failure(
            StopReason.MAXIMUM_COST,
            FailureCode.BUDGET,
            str(exc),
            phase=f"{role.replace(' ', '_')}_budget",
        ) from exc


def _verify_completion_budget_binding(
    reservation: CompletionCostReservation,
    *,
    model: ModelSnapshot,
    capabilities: ModelCapabilities,
    maximum_attempts: int,
    role: str,
) -> None:
    """Translate active reservation drift into a structured zero-dispatch budget failure.

    Args:
        reservation: Frozen request ceiling.
        model: Active exact model snapshot.
        capabilities: Active explicit prices and capacities.
        maximum_attempts: Active provider retry ceiling.
        role: Candidate or world-model diagnostic label.

    Raises:
        TextSimulationError: Active model, economics, capacity, or retry metadata drifted.
    """
    try:
        verify_completion_reservation(
            reservation,
            model=model,
            capabilities=capabilities,
            maximum_attempts=maximum_attempts,
        )
    except ValueError as exc:
        raise _text_failure(
            StopReason.MAXIMUM_COST,
            FailureCode.BUDGET,
            str(exc),
            phase=f"{role.replace(' ', '_')}_budget",
        ) from exc


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
            f"{role} response reached its maximum output budget; EXP did not truncate it",
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
    if (
        capabilities.maximum_output_tokens is not None
        and capabilities.maximum_output_tokens < budget
    ):
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


def _text_failure(
    stop_reason: StopReason,
    code: FailureCode,
    message: str,
    *,
    phase: str,
    exception_type: str | None = None,
    retryable: bool = False,
) -> TextSimulationError:
    """Build one non-secret structured simulator failure with a stable phase label.

    Args:
        stop_reason: Terminal classification recorded with the failed episode.
        code: Structured failure code persisted with the evidence.
        message: Non-secret operator-facing failure description.
        phase: Stable simulator phase label persisted in the failure details.
        exception_type: Optional exception class name retained for diagnostics.
        retryable: Whether resume may supersede this failure with a fresh attempt.

    Returns:
        One artifact-safe terminal simulator error.
    """
    return TextSimulationError(
        stop_reason,
        StructuredFailure(
            code=code,
            message=message,
            retryable=retryable,
            exception_type=exception_type,
            attribution=FailureAttribution.MODEL,
            details={"phase": phase},
        ),
    )
