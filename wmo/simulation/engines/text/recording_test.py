"""Tests for text-only candidate recording and preflight boundaries."""

from dataclasses import replace
from datetime import UTC, datetime
from typing import cast

import pytest

from wmo.common.models import (
    AssistantAction,
    EmbeddingCostReservation,
    ModelCapabilities,
    ModelFinishReason,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelSnapshot,
    NumericMeasurement,
    OperationEconomics,
    Usage,
)
from wmo.common.rollouts import StopReason
from wmo.common.tasks import TaskCase
from wmo.runtime.models import ResolvedModel
from wmo.simulation.engines.text.recording import (
    RecordingCandidateClient,
    TextSimulationError,
)
from wmo.simulation.retrieval import RAGMatch, RAGQuery, TraceRAGRetriever

_TIME = datetime(2026, 8, 12, tzinfo=UTC)


class _ScriptedClient:
    def __init__(self, responses: list[ModelResponse]) -> None:
        self.responses = list(responses)
        self.requests: list[ModelRequest] = []

    def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return self.responses.pop(0)


class _Retriever:
    def __init__(self) -> None:
        """Initialize an empty ordered query log."""
        self.queries: list[RAGQuery] = []

    def estimate_query_economics(
        self,
        query: RAGQuery,
        reservation: EmbeddingCostReservation,
    ) -> OperationEconomics:
        """Return deterministic query economics for recorder tests.

        Args:
            query: Canonical retrieval query being estimated.
            reservation: Immutable embedding price and retry reservation.

        Returns:
            Fixed conservative query cost.
        """
        del query
        del reservation
        return OperationEconomics(cost_usd=NumericMeasurement(value=0.01, provenance="estimated"))

    def retrieve(self, query: RAGQuery) -> tuple[RAGMatch, ...]:
        """Record a query and return no grounding examples.

        Args:
            query: Canonical retrieval query dispatched by the recorder.

        Returns:
            Empty deterministic result set.
        """
        self.queries.append(query)
        return ()


def _snapshot(name: str) -> ModelSnapshot:
    return ModelSnapshot(
        provider="test",
        model_id=name,
        capabilities_sha256="a" * 64,
        connection_sha256="b" * 64,
    )


def _response(
    content: str,
    *,
    model: ModelSnapshot,
    finish_reason: ModelFinishReason = ModelFinishReason.COMPLETED,
) -> ModelResponse:
    return ModelResponse(
        output=AssistantAction(content=content),
        model=model,
        economics=OperationEconomics(
            usage=Usage(input_tokens=4, output_tokens=3),
            cost_usd=NumericMeasurement(value=0.10, provenance="observed"),
        ),
        finish_reason=finish_reason,
    )


def _task() -> TaskCase:
    return TaskCase(
        task_id="task-a",
        lineage_group_id="lineage-a",
        partition="fit",
        instruction="Answer the customer question.",
        initial_context={"account": "safe"},
        workload_weight=1.0,
        source_trace_ids=("trace-a",),
    )


def _resolved(
    alias: str,
    client: _ScriptedClient,
    *,
    context_window_tokens: int = 100_000,
) -> ResolvedModel:
    return ResolvedModel(
        alias=alias,
        snapshot=_snapshot(alias),
        capabilities=ModelCapabilities(
            context_window_tokens=context_window_tokens,
            maximum_output_tokens=16_000,
        ),
        client=client,
        embedding_client=None,
    )


def _recorder(
    candidate_client: _ScriptedClient,
    world_client: _ScriptedClient,
    *,
    candidate_context_window: int = 100_000,
    candidate_served_model_id: str | None = None,
) -> RecordingCandidateClient:
    """Build a recorder with explicit fake candidate, world model, and retriever.

    Args:
        candidate_client: Scripted candidate provider client.
        world_client: Scripted world-model provider client.
        candidate_context_window: Candidate request context-window ceiling.
        candidate_served_model_id: Optional observed served-model identity.

    Returns:
        Recorder configured for one deterministic task.
    """
    candidate = _resolved(
        "candidate-a",
        candidate_client,
        context_window_tokens=candidate_context_window,
    )
    if candidate_served_model_id is not None:
        candidate = replace(candidate, served_model_id=candidate_served_model_id)
    return RecordingCandidateClient(
        task=_task(),
        candidate=candidate,
        world_model=_resolved("world-model-a", world_client),
        fit_retriever=cast(TraceRAGRetriever, _Retriever()),
        query_embedding=EmbeddingCostReservation(
            model=_snapshot("embedder-a"),
            input_usd_per_million_tokens=1.0,
            maximum_attempts=2,
            maximum_input_tokens=10_000,
        ),
        maximum_cost_usd=10.0,
        maximum_steps=2,
        maximum_output_tokens=16_000,
        redacted_field_names=frozenset(),
        clock=lambda: _TIME,
        token_counter=_Utf8Counter(),
    )


class _Utf8Counter:
    def count(self, request: ModelRequest) -> int:
        return len(request.model_dump_json().encode("utf-8"))


def test_recorder_keeps_candidate_and_world_calls_separate_and_tool_free() -> None:
    """A visible candidate turn becomes one strict JSON world transition without hidden transfer."""
    candidate_snapshot = _snapshot("candidate-a")
    world_snapshot = _snapshot("world-model-a")
    candidate_client = _ScriptedClient([_response("I can help.", model=candidate_snapshot)])
    world_client = _ScriptedClient(
        [
            _response(
                '{"message":"What is your order number?","terminal":false}',
                model=world_snapshot,
            )
        ]
    )
    recorder = _recorder(candidate_client, world_client)

    response = recorder.complete(
        ModelRequest(messages=(ModelMessage(role="user", content="My delivery is late."),))
    )

    assert response.output.content == "I can help."
    assert world_client.requests[0].tools == ()
    assert world_client.requests[0].tool_choice == "none"
    assert "candidate_hidden_reasoning" not in world_client.requests[0].model_dump_json()
    assert recorder.recorded.transitions[0].message == "What is your order number?"
    assert recorder.recorded.candidate_economics.cost_usd == NumericMeasurement(
        value=0.10,
        provenance="observed",
    )
    assert recorder.recorded.world_model_economics.cost_usd == NumericMeasurement(
        value=0.10,
        provenance="observed",
    )


def test_recorder_rejects_tool_requests_before_any_provider_call() -> None:
    """Text simulation does not quietly remove tool access from a candidate request."""
    candidate_client = _ScriptedClient([])
    world_client = _ScriptedClient([])
    recorder = _recorder(candidate_client, world_client)

    with pytest.raises(TextSimulationError, match="tool-free") as error:
        recorder.complete(
            ModelRequest(
                messages=(ModelMessage(role="user", content="Use the system."),),
                tool_choice="auto",
            )
        )

    assert error.value.stop_reason == StopReason.FAILURE
    assert candidate_client.requests == []
    assert world_client.requests == []


def test_recorder_fails_context_preflight_and_explicit_length_stops_without_truncation() -> None:
    """Provider calls are blocked before overflow, while explicit provider length stops persist."""
    candidate_client = _ScriptedClient([])
    world_client = _ScriptedClient([])
    overflow = _recorder(candidate_client, world_client, candidate_context_window=16_000)

    with pytest.raises(TextSimulationError) as overflow_error:
        overflow.complete(ModelRequest(messages=(ModelMessage(role="user", content="short"),)))

    assert overflow_error.value.stop_reason == StopReason.CONTEXT_OVERFLOW
    assert candidate_client.requests == []

    candidate_snapshot = _snapshot("candidate-a")
    length_client = _ScriptedClient(
        [
            _response(
                "unfinished response",
                model=candidate_snapshot,
                finish_reason=ModelFinishReason.LENGTH,
            )
        ]
    )
    length = _recorder(length_client, _ScriptedClient([]))

    with pytest.raises(TextSimulationError) as length_error:
        length.complete(ModelRequest(messages=(ModelMessage(role="user", content="short"),)))

    assert length_error.value.stop_reason == StopReason.LENGTH
    assert len(length_client.requests) == 1
    assert length.recorded.candidate_spans[0].payload["response"] == {
        "output": {"content": "unfinished response", "tool_calls": []},
        "finish_reason": "length",
    }


def test_recorder_fails_closed_on_rebound_response_identity_but_allows_explicit_served_id() -> None:
    """A provider alias cannot silently change identity after resolution."""
    wrong = _recorder(
        _ScriptedClient([_response("wrong", model=_snapshot("candidate-rebound"))]),
        _ScriptedClient([]),
    )

    with pytest.raises(TextSimulationError, match="identity") as error:
        wrong.complete(ModelRequest(messages=(ModelMessage(role="user", content="short"),)))

    assert error.value.failure.details["phase"] == "candidate_identity"
    accepted = _recorder(
        _ScriptedClient([_response("served", model=_snapshot("candidate-served"))]),
        _ScriptedClient(
            [
                _response(
                    '{"message":"done","terminal":true}',
                    model=_snapshot("world-model-a"),
                )
            ]
        ),
        candidate_served_model_id="candidate-served",
    )

    assert (
        accepted.complete(
            ModelRequest(messages=(ModelMessage(role="user", content="short"),))
        ).output.content
        == "served"
    )
