"""Tests for text-only candidate recording and preflight boundaries."""

import json
import logging
from dataclasses import replace
from datetime import UTC, datetime
from typing import cast

import pytest

from exp.common.core.artifacts import ArtifactInput, JsonObject
from exp.common.models import (
    AssistantAction,
    BillingSource,
    CompletionCostReservation,
    ConnectionConfig,
    EmbeddingCostReservation,
    ModelCapabilities,
    ModelCatalog,
    ModelFinishReason,
    ModelMessage,
    ModelRecord,
    ModelRequest,
    ModelResponse,
    ModelSnapshot,
    NumericMeasurement,
    OperationEconomics,
    ToolCall,
    Usage,
    completion_cost_reservation,
)
from exp.common.rollouts import StopReason
from exp.common.tasks import TaskCase, ToolSchema
from exp.runtime.models import ResolvedModel
from exp.runtime.models.providers.openai import openai_responses_response
from exp.runtime.models.providers.transport import ScriptedJsonTransport
from exp.runtime.models.registry import RuntimeModelCatalog
from exp.simulation.engines.text.recording import (
    RecordingCandidateClient,
    TextSimulationError,
    _require_response_identity,
)
from exp.simulation.retrieval import RAGMatch, RAGQuery, TraceRAGRetriever
from exp.simulation.world_model import GroundedWorldModel, GroundedWorldModelArtifact
from exp.simulation.world_model.artifact import (
    GROUNDED_WORLD_MODEL_PROMPT_VERSION,
    grounded_world_model_prompt_sha256,
)

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
        billing_source=BillingSource.CUSTOMER_MANAGED,
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


def _openai_payload(
    *,
    model: str,
    content: str,
    input_tokens: int,
    output_tokens: int,
    cached_tokens: int,
) -> JsonObject:
    """Return one native OpenAI Responses payload with usage but no dollar cost.

    Args:
        model: Served provider model identifier.
        content: Visible assistant output text.
        input_tokens: Provider-reported total input tokens.
        output_tokens: Provider-reported generated tokens.
        cached_tokens: Provider-reported cached-read input tokens.

    Returns:
        Schema-valid completed Responses payload.
    """
    return {
        "id": "resp_fixture",
        "object": "response",
        "created_at": 1.0,
        "status": "completed",
        "model": model,
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "output": [
            {
                "type": "message",
                "id": "msg_fixture",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": content, "annotations": []}],
            }
        ],
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "input_tokens_details": {"cached_tokens": cached_tokens, "cache_write_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
    }


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
    completion_pricing: bool = False,
    input_price: float | None = 1.0,
    output_limit: int | None = 16_000,
) -> ResolvedModel:
    """Resolve one scripted model with optional complete finite-cost metadata.

    Args:
        alias: Stable fixture alias.
        client: Scripted provider client.
        context_window_tokens: Declared request context capacity.
        completion_pricing: Whether to declare completion eligibility and prices.
        input_price: Explicit input price, or ``None`` for an unknown-price test.
        output_limit: Provider output capacity, or ``None`` when unpublished.

    Returns:
        Exact scripted runtime model.
    """
    return ResolvedModel(
        alias=alias,
        snapshot=_snapshot(alias),
        capabilities=ModelCapabilities(
            supports_completions=True if completion_pricing else None,
            context_window_tokens=context_window_tokens,
            maximum_output_tokens=output_limit,
            input_cost_per_million_tokens_usd=input_price if completion_pricing else None,
            output_cost_per_million_tokens_usd=2.0 if completion_pricing else None,
            cached_input_cost_per_million_tokens_usd=0.5 if completion_pricing else None,
            cache_write_cost_per_million_tokens_usd=1.5 if completion_pricing else None,
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
    resolved_world_client: _ScriptedClient | None = None,
    candidate_request: CompletionCostReservation | None = None,
    world_request: CompletionCostReservation | None = None,
    active_input_price: float | None = 1.0,
    maximum_cost_usd: float = 10.0,
    stop_on_overspend: bool = False,
    maximum_steps: int = 2,
    maximum_rollout_output_tokens: int = 1_000_000,
    output_limit: int | None = 16_000,
    world_model_json_object_output: bool = False,
    maximum_transition_attempts: int = 1,
    task: TaskCase | None = None,
    world_context_window: int = 100_000,
) -> RecordingCandidateClient:
    """Build a recorder with explicit fake candidate, world model, and retriever.

    Args:
        candidate_client: Scripted candidate provider client.
        world_client: Scripted world-model provider client.
        candidate_context_window: Candidate request context-window ceiling.
        candidate_served_model_id: Optional observed served-model identity.
        resolved_world_client: Optional decoy client retained only in the resolved identity.
        candidate_request: Optional secure candidate request reservation.
        world_request: Optional secure world-model request reservation.
        active_input_price: Active catalog input price for secure reservation tests.
        maximum_cost_usd: Reconciled provider-spend ceiling for the recorded cell.
        stop_on_overspend: Fail before the next paid dispatch once spend reaches the ceiling.
        output_limit: Published candidate and world output limit, or ``None``.
        world_model_json_object_output: Explicit frozen world-only JSON output control.
        maximum_transition_attempts: Permitted simulator replies for the same candidate turn.
        task: Optional task carrying declared tools for observation-boundary tests.
        world_context_window: Exact world-model context ceiling for retry admission tests.

    Returns:
        Recorder configured for one deterministic task.
    """
    candidate = _resolved(
        "candidate-a",
        candidate_client,
        context_window_tokens=candidate_context_window,
        completion_pricing=candidate_request is not None,
        input_price=active_input_price,
        output_limit=output_limit,
    )
    if candidate_served_model_id is not None:
        candidate = replace(candidate, served_model_id=candidate_served_model_id)
    retriever = cast(TraceRAGRetriever, _Retriever())
    serving_input = ArtifactInput(artifact_id="serving-rag", sha256="c" * 64)
    world_model = _resolved(
        "world-model-a",
        resolved_world_client or world_client,
        context_window_tokens=world_context_window,
        completion_pricing=world_request is not None,
        output_limit=output_limit,
    )
    grounded = GroundedWorldModel(
        artifact_input=ArtifactInput(artifact_id="grounded-world-model", sha256="d" * 64),
        artifact=GroundedWorldModelArtifact(
            schema_version=1,
            created_at=_TIME,
            inputs=(serving_input,),
            code_revision="test-revision",
            world_model_id="grounded-world-model",
            serving_rag=serving_input,
            model_alias="world-model-a",
            model=world_model.snapshot,
            prompt_version=GROUNDED_WORLD_MODEL_PROMPT_VERSION,
            prompt_sha256=grounded_world_model_prompt_sha256(),
            top_k=8,
        ),
        retriever=retriever,
        client=world_client,
    )
    return RecordingCandidateClient(
        task=task or _task(),
        candidate=candidate,
        world_model=world_model,
        grounded_world_model=grounded,
        query_embedding=EmbeddingCostReservation(
            model=_snapshot("embedder-a"),
            input_usd_per_million_tokens=1.0,
            maximum_attempts=2,
            maximum_input_tokens=10_000,
        ),
        candidate_request=candidate_request,
        world_model_request=world_request,
        completion_maximum_attempts=1,
        maximum_cost_usd=maximum_cost_usd,
        stop_on_overspend=stop_on_overspend,
        maximum_steps=maximum_steps,
        maximum_rollout_output_tokens=maximum_rollout_output_tokens,
        maximum_output_tokens=16_000,
        world_model_json_object_output=world_model_json_object_output,
        maximum_transition_attempts=maximum_transition_attempts,
        redacted_field_names=frozenset(),
        clock=lambda: _TIME,
        token_counter=_Utf8Counter(),
    )


def _completion_reservation(
    alias: str, *, maximum_input_tokens: int = 80_000, maximum_output_tokens: int = 16_000
) -> CompletionCostReservation:
    """Return a complete one-attempt request reservation for a scripted alias.

    Args:
        alias: Exact candidate or world-model alias.
        maximum_input_tokens: Full request input ceiling.
        maximum_output_tokens: Finite request output budget.

    Returns:
        Conservative completion request reservation.
    """
    return completion_cost_reservation(
        model=_snapshot(alias),
        input_usd_per_million_tokens=1,
        output_usd_per_million_tokens=2,
        cached_input_usd_per_million_tokens=0.5,
        cache_write_usd_per_million_tokens=1.5,
        maximum_attempts=1,
        maximum_input_tokens=maximum_input_tokens,
        maximum_output_tokens=maximum_output_tokens,
    )


class _Utf8Counter:
    def count(self, request: ModelRequest) -> int:
        return len(request.model_dump_json().encode("utf-8"))


@pytest.mark.parametrize("json_output", [False, True])
def test_recorder_keeps_candidate_and_world_calls_separate_and_tool_free(
    json_output: bool,
) -> None:
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
    recorder = _recorder(candidate_client, world_client, world_model_json_object_output=json_output)

    response = recorder.complete(
        ModelRequest(messages=(ModelMessage(role="user", content="My delivery is late."),))
    )

    assert response.output.content == "I can help."
    assert world_client.requests[0].tools == ()
    assert world_client.requests[0].tool_choice == "none"
    assert world_client.requests[0].json_object_output is json_output
    assert candidate_client.requests[0].json_object_output is False
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


@pytest.mark.parametrize("stop_on_overspend", [False, True])
def test_empty_candidate_response_is_preserved_without_retrieval_cost(
    stop_on_overspend: bool,
) -> None:
    """A completed blank reply is world-model evidence, not an embedding/provider failure."""
    candidate = _ScriptedClient([_response("", model=_snapshot("candidate-a"))])
    world = _ScriptedClient(
        [
            _response(
                '{"message":"No answer received.","terminal":true}',
                model=_snapshot("world-model-a"),
            )
        ]
    )
    recorder = _recorder(candidate, world, stop_on_overspend=stop_on_overspend)

    response = recorder.complete(
        ModelRequest(messages=(ModelMessage(role="user", content="Help."),))
    )

    assert response.output == AssistantAction(content="")
    assert len(candidate.requests) == len(world.requests) == 1
    evidence = json.loads(world.requests[0].messages[1].content or "")
    assert evidence["candidate_response"] == {"content": "", "tool_calls": []}
    assert recorder.terminal_error is None
    assert recorder.world_model_terminal
    assert recorder.recorded.retrieved_transition_ids == ((),)
    assert recorder.recorded.retrieval_economics.cost_usd == NumericMeasurement(
        value=0, provenance="estimated"
    )
    assert recorder.recorded.candidate_economics.cost_usd == NumericMeasurement(
        value=0.1, provenance="observed"
    )
    assert recorder.recorded.world_model_economics.cost_usd == NumericMeasurement(
        value=0.1, provenance="observed"
    )


def test_recorder_persists_estimated_cost_for_native_candidate_and_world_usage() -> None:
    """Price production-shaped candidate and world responses before rollout persistence."""
    candidate_snapshot = _snapshot("candidate-a")
    world_snapshot = _snapshot("world-model-a")
    candidate_response = openai_responses_response(
        _openai_payload(
            model="candidate-a",
            content="I can help.",
            input_tokens=10,
            output_tokens=2,
            cached_tokens=4,
        ),
        configured_model=candidate_snapshot,
        latency_seconds=0.1,
    )
    world_response = openai_responses_response(
        _openai_payload(
            model="world-model-a",
            content='{"message":"Done.","terminal":true}',
            input_tokens=20,
            output_tokens=3,
            cached_tokens=5,
        ),
        configured_model=world_snapshot,
        latency_seconds=0.2,
    )
    recorder = _recorder(
        _ScriptedClient([candidate_response]),
        _ScriptedClient([world_response]),
        candidate_request=_completion_reservation("candidate-a"),
        world_request=_completion_reservation("world-model-a"),
    )

    recorder.complete(ModelRequest(messages=(ModelMessage(role="user", content="Help me."),)))

    candidate_cost = recorder.recorded.candidate_economics.cost_usd
    world_cost = recorder.recorded.world_model_economics.cost_usd
    assert candidate_cost is not None and candidate_cost.provenance == "estimated"
    assert world_cost is not None and world_cost.provenance == "estimated"
    assert candidate_cost.value > 0
    assert world_cost.value > 0


def test_candidate_unknown_usage_retains_unknown_spend_after_dispatch() -> None:
    """Mark a paid candidate response unknown when its usage cannot be priced."""
    candidate = openai_responses_response(
        {
            "id": "resp_no_usage",
            "object": "response",
            "created_at": 1.0,
            "status": "completed",
            "model": "candidate-a",
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "output": [
                {
                    "type": "message",
                    "id": "msg_no_usage",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "I can help.", "annotations": []}],
                }
            ],
        },
        configured_model=_snapshot("candidate-a"),
        latency_seconds=0.1,
    )
    candidate_client = _ScriptedClient([candidate])
    world_client = _ScriptedClient([])
    recorder = _recorder(
        candidate_client,
        world_client,
        candidate_request=_completion_reservation("candidate-a"),
        world_request=_completion_reservation("world-model-a"),
    )

    with pytest.raises(TextSimulationError) as error:
        recorder.complete(ModelRequest(messages=(ModelMessage(role="user", content="Help."),)))

    assert error.value.failure.details["provider_dispatch_unknown_spend"] is True
    assert len(candidate_client.requests) == 1
    assert world_client.requests == []


def test_world_invalid_usage_retains_unknown_spend_after_dispatch() -> None:
    """Mark a paid world response unknown when cached usage is inconsistent."""
    candidate_client = _ScriptedClient([_response("I can help.", model=_snapshot("candidate-a"))])
    world = openai_responses_response(
        _openai_payload(
            model="world-model-a",
            content='{"message":"Done.","terminal":true}',
            input_tokens=10,
            output_tokens=2,
            cached_tokens=11,
        ),
        configured_model=_snapshot("world-model-a"),
        latency_seconds=0.1,
    )
    world_client = _ScriptedClient([world])
    recorder = _recorder(
        candidate_client,
        world_client,
        candidate_request=_completion_reservation("candidate-a"),
        world_request=_completion_reservation("world-model-a"),
    )

    with pytest.raises(TextSimulationError) as error:
        recorder.complete(ModelRequest(messages=(ModelMessage(role="user", content="Help."),)))

    assert error.value.failure.details["provider_dispatch_unknown_spend"] is True
    assert len(candidate_client.requests) == 1
    assert len(world_client.requests) == 1


def test_recorder_dispatches_only_through_the_artifact_bound_grounded_executor() -> None:
    """A distinct raw resolved client cannot bypass fit retrieval and grounded prompt framing."""
    candidate_client = _ScriptedClient([_response("I can help.", model=_snapshot("candidate-a"))])
    grounded_client = _ScriptedClient(
        [
            _response(
                '{"message":"Done.","terminal":true}',
                model=_snapshot("world-model-a"),
            )
        ]
    )
    raw_client = _ScriptedClient([])
    recorder = _recorder(
        candidate_client,
        grounded_client,
        resolved_world_client=raw_client,
    )

    recorder.complete(ModelRequest(messages=(ModelMessage(role="user", content="Help me."),)))

    assert len(grounded_client.requests) == 1
    assert raw_client.requests == []


def test_recorder_rejects_tool_requests_before_any_provider_call() -> None:
    """Text simulation does not quietly remove tool access from a candidate request."""
    candidate_client = _ScriptedClient([])
    world_client = _ScriptedClient([])
    recorder = _recorder(candidate_client, world_client)

    with pytest.raises(TextSimulationError, match="declared tool schemas") as error:
        recorder.complete(
            ModelRequest(
                messages=(ModelMessage(role="user", content="Use the system."),),
                tools=(ToolSchema(name="unknown", description="Unknown", input_schema={}),),
            )
        )

    assert error.value.stop_reason == StopReason.FAILURE
    assert candidate_client.requests == []
    assert world_client.requests == []


@pytest.mark.parametrize(
    ("maximum_input_tokens", "active_input_price", "message"),
    ((1, 1.0, "reserved input-token ceiling"), (80_000, None, "pricing is incomplete")),
)
def test_candidate_reservation_failure_blocks_every_provider_call(
    maximum_input_tokens: int,
    active_input_price: float | None,
    message: str,
) -> None:
    """Under-reserved or unpriced candidate calls fail before candidate and world dispatch.

    Args:
        maximum_input_tokens: Frozen candidate request ceiling.
        active_input_price: Active catalog input price or unknown marker.
        message: Expected fail-closed diagnostic.
    """
    candidate_client = _ScriptedClient([])
    world_client = _ScriptedClient([])
    recorder = _recorder(
        candidate_client,
        world_client,
        candidate_request=_completion_reservation(
            "candidate-a", maximum_input_tokens=maximum_input_tokens
        ),
        world_request=_completion_reservation("world-model-a"),
        active_input_price=active_input_price,
    )

    with pytest.raises(TextSimulationError, match=message):
        recorder.complete(ModelRequest(messages=(ModelMessage(role="user", content="Help"),)))

    assert candidate_client.requests == []
    assert world_client.requests == []


@pytest.mark.parametrize("output_limit", [None, 16_000])
def test_recorder_fails_context_preflight_and_explicit_length_stops_without_truncation(
    output_limit: int | None,
) -> None:
    """Provider calls are blocked before overflow, while explicit provider length stops persist."""
    candidate_client = _ScriptedClient([])
    world_client = _ScriptedClient([])
    overflow = _recorder(
        candidate_client, world_client, candidate_context_window=16_000, output_limit=output_limit
    )

    with pytest.raises(TextSimulationError) as overflow_error:
        overflow.complete(ModelRequest(messages=(ModelMessage(role="user", content="x" * 16_000),)))

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
    length = _recorder(length_client, _ScriptedClient([]), output_limit=output_limit)

    with pytest.raises(TextSimulationError) as length_error:
        length.complete(ModelRequest(messages=(ModelMessage(role="user", content="short"),)))

    assert length_error.value.stop_reason == StopReason.LENGTH
    assert len(length_client.requests) == 1
    assert length.recorded.candidate_spans[0].payload["response"] == {
        "output": {"content": "unfinished response", "tool_calls": []},
        "finish_reason": "length",
    }


def test_unpublished_output_limits_dispatch_full_requests_within_context_and_reservations() -> None:
    """Both completion roles work without an invented provider limit and keep exact inputs."""
    candidate = _ScriptedClient(
        [
            _response("Done.", model=_snapshot("candidate-a")).model_copy(
                update={
                    "economics": OperationEconomics(usage=Usage(input_tokens=4, output_tokens=3))
                }
            )
        ]
    )
    world = _ScriptedClient(
        [_response('{"message":"Done.","terminal":true}', model=_snapshot("world-model-a"))]
    )
    recorder = _recorder(
        candidate,
        world,
        candidate_context_window=10_000,
        output_limit=None,
        candidate_request=_completion_reservation(
            "candidate-a", maximum_input_tokens=10_000, maximum_output_tokens=10_000
        ),
        world_request=_completion_reservation("world-model-a", maximum_input_tokens=100_000),
    )
    request = ModelRequest(messages=(ModelMessage(role="user", content="Complete the task."),))
    recorder.complete(request)
    assert candidate.requests[0].messages == request.messages
    for dispatched, context in ((candidate.requests[0], 10_000), (world.requests[0], 100_000)):
        assert dispatched.maximum_output_tokens is not None
        assert _Utf8Counter().count(dispatched) + dispatched.maximum_output_tokens <= context
    assert recorder.world_model_terminal


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


def test_response_identity_accepts_a_catalog_pinned_served_model_id() -> None:
    """A catalog-resolved alias accepts responses named by its pinned served identity."""
    catalog = RuntimeModelCatalog(
        ModelCatalog(
            connections={
                "hosted": ConnectionConfig(
                    provider="openai-compatible",
                    base_url="https://models.example.test/v1",
                    api_key_env="FIXTURE_API_KEY",
                )
            },
            models={
                "candidate-a": ModelRecord(
                    connection="hosted",
                    model="deepseek-ai/DeepSeek-V4-Flash-0731",
                    served_model_id="deepseek-v4-flash",
                    billing_source=BillingSource.CUSTOMER_MANAGED,
                    capabilities=ModelCapabilities(
                        context_window_tokens=100_000,
                        maximum_output_tokens=16_000,
                    ),
                )
            },
        ),
        environment={"FIXTURE_API_KEY": "fixture-key"},
        transport_factory=ScriptedJsonTransport,
    )
    resolved = catalog.resolve("candidate-a")
    served = resolved.snapshot.model_copy(update={"model_id": "deepseek-v4-flash"})
    rebound = resolved.snapshot.model_copy(update={"model_id": "another-model"})

    _require_response_identity(_response("served", model=served), resolved, role="candidate")
    with pytest.raises(TextSimulationError, match="identity"):
        _require_response_identity(_response("wrong", model=rebound), resolved, role="candidate")


def test_default_recorder_warns_once_and_continues_after_spend_reaches_the_ceiling(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """By default reconciled spend crossing the ceiling logs one warning and keeps dispatching.

    Args:
        caplog: Captured recorder log records.
    """
    candidate_snapshot = _snapshot("candidate-a")
    world_snapshot = _snapshot("world-model-a")
    candidate_client = _ScriptedClient(
        [
            _response("first turn", model=candidate_snapshot),
            _response("second turn", model=candidate_snapshot),
        ]
    )
    world_client = _ScriptedClient(
        [
            _response('{"message":"continue","terminal":false}', model=world_snapshot),
            _response('{"message":"done","terminal":true}', model=world_snapshot),
        ]
    )
    recorder = _recorder(candidate_client, world_client, maximum_cost_usd=0.05)

    with caplog.at_level(logging.WARNING, logger="exp.simulation.engines.text.recording"):
        recorder.complete(
            ModelRequest(messages=(ModelMessage(role="user", content="My delivery is late."),))
        )
        response = recorder.complete(
            ModelRequest(messages=(ModelMessage(role="user", content="Any update?"),))
        )

    assert response.output.content == "second turn"
    assert len(candidate_client.requests) == 2
    warnings = [record for record in caplog.records if "already authorized" in record.message]
    assert len(warnings) == 1


def test_stop_mode_recorder_blocks_the_next_dispatch_after_spend_reaches_the_ceiling() -> None:
    """Stop mode fails closed before the next paid call once reconciled spend hits the ceiling."""
    candidate_snapshot = _snapshot("candidate-a")
    world_snapshot = _snapshot("world-model-a")
    candidate_client = _ScriptedClient([_response("first turn", model=candidate_snapshot)])
    world_client = _ScriptedClient(
        [_response('{"message":"continue","terminal":false}', model=world_snapshot)]
    )
    recorder = _recorder(
        candidate_client,
        world_client,
        maximum_cost_usd=0.15,
        stop_on_overspend=True,
    )

    recorder.complete(
        ModelRequest(messages=(ModelMessage(role="user", content="My delivery is late."),))
    )
    with pytest.raises(TextSimulationError) as error:
        recorder.complete(
            ModelRequest(messages=(ModelMessage(role="user", content="Any update?"),))
        )

    assert error.value.stop_reason == StopReason.MAXIMUM_COST
    assert len(candidate_client.requests) == 1


def test_rollout_token_budget_bounds_the_next_request_and_preserves_usage() -> None:
    """Only remaining total generated tokens are admitted on later candidate requests."""
    candidate = _ScriptedClient(
        [
            _response("first", model=_snapshot("candidate-a")),
            _response("second", model=_snapshot("candidate-a")),
        ]
    )
    world = _ScriptedClient(
        [
            _response('{"message":"continue","terminal":false}', model=_snapshot("world-model-a")),
            _response('{"message":"continue","terminal":false}', model=_snapshot("world-model-a")),
        ]
    )
    recorder = _recorder(candidate, world, maximum_steps=100, maximum_rollout_output_tokens=6)
    request = ModelRequest(messages=(ModelMessage(role="user", content="start"),))
    recorder.complete(request)
    recorder.complete(request)
    assert [item.maximum_output_tokens for item in candidate.requests] == [6, 3]
    with pytest.raises(TextSimulationError) as caught:
        recorder.complete(request)
    assert caught.value.stop_reason == StopReason.MAXIMUM_OUTPUT_TOKENS
    assert len(candidate.requests) == 2
    assert recorder.checkpoint() is not None


def test_missing_token_usage_blocks_further_dispatch() -> None:
    """Missing usage never silently replenishes a rollout's output budget."""
    response = _response("answer", model=_snapshot("candidate-a"))
    response = response.model_copy(
        update={"economics": response.economics.model_copy(update={"usage": None})}
    )
    candidate = _ScriptedClient([response])
    world = _ScriptedClient(
        [
            _response('{"message":"continue","terminal":false}', model=_snapshot("world-model-a")),
        ]
    )
    recorder = _recorder(candidate, world, maximum_steps=100)
    request = ModelRequest(messages=(ModelMessage(role="user", content="start"),))
    recorder.complete(request)
    with pytest.raises(TextSimulationError, match="usage is missing"):
        recorder.complete(request)
    assert len(candidate.requests) == 1


@pytest.mark.parametrize("invalid", ['{"message":', '{"message":"hello","extra":true}', "{}"])
def test_invalid_world_reply_retries_same_candidate_turn_and_accounts_all_costs(
    invalid: str,
) -> None:
    """A bad simulator reply never becomes a worker turn, free spend, or visible observation."""
    candidate = _ScriptedClient([_response("answer", model=_snapshot("candidate-a"))])
    world = _ScriptedClient(
        [
            _response(invalid, model=_snapshot("world-model-a")),
            _response('{"message":"","terminal":true}', model=_snapshot("world-model-a")),
        ]
    )
    recorder = _recorder(candidate, world, maximum_transition_attempts=3)
    request = ModelRequest(messages=(ModelMessage(role="user", content="question"),))
    response = recorder.complete(request)
    assert response.output.content == "answer"
    assert len(candidate.requests) == 1
    assert len(world.requests) == 2
    assert len(recorder.recorded.transitions) == 1
    assert len(recorder.recorded.world_model_spans) == 2
    assert recorder.recorded.world_model_spans[0].failure is not None
    assert recorder.recorded.world_model_spans[1].failure is None
    assert recorder.recorded.candidate_economics.cost_usd == NumericMeasurement(
        value=0.1, provenance="observed"
    )
    assert recorder.recorded.world_model_economics.cost_usd is not None
    assert recorder.recorded.world_model_economics.cost_usd.value == pytest.approx(0.2)
    assert world.requests[1].messages[:2] == world.requests[0].messages
    assert "invalid" in (world.requests[1].messages[-1].content or "")
    checkpoint = recorder.checkpoint()
    assert checkpoint is not None
    assert len(checkpoint.invalid_world_model_responses) == 1


def test_world_retry_exhaustion_is_invalid_and_retains_each_paid_response() -> None:
    """The frozen attempt count ends only the invalid cell, with all simulator spend retained."""
    candidate = _ScriptedClient([_response("answer", model=_snapshot("candidate-a"))])
    world = _ScriptedClient([_response("bad", model=_snapshot("world-model-a")) for _ in range(3)])
    recorder = _recorder(candidate, world, maximum_transition_attempts=3)
    with pytest.raises(TextSimulationError, match="JSON transition") as exc:
        recorder.complete(ModelRequest(messages=(ModelMessage(role="user", content="question"),)))
    assert exc.value.failure.retryable
    assert len(candidate.requests) == 1
    assert len(world.requests) == 3
    assert recorder.recorded.world_model_economics.cost_usd is not None
    assert recorder.recorded.world_model_economics.cost_usd.value == pytest.approx(0.3)
    assert recorder.recorded.transitions == ()
    assert recorder.checkpoint() is None


def test_world_retry_checks_spend_before_another_dispatch() -> None:
    """Protocol retries honor the cell's existing stop-on-overspend contract."""
    candidate = _ScriptedClient([_response("answer", model=_snapshot("candidate-a"))])
    world = _ScriptedClient([_response("bad", model=_snapshot("world-model-a"))])
    recorder = _recorder(
        candidate,
        world,
        maximum_transition_attempts=3,
        maximum_cost_usd=0.2,
        stop_on_overspend=True,
    )
    with pytest.raises(TextSimulationError, match="ceiling"):
        recorder.complete(ModelRequest(messages=(ModelMessage(role="user", content="question"),)))
    assert len(world.requests) == 1
    assert recorder.recorded.world_model_economics.cost_usd is not None
    assert recorder.recorded.world_model_economics.cost_usd.value == pytest.approx(0.1)


def test_checkpoint_after_world_retry_restores_valid_state_and_all_prior_spend() -> None:
    """Continuation restores accepted observations while preserving rejected simulator charges."""
    candidate = _ScriptedClient([_response("first", model=_snapshot("candidate-a"))])
    world = _ScriptedClient(
        [
            _response("bad", model=_snapshot("world-model-a")),
            _response(
                '{"message":"continue","state":{"step":1}}', model=_snapshot("world-model-a")
            ),
        ]
    )
    recorder = _recorder(candidate, world, maximum_transition_attempts=3)
    request = ModelRequest(messages=(ModelMessage(role="user", content="question"),))
    recorder.complete(request)
    checkpoint = recorder.checkpoint()
    assert checkpoint is not None
    next_candidate = _ScriptedClient([_response("second", model=_snapshot("candidate-a"))])
    next_world = _ScriptedClient(
        [
            _response('{"message":"","terminal":true}', model=_snapshot("world-model-a")),
        ]
    )
    resumed = _recorder(next_candidate, next_world, maximum_transition_attempts=3)
    resumed.restore(
        checkpoint, (*recorder.recorded.candidate_spans, *recorder.recorded.world_model_spans)
    )
    resumed.complete(request)
    assert len(next_candidate.requests) == len(next_world.requests) == 1
    assert json.loads(next_world.requests[0].messages[1].content or "")["environment_state"] == {
        "step": 1
    }
    assert resumed.recorded.candidate_economics.cost_usd is not None
    assert resumed.recorded.candidate_economics.cost_usd.value == pytest.approx(0.2)
    assert resumed.recorded.world_model_economics.cost_usd is not None
    assert resumed.recorded.world_model_economics.cost_usd.value == pytest.approx(0.3)
    spans = resumed.recorded.world_model_spans
    assert len({span.span_id for span in spans}) == len(spans) == 3
    assert resumed.checkpoint() is not None


def test_world_retry_does_not_repair_wrong_ids_or_deliver_rejected_tool_content() -> None:
    """The simulator must generate a valid exact-ID result; the engine never rewrites a bad ID."""
    tool = ToolSchema(name="lookup", description="Look up company facts.", input_schema={})
    task = _task().model_copy(update={"tools": (tool,)})
    call = ToolCall(call_id="chatcmpl-tool-exact", name="lookup", arguments={})
    worker_response = _response("", model=_snapshot("candidate-a")).model_copy(
        update={"output": AssistantAction(tool_calls=(call,))}
    )
    candidate = _ScriptedClient([worker_response])
    world = _ScriptedClient(
        [
            _response(
                '{"tool_results":[{"call_id":"chatmpl-tool-exact","content":"rejected"}]}',
                model=_snapshot("world-model-a"),
            ),
            _response(
                '{"tool_results":[{"call_id":"chatcmpl-tool-exact","content":"accepted"}]}',
                model=_snapshot("world-model-a"),
            ),
        ]
    )
    recorder = _recorder(candidate, world, maximum_transition_attempts=3, task=task)
    recorder.complete(
        ModelRequest(messages=(ModelMessage(role="user", content="question"),), tools=(tool,))
    )
    assert len(candidate.requests) == 1
    assert recorder.observe_tool(call).content == "accepted"
    assert "rejected" not in json.dumps(
        [m.model_dump(mode="json") for m in recorder.visible_transcript]
    )
    assert recorder.recorded.world_model_spans[0].failure is not None
    assert "chatcmpl-tool-exact" in (world.requests[1].messages[-1].content or "")
    assert recorder.checkpoint() is not None


@pytest.mark.parametrize("limited_by", ["context", "reservation"])
def test_world_retry_without_feedback_room_reuses_full_original_request(limited_by: str) -> None:
    """Retries remain usable at either input ceiling without discarding evidence or output space."""
    request = ModelRequest(messages=(ModelMessage(role="user", content="question"),))
    candidate_response = _response("answer", model=_snapshot("candidate-a"))
    invalid = _response("bad", model=_snapshot("world-model-a"))
    baseline_world = _ScriptedClient([invalid])
    baseline = _recorder(_ScriptedClient([candidate_response]), baseline_world)
    with pytest.raises(TextSimulationError):
        baseline.complete(request)
    input_tokens = _Utf8Counter().count(baseline_world.requests[0])
    world = _ScriptedClient(
        [
            invalid,
            _response('{"message":"","terminal":true}', model=_snapshot("world-model-a")),
        ]
    )
    world.responses = [
        response.model_copy(
            update={
                "economics": response.economics.model_copy(
                    update={"cost_usd": None, "provider_attempts": 1}
                )
            }
        )
        for response in world.responses
    ]
    candidate = _ScriptedClient([candidate_response])
    recorder = _recorder(
        candidate,
        world,
        maximum_transition_attempts=3,
        world_context_window=input_tokens + 16_000 if limited_by == "context" else 100_000,
        world_request=_completion_reservation("world-model-a", maximum_input_tokens=input_tokens)
        if limited_by == "reservation"
        else None,
    )
    recorder.complete(request)
    assert len(candidate.requests) == 1
    assert len(world.requests) == 2
    assert world.requests[1] == world.requests[0]
    assert recorder.world_model_terminal
