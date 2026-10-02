"""Executable grounded world-model runtime tests."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest

from exp.common.core.artifacts import (
    ArtifactInput,
    SourceIdentity,
    canonical_json_bytes,
    sha256_json,
)
from exp.common.models import (
    AssistantAction,
    BillingSource,
    Embedding,
    ModelCapabilities,
    ModelRequest,
    ModelResponse,
    ModelSnapshot,
    OperationEconomics,
    ToolCall,
)
from exp.common.project import ProjectConfig, ProjectStore, artifact_input
from exp.common.tasks import ToolSchema
from exp.common.traces import Trace, TraceDataset, TraceSource, TraceSpan
from exp.simulation.engines.text.prompt import (
    WORLD_MODEL_TEXT_SYSTEM_PROMPT,
    TextWorldModelProtocolError,
    text_prompt_sha256,
)
from exp.simulation.engines.text.recording_test import _grounding_example
from exp.simulation.engines.text.tokens import Utf8UpperBoundTokenCounter
from exp.simulation.retrieval import (
    RAGAction,
    RAGEmbedderBinding,
    RAGLineageBinding,
    RAGMatch,
    RAGQuery,
    TraceRAGRetriever,
    load_fit_rag_retriever,
    load_rag_index,
    persist_trace_rag,
)
from exp.simulation.world_model import (
    WorldModel,
    WorldModelSessionError,
    bind_fit_grounded_world_model,
    load_grounded_world_model,
    persist_grounded_world_model,
)
from exp.simulation.world_model.artifact import (
    GROUNDED_WORLD_MODEL_ARTIFACT_TYPE,
    GROUNDED_WORLD_MODEL_PROMPT_VERSION,
    GROUNDED_WORLD_MODEL_SYSTEM_PROMPT,
    WORLD_MODEL_ARTIFACT_PATH,
    GroundedWorldModelArtifact,
    grounded_world_model_prompt_sha256,
)
from exp.simulation.world_model.runtime import GroundedWorldModel


class _Embedder:
    """Stable local embedding client shared by index build and runtime query."""

    def __init__(self) -> None:
        """Retain every actual embedding batch for paid-work boundary checks."""
        self.requests: list[tuple[str, ...]] = []

    def embed(self, texts: Sequence[str]) -> tuple[Embedding, ...]:
        """Return deterministic unit vectors.

        Args:
            texts: Canonical query or transition texts.

        Returns:
            Stable unit vectors in input order.
        """
        self.requests.append(tuple(texts))
        embedded = []
        for text in texts:
            digest = hashlib.sha256(text.encode()).digest()
            raw = [float(value + 1) for value in digest[:8]]
            norm = math.sqrt(sum(value * value for value in raw))
            embedded.append(Embedding(values=tuple(value / norm for value in raw)))
        return tuple(embedded)


class _WorldClient:
    """Capture the grounded request and return one strict protocol transition."""

    def __init__(
        self,
        snapshot: ModelSnapshot,
        output: str = '{"message":"Use the saved email.","terminal":false}',
    ) -> None:
        """Record requests under one exact fixture model identity.

        Args:
            snapshot: Frozen world-model identity returned with every response.
            output: Simulated provider transition returned to the runtime.
        """
        self.snapshot = snapshot
        self.output = output
        self.requests: list[ModelRequest] = []

    def complete(self, request: ModelRequest) -> ModelResponse:
        """Return one visible environment observation.

        Args:
            request: Grounded world-model completion request to capture.

        Returns:
            Strict typed transition response under the configured fixture identity.
        """
        self.requests.append(request)
        return ModelResponse(
            output=AssistantAction(content=self.output),
            model=self.snapshot,
            economics=OperationEconomics(),
        )


def test_build_and_simulation_share_one_grounded_prompt_identity() -> None:
    """The persisted world-model protocol exactly matches active simulation framing."""
    assert GROUNDED_WORLD_MODEL_SYSTEM_PROMPT == WORLD_MODEL_TEXT_SYSTEM_PROMPT
    assert grounded_world_model_prompt_sha256() == text_prompt_sha256()


def test_loaded_world_model_retrieves_real_evidence_before_prediction(tmp_path: Path) -> None:
    """A completed build artifact executes with immutable observed-transition grounding.

    Args:
        tmp_path: Temporary project root containing the RAG and world-model artifacts.
    """
    store = ProjectStore(tmp_path / ".exp", "support")
    store.initialize(ProjectConfig(project_id="support"))
    created_at = datetime(2026, 8, 13, tzinfo=UTC)
    trace = _trace(created_at)
    trace_payload = trace.model_dump_json().encode() + b"\n"
    dataset = TraceDataset(
        schema_version=1,
        created_at=created_at,
        code_revision="fixture-revision",
        source=trace.source.identity,
        dataset_id="trace-source",
        semantic_convention_version="1.37.0",
        traces_path="traces.jsonl",
        traces_sha256=hashlib.sha256(trace_payload).hexdigest(),
        issues_path="normalization-issues.json",
        issues_sha256=hashlib.sha256(b"[]").hexdigest(),
        invalid_trace_count=0,
        trace_ids=(trace.trace_id,),
    )
    trace_manifest = store.artifacts.write(
        artifact_id=dataset.dataset_id,
        artifact_type="trace-dataset",
        envelope=dataset,
        files={
            "trace-dataset.json": dataset.model_dump_json().encode(),
            "traces.jsonl": trace_payload,
            "normalization-issues.json": b"[]",
        },
    )
    capabilities = ModelCapabilities(supports_embeddings=True)
    embedding_snapshot = ModelSnapshot(
        billing_source=BillingSource.CUSTOMER_MANAGED,
        provider="fixture",
        model_id="embed",
        capabilities_sha256=sha256_json(capabilities),
        connection_sha256=sha256_json({"connection": "fixture"}),
    )
    embedder = _Embedder()
    binding = RAGEmbedderBinding(client=embedder, snapshot=embedding_snapshot)
    rag = persist_trace_rag(
        store.artifacts,
        (artifact_input(trace_manifest),),
        (RAGLineageBinding(trace_id=trace.trace_id, lineage_id="lineage-a", partition="fit"),),
        created_at=created_at,
        code_revision="fixture-revision",
        embedder=binding,
        default_top_k=5,
        included_partitions=frozenset({"fit", "held_out"}),
    )
    world_capabilities = ModelCapabilities(
        context_window_tokens=32_768, maximum_output_tokens=8_192
    )
    world_snapshot = ModelSnapshot(
        billing_source=BillingSource.CUSTOMER_MANAGED,
        provider="fixture",
        model_id="world",
        capabilities_sha256=world_capabilities.identity_sha256(),
        connection_sha256=sha256_json({"connection": "world"}),
    )
    artifact = persist_grounded_world_model(
        store.artifacts,
        artifact_input(rag.manifest),
        model_alias="world",
        model=world_snapshot,
        created_at=created_at,
        code_revision="fixture-revision",
        top_k=5,
    )
    assert artifact.artifact.top_k == 5
    client = _WorldClient(world_snapshot)

    for context in (16_384, 65_536):
        with pytest.raises(ValueError, match="capabilities differ from.*artifact"):
            load_grounded_world_model(
                store.artifacts,
                artifact.artifact.world_model_id,
                client=client,
                embedder=binding,
                capabilities=world_capabilities.model_copy(
                    update={"context_window_tokens": context}
                ),
            )
    assert client.requests == []

    runtime = load_grounded_world_model(
        store.artifacts,
        artifact.artifact.world_model_id,
        client=client,
        embedder=binding,
        capabilities=world_capabilities,
    )
    assert runtime.capabilities == world_capabilities
    transition = runtime.step(
        task="Reset my password",
        action=AssistantAction(content="What email is associated with the account?"),
    )

    assert transition.message == "Use the saved email."
    assert transition.terminal is False
    assert len(client.requests) == 1
    content = client.requests[0].messages[1].content
    assert content is not None
    assert "grounded_examples" in content
    assert "customer@example.test" in content
    unchanged = load_rag_index(store.artifacts, rag.index.rag_id)
    assert unchanged.transitions == rag.transitions
    assert unchanged.vectors == rag.vectors

    fit_rag = persist_trace_rag(
        store.artifacts,
        (artifact_input(trace_manifest),),
        (RAGLineageBinding(trace_id=trace.trace_id, lineage_id="lineage-a", partition="fit"),),
        created_at=created_at,
        code_revision="fixture-revision",
        included_partitions=frozenset({"fit"}),
        embedder=binding,
        default_top_k=5,
    )
    fit_retriever = load_fit_rag_retriever(
        store.artifacts,
        artifact_input(fit_rag.manifest),
        embedder=binding,
    )
    fit_runtime = bind_fit_grounded_world_model(
        store.artifacts,
        artifact_input(artifact.manifest),
        client=client,
        capabilities=world_capabilities,
        fit_retriever=fit_retriever,
    )
    assert runtime.retriever.rag_input == artifact_input(rag.manifest)
    assert fit_runtime.retriever.rag_input == artifact_input(fit_rag.manifest)
    assert fit_runtime.artifact_input == artifact_input(artifact.manifest)
    assert fit_runtime.capabilities == world_capabilities
    fit_runtime.step(task="Reset my password", action=AssistantAction(content="What email?"))
    assert len(client.requests) == 2
    embedding_count = len(embedder.requests)
    for maximum_output_tokens in (8_193, 0, -1):
        with pytest.raises(ValueError):
            fit_runtime.step(
                task="Reset my password",
                action=AssistantAction(content="Help"),
                maximum_output_tokens=maximum_output_tokens,
            )
        assert len(embedder.requests) == embedding_count
    with pytest.raises(ValueError, match="required world-model input.*context"):
        fit_runtime.step(task="Required task " * 3_000, action=AssistantAction(content="Help"))
    assert len(embedder.requests) == embedding_count
    assert len(client.requests) == 2
    for context in (None, 16_384, 65_536):
        with pytest.raises(ValueError, match="capabilities differ from.*artifact"):
            bind_fit_grounded_world_model(
                store.artifacts,
                artifact_input(artifact.manifest),
                client=client,
                fit_retriever=fit_retriever,
                capabilities=world_capabilities.model_copy(
                    update={"context_window_tokens": context}
                ),
            )
    with pytest.raises(ValueError, match="fit-only"):
        bind_fit_grounded_world_model(
            store.artifacts,
            artifact_input(artifact.manifest),
            client=client,
            capabilities=world_capabilities,
            fit_retriever=runtime.retriever,
        )
    with pytest.raises(ValueError, match="manifest differs"):
        bind_fit_grounded_world_model(
            store.artifacts,
            artifact_input(artifact.manifest).model_copy(update={"sha256": "0" * 64}),
            client=client,
            capabilities=world_capabilities,
            fit_retriever=fit_retriever,
        )

    inconsistent = artifact.artifact.model_copy(update={"world_model_id": "inconsistent-world"})
    store.artifacts.write(
        artifact_id=inconsistent.world_model_id,
        artifact_type=GROUNDED_WORLD_MODEL_ARTIFACT_TYPE,
        envelope=inconsistent,
        files={WORLD_MODEL_ARTIFACT_PATH: canonical_json_bytes(inconsistent)},
    )
    with pytest.raises(ValueError, match="complete content"):
        load_grounded_world_model(
            store.artifacts,
            inconsistent.world_model_id,
            client=client,
            capabilities=world_capabilities,
            embedder=binding,
        )


def _trace(created_at: datetime) -> Trace:
    """Create one real assistant-to-user transition.

    Args:
        created_at: Fixture timestamp shared by the trace spans.

    Returns:
        Canonical two-span production trace.
    """
    source = TraceSource(
        identity=SourceIdentity(kind="otlp", source_id="fixture-source"),
        semantic_convention_version="1.37.0",
    )
    return Trace(
        trace_id="0" * 31 + "1",
        source=source,
        task="Reset my password",
        spans=(
            TraceSpan(
                span_id="0" * 15 + "1",
                parent_span_id=None,
                name="agent.model_call",
                started_at=created_at,
                ended_at=created_at,
                attributes={
                    "gen_ai.operation.name": "chat",
                    "gen_ai.output.messages": json.dumps(
                        [
                            {
                                "role": "assistant",
                                "content": "What email is associated with the account?",
                            }
                        ]
                    ),
                    "gen_ai.input.messages": '[{"role":"user","content":"Reset my password"}]',
                },
            ),
            TraceSpan(
                span_id="0" * 15 + "2",
                parent_span_id=None,
                name="agent.model_call",
                started_at=created_at,
                ended_at=created_at,
                attributes={
                    "gen_ai.operation.name": "chat",
                    "gen_ai.input.messages": json.dumps(
                        [
                            {
                                "role": "assistant",
                                "content": "What email is associated with the account?",
                            },
                            {"role": "user", "content": "customer@example.test"},
                        ]
                    ),
                },
            ),
        ),
    )


class _Retriever:
    """Record grounding queries without dispatching an embedding provider.

    Attributes:
        queries: Submitted retrieval queries in call order, initially empty.
    """

    def __init__(self) -> None:
        """Start with no observed grounding requests."""
        self.queries: list[RAGQuery] = []
        self.matches: tuple[RAGMatch, ...] = ()

    def retrieve(self, query: RAGQuery) -> tuple[RAGMatch, ...]:
        """Record the query and return the configured grounding corpus."""
        self.queries.append(query)
        return self.matches


def _runtime(
    output: str, *, capabilities: ModelCapabilities | None = None
) -> tuple[GroundedWorldModel, _Retriever, _WorldClient]:
    """Build an isolated runtime with observable retrieval and completion dispatches.

    Args:
        output: Exact response text returned by the simulated world-model provider.
        capabilities: Exact provider metadata frozen into the fixture model identity.

    Returns:
        Runtime, query recorder, and completion recorder.
    """
    capabilities = capabilities or ModelCapabilities()
    rag_input = ArtifactInput(artifact_id="serving-rag", sha256="a" * 64)
    snapshot = ModelSnapshot(
        billing_source=BillingSource.CUSTOMER_MANAGED,
        provider="fixture",
        model_id="world",
        capabilities_sha256=capabilities.identity_sha256(),
        connection_sha256=sha256_json({"connection": "world"}),
    )
    retriever = _Retriever()
    client = _WorldClient(snapshot, output)
    runtime = GroundedWorldModel(
        artifact_input=ArtifactInput(artifact_id="world-model", sha256="b" * 64),
        artifact=GroundedWorldModelArtifact(
            schema_version=1,
            created_at=datetime(2026, 8, 13, tzinfo=UTC),
            inputs=(rag_input,),
            code_revision="test-revision",
            world_model_id="world-model",
            serving_rag=rag_input,
            model_alias="world",
            model=snapshot,
            prompt_version=GROUNDED_WORLD_MODEL_PROMPT_VERSION,
            prompt_sha256=grounded_world_model_prompt_sha256(),
            top_k=5,
        ),
        retriever=cast(TraceRAGRetriever, retriever),
        client=client,
        capabilities=capabilities,
    )
    return runtime, retriever, client


@pytest.mark.parametrize("context", [None, 16_384, 65_536])
def test_direct_runtime_rejects_capabilities_outside_frozen_identity(context: int | None) -> None:
    """Direct construction cannot disable or widen the artifact's known context capacity."""
    runtime, retriever, client = _runtime(
        '{"message":"Next","terminal":false}',
        capabilities=ModelCapabilities(context_window_tokens=32_768, maximum_output_tokens=8_192),
    )
    with pytest.raises(ValueError, match="capabilities differ from.*artifact"):
        replace(
            runtime,
            capabilities=runtime.capabilities.model_copy(update={"context_window_tokens": context}),
        )
    assert retriever.queries == []
    assert client.requests == []


@pytest.mark.parametrize("field", ["prompt_version", "prompt_sha256"])
def test_direct_runtime_cannot_render_new_framing_under_stale_prompt_identity(field: str) -> None:
    """Direct construction has the same immutable prompt boundary as the artifact loader."""
    runtime, retriever, client = _runtime('{"message":"Next","terminal":false}')
    value = "text-world-model-v2" if field == "prompt_version" else "0" * 64
    with pytest.raises(ValueError, match="prompt"):
        replace(runtime, artifact=runtime.artifact.model_copy(update={field: value}))
    assert retriever.queries == []
    assert client.requests == []


def test_step_rejects_unsolicited_tool_results_for_text_action() -> None:
    """A valid transition envelope cannot invent tools the assistant never called."""
    runtime, _, _ = _runtime('{"tool_results":[{"call_id":"invented","content":"ok"}]}')
    with pytest.raises(TextWorldModelProtocolError, match="match every candidate call_id"):
        runtime.step(task="Research", action=AssistantAction(content="Done"))


@pytest.mark.parametrize("public_session", [False, True])
def test_public_steps_pack_examples_and_reject_required_overflow(public_session: bool) -> None:
    """Both public step surfaces fit optional evidence before dispatch, retaining required text."""
    runtime, retriever, client = _runtime(
        '{"message":"Next","terminal":false}',
        capabilities=ModelCapabilities(context_window_tokens=24_000, maximum_output_tokens=16_000),
    )
    retriever.matches = (_grounding_example("oversized", 30_000), _grounding_example("fits", 50))

    def step(task: str) -> None:
        """Drive the real shared dispatch through one of the two public entrypoints."""
        if public_session:
            world = WorldModel(runtime)
            session = world.new_session(task=task)
            world.step(session.id, {"role": "assistant", "content": "Required action"})
        else:
            runtime.step(task=task, action=AssistantAction(content="Required action"))

    step("Required task")
    request = client.requests[0]
    evidence = json.loads(request.messages[1].content or "")
    assert [item["transition_id"] for item in evidence["grounded_examples"]] == ["fits"]
    assert evidence["task"]["instruction"] == "Required task"
    assert request.maximum_output_tokens == 1_024
    assert Utf8UpperBoundTokenCounter().count(request) + 1_024 <= 24_000
    error = WorldModelSessionError if public_session else ValueError
    query_count = len(retriever.queries)
    with pytest.raises(error, match="required world-model input.*context"):
        step("Required task " * 3_000)
    assert len(retriever.queries) == query_count
    assert len(client.requests) == 1


def test_session_capacity_rejection_keeps_transcript_and_lock_reusable() -> None:
    """A local capacity failure keeps its public type and does not advance session evidence."""
    runtime, retriever, client = _runtime(
        '{"message":"Next","terminal":false}',
        capabilities=ModelCapabilities(context_window_tokens=24_000, maximum_output_tokens=16_000),
    )
    world = WorldModel(runtime)
    session = world.new_session(task="Required task")
    with pytest.raises(WorldModelSessionError, match="required world-model input.*context"):
        world.step(session.id, {"role": "assistant", "content": "Large action " * 3_000})
    assert retriever.queries == []
    assert client.requests == []
    world.step(session.id, {"role": "assistant", "content": "Valid action"})
    assert len(retriever.queries) == len(client.requests) == 1
    assert json.loads(client.requests[0].messages[1].content or "")["visible_conversation"] == []


def test_session_rejects_unsupported_output_before_retrieval() -> None:
    """The public fixed output request respects known output limits even with unknown context."""
    runtime, retriever, client = _runtime(
        '{"message":"Next","terminal":false}',
        capabilities=ModelCapabilities(maximum_output_tokens=1_023),
    )
    retriever.matches = (_grounding_example("eligible", 20),)
    world = WorldModel(runtime)
    session = world.new_session(task="Required task")
    with pytest.raises(WorldModelSessionError, match="choose a model with a larger output"):
        world.step(session.id, {"role": "assistant", "content": "Action"})
    assert retriever.queries == []
    assert client.requests == []


@pytest.mark.parametrize("phase", ["retrieve", "complete"])
def test_session_does_not_reclassify_provider_value_errors(
    phase: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only known local capacity errors become session errors; provider failures retain identity."""
    runtime, retriever, client = _runtime('{"message":"Next","terminal":false}')
    failure = ValueError("provider fixture failure")

    def fail(_request: ModelRequest | RAGQuery) -> None:
        """Raise the exact provider-side exception after the local preflight."""
        raise failure

    monkeypatch.setattr(retriever if phase == "retrieve" else client, phase, fail)
    world = WorldModel(runtime)
    session = world.new_session(task="Required task")
    with pytest.raises(ValueError) as raised:
        world.step(session.id, {"role": "assistant", "content": "Action"})
    assert raised.value is failure
    world.end_session(session.id)


def test_step_returns_exact_parallel_tool_results_with_canonical_grounding() -> None:
    """The convenience API retains schemas, call IDs, exclusion pins, and provider framing."""
    tool = ToolSchema(name="lookup", description="Look up a company", input_schema={})
    calls = tuple(
        ToolCall(call_id=identifier, name="lookup", arguments={"query": name})
        for identifier, name in (("a", "Acme"), ("b", "Beta"))
    )
    runtime, retriever, client = _runtime(
        '{"tool_results":[{"call_id":"a","content":"Acme"},'
        '{"call_id":"b","content":"Beta"}],"state":{"lookups":2}}'
    )
    action = AssistantAction(tool_calls=calls)
    transition = runtime.step(
        task="Research",
        action=action,
        tools=(tool,),
        initial_context={"tenant": "support"},
        excluded_lineage_ids=("source-lineage",),
        maximum_output_tokens=8_192,
    )
    assert [result.call_id for result in transition.tool_results] == ["a", "b"]
    assert [result.content for result in transition.tool_results] == ["Acme", "Beta"]
    assert transition.state == {"lookups": 2}
    assert not transition.terminal
    assert [query.action.tool_arguments for query in retriever.queries] == [
        {"query": "Acme"},
        {"query": "Beta"},
    ]
    assert all(query.excluded_lineage_ids == ("source-lineage",) for query in retriever.queries)
    request = client.requests[0]
    assert request.maximum_output_tokens == 8_192
    assert request.tools == ()
    assert request.tool_choice == "none"
    payload = json.loads(request.messages[1].content or "")
    assert payload["candidate_response"] == action.model_dump(mode="json", exclude_none=True)
    assert payload["task"]["tools"] == [tool.model_dump(mode="json")]
    assert payload["task"]["initial_context"] == {"tenant": "support"}


@pytest.mark.parametrize(
    "output",
    [
        '{"message":"wrong role","terminal":true}',
        '{"tool_results":[{"call_id":"a","content":"missing b"}]}',
        '{"tool_results":[{"call_id":"b","content":"b"},{"call_id":"a","content":"a"}]}',
        '{"tool_results":[{"call_id":"a","content":"a"},{"call_id":"a","content":"a"}]}',
        '{"tool_results":[{"call_id":"a","content":"a"},{"call_id":"b","content":"b"}],'
        '"terminal":true}',
    ],
)
def test_step_rejects_tool_results_that_do_not_match_the_action(output: str) -> None:
    """Missing, reordered, repeated, or terminal tool observations never escape validation."""
    runtime, _, _ = _runtime(output)
    with pytest.raises(TextWorldModelProtocolError):
        runtime.step(
            task="Research",
            action=AssistantAction(
                tool_calls=tuple(ToolCall(call_id=item, name="lookup") for item in ("a", "b"))
            ),
            tools=(ToolSchema(name="lookup", description="Look up a company", input_schema={}),),
        )


@pytest.mark.parametrize(
    "action",
    [
        RAGAction(kind="tool_call", tool_name="lookup", tool_arguments={}),
        AssistantAction(tool_calls=(ToolCall(call_id="a", name="undeclared"),)),
        AssistantAction(
            tool_calls=(ToolCall(call_id="a", name="lookup"), ToolCall(call_id="a", name="lookup"))
        ),
    ],
)
def test_step_rejects_invalid_action_before_any_paid_dispatch(
    action: RAGAction | AssistantAction,
) -> None:
    """Unknown tools, repeated IDs, and ID-free actions fail before retrieval or completion."""
    runtime, retriever, client = _runtime('{"message":"Done","terminal":true}')
    with pytest.raises((TypeError, ValueError)):
        runtime.step(
            task="Research",
            action=cast(AssistantAction, action),
            tools=(ToolSchema(name="lookup", description="Look up a company", input_schema={}),),
        )
    assert retriever.queries == []
    assert client.requests == []
