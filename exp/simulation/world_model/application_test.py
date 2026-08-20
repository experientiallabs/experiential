"""Public project-name world-model loader and bounded-session tests."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest
from openai.types.chat import ChatCompletionAssistantMessageParam

from exp.common.core.artifacts import ArtifactInput, JsonObject, sha256_json
from exp.common.models import (
    AssistantAction,
    BillingSource,
    Embedding,
    ModelCapabilities,
    ModelRequest,
    ModelResponse,
    ModelSnapshot,
    OperationEconomics,
)
from exp.common.project import (
    ProjectBuildArtifacts,
    ProjectConfig,
    ProjectModelConfiguration,
    ProjectStore,
)
from exp.runtime.models import CatalogRoleName, ResolvedModel, RuntimeModelCatalog
from exp.simulation.engines.text.prompt import WORLD_MODEL_TEXT_SYSTEM_PROMPT
from exp.simulation.retrieval import RAGEmbedderBinding, RAGMatch, RAGQuery, TraceRAGRetriever
from exp.simulation.world_model.application import (
    WorldModel,
    WorldModelLoadError,
    WorldModelSessionError,
    WorldModelSessionLimits,
    load_world_model,
)
from exp.simulation.world_model.artifact import (
    GROUNDED_WORLD_MODEL_PROMPT_VERSION,
    GroundedWorldModelArtifact,
    grounded_world_model_prompt_sha256,
)
from exp.simulation.world_model.runtime import GroundedWorldModel

_TIME = datetime(2026, 8, 13, tzinfo=UTC)


class _Retriever:
    """Capture public session queries without contacting an embedder."""

    def __init__(self, rag_input: ArtifactInput) -> None:
        """Bind the fake retriever to one serving-RAG pointer.

        Args:
            rag_input: Exact serving-RAG manifest reference.
        """
        self.rag_input = rag_input
        self.queries: list[RAGQuery] = []

    def retrieve(self, query: RAGQuery) -> tuple[RAGMatch, ...]:
        """Record one query and return no demonstrations.

        Args:
            query: Canonical serving-RAG query.

        Returns:
            Empty immutable retrieval result.
        """
        self.queries.append(query)
        return ()


class _WorldClient:
    """Capture canonical requests and return queued strict transitions."""

    def __init__(
        self,
        snapshot: ModelSnapshot,
        outputs: Sequence[str] = ('{"message":"Next","terminal":false}',),
    ) -> None:
        """Configure one exact response identity and output sequence.

        Args:
            snapshot: Provider identity returned with every response.
            outputs: Strict transition payloads returned in order.
        """
        self.snapshot = snapshot
        self.outputs = list(outputs)
        self.requests: list[ModelRequest] = []

    def complete(self, request: ModelRequest) -> ModelResponse:
        """Record a request and return the next configured transition.

        Args:
            request: Canonical artifact-bound request.

        Returns:
            Model response under the configured identity.
        """
        self.requests.append(request)
        return ModelResponse(
            output=AssistantAction(content=self.outputs.pop(0)),
            model=self.snapshot,
            economics=OperationEconomics(),
        )


class _Embedder:
    """Minimal embedding client used only to prove loader composition."""

    def embed(self, texts: Sequence[str]) -> tuple[Embedding, ...]:
        """Return one fixed vector for each input.

        Args:
            texts: Texts submitted for embedding.

        Returns:
            Stable one-dimensional unit vectors.
        """
        return tuple(Embedding(values=(1.0,)) for _ in texts)


class _Catalog:
    """Resolve the exact world-model and embedder aliases selected by a project."""

    def __init__(self, world: ResolvedModel, embedder: ResolvedModel) -> None:
        """Store exact resolved fixtures.

        Args:
            world: Completion-capable world-model resolution.
            embedder: Embedding-capable resolution.
        """
        self.world = world
        self.embedder = embedder
        self.resolved: list[str] = []

    def resolve(self, alias: str, *, role: CatalogRoleName | None = None) -> ResolvedModel:
        del role
        """Resolve only the configured world-model alias.

        Args:
            alias: Project-selected alias.

        Returns:
            Exact world-model fixture.
        """
        self.resolved.append(alias)
        return self.world

    def preflight(
        self,
        alias: str,
        requirement: object,
        *,
        role: CatalogRoleName | None = None,
    ) -> ResolvedModel:
        del role
        """Resolve the configured embedder after recording capability preflight.

        Args:
            alias: Project-selected embedder alias.
            requirement: Explicit embedding capability requirement.

        Returns:
            Exact embedder fixture.
        """
        del requirement
        self.resolved.append(alias)
        return self.embedder


def test_session_uses_canonical_prompt_and_exact_prior_transcript() -> None:
    """Every public step uses artifact framing.

    The second request contains each prior visible turn exactly once and preserves the detached
    initial context even after callers mutate both input and returned objects.
    """
    runtime, retriever, client = _runtime(
        outputs=(
            '{"message":"First observation","terminal":false}',
            '{"message":"Finished","terminal":true}',
        )
    )
    world_model = WorldModel(runtime, session_id_factory=lambda: "session-a")
    initial_context: JsonObject = {"tenant": {"name": "support"}}
    session = world_model.new_session(
        task="Reset the password",
        initial_context=initial_context,
    )
    cast(dict[str, object], initial_context["tenant"])["name"] = "mutated"
    detached = session.initial_context
    cast(dict[str, object], detached["tenant"])["name"] = "also-mutated"

    first = world_model.step(session.id, _action("Ask for the account email"))
    second = world_model.step(session.id, _action("Send the reset link"))

    assert first.message == {"role": "user", "content": "First observation"}
    assert first.terminal is False
    assert second.message == {"role": "user", "content": "Finished"}
    assert second.terminal is True
    assert len(retriever.queries) == 2
    assert len(client.requests) == 2
    assert client.requests[0].messages[0].content == WORLD_MODEL_TEXT_SYSTEM_PROMPT
    first_payload = _request_payload(client.requests[0])
    second_payload = _request_payload(client.requests[1])
    assert first_payload["visible_conversation"] == []
    assert second_payload["visible_conversation"] == [
        {"role": "assistant", "content": "Ask for the account email"},
        {"role": "user", "content": "First observation"},
    ]
    assert second_payload["candidate_response"] == "Send the reset link"
    task_payload = cast(dict[str, object], second_payload["task"])
    assert task_payload["instruction"] == "Reset the password"
    assert task_payload["initial_context"] == {"tenant": {"name": "support"}}
    assert session.initial_context == {"tenant": {"name": "support"}}
    with pytest.raises(WorldModelSessionError, match="closed"):
        world_model.step(session.id, _action("This must not dispatch"))
    assert len(retriever.queries) == 2
    assert len(client.requests) == 2
    world_model.end_session(session.id)
    with pytest.raises(WorldModelSessionError, match="unknown or expired"):
        world_model.step(session.id, _action("This must not dispatch either"))


def test_unsupported_action_and_preflight_overflow_dispatch_nothing() -> None:
    """Unsupported actions and known overflow dispatch nothing.

    Both failures occur before serving-RAG retrieval and world-model provider execution.
    """
    runtime, retriever, client = _runtime()
    limits = WorldModelSessionLimits(
        maximum_sessions=1,
        session_ttl_seconds=60,
        maximum_messages_per_session=2,
        maximum_transcript_bytes=200,
        maximum_observation_bytes=150,
    )
    world_model = WorldModel(runtime, limits=limits, session_id_factory=lambda: "session-a")
    session = world_model.new_session(task="Task")
    tool_action = cast(
        ChatCompletionAssistantMessageParam,
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": "{}"},
                }
            ],
        },
    )

    with pytest.raises(WorldModelSessionError, match="tool_calls"):
        world_model.step(session.id, tool_action)
    with pytest.raises(WorldModelSessionError, match="byte limit"):
        world_model.step(session.id, _action("x" * 100))

    assert retriever.queries == []
    assert client.requests == []


def test_failed_response_identity_or_size_never_advances_transcript() -> None:
    """Rejected provider results never advance the transcript.

    Oversized output leaves the next canonical request empty, while mismatched model identity is
    rejected before the invalid provider payload can reach protocol parsing.
    """
    runtime, _, client = _runtime(
        outputs=(
            '{"message":"This observation is too large","terminal":false}',
            '{"message":"ok","terminal":false}',
        )
    )
    limits = WorldModelSessionLimits(
        maximum_observation_bytes=10,
        maximum_transcript_bytes=1_000,
    )
    world_model = WorldModel(runtime, limits=limits, session_id_factory=lambda: "session-a")
    session = world_model.new_session(task="Task")

    with pytest.raises(WorldModelSessionError, match="provider-output byte limit"):
        world_model.step(session.id, _action("first"))
    result = world_model.step(session.id, _action("second"))

    assert result.message == {"role": "user", "content": "ok"}
    assert _request_payload(client.requests[1])["visible_conversation"] == []

    wrong_runtime, wrong_retriever, wrong_client = _runtime(
        response_snapshot=_snapshot("different"),
        outputs=("not-json",),
    )
    wrong_world = WorldModel(wrong_runtime, session_id_factory=lambda: "session-b")
    wrong_session = wrong_world.new_session(task="Task")
    with pytest.raises(WorldModelSessionError, match="response identity"):
        wrong_world.step(wrong_session.id, _action("ask"))
    assert len(wrong_retriever.queries) == 1
    assert len(wrong_client.requests) == 1


def test_session_capacity_expiry_and_end_remove_all_state() -> None:
    """Capacity, TTL, and explicit end bound retained session state.

    Expired and ended identities become unavailable, and released capacity can be reused without
    retrieval or provider execution.
    """
    now = [10.0]
    runtime, retriever, client = _runtime()
    world_model = WorldModel(
        runtime,
        limits=WorldModelSessionLimits(maximum_sessions=1, session_ttl_seconds=5),
        monotonic=lambda: now[0],
        session_id_factory=iter(("session-a", "session-b", "session-c")).__next__,
    )
    first = world_model.new_session(task="First")
    with pytest.raises(WorldModelSessionError, match="capacity"):
        world_model.new_session(task="Blocked")

    now[0] = 16.0
    second = world_model.new_session(task="Second")
    with pytest.raises(WorldModelSessionError, match="unknown or expired"):
        world_model.step(first.id, _action("ask"))
    world_model.end_session(second.id)
    third = world_model.new_session(task="Third")
    with pytest.raises(WorldModelSessionError, match="unknown or expired"):
        world_model.step(second.id, _action("ask"))

    assert third.id == "session-c"
    assert retriever.queries == []
    assert client.requests == []


def test_failed_provider_result_does_not_refresh_session_ttl() -> None:
    """A rejected provider result preserves the original expiry boundary.

    Advancing beyond that boundary removes the failed session without another retrieval or model
    dispatch.
    """
    now = [10.0]
    runtime, retriever, client = _runtime(outputs=('{"message":"too large","terminal":false}',))
    world_model = WorldModel(
        runtime,
        limits=WorldModelSessionLimits(
            session_ttl_seconds=5,
            maximum_observation_bytes=2,
            maximum_transcript_bytes=1_000,
        ),
        monotonic=lambda: now[0],
        session_id_factory=lambda: "session-a",
    )
    session = world_model.new_session(task="Task")

    with pytest.raises(WorldModelSessionError, match="provider-output byte limit"):
        world_model.step(session.id, _action("ask"))
    now[0] = 16.0
    with pytest.raises(WorldModelSessionError, match="unknown or expired"):
        world_model.step(session.id, _action("ask again"))

    assert len(retriever.queries) == 1
    assert len(client.requests) == 1


def test_loader_resolves_exact_serving_artifact_without_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Project-name load verifies serving pointers and constructs no paid operation.

    Args:
        tmp_path: Temporary local project root.
        monkeypatch: Fixture replacing only immutable artifact loading.
    """
    pointers = _build_pointers()
    store = ProjectStore(tmp_path / ".exp", "support")
    store.initialize(
        ProjectConfig(
            project_id="support",
            models=ProjectModelConfiguration(
                world_model="world",
                judge="judge",
                embedder="embedder",
            ),
            build=pointers,
        )
    )
    runtime, retriever, client = _runtime(
        artifact_input=pointers.world_model,
        serving_rag=pointers.serving_rag,
    )
    embedder_snapshot = _snapshot(
        "embedder",
        capabilities=ModelCapabilities(
            supports_embeddings=True,
            input_cost_per_million_tokens_usd=0.0,
        ),
    )
    embedder_client = _Embedder()
    catalog = _Catalog(
        ResolvedModel(
            alias="world",
            snapshot=runtime.artifact.model,
            capabilities=ModelCapabilities(),
            client=client,
            embedding_client=None,
        ),
        ResolvedModel(
            alias="embedder",
            snapshot=embedder_snapshot,
            capabilities=ModelCapabilities(
                supports_embeddings=True,
                input_cost_per_million_tokens_usd=0.0,
            ),
            client=client,
            embedding_client=embedder_client,
        ),
    )
    loaded: list[str] = []

    def fake_load(*args: object, **kwargs: object) -> GroundedWorldModel:
        """Capture the exact artifact ID selected by the project.

        Args:
            args: Positional artifact loader inputs.
            kwargs: Explicit client and embedder bindings.

        Returns:
            Verified serving-bound fixture runtime.
        """
        loaded.append(cast(str, args[1]))
        assert kwargs["client"] is client
        binding = cast(RAGEmbedderBinding, kwargs["embedder"])
        assert binding.client is embedder_client
        assert binding.snapshot == embedder_snapshot
        return runtime

    monkeypatch.setattr(
        "exp.simulation.world_model.application.load_grounded_world_model",
        fake_load,
    )

    public = load_world_model(
        "support",
        root=tmp_path / ".exp",
        runtime_catalog=cast(RuntimeModelCatalog, catalog),
    )

    assert isinstance(public, WorldModel)
    assert loaded == [pointers.world_model.artifact_id]
    assert catalog.resolved == ["world", "embedder"]
    assert retriever.queries == []
    assert client.requests == []
    assert store.artifacts.list_ids() == ()

    retriever.rag_input = pointers.fit_rag
    with pytest.raises(WorldModelLoadError, match="exact serving RAG"):
        load_world_model(
            "support",
            root=tmp_path / ".exp",
            runtime_catalog=cast(RuntimeModelCatalog, catalog),
        )


def _runtime(
    *,
    artifact_input: ArtifactInput | None = None,
    serving_rag: ArtifactInput | None = None,
    response_snapshot: ModelSnapshot | None = None,
    outputs: Sequence[str] = ('{"message":"Next","terminal":false}',),
) -> tuple[GroundedWorldModel, _Retriever, _WorldClient]:
    """Build one serving-bound grounded runtime with observable local seams.

    Args:
        artifact_input: Optional completed world-model manifest pointer.
        serving_rag: Optional exact serving-RAG pointer.
        response_snapshot: Optional provider response identity override.
        outputs: Strict transition payloads returned in order.

    Returns:
        Runtime, fake retriever, and fake provider client.
    """
    rag_input = serving_rag or ArtifactInput(artifact_id="serving-rag", sha256="a" * 64)
    snapshot = _snapshot("world")
    retriever = _Retriever(rag_input)
    client = _WorldClient(response_snapshot or snapshot, outputs)
    runtime = GroundedWorldModel(
        artifact_input=artifact_input or ArtifactInput(artifact_id="world-model", sha256="b" * 64),
        artifact=GroundedWorldModelArtifact(
            schema_version=1,
            created_at=_TIME,
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
    )
    return runtime, retriever, client


def _snapshot(
    model_id: str,
    *,
    capabilities: ModelCapabilities | None = None,
) -> ModelSnapshot:
    """Return one exact fixture model identity.

    Args:
        model_id: Stable provider model identifier.
        capabilities: Optional explicit capability snapshot.

    Returns:
        Secret-free immutable model snapshot.
    """
    resolved_capabilities = capabilities or ModelCapabilities()
    return ModelSnapshot(
        billing_source=BillingSource.CUSTOMER_MANAGED,
        provider="fixture",
        model_id=model_id,
        capabilities_sha256=sha256_json(resolved_capabilities),
        connection_sha256=sha256_json({"connection": model_id}),
    )


def _action(content: str) -> ChatCompletionAssistantMessageParam:
    """Return one official OpenAI text assistant message.

    Args:
        content: Visible assistant text.

    Returns:
        Official assistant-message parameter.
    """
    return {"role": "assistant", "content": content}


def _request_payload(request: ModelRequest) -> dict[str, object]:
    """Decode the canonical user prompt from one grounded request.

    Args:
        request: Artifact-bound world-model request.

    Returns:
        Decoded canonical prompt object.
    """
    content = request.messages[1].content
    assert content is not None
    return cast(dict[str, object], json.loads(content))


def _build_pointers() -> ProjectBuildArtifacts:
    """Return distinct completed-build pointers for loader tests.

    Returns:
        Exact immutable project build references.
    """
    return ProjectBuildArtifacts(
        trace_dataset=ArtifactInput(artifact_id="traces", sha256="1" * 64),
        task_set=ArtifactInput(artifact_id="tasks", sha256="2" * 64),
        serving_rag=ArtifactInput(artifact_id="serving-rag", sha256="3" * 64),
        fit_rag=ArtifactInput(artifact_id="fit-rag", sha256="4" * 64),
        world_model=ArtifactInput(artifact_id="world-model", sha256="5" * 64),
    )
