"""End-to-end and adversarial coverage for automatic runtime retrieval refresh."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

import pytest
from pydantic import ValidationError

from wmo.common.core.artifacts import (
    ArtifactInput,
    FailureAttribution,
    FailureCode,
    SourceIdentity,
    StructuredFailure,
    stable_id,
)
from wmo.common.models import (
    AssistantAction,
    BillingSource,
    Embedding,
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
    Usage,
)
from wmo.common.project import (
    ArtifactCorruptionError,
    ArtifactStore,
    ProjectPaths,
    artifact_input,
)
from wmo.common.routing import RouterFeatureExtractor, RoutingDecision
from wmo.common.traces import Trace, TraceOutcome, TraceSource, TraceSpan
from wmo.runtime.router import RuntimeAcceptedEvent, RuntimeInteractionJournal
from wmo.runtime.router.economics import (
    BillingSourceEconomics,
    RoutedCompletionEconomics,
    RoutedProviderComponent,
    RoutedProviderOperation,
    RoutedSpendDisposition,
    routed_completion_economics,
)
from wmo.runtime.router.journal import RuntimeAcceptance, _interaction_identity
from wmo.simulation.build import build_task_set
from wmo.simulation.ingest.dataset import PersistedTraceDataset, persist_trace_dataset
from wmo.simulation.ingest.otlp import TraceNormalizationResult
from wmo.simulation.mining.bindings import load_task_set_lineage_bindings
from wmo.simulation.mining.service import MiningSpec
from wmo.simulation.retrieval import (
    PersistedRuntimeRAGRefresh,
    RAGAction,
    RAGEmbedderBinding,
    RAGLineageBinding,
    RAGQuery,
    RuntimeRAGDatasetError,
    RuntimeRAGRefreshError,
    RuntimeTraceStitchingError,
    TraceRAGRetriever,
    load_rag_index,
    load_runtime_rag_refresh,
    refresh_runtime_trace_rag,
)

_DIGEST = "a" * 64
_TIME = datetime(2026, 8, 14, tzinfo=UTC)


class CountingEmbedder:
    """Deterministic embedding client that records paid-dispatch attempts."""

    def __init__(self, dimensions: int = 8) -> None:
        """Create a fixed-width client with no completed dispatches.

        Args:
            dimensions: Unit-vector width returned for every input.
        """
        self.dimensions = dimensions
        self.calls = 0
        self.inputs: list[tuple[str, ...]] = []

    def embed(self, texts: Sequence[str]) -> tuple[Embedding, ...]:
        """Return deterministic unit vectors and record the exact batch.

        Args:
            texts: Canonical retrieval keys dispatched together.

        Returns:
            One finite unit vector per input.
        """
        self.calls += 1
        self.inputs.append(tuple(texts))
        vector = (1.0, *(0.0 for _ in range(self.dimensions - 1)))
        return tuple(Embedding(values=vector) for _ in texts)


class NonFiniteEmbedder:
    """Malicious client that bypasses model construction to return a NaN vector."""

    def embed(self, texts: Sequence[str]) -> tuple[Embedding, ...]:
        """Return one structurally present but non-finite vector per input.

        Args:
            texts: Canonical retrieval keys dispatched together.

        Returns:
            Invalid embeddings used to prove fail-closed vector validation.
        """
        return tuple(Embedding.model_construct(values=(float("nan"), 0.0)) for _ in texts)


def _model() -> ModelSnapshot:
    """Build the fixed routed-model snapshot used by journal fixtures."""
    return ModelSnapshot(
        billing_source=BillingSource.CUSTOMER_MANAGED,
        provider="openai",
        model_id="gpt-test",
        capabilities_sha256=_DIGEST,
        connection_sha256="b" * 64,
    )


def _request(*messages: ModelMessage) -> ModelRequest:
    """Build one deterministic routed request from exact visible messages.

    Args:
        messages: Complete visible conversation passed to the routed model.

    Returns:
        Immutable provider-neutral request.
    """
    return ModelRequest(messages=messages)


def _response(content: str) -> ModelResponse:
    """Build one completed assistant response with observed usage.

    Args:
        content: Assistant text returned by the routed candidate.

    Returns:
        Deterministic successful model response.
    """
    return ModelResponse(
        output=AssistantAction(content=content),
        model=_model(),
        economics=OperationEconomics(
            usage=Usage(input_tokens=8, output_tokens=3, cached_input_tokens=0)
        ),
    )


def _economics() -> RoutedCompletionEconomics:
    """Return deterministic billing attribution for one routed test response."""
    return routed_completion_economics(
        (
            _operation(1, RoutedProviderComponent.ROUTER_EMBEDDING),
            _operation(2, RoutedProviderComponent.SELECTED_CANDIDATE),
        )
    )


def _operation(
    ordinal: int,
    component: RoutedProviderComponent,
) -> RoutedProviderOperation:
    """Build deterministic customer-managed operation evidence."""
    return RoutedProviderOperation(
        operation_id=f"routed-operation-{ordinal:020x}",
        operation_ordinal=ordinal,
        component=component,
        billing_source=BillingSource.CUSTOMER_MANAGED,
        disposition=RoutedSpendDisposition.LOCALLY_PRICED,
        operation_count=1,
        economics=OperationEconomics(
            usage=Usage(input_tokens=ordinal, output_tokens=0),
            cost_usd=NumericMeasurement(value=ordinal / 1_000_000, provenance="estimated"),
        ),
    )


def _decision(lineage_id: str, request: ModelRequest) -> RoutingDecision:
    """Build a content-addressed routing decision for one accepted request.

    Args:
        lineage_id: Hashed conversation lineage bound by the journal.
        request: Exact request whose routing feature is selected.

    Returns:
        Deterministic single-candidate routing decision.
    """
    feature = RouterFeatureExtractor().from_request(request)
    material = {
        "policy_id": "router-policy-a",
        "policy_sha256": _DIGEST,
        "request_sha256": hashlib.sha256(
            feature.encode("utf-8"), usedforsecurity=False
        ).hexdigest(),
        "episode_id_sha256": hashlib.sha256(
            lineage_id.encode("utf-8"), usedforsecurity=False
        ).hexdigest(),
        "selected_alias": "candidate-a",
        "baseline_alias": "candidate-a",
        "neighbor_count": 2,
        "paired_count": 2,
        "best_similarity": 1.0,
        "estimated_quality_difference": None,
        "uncertainty": None,
        "fallback_reason": None,
    }
    return RoutingDecision(
        decision_id=stable_id("routing-decision", material),
        **material,
    )


def _journal_and_store(
    tmp_path: Path,
) -> tuple[RuntimeInteractionJournal, ArtifactStore]:
    """Create one isolated project journal and immutable artifact store.

    Args:
        tmp_path: Pytest-owned directory for local state.

    Returns:
        Journal and store sharing the same project boundary.
    """
    paths = ProjectPaths(root=tmp_path / ".wmo", project_id="support-agent")
    return RuntimeInteractionJournal(paths), ArtifactStore(paths)


def _accept(
    journal: RuntimeInteractionJournal,
    *,
    key: str,
    conversation: str,
    request: ModelRequest,
    now: datetime,
) -> RuntimeAcceptedEvent:
    """Accept one exact interaction into the durable journal.

    Args:
        journal: Project journal receiving the acceptance.
        key: Stable idempotency key for the logical interaction.
        conversation: Caller conversation identity hashed by the journal.
        request: Complete visible request.
        now: Acceptance timestamp.

    Returns:
        Durable accepted event ready for a terminal result.
    """
    identity = _interaction_identity(journal.project_id, key, request, conversation)
    decision = _decision(identity.lineage_id, request)
    embedding = BillingSourceEconomics(
        billing_source=BillingSource.CUSTOMER_MANAGED,
        economics=_operation(1, RoutedProviderComponent.ROUTER_EMBEDDING).economics,
    )
    reserved = journal.reserve_selection(
        identity,
        embedding,
        now=now,
        stale_after=timedelta(minutes=5),
    )
    assert reserved.reservation is not None
    claim = journal.record_acceptance(
        identity,
        RuntimeAcceptance(
            decision=decision,
            selected_alias=decision.selected_alias,
            selected_model=_model(),
            router_embedding_billing_source=BillingSource.CUSTOMER_MANAGED,
            policy_input=ArtifactInput(
                artifact_id=decision.policy_id,
                sha256=decision.policy_sha256,
            ),
        ),
        reserved.reservation,
        _operation(1, RoutedProviderComponent.ROUTER_EMBEDDING),
        accepted_at=now,
    )
    assert claim.accepted is not None
    return claim.accepted


def _complete(
    journal: RuntimeInteractionJournal,
    *,
    key: str,
    conversation: str,
    request: ModelRequest,
    output: str | AssistantAction,
    now: datetime,
    finish_reason: ModelFinishReason = ModelFinishReason.COMPLETED,
) -> RuntimeAcceptedEvent:
    """Accept and complete one deterministic routed interaction.

    Args:
        journal: Project journal receiving both events.
        key: Stable idempotency key.
        conversation: Caller conversation identity.
        request: Complete visible request.
        output: Assistant response text or complete assistant action.
        now: Acceptance timestamp.
        finish_reason: Provider-reported terminal reason preserved in provenance.

    Returns:
        Accepted event named by the completion.
    """
    accepted = _accept(
        journal,
        key=key,
        conversation=conversation,
        request=request,
        now=now,
    )
    action = output if isinstance(output, AssistantAction) else AssistantAction(content=output)
    _reserve_candidate(journal, accepted, now=now)
    journal.record_completed(
        accepted,
        ModelResponse(
            output=action,
            model=_model(),
            economics=OperationEconomics(
                usage=Usage(input_tokens=8, output_tokens=3, cached_input_tokens=0)
            ),
            finish_reason=finish_reason,
        ),
        candidate_operation=_economics().operations[-1],
        completed_at=now + timedelta(seconds=1),
    )
    return accepted


def _reserve_candidate(
    journal: RuntimeInteractionJournal,
    accepted: RuntimeAcceptedEvent,
    *,
    now: datetime,
) -> None:
    """Persist one deterministic candidate reservation before low-level dispatch."""
    claim = journal.reserve_candidate(
        accepted,
        BillingSourceEconomics(
            billing_source=BillingSource.CUSTOMER_MANAGED,
            economics=_operation(2, RoutedProviderComponent.SELECTED_CANDIDATE).economics,
        ),
        now=now,
    )
    assert claim.status == "dispatch"


def _two_turn_journal(
    tmp_path: Path,
) -> tuple[RuntimeInteractionJournal, ArtifactStore, RuntimeAcceptedEvent]:
    """Create a two-turn same-lineage journal with one observed assistant turn.

    Args:
        tmp_path: Pytest-owned directory for local state.

    Returns:
        Journal, store, and first accepted event whose output has a later user observation.
    """
    journal, store = _journal_and_store(tmp_path)
    first_request = _request(ModelMessage(role="user", content="First question"))
    first = _complete(
        journal,
        key="first",
        conversation="conversation-a",
        request=first_request,
        output="First answer",
        now=_TIME,
    )
    second_request = _request(
        ModelMessage(role="user", content="First question"),
        ModelMessage(
            role="assistant",
            assistant_action=AssistantAction(content="First answer"),
        ),
        ModelMessage(role="user", content="Second question"),
    )
    _complete(
        journal,
        key="second",
        conversation="conversation-a",
        request=second_request,
        output="Second answer",
        now=_TIME + timedelta(minutes=1),
    )
    return journal, store, first


def _binding(client: CountingEmbedder | NonFiniteEmbedder) -> RAGEmbedderBinding:
    """Bind one fixture client to explicit configured identity, retry, and price.

    Args:
        client: Deterministic or adversarial embedding implementation.

    Returns:
        Explicit semantic embedding binding for refresh.
    """
    capabilities = ModelCapabilities(supports_embeddings=True)
    return RAGEmbedderBinding(
        client=client,
        snapshot=ModelSnapshot(
            billing_source=BillingSource.CUSTOMER_MANAGED,
            provider="openai",
            model_id="text-embedding-test",
            revision="1",
            capabilities_sha256=hashlib.sha256(
                capabilities.model_dump_json().encode(), usedforsecurity=False
            ).hexdigest(),
            connection_sha256="c" * 64,
        ),
        maximum_attempts=3,
        input_usd_per_million_tokens=2.0,
    )


def _reservation(binding: RAGEmbedderBinding) -> EmbeddingCostReservation:
    """Create a generous explicit retry-inclusive reservation for one test refresh.

    Args:
        binding: Active embedder whose exact model, retry, and price are reserved.

    Returns:
        Finite reservation large enough for fixture transition keys.
    """
    return EmbeddingCostReservation(
        model=binding.snapshot,
        input_usd_per_million_tokens=binding.input_usd_per_million_tokens,
        maximum_attempts=binding.maximum_attempts,
        maximum_input_tokens=10_000,
    )


def _refresh(
    journal: RuntimeInteractionJournal,
    store: ArtifactStore,
    binding: RAGEmbedderBinding,
    *,
    created_at: datetime = _TIME + timedelta(hours=1),
    maximum_embedding_cost_usd: float = 1.0,
) -> PersistedRuntimeRAGRefresh:
    """Run one fixture refresh with no imported source datasets.

    Args:
        journal: Journal containing at least one observed transition.
        store: Same-project immutable artifact store.
        binding: Explicit configured embedder.
        created_at: Materialization time for new artifacts.
        maximum_embedding_cost_usd: Caller-authorized total embedding ceiling.

    Returns:
        Completed runtime retrieval refresh result.
    """
    return refresh_runtime_trace_rag(
        journal,
        store,
        (),
        (),
        embedder=binding,
        embedding_reservation=_reservation(binding),
        maximum_embedding_cost_usd=maximum_embedding_cost_usd,
        created_at=created_at,
        code_revision="test-revision",
    )


def test_zero_cost_int_and_float_share_identity_and_replay_without_dispatch(
    tmp_path: Path,
) -> None:
    """Canonicalize a public integer zero before identity and durable materialization.

    Args:
        tmp_path: Pytest-owned project directory.
    """
    journal, store, _first = _two_turn_journal(tmp_path)
    client = CountingEmbedder()
    configured = _binding(client)
    binding = RAGEmbedderBinding(
        client=client,
        snapshot=configured.snapshot,
        maximum_attempts=configured.maximum_attempts,
        input_usd_per_million_tokens=0.0,
    )

    first = _refresh(
        journal,
        store,
        binding,
        maximum_embedding_cost_usd=0,
    )
    completed_artifacts = {path.name for path in (store.project_directory / "artifacts").iterdir()}
    replay = _refresh(
        journal,
        store,
        binding,
        created_at=_TIME + timedelta(days=1),
        maximum_embedding_cost_usd=0.0,
    )

    assert first.refresh.maximum_embedding_cost_usd == 0.0
    assert isinstance(first.refresh.maximum_embedding_cost_usd, float)
    assert replay.refresh == first.refresh
    assert replay.retrieval.index == first.retrieval.index
    assert client.calls == 1
    assert {
        path.name for path in (store.project_directory / "artifacts").iterdir()
    } == completed_artifacts


@pytest.mark.parametrize(
    "ceiling",
    (True, -0.01, float("nan"), float("inf"), float("-inf")),
)
def test_invalid_cost_ceiling_fails_before_artifact_or_embedding(
    tmp_path: Path,
    ceiling: float,
) -> None:
    """Reject invalid public ceilings before sealing evidence or spending.

    Args:
        tmp_path: Pytest-owned project directory.
        ceiling: Boolean, negative, or non-finite caller input.
    """
    journal, store, _first = _two_turn_journal(tmp_path)
    client = CountingEmbedder()

    with pytest.raises(
        RuntimeRAGRefreshError,
        match="maximum_embedding_cost_usd must be finite and nonnegative",
    ):
        _refresh(
            journal,
            store,
            _binding(client),
            maximum_embedding_cost_usd=ceiling,
        )

    assert client.calls == 0
    assert not (store.project_directory / "artifacts").exists()


def test_two_turn_refresh_indexes_observed_turn_and_excludes_terminal_output(
    tmp_path: Path,
) -> None:
    """Admit the prior answer only after a later user turn.

    Args:
        tmp_path: Pytest-owned project directory.
    """
    journal, store, first = _two_turn_journal(tmp_path)
    client = CountingEmbedder()
    binding = _binding(client)

    result = _refresh(journal, store, binding)

    assert client.calls == 1
    assert result.refresh.last_ordinal == 4
    assert result.retrieval.index.transition_count == 1
    transition = result.retrieval.transitions[0]
    assert transition.trace_id == first.interaction_id
    assert transition.lineage_id == first.identity.lineage_id
    assert transition.action.content == "First answer"
    assert transition.observation.content == "Second question"
    assert "Second answer" not in tuple(
        item.action.content for item in result.retrieval.transitions
    )
    provenance = transition.initial_context["runtime_observation_provenance"]
    assert isinstance(provenance, dict)
    assert provenance["interaction_id"] == first.interaction_id
    assert provenance["accepted_ordinal"] == 1
    assert provenance["completed_ordinal"] == 2
    assert provenance["runtime_snapshot_input"] == artifact_input(
        result.snapshot_export.snapshot_manifest
    ).model_dump(mode="json")


def test_exact_replay_dispatches_zero_embeds_and_append_creates_new_siblings(
    tmp_path: Path,
) -> None:
    """Keep completed replay free and materialize a longer prefix as new siblings.

    Args:
        tmp_path: Pytest-owned project directory.
    """
    journal, store, _first = _two_turn_journal(tmp_path)
    client = CountingEmbedder()
    binding = _binding(client)
    first = _refresh(journal, store, binding)
    first_ids = (
        first.snapshot_export.snapshot.snapshot_id,
        first.dataset.dataset.dataset_id,
        first.retrieval.index.rag_id,
        first.refresh.refresh_id,
    )
    first_payloads = {
        artifact_id: {
            path.name: path.read_bytes()
            for path in store.read(artifact_id).directory.iterdir()
            if path.is_file()
        }
        for artifact_id in first_ids
    }

    replay = _refresh(
        journal,
        store,
        binding,
        created_at=_TIME + timedelta(days=1),
    )

    assert replay.refresh == first.refresh
    assert client.calls == 1

    third_request = _request(
        ModelMessage(role="user", content="First question"),
        ModelMessage(
            role="assistant",
            assistant_action=AssistantAction(content="First answer"),
        ),
        ModelMessage(role="user", content="Second question"),
        ModelMessage(
            role="assistant",
            assistant_action=AssistantAction(content="Second answer"),
        ),
        ModelMessage(role="user", content="Third question"),
    )
    _complete(
        journal,
        key="third",
        conversation="conversation-a",
        request=third_request,
        output="Third answer",
        now=_TIME + timedelta(hours=2),
    )
    later = _refresh(
        journal,
        store,
        binding,
        created_at=_TIME + timedelta(hours=3),
    )

    assert client.calls == 2
    assert later.refresh.refresh_id != first.refresh.refresh_id
    assert later.dataset.dataset.dataset_id != first.dataset.dataset.dataset_id
    assert later.retrieval.index.rag_id != first.retrieval.index.rag_id
    assert later.retrieval.index.transition_count == 2
    for artifact_id, payloads in first_payloads.items():
        assert {
            path.name: path.read_bytes()
            for path in store.read(artifact_id).directory.iterdir()
            if path.is_file()
        } == payloads


def test_interleaved_conversations_stitch_only_within_each_lineage(tmp_path: Path) -> None:
    """Prevent interleaved conversations from crossing observation lineages.

    Args:
        tmp_path: Pytest-owned project directory.
    """
    journal, store = _journal_and_store(tmp_path)
    first_a = _complete(
        journal,
        key="a-1",
        conversation="conversation-a",
        request=_request(ModelMessage(role="user", content="A1")),
        output="answer-a",
        now=_TIME,
    )
    first_b = _complete(
        journal,
        key="b-1",
        conversation="conversation-b",
        request=_request(ModelMessage(role="user", content="B1")),
        output="answer-b",
        now=_TIME + timedelta(minutes=1),
    )
    _complete(
        journal,
        key="a-2",
        conversation="conversation-a",
        request=_request(
            ModelMessage(role="user", content="A1"),
            ModelMessage(role="assistant", assistant_action=AssistantAction(content="answer-a")),
            ModelMessage(role="user", content="A2"),
        ),
        output="terminal-a",
        now=_TIME + timedelta(minutes=2),
    )
    _complete(
        journal,
        key="b-2",
        conversation="conversation-b",
        request=_request(
            ModelMessage(role="user", content="B1"),
            ModelMessage(role="assistant", assistant_action=AssistantAction(content="answer-b")),
            ModelMessage(role="user", content="B2"),
        ),
        output="terminal-b",
        now=_TIME + timedelta(minutes=3),
    )

    result = _refresh(journal, store, _binding(CountingEmbedder()))
    observed = {
        item.trace_id: (item.lineage_id, item.observation.content)
        for item in result.retrieval.transitions
    }

    assert observed == {
        first_a.interaction_id: (first_a.identity.lineage_id, "A2"),
        first_b.interaction_id: (first_b.identity.lineage_id, "B2"),
    }


def test_retry_attempts_produce_one_completed_transition(tmp_path: Path) -> None:
    """Preserve retry provenance without duplicating the completed target.

    Args:
        tmp_path: Pytest-owned project directory.
    """
    journal, store = _journal_and_store(tmp_path)
    request = _request(ModelMessage(role="user", content="Retry me"))
    first_attempt = _accept(
        journal,
        key="retry",
        conversation="conversation-a",
        request=request,
        now=_TIME,
    )
    _reserve_candidate(journal, first_attempt, now=_TIME)
    journal.record_failure(
        first_attempt,
        StructuredFailure(
            code=FailureCode.PROVIDER,
            message="temporary",
            retryable=True,
            attribution=FailureAttribution.MODEL,
        ),
        failed_at=_TIME + timedelta(seconds=1),
    )
    identity = _interaction_identity(journal.project_id, "retry", request, "conversation-a")
    retry_claim = journal.claim(
        identity,
        now=_TIME + timedelta(seconds=2),
        stale_after=timedelta(minutes=5),
    )
    assert retry_claim.accepted is not None
    _reserve_candidate(journal, retry_claim.accepted, now=_TIME + timedelta(seconds=2))
    journal.record_completed(
        retry_claim.accepted,
        _response("Recovered"),
        candidate_operation=_economics().operations[-1],
        completed_at=_TIME + timedelta(seconds=3),
    )
    _complete(
        journal,
        key="later",
        conversation="conversation-a",
        request=_request(
            ModelMessage(role="user", content="Retry me"),
            ModelMessage(role="assistant", assistant_action=AssistantAction(content="Recovered")),
            ModelMessage(role="user", content="Thanks"),
        ),
        output="Terminal",
        now=_TIME + timedelta(minutes=1),
    )

    result = _refresh(journal, store, _binding(CountingEmbedder()))

    assert len(result.snapshot_export.interactions[0].attempts) == 2
    assert result.retrieval.index.transition_count == 1
    assert result.retrieval.transitions[0].observation.content == "Thanks"


def test_failed_incomplete_interaction_is_skipped_for_later_real_observation(
    tmp_path: Path,
) -> None:
    """Continue within one lineage after an incomplete request omits the prior output.

    Args:
        tmp_path: Pytest-owned project directory.
    """
    journal, store = _journal_and_store(tmp_path)
    first = _complete(
        journal,
        key="first",
        conversation="conversation-a",
        request=_request(ModelMessage(role="user", content="First")),
        output="Observed answer",
        now=_TIME,
    )
    incomplete = _accept(
        journal,
        key="failed",
        conversation="conversation-a",
        request=_request(ModelMessage(role="user", content="Transient retry")),
        now=_TIME + timedelta(minutes=1),
    )
    _reserve_candidate(journal, incomplete, now=_TIME + timedelta(minutes=1))
    journal.record_failure(
        incomplete,
        StructuredFailure(
            code=FailureCode.PROVIDER,
            message="temporary",
            retryable=True,
            attribution=FailureAttribution.MODEL,
        ),
        failed_at=_TIME + timedelta(minutes=1, seconds=1),
    )
    _complete(
        journal,
        key="later",
        conversation="conversation-a",
        request=_request(
            ModelMessage(role="user", content="First"),
            ModelMessage(
                role="assistant",
                assistant_action=AssistantAction(content="Observed answer"),
            ),
            ModelMessage(role="user", content="Observation after failure"),
        ),
        output="Terminal",
        now=_TIME + timedelta(minutes=2),
    )

    result = _refresh(journal, store, _binding(CountingEmbedder()))

    assert result.retrieval.index.transition_count == 1
    transition = result.retrieval.transitions[0]
    assert transition.trace_id == first.interaction_id
    assert transition.observation.content == "Observation after failure"


def test_overlapping_acceptance_is_skipped_for_later_real_observation(tmp_path: Path) -> None:
    """Skip a request accepted before the preceding same-lineage response completed.

    Args:
        tmp_path: Pytest-owned project directory.
    """
    journal, store = _journal_and_store(tmp_path)
    first_request = _request(ModelMessage(role="user", content="First"))
    first = _accept(
        journal,
        key="first",
        conversation="conversation-a",
        request=first_request,
        now=_TIME,
    )
    _accept(
        journal,
        key="overlap",
        conversation="conversation-a",
        request=_request(ModelMessage(role="user", content="Concurrent request")),
        now=_TIME + timedelta(seconds=1),
    )
    _reserve_candidate(journal, first, now=_TIME)
    journal.record_completed(
        first,
        _response("Observed answer"),
        candidate_operation=_economics().operations[-1],
        completed_at=_TIME + timedelta(seconds=2),
    )
    _complete(
        journal,
        key="later",
        conversation="conversation-a",
        request=_request(
            ModelMessage(role="user", content="First"),
            ModelMessage(
                role="assistant",
                assistant_action=AssistantAction(content="Observed answer"),
            ),
            ModelMessage(role="user", content="Observation after overlap"),
        ),
        output="Terminal",
        now=_TIME + timedelta(seconds=3),
    )

    result = _refresh(journal, store, _binding(CountingEmbedder()))

    assert result.retrieval.index.transition_count == 1
    transition = result.retrieval.transitions[0]
    assert transition.trace_id == first.interaction_id
    assert transition.observation.content == "Observation after overlap"


def test_own_lineage_query_exclusion_removes_runtime_demonstration(tmp_path: Path) -> None:
    """Exclude demonstrations from the serving query's own production lineage.

    Args:
        tmp_path: Pytest-owned project directory.
    """
    journal, store, first = _two_turn_journal(tmp_path)
    binding = _binding(CountingEmbedder())
    result = _refresh(journal, store, binding)
    retriever = TraceRAGRetriever(
        load_rag_index(store, result.retrieval.index.rag_id),
        embedder=binding,
    )
    transition = result.retrieval.transitions[0]

    matches = retriever.retrieve(
        RAGQuery(
            task=transition.task,
            initial_context=transition.initial_context,
            action=RAGAction(kind="message", content="First answer"),
            excluded_lineage_ids=(first.identity.lineage_id,),
        )
    )

    assert matches == ()


def test_same_lineage_request_without_prior_output_fails_before_embed(tmp_path: Path) -> None:
    """Reject a claimed continuation that omits the prior output as source drift.

    Args:
        tmp_path: Pytest-owned project directory.
    """
    journal, store = _journal_and_store(tmp_path)
    _complete(
        journal,
        key="first",
        conversation="conversation-a",
        request=_request(ModelMessage(role="user", content="First")),
        output="Prior answer",
        now=_TIME,
    )
    _complete(
        journal,
        key="second",
        conversation="conversation-a",
        request=_request(ModelMessage(role="user", content="Unbound continuation")),
        output="Terminal",
        now=_TIME + timedelta(minutes=1),
    )
    client = CountingEmbedder()

    with pytest.raises(RuntimeTraceStitchingError, match="does not preserve"):
        _refresh(journal, store, _binding(client))

    assert client.calls == 0


def test_reservation_drift_and_nonfinite_vectors_fail_closed(tmp_path: Path) -> None:
    """Reject unknown price identity and non-finite vectors before completion.

    Args:
        tmp_path: Pytest-owned project directory.
    """
    journal, store, _first = _two_turn_journal(tmp_path)
    client = CountingEmbedder()
    binding = _binding(client)
    drifted = _reservation(binding).model_copy(update={"input_usd_per_million_tokens": 3.0})

    with pytest.raises(RuntimeRAGRefreshError, match="price differs"):
        refresh_runtime_trace_rag(
            journal,
            store,
            (),
            (),
            embedder=binding,
            embedding_reservation=drifted,
            maximum_embedding_cost_usd=1.0,
            created_at=_TIME + timedelta(hours=1),
            code_revision="test-revision",
        )
    assert client.calls == 0

    malicious = _binding(NonFiniteEmbedder())
    with pytest.raises(ValidationError, match="finite"):
        _refresh(journal, store, malicious)


def test_generated_import_is_rejected_before_embed(tmp_path: Path) -> None:
    """Keep generated or simulated provenance out of the refreshed retrieval corpus.

    Args:
        tmp_path: Pytest-owned project directory.
    """
    journal, store, _first = _two_turn_journal(tmp_path)
    generated = _persist_imported_trace(store, source_kind="generated")
    client = CountingEmbedder()
    binding = _binding(client)

    with pytest.raises(RuntimeRAGDatasetError, match="not observed real evidence"):
        refresh_runtime_trace_rag(
            journal,
            store,
            (artifact_input(generated.manifest),),
            (
                RAGLineageBinding(
                    trace_id=generated.traces[0].trace_id,
                    lineage_id="imported-lineage",
                    partition="fit",
                ),
            ),
            embedder=binding,
            embedding_reservation=_reservation(binding),
            maximum_embedding_cost_usd=1.0,
            created_at=_TIME + timedelta(hours=1),
            code_revision="test-revision",
        )

    assert client.calls == 0


def test_real_import_and_runtime_snapshot_build_one_new_union_and_index(tmp_path: Path) -> None:
    """Combine imported and routed observations into one immutable refreshed corpus.

    Args:
        tmp_path: Pytest-owned project directory.
    """
    journal, store, first = _two_turn_journal(tmp_path)
    built = build_task_set(
        TraceNormalizationResult(traces=(_imported_trace(source_kind="otlp"),), issues=()),
        store,
        created_at=_TIME,
        code_revision="test-revision",
        mining_spec=MiningSpec(fit_task_budget=1, held_out_task_budget=0),
    )
    imported = built.trace_dataset
    imported_input = artifact_input(imported.manifest)
    lineage_payload = load_task_set_lineage_bindings(store, built.task_set.task_set_id)
    imported_bindings = tuple(
        RAGLineageBinding(
            trace_id=item.trace_id,
            lineage_id=item.lineage_id,
            partition=item.partition,
        )
        for item in lineage_payload.bindings
    )
    binding = _binding(CountingEmbedder())

    result = refresh_runtime_trace_rag(
        journal,
        store,
        (imported_input,),
        imported_bindings,
        embedder=binding,
        embedding_reservation=_reservation(binding),
        maximum_embedding_cost_usd=1.0,
        created_at=_TIME + timedelta(hours=1),
        code_revision="test-revision",
    )

    assert result.dataset.dataset.inputs == tuple(
        sorted(
            (imported_input, artifact_input(result.snapshot_export.dataset_manifest)),
            key=lambda item: item.artifact_id,
        )
    )
    assert set(result.dataset.dataset.trace_ids) == {
        imported.traces[0].trace_id,
        *(trace.trace_id for trace in result.snapshot_export.traces),
    }
    assert {
        (transition.trace_id, transition.lineage_id) for transition in result.retrieval.transitions
    } == {
        (imported.traces[0].trace_id, imported_bindings[0].lineage_id),
        (first.interaction_id, first.identity.lineage_id),
    }


def test_lineage_assignment_change_creates_a_distinct_refresh_and_index(tmp_path: Path) -> None:
    """Bind replay identity to the fit and held-out assignment for every source trace.

    Args:
        tmp_path: Pytest-owned project directory.
    """
    journal, store, first = _two_turn_journal(tmp_path)
    imported = _persist_imported_trace(store, source_kind="otlp")
    imported_input = artifact_input(imported.manifest)
    client = CountingEmbedder()
    binding = _binding(client)
    fit_binding = RAGLineageBinding(
        trace_id=imported.traces[0].trace_id,
        lineage_id="imported-lineage",
        partition="fit",
    )
    held_out_binding = fit_binding.model_copy(update={"partition": "held_out"})

    fit_refresh = refresh_runtime_trace_rag(
        journal,
        store,
        (imported_input,),
        (fit_binding,),
        embedder=binding,
        embedding_reservation=_reservation(binding),
        maximum_embedding_cost_usd=1.0,
        created_at=_TIME + timedelta(hours=1),
        code_revision="test-revision",
    )
    held_out_refresh = refresh_runtime_trace_rag(
        journal,
        store,
        (imported_input,),
        (held_out_binding,),
        embedder=binding,
        embedding_reservation=_reservation(binding),
        maximum_embedding_cost_usd=1.0,
        created_at=_TIME + timedelta(hours=2),
        code_revision="test-revision",
    )

    assert client.calls == 2
    assert fit_refresh.refresh.refresh_id != held_out_refresh.refresh.refresh_id
    assert fit_refresh.retrieval.index.rag_id != held_out_refresh.retrieval.index.rag_id
    assert {item.trace_id for item in fit_refresh.retrieval.transitions} == {
        imported.traces[0].trace_id,
        first.interaction_id,
    }
    assert {item.trace_id for item in held_out_refresh.retrieval.transitions} == {
        first.interaction_id
    }


def test_retry_inclusive_cost_ceiling_blocks_before_embedding(tmp_path: Path) -> None:
    """Require the complete retry reservation to fit the ceiling before dispatch.

    Args:
        tmp_path: Pytest-owned project directory.
    """
    journal, store, _first = _two_turn_journal(tmp_path)
    client = CountingEmbedder()
    binding = _binding(client)

    with pytest.raises(RuntimeRAGRefreshError, match="exceeds the refresh cost ceiling"):
        refresh_runtime_trace_rag(
            journal,
            store,
            (),
            (),
            embedder=binding,
            embedding_reservation=_reservation(binding),
            maximum_embedding_cost_usd=0.01,
            created_at=_TIME + timedelta(hours=1),
            code_revision="test-revision",
        )

    assert client.calls == 0


def test_corrupt_snapshot_and_index_hash_fail_closed_on_replay(tmp_path: Path) -> None:
    """Reject altered snapshot and index bytes without another replay dispatch.

    Args:
        tmp_path: Pytest-owned project directory.
    """
    journal, store, _first = _two_turn_journal(tmp_path)
    client = CountingEmbedder()
    binding = _binding(client)
    result = _refresh(journal, store, binding)
    snapshot_path = (
        store.read(result.snapshot_export.snapshot.snapshot_id).directory / "interactions.jsonl"
    )
    original_snapshot = snapshot_path.read_bytes()
    snapshot_path.write_bytes(original_snapshot + b"{}\n")

    with pytest.raises(ArtifactCorruptionError, match="digest mismatch"):
        _refresh(journal, store, binding, created_at=_TIME + timedelta(days=1))
    assert client.calls == 1

    snapshot_path.write_bytes(original_snapshot)
    vectors_path = store.read(result.retrieval.index.rag_id).directory / "vectors.jsonl"
    vectors_path.write_bytes(vectors_path.read_bytes() + b"{}\n")
    with pytest.raises(ArtifactCorruptionError, match="digest mismatch"):
        load_runtime_rag_refresh(store, result.refresh.refresh_id, embedder=binding)


def test_plain_text_assistant_history_is_eligible_without_tool_inference(tmp_path: Path) -> None:
    """Match programmatic plain-text history to exact text-only completed output.

    Args:
        tmp_path: Pytest-owned project directory.
    """
    journal, store = _journal_and_store(tmp_path)
    first = _complete(
        journal,
        key="first",
        conversation="conversation-a",
        request=_request(ModelMessage(role="user", content="First")),
        output="Plain answer",
        now=_TIME,
    )
    _complete(
        journal,
        key="second",
        conversation="conversation-a",
        request=_request(
            ModelMessage(role="user", content="First"),
            ModelMessage(role="assistant", content="Plain answer"),
            ModelMessage(role="user", content="Observed"),
        ),
        output="Terminal",
        now=_TIME + timedelta(minutes=1),
    )

    result = _refresh(journal, store, _binding(CountingEmbedder()))

    assert result.retrieval.transitions[0].trace_id == first.interaction_id
    assert result.retrieval.transitions[0].observation.content == "Observed"


def test_tool_result_is_paired_only_with_its_exact_call_id(tmp_path: Path) -> None:
    """Admit a routed tool call only through its later matching real result.

    Args:
        tmp_path: Pytest-owned project directory.
    """
    journal, store = _journal_and_store(tmp_path)
    first_request = _request(ModelMessage(role="user", content="Look it up"))
    first = _accept(
        journal,
        key="first",
        conversation="conversation-a",
        request=first_request,
        now=_TIME,
    )
    action = AssistantAction(
        tool_calls=(ToolCall(call_id="call-1", name="lookup", arguments={"id": "A-1"}),)
    )
    _reserve_candidate(journal, first, now=_TIME)
    journal.record_completed(
        first,
        ModelResponse(
            output=action,
            model=_model(),
            economics=OperationEconomics(
                usage=Usage(input_tokens=8, output_tokens=3, cached_input_tokens=0)
            ),
        ),
        candidate_operation=_economics().operations[-1],
        completed_at=_TIME + timedelta(seconds=1),
    )
    _complete(
        journal,
        key="second",
        conversation="conversation-a",
        request=_request(
            ModelMessage(role="user", content="Look it up"),
            ModelMessage(role="assistant", assistant_action=action),
            ModelMessage(role="tool", content="record A-1", tool_call_id="call-1"),
            ModelMessage(role="user", content="Continue"),
        ),
        output="Terminal",
        now=_TIME + timedelta(minutes=1),
    )

    result = _refresh(journal, store, _binding(CountingEmbedder()))

    transition = result.retrieval.transitions[0]
    assert transition.action.kind == "tool_call"
    assert transition.action.tool_name == "lookup"
    assert transition.observation.kind == "tool_result"
    assert transition.observation.content == "record A-1"


def _persist_imported_trace(
    store: ArtifactStore,
    *,
    source_kind: Literal["file", "otlp", "production", "simulation", "manual", "generated"],
) -> PersistedTraceDataset:
    """Persist one small imported trace dataset with selectable source provenance.

    Args:
        store: Project artifact store receiving the imported dataset.
        source_kind: SourceIdentity kind used for envelope and trace provenance.

    Returns:
        Completed canonical imported dataset.
    """
    trace = _imported_trace(source_kind=source_kind)
    return persist_trace_dataset(
        TraceNormalizationResult(traces=(trace,), issues=()),
        store,
        created_at=_TIME - timedelta(minutes=1),
        code_revision="import-revision",
    )


def _imported_trace(
    *,
    source_kind: Literal["file", "otlp", "production", "simulation", "manual", "generated"],
) -> Trace:
    """Create one imported trace with selectable source provenance.

    Args:
        source_kind: SourceIdentity kind used for trace provenance.

    Returns:
        Canonical imported trace with one observed transition.
    """
    source = TraceSource(
        identity=SourceIdentity(
            kind=source_kind,
            source_id=f"{source_kind}-fixture",
            sha256="d" * 64,
        ),
        semantic_convention_version="1.0",
    )
    return Trace(
        trace_id=f"{source_kind}-trace",
        conversation_id="imported-lineage",
        task="Imported task",
        spans=(
            TraceSpan(
                span_id="action",
                name="model",
                started_at=_TIME - timedelta(minutes=2),
                ended_at=_TIME - timedelta(minutes=1, seconds=59),
                attributes={
                    "gen_ai.output.messages": [{"role": "assistant", "content": "Imported answer"}]
                },
            ),
            TraceSpan(
                span_id="observation",
                parent_span_id="action",
                name="user",
                started_at=_TIME - timedelta(minutes=1, seconds=58),
                ended_at=_TIME - timedelta(minutes=1, seconds=58),
                attributes={
                    "gen_ai.input.messages": [{"role": "user", "content": "Imported followup"}]
                },
            ),
        ),
        outcome=TraceOutcome(status="success"),
        source=source,
    )
