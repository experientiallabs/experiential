"""Behavior and provenance tests for routed-interaction SFT dataset sources."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from wmo.common.core.artifacts import (
    ArtifactEnvelope,
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
from wmo.common.project import ProjectConfig, ProjectStore
from wmo.common.routing import RouterFeatureExtractor, RoutingDecision
from wmo.optimize.model.sft import (
    AssistantActionEvent,
    RuntimeInteractionExampleSource,
    RuntimeSFTSource,
    SFTBuildError,
    SFTBuildSpec,
    SFTDatasetArtifact,
    ToolEvent,
    build_sft_dataset,
    load_verified_sft_dataset,
    write_sft_dataset,
)
from wmo.runtime.router import RuntimeAcceptedEvent, RuntimeInteractionJournal
from wmo.runtime.router.economics import (
    BillingSourceEconomics,
    RoutedProviderComponent,
    RoutedProviderOperation,
    RoutedSpendDisposition,
)
from wmo.runtime.router.journal import RuntimeAcceptance, _interaction_identity
from wmo.runtime.router.snapshot import seal_runtime_trace_snapshot

_DIGEST = "a" * 64
_TIME = datetime(2026, 8, 14, tzinfo=UTC)


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
        candidate_operation=_operation(2, RoutedProviderComponent.SELECTED_CANDIDATE),
        completed_at=now + timedelta(seconds=1),
    )
    return accepted


def _store(tmp_path: Path) -> ProjectStore:
    """Create one initialized project store and its matching runtime journal.

    Args:
        tmp_path: Pytest-owned state directory.

    Returns:
        Initialized project-local immutable store.
    """
    store = ProjectStore(tmp_path / ".wmo", "support-agent")
    store.initialize(ProjectConfig(project_id="support-agent"))
    return store


def _snapshot(
    store: ProjectStore,
    journal: RuntimeInteractionJournal,
    *,
    created_at: datetime = _TIME + timedelta(hours=1),
) -> RuntimeSFTSource:
    """Seal the current journal and return its W12 runtime source pointer.

    Args:
        store: Store receiving the immutable runtime snapshot.
        journal: Same-project routed interaction journal.
        created_at: Snapshot materialization time.

    Returns:
        Runtime SFT source naming the sealed prefix.
    """
    export = seal_runtime_trace_snapshot(
        journal,
        store.artifacts,
        created_at=created_at,
        code_revision="runtime-source-test",
    )
    return RuntimeSFTSource(snapshot_id=export.snapshot.snapshot_id)


def _build(
    store: ProjectStore,
    *runtime_sources: RuntimeSFTSource,
    created_at: datetime = _TIME + timedelta(hours=2),
) -> SFTDatasetArtifact:
    """Build one W12 dataset from routed snapshots only.

    Args:
        store: Store owning all supplied snapshots.
        runtime_sources: Exact routed snapshot pointers.
        created_at: Dataset materialization time.

    Returns:
        Complete in-memory W12 dataset artifact.
    """
    return build_sft_dataset(
        store=store,
        production_sources=(),
        teacher_sources=(),
        runtime_sources=runtime_sources,
        spec=SFTBuildSpec(held_out_fraction=0.5, representative_sample_count=2),
        created_at=created_at,
        code_revision="runtime-source-test",
    )


def test_completed_multiplicity_is_retained_without_quality_filter(
    tmp_path: Path,
) -> None:
    """Keep every completed interaction while excluding failed and open attempts.

    Args:
        tmp_path: Pytest-owned state directory.
    """
    store = _store(tmp_path)
    journal = RuntimeInteractionJournal(store.paths)
    request = _request(ModelMessage(role="user", content="Repeat this"))
    target = AssistantAction(
        content="Calling lookup",
        tool_calls=(ToolCall(call_id="call-1", name="lookup", arguments={"id": "A-1"}),),
    )
    first = _complete(
        journal,
        key="first",
        conversation="conversation-a",
        request=request,
        output=target,
        now=_TIME,
        finish_reason=ModelFinishReason.LENGTH,
    )
    second = _complete(
        journal,
        key="second",
        conversation="conversation-a",
        request=request,
        output=target,
        now=_TIME + timedelta(minutes=1),
    )
    failed = _accept(
        journal,
        key="failed",
        conversation="conversation-a",
        request=_request(ModelMessage(role="user", content="Provider failure")),
        now=_TIME + timedelta(minutes=2),
    )
    journal.record_failure(
        failed,
        StructuredFailure(
            code=FailureCode.PROVIDER,
            message="permanent provider failure",
            retryable=False,
            attribution=FailureAttribution.MODEL,
        ),
        failed_at=_TIME + timedelta(minutes=2, seconds=1),
    )
    open_event = _accept(
        journal,
        key="open",
        conversation="conversation-a",
        request=_request(ModelMessage(role="user", content="Disconnected")),
        now=_TIME + timedelta(minutes=3),
    )

    artifact = _build(store, _snapshot(store, journal))

    assert len(artifact.rows) == 2
    assert len({row.example.example_id for row in artifact.rows}) == 2
    assert len({row.fingerprint for row in artifact.rows}) == 1
    runtime_example_sources = tuple(
        row.example.source
        for row in artifact.rows
        if isinstance(row.example.source, RuntimeInteractionExampleSource)
    )
    assert len(runtime_example_sources) == 2
    assert {source.interaction_id for source in runtime_example_sources} == {
        first.interaction_id,
        second.interaction_id,
    }
    assert all(row.example.target == target for row in artifact.rows)
    assert {row.example.leakage_group_id for row in artifact.rows} == {
        artifact.rows[0].example.leakage_group_id
    }
    assert artifact.dataset.acceptance_rule_ids == ()
    assert artifact.dataset.acceptance_evidence_ids == ()
    assert artifact.inspection.source_count == 4
    assert artifact.inspection.accepted_source_count == 2
    assert {(item.source_id, item.reason) for item in artifact.inspection.exclusions} == {
        (failed.interaction_id, "runtime_interaction_failed"),
        (open_event.interaction_id, "runtime_interaction_incomplete"),
    }


def test_request_history_and_completed_tool_action_are_preserved(tmp_path: Path) -> None:
    """Retain visible assistant tool calls and matching tool results in canonical rows.

    Args:
        tmp_path: Pytest-owned state directory.
    """
    store = _store(tmp_path)
    journal = RuntimeInteractionJournal(store.paths)
    prior_action = AssistantAction(
        tool_calls=(ToolCall(call_id="prior-call", name="lookup", arguments={"id": "A-1"}),)
    )
    target = AssistantAction(
        content="I found it",
        tool_calls=(ToolCall(call_id="next-call", name="notify", arguments={"ok": True}),),
    )
    accepted = _complete(
        journal,
        key="tool-turn",
        conversation="conversation-tools",
        request=_request(
            ModelMessage(role="system", content="Use tools carefully"),
            ModelMessage(role="user", content="Find A-1"),
            ModelMessage(role="assistant", assistant_action=prior_action),
            ModelMessage(role="tool", content="record A-1", tool_call_id="prior-call"),
            ModelMessage(role="user", content="Now notify me"),
        ),
        output=target,
        now=_TIME,
    )

    artifact = _build(store, _snapshot(store, journal))

    assert len(artifact.rows) == 1
    row = artifact.rows[0].example
    assert row.target == target
    assert row.task == "Now notify me"
    assert isinstance(row.source, RuntimeInteractionExampleSource)
    assert row.source.interaction_id == accepted.interaction_id
    assert tuple(event for event in row.history if isinstance(event, AssistantActionEvent)) == (
        AssistantActionEvent(action=prior_action, approved=True),
    )
    assert tuple(event for event in row.history if isinstance(event, ToolEvent)) == (
        ToolEvent(
            tool_call_id="prior-call",
            content="record A-1",
            tool_name="lookup",
        ),
    )


def test_identical_rows_across_lineages_share_one_partition_component(tmp_path: Path) -> None:
    """Link duplicate fingerprints across lineages without dropping their multiplicity.

    Args:
        tmp_path: Pytest-owned state directory.
    """
    store = _store(tmp_path)
    journal = RuntimeInteractionJournal(store.paths)
    request = _request(ModelMessage(role="user", content="Same request"))
    target = AssistantAction(content="Same response")
    _complete(
        journal,
        key="lineage-a",
        conversation="conversation-a",
        request=request,
        output=target,
        now=_TIME,
    )
    _complete(
        journal,
        key="lineage-b",
        conversation="conversation-b",
        request=request,
        output=target,
        now=_TIME + timedelta(minutes=1),
    )

    artifact = _build(store, _snapshot(store, journal))

    assert len(artifact.rows) == 2
    assert len({row.fingerprint for row in artifact.rows}) == 1
    assert len({row.example.leakage_group_id for row in artifact.rows}) == 2
    assert len(artifact.partitions) == 1
    assert set(artifact.partitions[0].leakage_group_ids) == {
        row.example.leakage_group_id for row in artifact.rows
    }
    assert len({row.partition for row in artifact.rows}) == 1


def test_runtime_dataset_replay_is_exact_and_overlapping_snapshots_are_rejected(
    tmp_path: Path,
) -> None:
    """Reuse one exact W12 artifact and reject duplicate interactions across prefixes.

    Args:
        tmp_path: Pytest-owned state directory.
    """
    store = _store(tmp_path)
    journal = RuntimeInteractionJournal(store.paths)
    _complete(
        journal,
        key="first",
        conversation="conversation-a",
        request=_request(ModelMessage(role="user", content="First")),
        output=AssistantAction(content="Answer"),
        now=_TIME,
    )
    first_source = _snapshot(store, journal)
    first_build = _build(store, first_source)
    persisted = write_sft_dataset(store, first_build)
    replay = write_sft_dataset(
        store,
        _build(store, first_source, created_at=_TIME + timedelta(days=1)),
    )

    assert replay == persisted
    assert load_verified_sft_dataset(store, persisted.dataset.dataset_id) == persisted

    _complete(
        journal,
        key="second",
        conversation="conversation-a",
        request=_request(ModelMessage(role="user", content="Second")),
        output=AssistantAction(content="Another answer"),
        now=_TIME + timedelta(minutes=1),
    )
    longer_source = _snapshot(
        store,
        journal,
        created_at=_TIME + timedelta(hours=3),
    )
    longer_build = _build(store, longer_source)
    assert longer_build.dataset.dataset_id != persisted.dataset.dataset_id
    with pytest.raises(SFTBuildError, match="repeats source runtime_interaction"):
        _build(store, first_source, longer_source)


def test_generated_or_non_snapshot_artifact_cannot_enter_runtime_sft(tmp_path: Path) -> None:
    """Reject a generated artifact before it can become a routed SFT source.

    Args:
        tmp_path: Pytest-owned state directory.
    """
    store = _store(tmp_path)
    generated_id = "generated-runtime-source"
    envelope = ArtifactEnvelope(
        schema_version=1,
        created_at=_TIME,
        inputs=(),
        code_revision="runtime-source-test",
        source=SourceIdentity(kind="simulation", source_id="generated", sha256="c" * 64),
    )
    store.artifacts.write(
        artifact_id=generated_id,
        artifact_type="simulation-output",
        envelope=envelope,
        files={"generated.json": b"{}"},
    )

    with pytest.raises(SFTBuildError, match="not verified production evidence"):
        _build(store, RuntimeSFTSource(snapshot_id=generated_id))
