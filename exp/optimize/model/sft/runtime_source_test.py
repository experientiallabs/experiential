"""Behavior and provenance tests for routed-interaction SFT dataset sources."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    FailureAttribution,
    FailureCode,
    SourceIdentity,
    StructuredFailure,
)
from exp.common.models import (
    AssistantAction,
    ModelFinishReason,
    ModelMessage,
    ToolCall,
)
from exp.common.project import ProjectConfig, ProjectStore
from exp.optimize.model.sft import (
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
from exp.runtime.router import RuntimeInteractionJournal
from exp.runtime.router.snapshot import seal_runtime_trace_snapshot
from exp.simulation.retrieval.refresh_test import _accept, _complete, _request

_TIME = datetime(2026, 8, 14, tzinfo=UTC)


def _store(tmp_path: Path) -> ProjectStore:
    """Create one initialized project store and its matching runtime journal.

    Args:
        tmp_path: Pytest-owned state directory.

    Returns:
        Initialized project-local immutable store.
    """
    store = ProjectStore(tmp_path / ".exp", "support-agent")
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
