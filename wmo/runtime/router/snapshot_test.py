"""Adversarial coverage for immutable runtime trace snapshots."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from wmo.common.core.artifacts import (
    ArtifactInput,
    FailureAttribution,
    FailureCode,
    StructuredFailure,
    canonical_json_bytes,
    stable_id,
)
from wmo.common.models import (
    AssistantAction,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelSnapshot,
    OperationEconomics,
    Usage,
)
from wmo.common.project import (
    ArtifactCorruptionError,
    ArtifactStore,
    ArtifactStoreError,
    ProjectPaths,
)
from wmo.common.routing import RouterFeatureExtractor, RoutingDecision
from wmo.common.traces import load_trace_dataset
from wmo.runtime.router.journal import (
    RuntimeAcceptance,
    RuntimeAcceptedEvent,
    RuntimeInteractionJournal,
    RuntimeJournalError,
    _interaction_identity,
)
from wmo.runtime.router.snapshot import (
    RuntimeTraceAttempt,
    RuntimeTraceSnapshotError,
    load_runtime_trace_snapshot,
    seal_runtime_trace_snapshot,
)

_DIGEST = "a" * 64
_TIME = datetime(2026, 8, 13, tzinfo=UTC)


def _snapshot(*, model_id: str = "gpt-test") -> ModelSnapshot:
    """Build a frozen test model snapshot.

    Args:
        model_id: Provider model identity to preserve in the fixture.

    Returns:
        Deterministic model snapshot with fixed capability and connection digests.
    """
    return ModelSnapshot(
        provider="openai",
        model_id=model_id,
        capabilities_sha256=_DIGEST,
        connection_sha256="b" * 64,
    )


def _request(content: str = "Help me") -> ModelRequest:
    """Build a deterministic two-message model request.

    Args:
        content: User message included after the fixed system instruction.

    Returns:
        Provider-neutral request used by runtime snapshot fixtures.
    """
    return ModelRequest(
        messages=(
            ModelMessage(role="system", content="You are helpful."),
            ModelMessage(role="user", content=content),
        )
    )


def _response(*, model: ModelSnapshot | None = None, content: str = "Done") -> ModelResponse:
    """Build a deterministic successful model response.

    Args:
        model: Optional provider-resolved model identity.
        content: Assistant text returned by the fixture.

    Returns:
        Successful response with fixed token economics.
    """
    return ModelResponse(
        output=AssistantAction(content=content),
        model=model or _snapshot(),
        economics=OperationEconomics(
            usage=Usage(input_tokens=12, output_tokens=3, cached_input_tokens=2)
        ),
    )


def _decision(lineage_id: str, request: ModelRequest) -> RoutingDecision:
    """Build a content-addressed routing decision for one request lineage.

    Args:
        lineage_id: Stable conversation lineage used by the runtime journal.
        request: Request whose extracted feature defines the decision identity.

    Returns:
        Deterministic candidate selection with canonical content identity.
    """
    episode_sha256 = hashlib.sha256(lineage_id.encode("utf-8"), usedforsecurity=False).hexdigest()
    feature = RouterFeatureExtractor().from_request(request)
    material = {
        "policy_id": "router-policy-a",
        "policy_sha256": _DIGEST,
        "request_sha256": hashlib.sha256(
            feature.encode("utf-8"), usedforsecurity=False
        ).hexdigest(),
        "episode_id_sha256": episode_sha256,
        "selected_alias": "candidate-a",
        "baseline_alias": "candidate-a",
        "neighbor_count": 8,
        "paired_count": 8,
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
    """Build an isolated runtime journal and artifact store."""
    paths = ProjectPaths(root=tmp_path / ".wmo", project_id="support-agent")
    return RuntimeInteractionJournal(paths), ArtifactStore(paths)


def _accept(
    journal: RuntimeInteractionJournal,
    *,
    key: str,
    request: ModelRequest | None = None,
    conversation_id: str | None = None,
    now: datetime = _TIME,
) -> RuntimeAcceptedEvent:
    """Accept one deterministic interaction into the runtime journal.

    Args:
        journal: Mutable runtime journal receiving the acceptance.
        key: Caller idempotency key used only to derive the interaction identity.
        request: Optional request override.
        conversation_id: Optional stable routed conversation identity.
        now: Acceptance timestamp.

    Returns:
        Durable accepted event produced by the journal claim.
    """
    routed_request = request or _request()
    identity = _interaction_identity(
        journal.project_id,
        key,
        routed_request,
        conversation_id,
    )
    decision = _decision(identity.lineage_id, routed_request)
    claim = journal.claim(
        identity,
        RuntimeAcceptance(
            decision=decision,
            selected_alias=decision.selected_alias,
            selected_model=_snapshot(),
            policy_input=ArtifactInput(
                artifact_id=decision.policy_id,
                sha256=decision.policy_sha256,
            ),
        ),
        now=now,
        stale_after=timedelta(minutes=5),
    )
    assert claim.accepted is not None
    return claim.accepted


def _complete(
    journal: RuntimeInteractionJournal,
    *,
    key: str,
    request: ModelRequest | None = None,
    conversation_id: str | None = None,
    now: datetime = _TIME,
    response: ModelResponse | None = None,
) -> RuntimeAcceptedEvent:
    """Accept and complete one deterministic journal interaction.

    Args:
        journal: Mutable runtime journal receiving both events.
        key: Caller idempotency key used to derive the interaction identity.
        request: Optional request override.
        conversation_id: Optional stable routed conversation identity.
        now: Acceptance timestamp used to derive completion time.
        response: Optional provider result override.

    Returns:
        Accepted event named by the recorded completion.
    """
    accepted = _accept(
        journal,
        key=key,
        request=request,
        conversation_id=conversation_id,
        now=now,
    )
    journal.record_completed(
        accepted,
        response or _response(),
        completed_at=now + timedelta(seconds=1),
    )
    return accepted


def _artifact_bytes(store: ArtifactStore, artifact_id: str) -> dict[str, bytes]:
    """Read every file in an artifact directory as stable relative-path bytes.

    Args:
        store: Project artifact store containing the target artifact.
        artifact_id: Immutable artifact identity to inspect.

    Returns:
        Mapping from relative file paths to exact bytes.
    """
    directory = store.read(artifact_id).directory
    return {
        path.relative_to(directory).as_posix(): path.read_bytes()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def test_identical_replay_and_later_append_preserve_old_snapshot(tmp_path: Path) -> None:
    """Prove exact replay and later append preserve the original snapshot.

    The same prefix reuses identical artifact bytes, while a newly appended journal event creates
    distinct immutable snapshot and dataset siblings without changing the first export.
    """
    journal, store = _journal_and_store(tmp_path)
    _complete(journal, key="first-key", conversation_id="conversation-a")

    first = seal_runtime_trace_snapshot(
        journal,
        store,
        created_at=_TIME + timedelta(minutes=1),
        code_revision="test-revision",
    )
    first_snapshot_bytes = _artifact_bytes(store, first.snapshot.snapshot_id)
    first_dataset_bytes = _artifact_bytes(store, first.dataset.dataset_id)
    replay = seal_runtime_trace_snapshot(
        journal,
        store,
        created_at=_TIME + timedelta(days=1),
        code_revision="test-revision",
    )

    assert replay == first
    assert _artifact_bytes(store, first.snapshot.snapshot_id) == first_snapshot_bytes
    assert _artifact_bytes(store, first.dataset.dataset_id) == first_dataset_bytes

    _complete(
        journal,
        key="second-key",
        conversation_id="conversation-b",
        now=_TIME + timedelta(minutes=2),
    )
    later = seal_runtime_trace_snapshot(
        journal,
        store,
        created_at=_TIME + timedelta(minutes=3),
        code_revision="test-revision",
    )
    explicit_old = seal_runtime_trace_snapshot(
        journal,
        store,
        created_at=_TIME + timedelta(days=2),
        code_revision="test-revision",
        last_ordinal=first.snapshot.last_ordinal,
    )

    assert explicit_old == first
    assert later.snapshot.snapshot_id != first.snapshot.snapshot_id
    assert later.dataset.dataset_id != first.dataset.dataset_id
    assert later.snapshot.interaction_ids == (
        first.snapshot.interaction_ids[0],
        later.snapshot.interaction_ids[1],
    )
    assert later.dataset.trace_ids == later.snapshot.interaction_ids
    assert _artifact_bytes(store, first.snapshot.snapshot_id) == first_snapshot_bytes
    assert _artifact_bytes(store, first.dataset.dataset_id) == first_dataset_bytes


def test_equivalent_prefixes_produce_identical_artifact_files(tmp_path: Path) -> None:
    """Prove equivalent prefixes produce identical artifacts across project roots.

    Independently materialized journals with equal events yield the same content identities and
    byte-for-byte artifact files.
    """
    first_journal, first_store = _journal_and_store(tmp_path / "first")
    second_journal, second_store = _journal_and_store(tmp_path / "second")
    for journal in (first_journal, second_journal):
        _complete(journal, key="same-key", conversation_id="same-lineage")

    first = seal_runtime_trace_snapshot(
        first_journal,
        first_store,
        created_at=_TIME + timedelta(minutes=1),
        code_revision="test-revision",
    )
    second = seal_runtime_trace_snapshot(
        second_journal,
        second_store,
        created_at=_TIME + timedelta(minutes=1),
        code_revision="test-revision",
    )

    assert second.snapshot == first.snapshot
    assert second.dataset == first.dataset
    assert _artifact_bytes(first_store, first.snapshot.snapshot_id) == _artifact_bytes(
        second_store, second.snapshot.snapshot_id
    )
    assert _artifact_bytes(first_store, first.dataset.dataset_id) == _artifact_bytes(
        second_store, second.dataset.dataset_id
    )


def test_code_revision_drift_creates_distinct_snapshot_and_dataset_ids(tmp_path: Path) -> None:
    """Prove revision-sensitive envelopes receive distinct content identities.

    Sealing the same journal prefix from another code revision produces immutable snapshot and
    dataset siblings instead of colliding with the original envelopes.
    """
    journal, store = _journal_and_store(tmp_path)
    _complete(journal, key="revision-key")

    first = seal_runtime_trace_snapshot(
        journal,
        store,
        created_at=_TIME + timedelta(minutes=1),
        code_revision="revision-a",
    )
    second = seal_runtime_trace_snapshot(
        journal,
        store,
        created_at=_TIME + timedelta(minutes=2),
        code_revision="revision-b",
    )

    assert first.snapshot.snapshot_id != second.snapshot.snapshot_id
    assert first.dataset.dataset_id != second.dataset.dataset_id
    assert first.snapshot.prefix_sha256 == second.snapshot.prefix_sha256
    assert first.dataset.trace_ids == second.dataset.trace_ids
    assert first.traces[0].model_dump(exclude={"source"}) == second.traces[0].model_dump(
        exclude={"source"}
    )
    assert first.traces[0].source != second.traces[0].source


@pytest.mark.parametrize("mutation", ["inputs", "source-id", "schema-version", "path"])
def test_loader_rejects_forged_snapshot_provenance(
    tmp_path: Path,
    mutation: str,
) -> None:
    """Prove verified artifact files cannot attach unbound snapshot provenance.

    Args:
        tmp_path: Isolated project root.
        mutation: Shared envelope and manifest field rewritten with valid digests.
    """
    journal, store = _journal_and_store(tmp_path)
    _complete(journal, key="forged-provenance-key")
    exported = seal_runtime_trace_snapshot(
        journal,
        store,
        created_at=_TIME + timedelta(minutes=1),
        code_revision="test-revision",
    )
    directory = store.read(exported.snapshot.snapshot_id).directory
    snapshot_path = directory / "runtime-trace-snapshot.json"
    manifest_path = directory / "manifest.json"
    snapshot = json.loads(snapshot_path.read_bytes())
    manifest = json.loads(manifest_path.read_bytes())
    if mutation == "inputs":
        forged_inputs = [
            ArtifactInput(artifact_id="forged-input", sha256="f" * 64).model_dump(mode="json")
        ]
        snapshot["inputs"] = forged_inputs
        manifest["inputs"] = forged_inputs
    elif mutation == "source-id":
        snapshot["source"]["source_id"] = "forged/runtime/interactions"
        manifest["source"]["source_id"] = "forged/runtime/interactions"
    elif mutation == "schema-version":
        snapshot["schema_version"] = 2
        manifest["schema_version"] = 2
    else:
        snapshot["interactions_path"] = "forged-interactions.jsonl"
    snapshot_payload = canonical_json_bytes(snapshot)
    snapshot_path.write_bytes(snapshot_payload)
    snapshot_entry = next(
        entry for entry in manifest["files"] if entry["path"] == "runtime-trace-snapshot.json"
    )
    snapshot_entry["sha256"] = hashlib.sha256(snapshot_payload).hexdigest()
    snapshot_entry["size_bytes"] = len(snapshot_payload)
    manifest_path.write_bytes(canonical_json_bytes(manifest))

    with pytest.raises(ArtifactCorruptionError, match="invalid envelope"):
        load_runtime_trace_snapshot(store, exported.snapshot.snapshot_id)


@pytest.mark.parametrize("mutation", ["schema-version", "path"])
def test_loader_rejects_reidentified_noncanonical_snapshot_constants(
    tmp_path: Path,
    mutation: str,
) -> None:
    """Prove a recomputed ID cannot legitimize unsupported snapshot constants.

    Args:
        tmp_path: Isolated project root.
        mutation: Schema or interactions-path constant rewritten before identity recomputation.
    """
    journal, store = _journal_and_store(tmp_path)
    _complete(journal, key="reidentified-constant-key")
    exported = seal_runtime_trace_snapshot(
        journal,
        store,
        created_at=_TIME + timedelta(minutes=1),
        code_revision="test-revision",
    )
    directory = store.read(exported.snapshot.snapshot_id).directory
    snapshot_path = directory / "runtime-trace-snapshot.json"
    manifest_path = directory / "manifest.json"
    snapshot = json.loads(snapshot_path.read_bytes())
    manifest = json.loads(manifest_path.read_bytes())
    if mutation == "schema-version":
        snapshot["schema_version"] = 2
        manifest["schema_version"] = 2
    else:
        snapshot["interactions_path"] = "alternate-interactions.jsonl"
    identity_material = {
        key: value for key, value in snapshot.items() if key not in {"created_at", "snapshot_id"}
    }
    forged_id = stable_id("runtime-trace-snapshot", identity_material)
    snapshot["snapshot_id"] = forged_id
    manifest["artifact_id"] = forged_id
    snapshot_payload = canonical_json_bytes(snapshot)
    snapshot_path.write_bytes(snapshot_payload)
    snapshot_entry = next(
        entry for entry in manifest["files"] if entry["path"] == "runtime-trace-snapshot.json"
    )
    snapshot_entry["sha256"] = hashlib.sha256(snapshot_payload).hexdigest()
    snapshot_entry["size_bytes"] = len(snapshot_payload)
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    directory.rename(directory.parent / forged_id)

    with pytest.raises(ArtifactCorruptionError, match="invalid envelope"):
        load_runtime_trace_snapshot(store, forged_id)


@pytest.mark.parametrize(
    "mutation",
    [
        "schema-version",
        "traces-path",
        "semantic-convention",
        "inputs",
        "code-revision",
        "trace-ids",
        "source",
        "traces-digest",
        "issues",
    ],
)
def test_dataset_replay_rejects_envelope_fields_outside_its_identity(
    tmp_path: Path,
    mutation: str,
) -> None:
    """Prove every replay-sensitive dataset field is bound to its content identity.

    Args:
        tmp_path: Isolated project root.
        mutation: Valid envelope field rewrite applied with updated file and manifest digests.
    """
    journal, store = _journal_and_store(tmp_path)
    _complete(journal, key="forged-dataset-key")
    exported = seal_runtime_trace_snapshot(
        journal,
        store,
        created_at=_TIME + timedelta(minutes=1),
        code_revision="test-revision",
    )
    directory = store.read(exported.dataset.dataset_id).directory
    dataset_path = directory / "trace-dataset.json"
    manifest_path = directory / "manifest.json"
    dataset = json.loads(dataset_path.read_bytes())
    manifest = json.loads(manifest_path.read_bytes())
    if mutation == "schema-version":
        dataset["schema_version"] = 2
        manifest["schema_version"] = 2
    elif mutation == "traces-path":
        dataset["traces_path"] = "alternate-traces.jsonl"
    elif mutation == "semantic-convention":
        dataset["semantic_convention_version"] = "wmo.runtime.router.v2"
    elif mutation == "inputs":
        forged_inputs = [
            ArtifactInput(artifact_id="forged-input", sha256="f" * 64).model_dump(mode="json")
        ]
        dataset["inputs"] = forged_inputs
        manifest["inputs"] = forged_inputs
    elif mutation == "code-revision":
        dataset["code_revision"] = "forged-revision"
        manifest["code_revision"] = "forged-revision"
    elif mutation == "trace-ids":
        dataset["trace_ids"] = ["forged-trace"]
    elif mutation == "source":
        dataset["source"]["source_id"] = "forged-snapshot"
        manifest["source"]["source_id"] = "forged-snapshot"
    elif mutation == "traces-digest":
        dataset["traces_sha256"] = "f" * 64
    else:
        dataset["issues_path"] = "issues.jsonl"
        dataset["issues_sha256"] = "f" * 64
        dataset["invalid_trace_count"] = 1
    dataset_payload = canonical_json_bytes(dataset)
    dataset_path.write_bytes(dataset_payload)
    dataset_entry = next(
        entry for entry in manifest["files"] if entry["path"] == "trace-dataset.json"
    )
    dataset_entry["sha256"] = hashlib.sha256(dataset_payload).hexdigest()
    dataset_entry["size_bytes"] = len(dataset_payload)
    manifest_path.write_bytes(canonical_json_bytes(manifest))

    with pytest.raises(RuntimeTraceSnapshotError, match="differs from"):
        seal_runtime_trace_snapshot(
            journal,
            store,
            created_at=_TIME + timedelta(minutes=2),
            code_revision="test-revision",
        )


def test_retry_success_is_one_target_and_failures_remain_prefix_provenance(
    tmp_path: Path,
) -> None:
    """Prove retry failures remain provenance while one success becomes the target.

    The snapshot retains both failed and completed attempts, but normalized traces contain exactly
    the eventual completed response for the logical interaction.
    """
    journal, store = _journal_and_store(tmp_path)
    request = _request("Reset my password")
    first = _accept(
        journal,
        key="retry-key",
        request=request,
        conversation_id="customer-thread",
    )
    journal.record_failure(
        first,
        StructuredFailure(
            code=FailureCode.TIMEOUT,
            message="routed model provider attempt failed",
            retryable=True,
            attribution=FailureAttribution.MODEL,
        ),
        failed_at=_TIME + timedelta(seconds=1),
    )
    identity = _interaction_identity(
        journal.project_id,
        "retry-key",
        request,
        "customer-thread",
    )
    retry = journal.claim(
        identity,
        None,
        now=_TIME + timedelta(seconds=2),
        stale_after=timedelta(minutes=5),
    )
    assert retry.accepted is not None
    journal.record_completed(
        retry.accepted,
        _response(content="Use the reset link."),
        completed_at=_TIME + timedelta(seconds=3),
    )

    permanent = _accept(
        journal,
        key="permanent-key",
        now=_TIME + timedelta(seconds=4),
    )
    journal.record_failure(
        permanent,
        StructuredFailure(
            code=FailureCode.PROVIDER,
            message="routed model provider attempt failed permanently",
            retryable=False,
            attribution=FailureAttribution.MODEL,
        ),
        failed_at=_TIME + timedelta(seconds=5),
    )

    exported = seal_runtime_trace_snapshot(
        journal,
        store,
        created_at=_TIME + timedelta(minutes=1),
        code_revision="test-revision",
    )

    assert exported.snapshot.completed_target_count == 1
    assert exported.snapshot.failed_attempt_count == 2
    assert len(exported.snapshot.interaction_ids) == 2
    assert exported.snapshot.prefix_sha256 != exported.snapshot.interactions_sha256
    assert len(
        store.read_bytes(exported.snapshot.snapshot_id, "interactions.jsonl").splitlines()
    ) == len(exported.snapshot.interaction_ids)
    assert [
        [attempt.disposition for attempt in interaction.attempts]
        for interaction in exported.interactions
    ] == [
        ["retryable_failure", "completed"],
        ["permanent_failure"],
    ]
    assert exported.dataset.trace_ids == (first.interaction_id,)
    assert len(exported.traces) == 1
    trace = exported.traces[0]
    assert trace.trace_id == first.interaction_id
    assert trace.conversation_id == first.identity.lineage_id
    assert trace.task == "Reset my password"
    assert trace.spans[0].started_at == retry.accepted.attempt_started_at
    assert trace.spans[0].attributes["runtime.attempt_ordinal"] == 2
    assert trace.source.identity.source_id == exported.snapshot.snapshot_id
    assert trace.source.identity.sha256 == exported.snapshot.prefix_sha256
    assert load_trace_dataset(store, exported.dataset.dataset_id).traces == exported.traces


def test_late_original_success_preserves_failure_and_superseded_retry(tmp_path: Path) -> None:
    """Prove a late original success preserves failure and superseded retry evidence.

    A retry accepted after a retryable failure becomes superseded when the original attempt later
    completes, and the original acceptance time remains the trace span start.
    """
    journal, store = _journal_and_store(tmp_path)
    request = _request("Summarize the account")
    original = _accept(journal, key="late-key", request=request)
    journal.record_failure(
        original,
        StructuredFailure(
            code=FailureCode.TIMEOUT,
            message="routed model provider attempt became stale",
            retryable=True,
            attribution=FailureAttribution.MODEL,
        ),
        failed_at=_TIME + timedelta(seconds=1),
    )
    identity = _interaction_identity(journal.project_id, "late-key", request, None)
    replacement = journal.claim(
        identity,
        None,
        now=_TIME + timedelta(seconds=2),
        stale_after=timedelta(minutes=5),
    )
    assert replacement.accepted is not None
    journal.record_completed(
        original,
        _response(content="Account summary"),
        completed_at=_TIME + timedelta(seconds=3),
    )

    exported = seal_runtime_trace_snapshot(
        journal,
        store,
        created_at=_TIME + timedelta(minutes=1),
        code_revision="test-revision",
    )

    interaction = exported.interactions[0]
    assert interaction.completed_attempt_ordinal == 1
    assert [attempt.disposition for attempt in interaction.attempts] == [
        "completed",
        "superseded",
    ]
    assert [event.event for event in interaction.attempts[0].terminal_events] == [
        "attempt_failed",
        "completed",
    ]
    assert interaction.attempts[1].terminal_events == ()
    assert exported.snapshot.failed_attempt_count == 1
    assert exported.snapshot.completed_target_count == 1
    assert exported.dataset.trace_ids == (original.interaction_id,)
    assert len(exported.traces) == 1
    assert exported.traces[0].spans[0].started_at == original.attempt_started_at


@pytest.mark.parametrize(
    "corruption",
    ["duplicate", "event-id", "request-hash", "response-hash"],
)
def test_malformed_ids_digests_and_transitions_fail_closed(tmp_path: Path, corruption: str) -> None:
    """Prove structural rewrites cannot bypass journal identity and transition checks.

    Args:
        tmp_path: Isolated project root.
        corruption: Event mutation applied before sealing the journal.
    """
    journal, store = _journal_and_store(tmp_path)
    _complete(journal, key="corrupt-key")
    lines = [json.loads(line) for line in journal.path.read_text(encoding="utf-8").splitlines()]
    if corruption == "duplicate":
        lines.append(lines[1])
    elif corruption == "event-id":
        lines[0]["event_id"] = "runtime-event-ffffffffffffffffffff"
    elif corruption == "request-hash":
        lines[0]["identity"]["request_sha256"] = "f" * 64
    else:
        lines[1]["response_sha256"] = "f" * 64
    journal.path.write_bytes(b"\n".join(canonical_json_bytes(line) for line in lines) + b"\n")

    with pytest.raises(RuntimeJournalError):
        seal_runtime_trace_snapshot(
            journal,
            store,
            created_at=_TIME + timedelta(minutes=1),
            code_revision="test-revision",
        )


def test_provider_resolved_response_model_preserves_both_model_identities(
    tmp_path: Path,
) -> None:
    """Prove routed and provider-resolved model identities remain distinct provenance.

    The normalized trace records both the selected frozen model and the exact model identity
    returned by the provider rather than requiring them to be equal.
    """
    journal, store = _journal_and_store(tmp_path)
    _complete(
        journal,
        key="wrong-model-key",
        response=_response(model=_snapshot(model_id="different-model")),
    )

    exported = seal_runtime_trace_snapshot(
        journal,
        store,
        created_at=_TIME + timedelta(minutes=1),
        code_revision="test-revision",
    )

    span = exported.traces[0].spans[0]
    assert span.model == _snapshot(model_id="different-model")
    assert span.attributes["runtime.selected_model"] == _snapshot().model_dump(mode="json")
    assert span.attributes["runtime.response_model"] == _snapshot(
        model_id="different-model"
    ).model_dump(mode="json")


def test_torn_tail_is_ignored_but_newline_terminated_corruption_is_rejected(
    tmp_path: Path,
) -> None:
    """Prove only an explicit non-newline journal tail is excluded from sealing.

    A torn final write is ignored by journal recovery, while newline-terminated malformed content
    remains part of the durable prefix and fails closed.
    """
    journal, store = _journal_and_store(tmp_path)
    _complete(journal, key="tail-key")
    clean = journal.path.read_bytes()
    journal.path.write_bytes(clean + b'{"event":"accepted"')

    exported = seal_runtime_trace_snapshot(
        journal,
        store,
        created_at=_TIME + timedelta(minutes=1),
        code_revision="test-revision",
    )
    loaded = load_runtime_trace_snapshot(store, exported.snapshot.snapshot_id)

    assert loaded.interactions == exported.interactions
    assert (
        exported.snapshot.prefix_sha256 == hashlib.sha256(clean, usedforsecurity=False).hexdigest()
    )
    assert (
        len(store.read_bytes(exported.snapshot.snapshot_id, "interactions.jsonl").splitlines()) == 1
    )

    journal.path.write_bytes(clean + b'{"event":"accepted"\n')
    with pytest.raises(RuntimeJournalError, match="invalid interior line"):
        seal_runtime_trace_snapshot(
            journal,
            store,
            created_at=_TIME + timedelta(minutes=2),
            code_revision="test-revision",
        )


def test_secret_idempotency_key_is_hashed_and_secret_prompts_are_rejected(
    tmp_path: Path,
) -> None:
    """Prove runtime artifacts hash caller keys and reject credentials in prompts.

    A benign interaction leaves no raw idempotency key in artifact bytes, while a detectable secret
    in request content prevents canonical trace persistence.
    """
    journal, store = _journal_and_store(tmp_path / "safe")
    raw_key = "sk-this-is-fake-secret-material-123456789"
    _complete(journal, key=raw_key)
    exported = seal_runtime_trace_snapshot(
        journal,
        store,
        created_at=_TIME + timedelta(minutes=1),
        code_revision="test-revision",
    )
    artifact_payload = b"".join(
        _artifact_bytes(store, artifact_id)[path]
        for artifact_id in (exported.snapshot.snapshot_id, exported.dataset.dataset_id)
        for path in sorted(_artifact_bytes(store, artifact_id))
    )
    assert raw_key.encode() not in journal.path.read_bytes()
    assert raw_key.encode() not in artifact_payload

    unsafe_journal, unsafe_store = _journal_and_store(tmp_path / "unsafe")
    _complete(
        unsafe_journal,
        key="safe-key",
        request=_request("Use sk-this-is-fake-secret-material-987654321"),
    )
    with pytest.raises(ArtifactStoreError, match="secret boundary"):
        seal_runtime_trace_snapshot(
            unsafe_journal,
            unsafe_store,
            created_at=_TIME + timedelta(minutes=1),
            code_revision="test-revision",
        )
    assert unsafe_store.list_ids() == ()


def test_empty_or_failed_only_prefix_has_no_canonical_target_dataset(tmp_path: Path) -> None:
    """Prove a dataset requires a completed target in the selected prefix.

    Empty journals and prefixes containing only accepted or failed attempts fail before a canonical
    target dataset can be materialized.
    """
    journal, store = _journal_and_store(tmp_path)
    with pytest.raises(RuntimeTraceSnapshotError, match="no durable events"):
        seal_runtime_trace_snapshot(
            journal,
            store,
            created_at=_TIME,
            code_revision="test-revision",
        )

    accepted = _accept(journal, key="failed-only")
    journal.record_failure(
        accepted,
        StructuredFailure(
            code=FailureCode.PROVIDER,
            message="routed model provider attempt failed permanently",
            retryable=False,
            attribution=FailureAttribution.MODEL,
        ),
        failed_at=_TIME + timedelta(seconds=1),
    )
    with pytest.raises(RuntimeTraceSnapshotError, match="completed routed interaction"):
        seal_runtime_trace_snapshot(
            journal,
            store,
            created_at=_TIME + timedelta(minutes=1),
            code_revision="test-revision",
        )
    assert store.list_ids() == ()


def test_attempt_contract_requires_terminal_evidence_for_terminal_dispositions(
    tmp_path: Path,
) -> None:
    """Prove terminal attempt dispositions require corresponding terminal evidence.

    Pydantic validation rejects completed and failed dispositions when the attempt contains only
    its acceptance event.
    """
    journal, _ = _journal_and_store(tmp_path)
    accepted = _accept(journal, key="contract-key")
    with pytest.raises(ValidationError, match="must be open or superseded"):
        RuntimeTraceAttempt(disposition="completed", accepted=accepted)
