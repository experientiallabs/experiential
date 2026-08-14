"""Immutable snapshots of routed interactions and their canonical production traces."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    Field,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_validator,
)

from wmo.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactId,
    ContractModel,
    JsonObject,
    Sha256,
    SourceIdentity,
    canonical_json_bytes,
    stable_id,
    validate_artifact_file_path,
)
from wmo.common.project import (
    ArtifactAlreadyExistsError,
    ArtifactCorruptionError,
    ArtifactManifest,
    ArtifactStore,
    artifact_input,
)
from wmo.common.traces import Trace, TraceDataset, TraceOutcome, TraceSource, TraceSpan
from wmo.runtime.router.journal import (
    RuntimeAcceptedEvent,
    RuntimeAttemptFailedEvent,
    RuntimeCompletedEvent,
    RuntimeInteractionJournal,
    RuntimeJournalError,
    RuntimeJournalEvent,
    _validate_events,
)

_SNAPSHOT_ARTIFACT_TYPE = "runtime-trace-snapshot"
_SNAPSHOT_PATH = "runtime-trace-snapshot.json"
_INTERACTIONS_PATH = "interactions.jsonl"
_TRACE_DATASET_ARTIFACT_TYPE = "trace-dataset"
_TRACE_DATASET_PATH = "trace-dataset.json"
_TRACES_PATH = "traces.jsonl"
_RUNTIME_TRACE_CONVENTION = "wmo.runtime.router.v1"
_JSON_OBJECT_ADAPTER = TypeAdapter(JsonObject)
RuntimeTerminalEvent = Annotated[
    RuntimeAttemptFailedEvent | RuntimeCompletedEvent,
    Field(discriminator="event"),
]


class RuntimeTraceSnapshotError(ValueError):
    """A runtime journal cannot be sealed as canonical production trace evidence."""


class RuntimeTraceAttempt(ContractModel):
    """One accepted routed attempt and every terminal event that names it."""

    disposition: Literal[
        "open",
        "retryable_failure",
        "permanent_failure",
        "completed",
        "superseded",
    ]
    accepted: RuntimeAcceptedEvent
    terminal_events: tuple[RuntimeTerminalEvent, ...] = ()

    @model_validator(mode="after")
    def _require_coherent_attempt(self) -> RuntimeTraceAttempt:
        event_ordinals = tuple(event.ordinal for event in self.terminal_events)
        if event_ordinals != tuple(sorted(event_ordinals)) or len(set(event_ordinals)) != len(
            event_ordinals
        ):
            raise ValueError("runtime attempt terminal events must have unique ordered ordinals")
        failures = []
        completions = []
        for event in self.terminal_events:
            if (
                event.interaction_id != self.accepted.interaction_id
                or event.attempt_ordinal != self.accepted.attempt_ordinal
                or event.ordinal <= self.accepted.ordinal
            ):
                raise ValueError("runtime attempt terminal event differs from its acceptance")
            if isinstance(event, RuntimeAttemptFailedEvent):
                failures.append(event)
            else:
                completions.append(event)
        if len(failures) > 1 or len(completions) > 1:
            raise ValueError("runtime attempt cannot repeat failure or completion events")
        if failures and completions and not failures[0].retryable:
            raise ValueError("runtime attempt completion cannot follow a permanent failure")
        expected_disposition: str
        if completions:
            expected_disposition = "completed"
        elif failures:
            expected_disposition = (
                "retryable_failure" if failures[0].retryable else "permanent_failure"
            )
        else:
            if self.disposition not in {"open", "superseded"}:
                raise ValueError(
                    "runtime attempts without terminal events must be open or superseded"
                )
            expected_disposition = self.disposition
        if self.disposition != expected_disposition:
            raise ValueError("runtime attempt disposition differs from its terminal events")
        return self


class RuntimeTraceInteraction(ContractModel):
    """One logical routed interaction with complete retry and terminal provenance."""

    interaction_id: ArtifactId
    attempts: tuple[RuntimeTraceAttempt, ...] = Field(min_length=1)
    completed_attempt_ordinal: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _require_coherent_interaction(self) -> RuntimeTraceInteraction:
        accepted = tuple(attempt.accepted for attempt in self.attempts)
        if any(event.interaction_id != self.interaction_id for event in accepted):
            raise ValueError("runtime interaction attempts name another interaction")
        attempt_ordinals = tuple(event.attempt_ordinal for event in accepted)
        if attempt_ordinals != tuple(range(1, len(accepted) + 1)):
            raise ValueError("runtime interaction attempt ordinals must be contiguous")
        original_pins = _accepted_pins(accepted[0])
        if any(_accepted_pins(event) != original_pins for event in accepted[1:]):
            raise ValueError("runtime interaction retry pins differ from the original acceptance")
        completed_attempts = tuple(
            attempt
            for attempt in self.attempts
            if any(isinstance(event, RuntimeCompletedEvent) for event in attempt.terminal_events)
        )
        if len(completed_attempts) > 1:
            raise ValueError("runtime interaction cannot contain multiple completed targets")
        expected_completed_ordinal = (
            completed_attempts[0].accepted.attempt_ordinal if completed_attempts else None
        )
        if self.completed_attempt_ordinal != expected_completed_ordinal:
            raise ValueError("completed attempt ordinal differs from interaction events")
        unterminated = tuple(attempt for attempt in self.attempts if not attempt.terminal_events)
        if completed_attempts:
            if any(attempt.disposition != "superseded" for attempt in unterminated):
                raise ValueError("unterminated attempts must be superseded after completion")
        elif unterminated:
            if unterminated != (self.attempts[-1],) or unterminated[0].disposition != "open":
                raise ValueError("only the latest incomplete runtime attempt may remain open")
        return self


_INTERACTION_ADAPTER = TypeAdapter(RuntimeTraceInteraction)


class RuntimeTraceSnapshot(ArtifactEnvelope):
    """One immutable, content-addressed prefix of a routed-interaction journal."""

    snapshot_id: ArtifactId
    project_id: ArtifactId
    last_ordinal: int = Field(gt=0)
    prefix_sha256: Sha256
    interactions_path: str = Field(min_length=1)
    interactions_sha256: Sha256
    interaction_ids: tuple[ArtifactId, ...]
    completed_target_count: int = Field(ge=0)
    failed_attempt_count: int = Field(ge=0)

    @field_validator("interactions_path")
    @classmethod
    def _require_safe_interactions_path(cls, value: str) -> str:
        return validate_artifact_file_path(value).as_posix()

    @field_validator("interaction_ids")
    @classmethod
    def _require_unique_interactions(cls, value: tuple[ArtifactId, ...]) -> tuple[ArtifactId, ...]:
        if not value:
            raise ValueError("a runtime trace snapshot needs at least one interaction")
        if len(set(value)) != len(value):
            raise ValueError("runtime trace snapshot interaction IDs must be unique")
        return value

    @model_validator(mode="after")
    def _require_content_identity(self) -> RuntimeTraceSnapshot:
        if self.completed_target_count > len(self.interaction_ids):
            raise ValueError("completed target count exceeds the interaction count")
        expected_id = stable_id(
            "runtime-trace-snapshot",
            {
                "project_id": self.project_id,
                "last_ordinal": self.last_ordinal,
                "prefix_sha256": self.prefix_sha256,
            },
        )
        if self.snapshot_id != expected_id:
            raise ValueError("runtime trace snapshot ID differs from its canonical prefix")
        if (
            self.source is None
            or self.source.kind != "production"
            or self.source.sha256 != self.prefix_sha256
        ):
            raise ValueError("runtime trace snapshot source must name its production prefix")
        return self


@dataclass(frozen=True)
class LoadedRuntimeTraceSnapshot:
    """One verified runtime snapshot and its ordered logical interactions."""

    snapshot: RuntimeTraceSnapshot
    manifest: ArtifactManifest
    interactions: tuple[RuntimeTraceInteraction, ...]


@dataclass(frozen=True)
class PersistedRuntimeTraceExport:
    """One runtime snapshot and the canonical trace dataset derived from its targets."""

    snapshot: RuntimeTraceSnapshot
    snapshot_manifest: ArtifactManifest
    interactions: tuple[RuntimeTraceInteraction, ...]
    dataset: TraceDataset
    dataset_manifest: ArtifactManifest
    traces: tuple[Trace, ...]


def seal_runtime_trace_snapshot(
    journal: RuntimeInteractionJournal,
    store: ArtifactStore,
    *,
    created_at: datetime,
    code_revision: str,
    last_ordinal: int | None = None,
) -> PersistedRuntimeTraceExport:
    """Seal a validated journal prefix and derive completed targets as canonical traces.

    Args:
        journal: Project runtime journal whose prefix is being sealed.
        store: Immutable artifact store for the same project.
        created_at: Time the new immutable artifact is materialized.
        code_revision: Exact WMO revision producing the artifact.
        last_ordinal: Optional inclusive prefix boundary. The full durable journal is used when
            omitted.

    Returns:
        The immutable prefix snapshot and its sibling canonical trace dataset.

    Raises:
        RuntimeTraceSnapshotError: The journal is empty, has no completed target in the selected
            prefix, does not belong to the artifact store, or contains inconsistent routed output.
        RuntimeJournalError: The journal prefix violates its event or transition contracts.
    """
    _require_same_project(journal, store)
    all_events = journal.read_events()
    events = _select_prefix(all_events, last_ordinal)
    prefix_sha256 = _sha256(_jsonl_bytes(events))
    interactions = _build_interactions(events)
    interactions_payload = _jsonl_bytes(interactions)
    source = SourceIdentity(
        kind="production",
        source_id=f"{journal.project_id}/runtime/interactions",
        sha256=prefix_sha256,
    )
    interaction_ids = tuple(interaction.interaction_id for interaction in interactions)
    completed_target_count = sum(
        interaction.completed_attempt_ordinal is not None for interaction in interactions
    )
    if completed_target_count == 0:
        raise RuntimeTraceSnapshotError(
            "runtime trace snapshots require at least one completed routed interaction"
        )
    failed_attempt_count = sum(
        isinstance(event, RuntimeAttemptFailedEvent)
        for interaction in interactions
        for attempt in interaction.attempts
        for event in attempt.terminal_events
    )
    snapshot_id = stable_id(
        "runtime-trace-snapshot",
        {
            "project_id": journal.project_id,
            "last_ordinal": events[-1].ordinal,
            "prefix_sha256": prefix_sha256,
        },
    )
    snapshot = RuntimeTraceSnapshot(
        schema_version=1,
        created_at=created_at,
        code_revision=code_revision,
        source=source,
        snapshot_id=snapshot_id,
        project_id=journal.project_id,
        last_ordinal=events[-1].ordinal,
        prefix_sha256=prefix_sha256,
        interactions_path=_INTERACTIONS_PATH,
        interactions_sha256=_sha256(interactions_payload),
        interaction_ids=interaction_ids,
        completed_target_count=completed_target_count,
        failed_attempt_count=failed_attempt_count,
    )
    traces = _derive_traces(interactions, snapshot)
    loaded_snapshot = _persist_snapshot(store, snapshot, interactions, interactions_payload)
    dataset, dataset_manifest = _persist_dataset(
        store,
        traces,
        loaded_snapshot,
    )
    return PersistedRuntimeTraceExport(
        snapshot=loaded_snapshot.snapshot,
        snapshot_manifest=loaded_snapshot.manifest,
        interactions=loaded_snapshot.interactions,
        dataset=dataset,
        dataset_manifest=dataset_manifest,
        traces=traces,
    )


def load_runtime_trace_snapshot(
    store: ArtifactStore, snapshot_id: str
) -> LoadedRuntimeTraceSnapshot:
    """Load and fully validate one immutable runtime journal prefix."""
    stored = store.read(snapshot_id)
    if stored.manifest.artifact_type != _SNAPSHOT_ARTIFACT_TYPE:
        raise ArtifactCorruptionError(f"artifact {snapshot_id} is not a runtime trace snapshot")
    try:
        snapshot = RuntimeTraceSnapshot.model_validate_json(
            store.read_bytes(snapshot_id, _SNAPSHOT_PATH)
        )
    except (ValidationError, ValueError) as exc:
        raise ArtifactCorruptionError(
            f"runtime trace snapshot {snapshot_id} has an invalid envelope"
        ) from exc
    if snapshot.snapshot_id != snapshot_id:
        raise ArtifactCorruptionError(
            f"runtime trace snapshot envelope ID differs from artifact {snapshot_id}"
        )
    payload = store.read_bytes(snapshot_id, snapshot.interactions_path)
    if _sha256(payload) != snapshot.interactions_sha256:
        raise ArtifactCorruptionError(
            f"runtime trace snapshot {snapshot_id} interaction digest differs from envelope"
        )
    try:
        interactions = _parse_canonical_interactions(payload)
        events = _events_from_interactions(interactions)
    except (RuntimeJournalError, ValidationError, ValueError) as exc:
        raise ArtifactCorruptionError(
            f"runtime trace snapshot {snapshot_id} has invalid interaction events"
        ) from exc
    if not events or events[-1].ordinal != snapshot.last_ordinal:
        raise ArtifactCorruptionError(
            f"runtime trace snapshot {snapshot_id} event boundary differs from envelope"
        )
    if _sha256(_jsonl_bytes(events)) != snapshot.prefix_sha256:
        raise ArtifactCorruptionError(
            f"runtime trace snapshot {snapshot_id} journal prefix differs from envelope"
        )
    interaction_ids = tuple(interaction.interaction_id for interaction in interactions)
    if interaction_ids != snapshot.interaction_ids:
        raise ArtifactCorruptionError(
            f"runtime trace snapshot {snapshot_id} interaction order differs from envelope"
        )
    if (
        sum(interaction.completed_attempt_ordinal is not None for interaction in interactions)
        != snapshot.completed_target_count
    ):
        raise ArtifactCorruptionError(
            f"runtime trace snapshot {snapshot_id} completed target count differs from envelope"
        )
    if sum(event.event == "attempt_failed" for event in events) != snapshot.failed_attempt_count:
        raise ArtifactCorruptionError(
            f"runtime trace snapshot {snapshot_id} failed attempt count differs from envelope"
        )
    _require_envelope_matches_manifest(snapshot, stored.manifest)
    return LoadedRuntimeTraceSnapshot(
        snapshot=snapshot,
        manifest=stored.manifest,
        interactions=interactions,
    )


def _select_prefix(
    events: tuple[RuntimeJournalEvent, ...], last_ordinal: int | None
) -> tuple[RuntimeJournalEvent, ...]:
    """Return one inclusive validated prefix boundary."""
    if not events:
        raise RuntimeTraceSnapshotError("runtime journal has no durable events to snapshot")
    if last_ordinal is None:
        return events
    if last_ordinal <= 0 or last_ordinal > len(events):
        raise RuntimeTraceSnapshotError(
            f"last_ordinal must be between 1 and {len(events)} inclusive"
        )
    prefix = events[:last_ordinal]
    _validate_events(prefix)
    return prefix


def _build_interactions(
    events: tuple[RuntimeJournalEvent, ...],
) -> tuple[RuntimeTraceInteraction, ...]:
    """Group globally ordered journal events into canonical logical interactions."""
    accepted_by_interaction: dict[str, list[RuntimeAcceptedEvent]] = {}
    terminals_by_attempt: dict[tuple[str, int], list[RuntimeTerminalEvent]] = {}
    interaction_ids = []
    for event in events:
        if isinstance(event, RuntimeAcceptedEvent):
            attempts = accepted_by_interaction.setdefault(event.interaction_id, [])
            if not attempts:
                interaction_ids.append(event.interaction_id)
            attempts.append(event)
        else:
            terminals_by_attempt.setdefault(
                (event.interaction_id, event.attempt_ordinal), []
            ).append(event)
    interactions = []
    for interaction_id in interaction_ids:
        accepted_events = accepted_by_interaction[interaction_id]
        completed_ordinal = next(
            (
                accepted.attempt_ordinal
                for accepted in accepted_events
                if any(
                    isinstance(event, RuntimeCompletedEvent)
                    for event in terminals_by_attempt.get(
                        (interaction_id, accepted.attempt_ordinal), []
                    )
                )
            ),
            None,
        )
        attempts = []
        for accepted in accepted_events:
            terminal_events = tuple(
                terminals_by_attempt.get(
                    (interaction_id, accepted.attempt_ordinal),
                    [],
                )
            )
            completion = next(
                (event for event in terminal_events if isinstance(event, RuntimeCompletedEvent)),
                None,
            )
            failure = next(
                (
                    event
                    for event in terminal_events
                    if isinstance(event, RuntimeAttemptFailedEvent)
                ),
                None,
            )
            disposition: Literal[
                "open",
                "retryable_failure",
                "permanent_failure",
                "completed",
                "superseded",
            ]
            if completion is not None:
                disposition = "completed"
            elif failure is not None:
                disposition = "retryable_failure" if failure.retryable else "permanent_failure"
            elif completed_ordinal is not None:
                disposition = "superseded"
            else:
                disposition = "open"
            attempts.append(
                RuntimeTraceAttempt(
                    disposition=disposition,
                    accepted=accepted,
                    terminal_events=terminal_events,
                )
            )
        interactions.append(
            RuntimeTraceInteraction(
                interaction_id=interaction_id,
                attempts=tuple(attempts),
                completed_attempt_ordinal=completed_ordinal,
            )
        )
    result = tuple(interactions)
    if _events_from_interactions(result) != events:
        raise RuntimeTraceSnapshotError(
            "canonical interactions do not preserve the exact runtime journal prefix"
        )
    return result


def _derive_traces(
    interactions: tuple[RuntimeTraceInteraction, ...], snapshot: RuntimeTraceSnapshot
) -> tuple[Trace, ...]:
    """Map one completed target per interaction into the shared trace contracts."""
    traces = []
    trace_source = TraceSource(
        identity=SourceIdentity(
            kind="production",
            source_id=snapshot.snapshot_id,
            sha256=snapshot.prefix_sha256,
        ),
        semantic_convention_version=_RUNTIME_TRACE_CONVENTION,
    )
    for interaction in interactions:
        if interaction.completed_attempt_ordinal is None:
            continue
        completed_attempt = interaction.attempts[interaction.completed_attempt_ordinal - 1]
        accepted = completed_attempt.accepted
        completed = next(
            event
            for event in completed_attempt.terminal_events
            if isinstance(event, RuntimeCompletedEvent)
        )
        request_context = _JSON_OBJECT_ADAPTER.validate_python(
            {
                "model_request": accepted.request.model_dump(mode="json"),
                "routing_decision": accepted.decision.model_dump(mode="json"),
            }
        )
        attributes = _JSON_OBJECT_ADAPTER.validate_python(
            {
                "runtime.interaction_id": interaction.interaction_id,
                "runtime.lineage_id": accepted.lineage_id,
                "runtime.request_sha256": accepted.request_sha256,
                "runtime.response_sha256": completed.response_sha256,
                "runtime.attempt_ordinal": completed.attempt_ordinal,
                "runtime.selected_alias": accepted.selected_alias,
                "runtime.selected_model": accepted.selected_model.model_dump(mode="json"),
                "runtime.response_model": completed.response.model.model_dump(mode="json"),
                "runtime.policy_id": accepted.policy_input.artifact_id,
                "runtime.policy_sha256": accepted.policy_input.sha256,
                "runtime.finish_reason": completed.response.finish_reason,
                "runtime.response_output": completed.response.output.model_dump(mode="json"),
                "runtime.economics": completed.response.economics.model_dump(mode="json"),
            }
        )
        traces.append(
            Trace(
                trace_id=interaction.interaction_id,
                conversation_id=accepted.lineage_id,
                task=_task_text(accepted),
                initial_context=request_context,
                tools=accepted.request.tools,
                spans=(
                    TraceSpan(
                        span_id=completed.event_id,
                        name="wmo.router.completion",
                        started_at=accepted.attempt_started_at,
                        ended_at=completed.completed_at,
                        attributes=attributes,
                        model=completed.response.model,
                        usage=completed.response.economics.usage,
                    ),
                ),
                outcome=TraceOutcome(status="success", outcome_name="routed_completion"),
                source=trace_source,
            )
        )
    if len(traces) != snapshot.completed_target_count:
        raise RuntimeTraceSnapshotError(
            "runtime trace derivation did not produce exactly one target per completion"
        )
    return tuple(traces)


def _task_text(accepted: RuntimeAcceptedEvent) -> str:
    """Choose a stable human-readable task while retaining the full request context."""
    for message in reversed(accepted.request.messages):
        if message.role == "user" and message.content:
            return message.content
    for message in accepted.request.messages:
        if message.content:
            return message.content
    return "Complete the routed model request."


def _persist_snapshot(
    store: ArtifactStore,
    snapshot: RuntimeTraceSnapshot,
    interactions: tuple[RuntimeTraceInteraction, ...],
    interactions_payload: bytes,
) -> LoadedRuntimeTraceSnapshot:
    """Write a new prefix snapshot or return its exact existing materialization."""
    try:
        store.write(
            artifact_id=snapshot.snapshot_id,
            artifact_type=_SNAPSHOT_ARTIFACT_TYPE,
            envelope=snapshot,
            files={
                _INTERACTIONS_PATH: interactions_payload,
                _SNAPSHOT_PATH: canonical_json_bytes(snapshot),
            },
        )
    except ArtifactAlreadyExistsError:
        loaded = load_runtime_trace_snapshot(store, snapshot.snapshot_id)
        replay = snapshot.model_copy(update={"created_at": loaded.snapshot.created_at})
        if loaded.snapshot != replay or loaded.interactions != interactions:
            raise RuntimeTraceSnapshotError(
                "existing runtime trace snapshot differs from the journal prefix replay"
            ) from None
        return loaded
    return load_runtime_trace_snapshot(store, snapshot.snapshot_id)


def _persist_dataset(
    store: ArtifactStore,
    traces: tuple[Trace, ...],
    loaded_snapshot: LoadedRuntimeTraceSnapshot,
) -> tuple[TraceDataset, ArtifactManifest]:
    """Persist completed runtime targets through the shared canonical trace envelope."""
    traces_payload = _jsonl_bytes(traces)
    snapshot_input = artifact_input(loaded_snapshot.manifest)
    dataset_id = stable_id(
        "trace-dataset",
        {
            "snapshot": snapshot_input.model_dump(mode="json"),
            "traces_sha256": _sha256(traces_payload),
        },
    )
    dataset = TraceDataset(
        schema_version=1,
        created_at=loaded_snapshot.snapshot.created_at,
        inputs=(snapshot_input,),
        code_revision=loaded_snapshot.snapshot.code_revision,
        source=SourceIdentity(
            kind="production",
            source_id=loaded_snapshot.snapshot.snapshot_id,
            sha256=loaded_snapshot.snapshot.prefix_sha256,
        ),
        dataset_id=dataset_id,
        semantic_convention_version=_RUNTIME_TRACE_CONVENTION,
        traces_path=_TRACES_PATH,
        traces_sha256=_sha256(traces_payload),
        trace_ids=tuple(trace.trace_id for trace in traces),
    )
    try:
        manifest = store.write(
            artifact_id=dataset.dataset_id,
            artifact_type=_TRACE_DATASET_ARTIFACT_TYPE,
            envelope=dataset,
            files={
                _TRACE_DATASET_PATH: canonical_json_bytes(dataset),
                _TRACES_PATH: traces_payload,
            },
        )
    except ArtifactAlreadyExistsError:
        return _load_exact_dataset_replay(store, dataset, traces_payload)
    return dataset, manifest


def _load_exact_dataset_replay(
    store: ArtifactStore,
    expected: TraceDataset,
    traces_payload: bytes,
) -> tuple[TraceDataset, ArtifactManifest]:
    """Return an existing byte-identical dataset derived from the same snapshot."""
    stored = store.read(expected.dataset_id)
    if stored.manifest.artifact_type != _TRACE_DATASET_ARTIFACT_TYPE:
        raise RuntimeTraceSnapshotError(
            f"existing artifact {expected.dataset_id} is not a trace dataset"
        )
    try:
        existing = TraceDataset.model_validate_json(
            store.read_bytes(expected.dataset_id, _TRACE_DATASET_PATH)
        )
    except (ValidationError, ValueError) as exc:
        raise RuntimeTraceSnapshotError(
            f"existing trace dataset {expected.dataset_id} has an invalid envelope"
        ) from exc
    if existing != expected:
        raise RuntimeTraceSnapshotError(
            "existing runtime trace dataset differs from the snapshot replay"
        )
    if store.read_bytes(expected.dataset_id, _TRACES_PATH) != traces_payload:
        raise RuntimeTraceSnapshotError(
            "existing runtime trace dataset records differ from the snapshot replay"
        )
    _require_envelope_matches_manifest(existing, stored.manifest)
    return existing, stored.manifest


def _parse_canonical_interactions(
    payload: bytes,
) -> tuple[RuntimeTraceInteraction, ...]:
    """Parse canonical logical interactions from newline-terminated JSONL."""
    if payload and not payload.endswith(b"\n"):
        raise RuntimeJournalError("snapshot interaction JSONL must end with a newline")
    interactions = []
    for index, line in enumerate(payload.splitlines(), start=1):
        if not line:
            raise RuntimeJournalError(f"snapshot interaction JSONL has blank line {index}")
        try:
            interactions.append(_INTERACTION_ADAPTER.validate_json(line))
        except (UnicodeDecodeError, ValidationError) as exc:
            raise RuntimeJournalError(
                f"snapshot interaction JSONL has invalid line {index}"
            ) from exc
    canonical = _jsonl_bytes(interactions)
    if canonical != payload:
        raise RuntimeJournalError("snapshot interaction JSONL is not canonical")
    return tuple(interactions)


def _events_from_interactions(
    interactions: tuple[RuntimeTraceInteraction, ...],
) -> tuple[RuntimeJournalEvent, ...]:
    """Restore the exact global event prefix from canonical interaction provenance."""
    events: list[RuntimeJournalEvent] = []
    for interaction in interactions:
        for attempt in interaction.attempts:
            events.append(attempt.accepted)
            events.extend(attempt.terminal_events)
    events.sort(key=lambda event: event.ordinal)
    _validate_events(events)
    expected_interaction_ids = tuple(
        dict.fromkeys(
            event.interaction_id for event in events if isinstance(event, RuntimeAcceptedEvent)
        )
    )
    actual_interaction_ids = tuple(interaction.interaction_id for interaction in interactions)
    if actual_interaction_ids != expected_interaction_ids:
        raise RuntimeJournalError(
            "snapshot interaction rows differ from first-acceptance journal order"
        )
    return tuple(events)


def _accepted_pins(event: RuntimeAcceptedEvent) -> JsonObject:
    """Return the immutable interaction fields that every retry must preserve."""
    material = event.model_dump(mode="json")
    for field in ("event_id", "ordinal", "attempt_ordinal", "attempt_started_at"):
        del material[field]
    return _JSON_OBJECT_ADAPTER.validate_python(material)


def _require_same_project(journal: RuntimeInteractionJournal, store: ArtifactStore) -> None:
    """Require mutable and immutable paths to belong to one project boundary."""
    expected_path = store.project_directory / "runtime" / "interactions.jsonl"
    if journal.project_id != store.project_directory.name or (
        journal.path.resolve() != expected_path.resolve()
    ):
        raise RuntimeTraceSnapshotError(
            "runtime journal and artifact store must belong to the same project"
        )


def _require_envelope_matches_manifest(
    envelope: ArtifactEnvelope, manifest: ArtifactManifest
) -> None:
    """Require shared provenance fields to match the immutable artifact manifest."""
    if (
        manifest.schema_version,
        manifest.created_at,
        manifest.inputs,
        manifest.code_revision,
        manifest.source,
    ) != (
        envelope.schema_version,
        envelope.created_at,
        envelope.inputs,
        envelope.code_revision,
        envelope.source,
    ):
        raise ArtifactCorruptionError("artifact envelope differs from its manifest")


def _jsonl_bytes(records: Sequence[BaseModel]) -> bytes:
    """Serialize ordered contract records as deterministic newline-terminated JSONL."""
    payload = b"\n".join(canonical_json_bytes(record) for record in records)
    return payload + b"\n" if payload else b""


def _sha256(payload: bytes) -> str:
    """Return a content digest for immutable runtime evidence."""
    return hashlib.sha256(payload, usedforsecurity=False).hexdigest()
