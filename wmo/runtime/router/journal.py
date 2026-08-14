"""Durable routed-interaction journal and local idempotency boundary.

The journal guarantees one local logical interaction and one durable target for each project and
idempotency key. Provider dispatch can still be at-least-once if the process crashes after the
provider succeeds but before the completion record reaches disk. Remote exactly-once behavior is
available only when the selected provider honors the explicitly forwarded idempotency key.
"""

from __future__ import annotations

import hashlib
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, TypeAdapter, ValidationError, field_validator, model_validator

from wmo.common.core.artifacts import (
    ArtifactInput,
    ContractModel,
    FailureAttribution,
    FailureCode,
    Sha256,
    StructuredFailure,
    canonical_json_bytes,
    sha256_json,
    stable_id,
)
from wmo.common.core.locks import file_write_lock
from wmo.common.models import ModelRequest, ModelResponse, ModelSnapshot
from wmo.common.project import ProjectPaths
from wmo.common.routing import RoutingDecision
from wmo.runtime.models.providers.transport import ProviderTransportError
from wmo.runtime.router.runtime import (
    RoutedModelResponse,
    RouterModelCapabilityError,
    RouterRuntime,
    RouterRuntimeIntegrityError,
)


class RuntimeJournalError(ValueError):
    """The runtime journal is corrupt or an attempted transition is invalid."""


class RuntimeIdempotencyConflictError(ValueError):
    """An idempotency key was reused for different request or lineage content."""


class RuntimeInteractionInProgressError(RuntimeError):
    """Another process still owns the live provider attempt for this interaction."""

    retryable = True


class RuntimeInteractionFailedError(RuntimeError):
    """A prior non-retryable provider attempt ended without a completed response."""

    retryable = False

    def __init__(self, failure: StructuredFailure) -> None:
        super().__init__(failure.message)
        self.failure = failure


class RuntimeInteractionIdentity(ContractModel):
    """Secret-free identity used to find and compare one logical interaction."""

    interaction_id: str = Field(pattern=r"^interaction-[0-9a-f]{20}$")
    project_id: str = Field(pattern=r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*$")
    idempotency_key_sha256: Sha256
    request: ModelRequest
    request_sha256: Sha256
    lineage_id: str = Field(pattern=r"^lineage-[0-9a-f]{20}$")

    @model_validator(mode="after")
    def _require_request_digest(self) -> RuntimeInteractionIdentity:
        if self.request_sha256 != sha256_json(self.request):
            raise ValueError("interaction request digest differs from canonical request")
        return self


class RuntimeAcceptance(ContractModel):
    """Immutable route pins proposed before the first target dispatch."""

    decision: RoutingDecision
    selected_alias: str = Field(min_length=1, max_length=128)
    selected_model: ModelSnapshot
    policy_input: ArtifactInput

    @model_validator(mode="after")
    def _require_matching_route_pins(self) -> RuntimeAcceptance:
        if self.selected_alias != self.decision.selected_alias:
            raise ValueError("accepted selected alias differs from the routing decision")
        if (
            self.policy_input.artifact_id != self.decision.policy_id
            or self.policy_input.sha256 != self.decision.policy_sha256
        ):
            raise ValueError("accepted policy input differs from the routing decision")
        return self


class RuntimeAcceptedEvent(ContractModel):
    """One initial or retry attempt accepted against immutable request and route pins."""

    event: Literal["accepted"] = "accepted"
    event_id: str = Field(pattern=r"^runtime-event-[0-9a-f]{20}$")
    ordinal: int = Field(gt=0)
    attempt_ordinal: int = Field(gt=0)
    interaction_id: str = Field(pattern=r"^interaction-[0-9a-f]{20}$")
    project_id: str = Field(pattern=r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*$")
    idempotency_key_sha256: Sha256
    request: ModelRequest
    request_sha256: Sha256
    lineage_id: str = Field(pattern=r"^lineage-[0-9a-f]{20}$")
    decision: RoutingDecision
    selected_alias: str = Field(min_length=1, max_length=128)
    selected_model: ModelSnapshot
    policy_input: ArtifactInput
    received_at: datetime
    attempt_started_at: datetime

    @field_validator("received_at", "attempt_started_at")
    @classmethod
    def _require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("runtime journal timestamps must include a timezone")
        return value

    @model_validator(mode="after")
    def _require_matching_pins(self) -> RuntimeAcceptedEvent:
        if self.request_sha256 != sha256_json(self.request):
            raise ValueError("accepted request digest differs from canonical request")
        if self.attempt_started_at < self.received_at:
            raise ValueError("accepted attempt start precedes request receipt")
        RuntimeAcceptance(
            decision=self.decision,
            selected_alias=self.selected_alias,
            selected_model=self.selected_model,
            policy_input=self.policy_input,
        )
        return self


class RuntimeAttemptFailedEvent(ContractModel):
    """One completion attempt that ended without a response target."""

    event: Literal["attempt_failed"] = "attempt_failed"
    event_id: str = Field(pattern=r"^runtime-event-[0-9a-f]{20}$")
    ordinal: int = Field(gt=0)
    attempt_ordinal: int = Field(gt=0)
    interaction_id: str = Field(pattern=r"^interaction-[0-9a-f]{20}$")
    failure: StructuredFailure
    retryable: bool
    attempt_started_at: datetime
    failed_at: datetime

    @field_validator("attempt_started_at", "failed_at")
    @classmethod
    def _require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("runtime journal timestamps must include a timezone")
        return value

    @model_validator(mode="after")
    def _require_matching_failure(self) -> RuntimeAttemptFailedEvent:
        if self.retryable != self.failure.retryable:
            raise ValueError("attempt retryability differs from its structured failure")
        if self.failed_at < self.attempt_started_at:
            raise ValueError("attempt failure precedes its start")
        return self


class RuntimeCompletedEvent(ContractModel):
    """One completed canonical response committed after provider success."""

    event: Literal["completed"] = "completed"
    event_id: str = Field(pattern=r"^runtime-event-[0-9a-f]{20}$")
    ordinal: int = Field(gt=0)
    attempt_ordinal: int = Field(gt=0)
    interaction_id: str = Field(pattern=r"^interaction-[0-9a-f]{20}$")
    response: ModelResponse
    response_sha256: Sha256
    completed_at: datetime

    @field_validator("completed_at")
    @classmethod
    def _require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("runtime journal timestamps must include a timezone")
        return value

    @model_validator(mode="after")
    def _require_response_digest(self) -> RuntimeCompletedEvent:
        if self.response_sha256 != sha256_json(self.response):
            raise ValueError("completed response digest differs from canonical response")
        return self


RuntimeJournalEvent = Annotated[
    RuntimeAcceptedEvent | RuntimeAttemptFailedEvent | RuntimeCompletedEvent,
    Field(discriminator="event"),
]
_EVENT_ADAPTER: TypeAdapter[RuntimeJournalEvent] = TypeAdapter(RuntimeJournalEvent)
_SELECTION_LOCKS = tuple(threading.RLock() for _ in range(64))


@dataclass(frozen=True)
class _InteractionState:
    """Validated latest state for one interaction."""

    accepted: RuntimeAcceptedEvent
    terminal: RuntimeAttemptFailedEvent | RuntimeCompletedEvent | None


@dataclass(frozen=True)
class JournalClaim:
    """Result of one atomic attempt to claim an interaction."""

    status: Literal["needs_selection", "dispatch", "live", "completed", "failed"]
    accepted: RuntimeAcceptedEvent | None = None
    completed: RuntimeCompletedEvent | None = None
    failure: RuntimeAttemptFailedEvent | None = None


class RuntimeInteractionJournal:
    """Strict append-only JSONL state for routed interactions in one project."""

    def __init__(self, paths: ProjectPaths) -> None:
        """Bind the journal to one project's canonical mutable runtime path."""
        self.path = paths.runtime_journal
        self.project_id = paths.project_id
        self._durability_directories = (
            paths.runtime_directory,
            paths.project_directory,
            paths.projects_directory,
            paths.root,
        )

    def read_events(self) -> tuple[RuntimeJournalEvent, ...]:
        """Read and fully validate all durable records, ignoring only a torn final line."""
        with file_write_lock(self.path, what="the routed-interaction journal"):
            return self._read_unlocked()

    def claim(
        self,
        identity: RuntimeInteractionIdentity,
        acceptance: RuntimeAcceptance | None,
        *,
        now: datetime,
        stale_after: timedelta,
    ) -> JournalClaim:
        """Atomically inspect, create, or retry one provider attempt.

        Args:
            identity: Secret-free interaction identity derived from caller input.
            acceptance: Route pins selected outside the journal lock, if selection was needed.
            now: Current timezone-aware time.
            stale_after: Age after which an unclosed attempt may be retried.

        Returns:
            A claim telling the caller to select, dispatch, wait, replay, or fail.
        """
        _require_timezone(now)
        if stale_after.total_seconds() <= 0:
            raise ValueError("stale_after must be positive")
        with file_write_lock(self.path, what="the routed-interaction journal"):
            events = list(self._read_unlocked())
            states = _validate_events(events)
            state = states.get(identity.interaction_id)
            if state is None:
                if acceptance is None:
                    return JournalClaim("needs_selection")
                accepted = _accepted_event(
                    identity,
                    acceptance,
                    ordinal=len(events) + 1,
                    attempt_ordinal=1,
                    received_at=now,
                    attempt_started_at=now,
                )
                self._append_unlocked(accepted)
                return JournalClaim("dispatch", accepted=accepted)
            _require_identity(state.accepted, identity)
            terminal = state.terminal
            if isinstance(terminal, RuntimeCompletedEvent):
                return JournalClaim("completed", accepted=state.accepted, completed=terminal)
            if isinstance(terminal, RuntimeAttemptFailedEvent) and not terminal.retryable:
                return JournalClaim("failed", accepted=state.accepted, failure=terminal)
            if terminal is None and now - state.accepted.attempt_started_at < stale_after:
                return JournalClaim("live", accepted=state.accepted)
            if terminal is None:
                stale_failure = StructuredFailure(
                    code=FailureCode.TIMEOUT,
                    message="prior routed model attempt became stale",
                    retryable=True,
                    attribution=FailureAttribution.MODEL,
                )
                failed = _failed_event(
                    state.accepted,
                    stale_failure,
                    ordinal=len(events) + 1,
                    failed_at=now,
                )
                self._append_unlocked(failed)
                events.append(failed)
            accepted = _accepted_event(
                identity,
                RuntimeAcceptance(
                    decision=state.accepted.decision,
                    selected_alias=state.accepted.selected_alias,
                    selected_model=state.accepted.selected_model,
                    policy_input=state.accepted.policy_input,
                ),
                ordinal=len(events) + 1,
                attempt_ordinal=state.accepted.attempt_ordinal + 1,
                received_at=state.accepted.received_at,
                attempt_started_at=now,
            )
            self._append_unlocked(accepted)
            return JournalClaim("dispatch", accepted=accepted)

    def record_failure(
        self,
        accepted: RuntimeAcceptedEvent,
        failure: StructuredFailure,
        *,
        failed_at: datetime,
    ) -> RuntimeAttemptFailedEvent | RuntimeCompletedEvent:
        """Append a live failure or reconcile with a concurrent winning outcome."""
        _require_timezone(failed_at)
        with file_write_lock(self.path, what="the routed-interaction journal"):
            events = list(self._read_unlocked())
            state = _validate_events(events).get(accepted.interaction_id)
            if state is None or accepted not in events:
                raise RuntimeJournalError(
                    "cannot fail an interaction attempt that was not accepted"
                )
            if isinstance(state.terminal, RuntimeCompletedEvent):
                return state.terminal
            if state.accepted != accepted:
                prior = _attempt_failure(events, accepted)
                if prior is None:
                    raise RuntimeJournalError("superseded attempt has no durable failure")
                return prior
            if state.terminal is not None:
                if isinstance(state.terminal, RuntimeAttemptFailedEvent):
                    return state.terminal
                raise RuntimeJournalError("cannot fail an interaction after completion")
            event = _failed_event(
                accepted,
                failure,
                ordinal=len(events) + 1,
                failed_at=failed_at,
            )
            self._append_unlocked(event)
            return event

    def record_completed(
        self,
        accepted: RuntimeAcceptedEvent,
        response: ModelResponse,
        *,
        completed_at: datetime,
    ) -> RuntimeAttemptFailedEvent | RuntimeCompletedEvent:
        """Commit the first response or observe an earlier permanent failure."""
        _require_timezone(completed_at)
        with file_write_lock(self.path, what="the routed-interaction journal"):
            events = list(self._read_unlocked())
            state = _validate_events(events).get(accepted.interaction_id)
            if state is None or accepted not in events:
                raise RuntimeJournalError(
                    "cannot complete an interaction attempt that was not accepted"
                )
            if isinstance(state.terminal, RuntimeCompletedEvent):
                return state.terminal
            if (
                isinstance(state.terminal, RuntimeAttemptFailedEvent)
                and not state.terminal.retryable
            ):
                return state.terminal
            if state.accepted != accepted:
                prior = _attempt_failure(events, accepted)
                if prior is None or not prior.retryable:
                    raise RuntimeJournalError(
                        "superseded attempt lacks a retryable durable failure"
                    )
            elif state.terminal is not None:
                raise RuntimeJournalError("cannot complete a terminal interaction attempt")
            event = _completed_event(
                accepted,
                response,
                ordinal=len(events) + 1,
                completed_at=completed_at,
            )
            self._append_unlocked(event)
            return event

    def _read_unlocked(self) -> tuple[RuntimeJournalEvent, ...]:
        try:
            payload = self.path.read_bytes()
        except FileNotFoundError:
            return ()
        except OSError as exc:
            raise RuntimeJournalError(f"cannot read runtime journal {self.path}") from exc
        lines = payload.splitlines(keepends=True)
        if lines and not lines[-1].endswith(b"\n"):
            lines.pop()
        events: list[RuntimeJournalEvent] = []
        for index, line in enumerate(lines, start=1):
            if not line.strip():
                raise RuntimeJournalError(f"runtime journal has a blank interior line {index}")
            try:
                events.append(_EVENT_ADAPTER.validate_json(line))
            except (UnicodeDecodeError, ValidationError) as exc:
                raise RuntimeJournalError(
                    f"runtime journal has invalid interior line {index}"
                ) from exc
        _validate_events(events)
        return tuple(events)

    def _append_unlocked(self, event: RuntimeJournalEvent) -> None:
        if event.event_id != _event_content_id(event):
            raise RuntimeJournalError("runtime event ID differs from its canonical content")
        _prepare_runtime_directory(self.path)
        _truncate_torn_tail(self.path)
        flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(self.path, flags, 0o600)
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "ab", closefd=True) as handle:
                handle.write(canonical_json_bytes(event) + b"\n")
                handle.flush()
                os.fsync(handle.fileno())
            _fsync_directories(self._durability_directories)
        except OSError as exc:
            raise RuntimeJournalError(f"cannot append runtime journal {self.path}") from exc


class JournaledRouterRuntime:
    """Idempotent completion service around one verified ``RouterRuntime``.

    WMO guarantees one local logical interaction and one pinned target. A provider dispatch can
    happen more than once only in the crash-after-success, before-journal window, unless the
    provider itself honors the forwarded idempotency key.
    """

    def __init__(
        self,
        runtime: RouterRuntime,
        journal: RuntimeInteractionJournal,
        *,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        wait_timeout_seconds: float = 10.0,
        stale_after_seconds: float = 120.0,
    ) -> None:
        """Create a durable wrapper with bounded live-attempt waiting."""
        if wait_timeout_seconds < 0 or stale_after_seconds <= 0:
            raise ValueError("journal wait must be non-negative and stale age must be positive")
        self.runtime = runtime
        self.journal = journal
        self._clock = clock
        self._monotonic = monotonic
        self._sleep = sleep
        self._wait_timeout_seconds = wait_timeout_seconds
        self._stale_after = timedelta(seconds=stale_after_seconds)

    def complete(
        self,
        request: ModelRequest,
        *,
        idempotency_key: str,
        conversation_id: str | None = None,
    ) -> RoutedModelResponse:
        """Return one durable response for a caller-owned idempotency key.

        Args:
            request: Canonical provider-neutral model request.
            idempotency_key: Standard request key. Only its SHA-256 digest is persisted.
            conversation_id: Optional caller identity for sticky routing. Only a stable digest ID
                is persisted. The idempotency key defines a single interaction, not a conversation.

        Returns:
            The pinned routing decision and completed or replayed model response.

        Raises:
            RuntimeIdempotencyConflictError: The key was reused for different request content.
            RuntimeInteractionInProgressError: A live attempt exceeded the bounded local wait.
            RuntimeInteractionFailedError: A prior attempt failed permanently.
        """
        _validate_external_id(idempotency_key, label="idempotency key", visible_ascii=True)
        if conversation_id is not None:
            _validate_external_id(conversation_id, label="conversation ID", visible_ascii=False)
        identity = _interaction_identity(
            self.journal.project_id,
            idempotency_key,
            request,
            conversation_id,
        )
        deadline = self._monotonic() + self._wait_timeout_seconds
        while True:
            claim = self.journal.claim(
                identity,
                None,
                now=self._clock(),
                stale_after=self._stale_after,
            )
            if claim.status == "needs_selection":
                selection_lock = _selection_lock(identity.interaction_id)
                remaining = deadline - self._monotonic()
                if remaining <= 0 or not selection_lock.acquire(timeout=remaining):
                    raise RuntimeInteractionInProgressError(
                        "another local request is selecting an idempotent interaction; retry"
                    )
                try:
                    claim = self.journal.claim(
                        identity,
                        None,
                        now=self._clock(),
                        stale_after=self._stale_after,
                    )
                    if claim.status == "needs_selection":
                        decision = self.runtime.select(request, episode_id=identity.lineage_id)
                        acceptance = RuntimeAcceptance(
                            decision=decision,
                            selected_alias=decision.selected_alias,
                            selected_model=_selected_model(self.runtime, decision.selected_alias),
                            policy_input=ArtifactInput(
                                artifact_id=decision.policy_id,
                                sha256=decision.policy_sha256,
                            ),
                        )
                        claim = self.journal.claim(
                            identity,
                            acceptance,
                            now=self._clock(),
                            stale_after=self._stale_after,
                        )
                finally:
                    selection_lock.release()
            if claim.status == "completed":
                if claim.accepted is None or claim.completed is None:
                    raise RuntimeJournalError("completed journal claim omitted its records")
                return RoutedModelResponse(
                    decision=claim.accepted.decision,
                    response=claim.completed.response,
                )
            if claim.status == "failed":
                if claim.failure is None:
                    raise RuntimeJournalError("failed journal claim omitted its failure")
                if claim.failure.failure.code == FailureCode.UNSUPPORTED:
                    raise RouterModelCapabilityError(claim.failure.failure.message)
                raise RuntimeInteractionFailedError(claim.failure.failure)
            if claim.status == "live":
                remaining = deadline - self._monotonic()
                if remaining <= 0:
                    raise RuntimeInteractionInProgressError(
                        "another process is completing this idempotent interaction; retry"
                    )
                self._sleep(min(0.05, remaining))
                continue
            if claim.accepted is None:
                raise RuntimeJournalError("dispatch journal claim omitted its accepted record")
            accepted = claim.accepted
            break
        try:
            routed = self.runtime.complete(
                request,
                episode_id=accepted.lineage_id,
                decision=accepted.decision,
                provider_idempotency_key=idempotency_key,
            )
        except Exception as exc:
            failure = _structured_completion_failure(exc)
            terminal = self.journal.record_failure(accepted, failure, failed_at=self._clock())
            if isinstance(terminal, RuntimeCompletedEvent):
                return RoutedModelResponse(
                    decision=accepted.decision,
                    response=terminal.response,
                )
            raise
        completed = self.journal.record_completed(
            accepted,
            routed.response,
            completed_at=self._clock(),
        )
        if isinstance(completed, RuntimeAttemptFailedEvent):
            raise RuntimeInteractionFailedError(completed.failure)
        return RoutedModelResponse(
            decision=accepted.decision,
            response=completed.response,
        )


def _selection_lock(interaction_id: str) -> threading.RLock:
    """Return a bounded process-wide lock for one interaction selection handshake.

    Args:
        interaction_id: Canonical content-derived runtime interaction identity.

    Returns:
        Stable lock stripe shared by every local runtime instance.
    """
    digest = int(interaction_id.rsplit("-", maxsplit=1)[-1], 16)
    return _SELECTION_LOCKS[digest % len(_SELECTION_LOCKS)]


def _interaction_identity(
    project_id: str,
    idempotency_key: str,
    request: ModelRequest,
    conversation_id: str | None,
) -> RuntimeInteractionIdentity:
    """Build stable secret-free interaction and lineage identities."""
    key_sha256 = hashlib.sha256(idempotency_key.encode("utf-8"), usedforsecurity=False).hexdigest()
    lineage_value = conversation_id or idempotency_key
    lineage_sha256 = hashlib.sha256(
        lineage_value.encode("utf-8"), usedforsecurity=False
    ).hexdigest()
    lineage_id = stable_id("lineage", {"project_id": project_id, "lineage_sha256": lineage_sha256})
    interaction_id = stable_id(
        "interaction", {"project_id": project_id, "idempotency_key_sha256": key_sha256}
    )
    return RuntimeInteractionIdentity(
        interaction_id=interaction_id,
        project_id=project_id,
        idempotency_key_sha256=key_sha256,
        request=request,
        request_sha256=sha256_json(request),
        lineage_id=lineage_id,
    )


def _accepted_event(
    identity: RuntimeInteractionIdentity,
    acceptance: RuntimeAcceptance,
    *,
    ordinal: int,
    attempt_ordinal: int,
    received_at: datetime,
    attempt_started_at: datetime,
) -> RuntimeAcceptedEvent:
    """Create one content-addressed accepted attempt with exact immutable pins."""
    provisional = RuntimeAcceptedEvent(
        event_id="runtime-event-00000000000000000000",
        ordinal=ordinal,
        attempt_ordinal=attempt_ordinal,
        interaction_id=identity.interaction_id,
        project_id=identity.project_id,
        idempotency_key_sha256=identity.idempotency_key_sha256,
        request=identity.request,
        request_sha256=identity.request_sha256,
        lineage_id=identity.lineage_id,
        decision=acceptance.decision,
        selected_alias=acceptance.selected_alias,
        selected_model=acceptance.selected_model,
        policy_input=acceptance.policy_input,
        received_at=received_at,
        attempt_started_at=attempt_started_at,
    )
    return provisional.model_copy(update={"event_id": _event_content_id(provisional)})


def _failed_event(
    accepted: RuntimeAcceptedEvent,
    failure: StructuredFailure,
    *,
    ordinal: int,
    failed_at: datetime,
) -> RuntimeAttemptFailedEvent:
    """Create one content-addressed terminal record for a failed attempt."""
    provisional = RuntimeAttemptFailedEvent(
        event_id="runtime-event-00000000000000000000",
        ordinal=ordinal,
        attempt_ordinal=accepted.attempt_ordinal,
        interaction_id=accepted.interaction_id,
        failure=failure,
        retryable=failure.retryable,
        attempt_started_at=accepted.attempt_started_at,
        failed_at=failed_at,
    )
    return provisional.model_copy(update={"event_id": _event_content_id(provisional)})


def _completed_event(
    accepted: RuntimeAcceptedEvent,
    response: ModelResponse,
    *,
    ordinal: int,
    completed_at: datetime,
) -> RuntimeCompletedEvent:
    """Create one content-addressed completed response record."""
    provisional = RuntimeCompletedEvent(
        event_id="runtime-event-00000000000000000000",
        ordinal=ordinal,
        attempt_ordinal=accepted.attempt_ordinal,
        interaction_id=accepted.interaction_id,
        response=response,
        response_sha256=sha256_json(response),
        completed_at=completed_at,
    )
    return provisional.model_copy(update={"event_id": _event_content_id(provisional)})


def _event_content_id(event: RuntimeJournalEvent) -> str:
    """Return the stable content identity for one event, excluding its own ID."""
    material = event.model_dump(mode="json")
    del material["event_id"]
    return stable_id("runtime-event", material)


def _validate_events(
    events: list[RuntimeJournalEvent] | tuple[RuntimeJournalEvent, ...],
) -> dict[str, _InteractionState]:
    """Validate global order, content digests, and every interaction transition."""
    states: dict[str, _InteractionState] = {}
    accepted_attempts: dict[tuple[str, int], RuntimeAcceptedEvent] = {}
    failed_attempts: dict[tuple[str, int], RuntimeAttemptFailedEvent] = {}
    key_owners: dict[tuple[str, str], str] = {}
    seen_event_ids: set[str] = set()
    for expected_ordinal, event in enumerate(events, start=1):
        if event.ordinal != expected_ordinal:
            raise RuntimeJournalError("runtime journal ordinals are not contiguous")
        if event.event_id in seen_event_ids:
            raise RuntimeJournalError("runtime journal repeats an event ID")
        if event.event_id != _event_content_id(event):
            raise RuntimeJournalError("runtime event ID differs from its canonical content")
        seen_event_ids.add(event.event_id)
        state = states.get(event.interaction_id)
        if isinstance(event, RuntimeAcceptedEvent):
            expected_interaction_id = stable_id(
                "interaction",
                {
                    "project_id": event.project_id,
                    "idempotency_key_sha256": event.idempotency_key_sha256,
                },
            )
            if event.interaction_id != expected_interaction_id:
                raise RuntimeJournalError("interaction ID differs from project and key digest")
            expected_episode_sha256 = hashlib.sha256(
                event.lineage_id.encode("utf-8"), usedforsecurity=False
            ).hexdigest()
            if event.decision.episode_id_sha256 != expected_episode_sha256:
                raise RuntimeJournalError("routing decision differs from accepted lineage")
            if event.decision.decision_id != _routing_decision_content_id(event.decision):
                raise RuntimeJournalError("routing decision ID differs from its canonical content")
            key = (event.project_id, event.idempotency_key_sha256)
            owner = key_owners.setdefault(key, event.interaction_id)
            if owner != event.interaction_id:
                raise RuntimeJournalError("idempotency key digest maps to multiple interactions")
            if state is None:
                if event.attempt_ordinal != 1:
                    raise RuntimeJournalError("first interaction attempt must have ordinal one")
            else:
                if not isinstance(state.terminal, RuntimeAttemptFailedEvent):
                    raise RuntimeJournalError("accepted retry does not follow a failed attempt")
                if not state.terminal.retryable:
                    raise RuntimeJournalError("accepted retry follows a permanent failure")
                if event.attempt_ordinal != state.accepted.attempt_ordinal + 1:
                    raise RuntimeJournalError("interaction attempt ordinals are not contiguous")
                if _acceptance_pins(event) != _acceptance_pins(state.accepted):
                    raise RuntimeJournalError("retry drifted from the original accepted pins")
            states[event.interaction_id] = _InteractionState(event, None)
            accepted_attempts[(event.interaction_id, event.attempt_ordinal)] = event
            continue
        if state is None:
            raise RuntimeJournalError("terminal runtime event has no accepted attempt")
        accepted = accepted_attempts.get((event.interaction_id, event.attempt_ordinal))
        if accepted is None:
            raise RuntimeJournalError("terminal event names an unaccepted attempt ordinal")
        if isinstance(event, RuntimeAttemptFailedEvent):
            if state.terminal is not None:
                raise RuntimeJournalError("runtime attempt has more than one terminal event")
            if accepted != state.accepted:
                raise RuntimeJournalError("failure event names a superseded attempt")
            if event.attempt_started_at != accepted.attempt_started_at:
                raise RuntimeJournalError("failure start time differs from accepted attempt")
            failed_attempts[(event.interaction_id, event.attempt_ordinal)] = event
            states[event.interaction_id] = _InteractionState(state.accepted, event)
        elif event.completed_at < accepted.attempt_started_at:
            raise RuntimeJournalError("completion precedes its accepted attempt")
        elif isinstance(state.terminal, RuntimeCompletedEvent):
            raise RuntimeJournalError("interaction has more than one completed response")
        elif isinstance(state.terminal, RuntimeAttemptFailedEvent) and accepted == state.accepted:
            raise RuntimeJournalError("runtime attempt has more than one terminal event")
        elif accepted != state.accepted:
            if (
                isinstance(state.terminal, RuntimeAttemptFailedEvent)
                and not state.terminal.retryable
            ):
                raise RuntimeJournalError("completion follows a permanent interaction failure")
            prior = failed_attempts.get((event.interaction_id, event.attempt_ordinal))
            if prior is None or not prior.retryable:
                raise RuntimeJournalError("superseded completion lacks a retryable durable failure")
            states[event.interaction_id] = _InteractionState(state.accepted, event)
        else:
            states[event.interaction_id] = _InteractionState(accepted, event)
    return states


def _acceptance_pins(event: RuntimeAcceptedEvent) -> tuple[object, ...]:
    """Return fields that retries must preserve exactly."""
    return (
        event.interaction_id,
        event.project_id,
        event.idempotency_key_sha256,
        event.request,
        event.request_sha256,
        event.lineage_id,
        event.decision,
        event.selected_alias,
        event.selected_model,
        event.policy_input,
        event.received_at,
    )


def _require_identity(accepted: RuntimeAcceptedEvent, identity: RuntimeInteractionIdentity) -> None:
    """Reject reuse of one key with a different request, project, or lineage."""
    actual = (
        accepted.interaction_id,
        accepted.project_id,
        accepted.idempotency_key_sha256,
        accepted.request,
        accepted.request_sha256,
        accepted.lineage_id,
    )
    expected = (
        identity.interaction_id,
        identity.project_id,
        identity.idempotency_key_sha256,
        identity.request,
        identity.request_sha256,
        identity.lineage_id,
    )
    if actual != expected:
        raise RuntimeIdempotencyConflictError(
            "idempotency key was already used for different request or conversation content"
        )


def _selected_model(runtime: RouterRuntime, alias: str) -> ModelSnapshot:
    """Return the policy-pinned model snapshot for one selected candidate alias."""
    for candidate in runtime.policy.candidates:
        if candidate.alias == alias:
            return candidate.model
    raise RuntimeJournalError("routing decision selected an alias outside the frozen policy")


def _structured_completion_failure(exception: Exception) -> StructuredFailure:
    """Normalize a completion exception without retaining provider secrets."""
    if isinstance(exception, RouterModelCapabilityError):
        return StructuredFailure(
            code=FailureCode.UNSUPPORTED,
            message=str(exception),
            retryable=False,
            exception_type=type(exception).__name__,
            attribution=FailureAttribution.MODEL,
        )
    if isinstance(exception, (RouterRuntimeIntegrityError, ValueError)):
        return StructuredFailure(
            code=FailureCode.INTERNAL,
            message="routed model runtime rejected the accepted target",
            retryable=False,
            exception_type=type(exception).__name__,
            attribution=FailureAttribution.MODEL,
        )
    retryable = isinstance(exception, TimeoutError)
    code = FailureCode.TIMEOUT if retryable else FailureCode.PROVIDER
    if isinstance(exception, ProviderTransportError):
        retryable = (
            exception.status_code is None
            or exception.status_code
            in {
                408,
                409,
                425,
                429,
            }
            or (exception.status_code is not None and exception.status_code >= 500)
        )
        code = FailureCode.PROVIDER
    return StructuredFailure(
        code=code,
        message="routed model provider attempt failed",
        retryable=retryable,
        exception_type=type(exception).__name__,
        attribution=FailureAttribution.MODEL,
    )


def _routing_decision_content_id(decision: RoutingDecision) -> str:
    """Return the canonical identity required by durable routing decisions."""
    material = decision.model_dump(mode="json")
    del material["decision_id"]
    return stable_id("routing-decision", material)


def _prepare_runtime_directory(path: Path) -> None:
    """Create a private runtime directory and reject symlinked journal targets."""
    directory = path.parent
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    if directory.is_symlink() or not directory.is_dir():
        raise RuntimeJournalError("runtime journal directory must be a real directory")
    if path.is_symlink():
        raise RuntimeJournalError("runtime journal path cannot be a symbolic link")


def _attempt_failure(
    events: list[RuntimeJournalEvent], accepted: RuntimeAcceptedEvent
) -> RuntimeAttemptFailedEvent | None:
    """Return the durable failure that closed one accepted attempt, if present."""
    for event in events:
        if (
            isinstance(event, RuntimeAttemptFailedEvent)
            and event.interaction_id == accepted.interaction_id
            and event.attempt_ordinal == accepted.attempt_ordinal
        ):
            return event
    return None


def _fsync_directories(directories: tuple[Path, ...]) -> None:
    """Persist the journal and every project directory entry that names it."""
    if os.name == "nt":
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    for directory in directories:
        try:
            descriptor = os.open(directory, flags)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        except OSError as exc:
            raise RuntimeJournalError(
                f"cannot persist runtime journal directory {directory}"
            ) from exc


def _truncate_torn_tail(path: Path) -> None:
    """Remove only a non-newline-terminated final record before the next append."""
    try:
        with path.open("r+b") as handle:
            payload = handle.read()
            if not payload or payload.endswith(b"\n"):
                return
            last_newline = payload.rfind(b"\n")
            handle.seek(last_newline + 1)
            handle.truncate()
            handle.flush()
            os.fsync(handle.fileno())
    except FileNotFoundError:
        return
    except OSError as exc:
        raise RuntimeJournalError(f"cannot repair torn runtime journal {path}") from exc


def _validate_external_id(value: str, *, label: str, visible_ascii: bool) -> None:
    """Validate a caller identity before hashing and provider forwarding."""
    if not value or len(value) > 512 or value.strip() != value:
        raise ValueError(f"{label} must be 1 to 512 non-blank characters")
    if visible_ascii and any(ord(character) < 33 or ord(character) > 126 for character in value):
        raise ValueError(f"{label} must contain only visible ASCII characters")


def _require_timezone(value: datetime) -> None:
    """Require an aware timestamp at the journal boundary."""
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("runtime journal timestamps must include a timezone")
