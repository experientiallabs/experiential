"""Validated read surface and event contracts for routed-interaction journals.

The journal binds one project's append-only JSONL interaction records and their provider-spend
sidecar, cross-validates both files on every read, and defines the canonical event models
consumed by runtime trace snapshots and managed SFT dataset builds.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from typing import Annotated, Literal

from pydantic import AwareDatetime, Field, TypeAdapter, ValidationError, model_validator

from wmo.common.core.artifacts import (
    ArtifactInput,
    ContractModel,
    Sha256,
    StructuredFailure,
    sha256_json,
    stable_id,
)
from wmo.common.core.locks import file_write_lock
from wmo.common.models import BillingSource, ModelRequest, ModelResponse, ModelSnapshot
from wmo.common.project import ProjectPaths
from wmo.common.routing import RoutingDecision
from wmo.runtime.router.economics import RoutedCompletionEconomics, RoutedSpendLedger
from wmo.runtime.router.journal_io import RuntimeJournalError
from wmo.runtime.router.journal_spend import (
    RuntimeSpendCheckpointEvent,
    parse_spend_event,
    validate_spend_events,
)
from wmo.runtime.router.journal_validation import (
    acceptance_pins as acceptance_pins,
)
from wmo.runtime.router.journal_validation import (
    validate_combined_spend as _validate_combined_spend,
)
from wmo.runtime.router.journal_validation import (
    validate_events,
)


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
    router_embedding_billing_source: BillingSource
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
    identity: RuntimeInteractionIdentity
    acceptance: RuntimeAcceptance
    spend: RoutedSpendLedger
    received_at: AwareDatetime
    attempt_started_at: AwareDatetime

    @model_validator(mode="after")
    def _require_matching_pins(self) -> RuntimeAcceptedEvent:
        if self.attempt_started_at < self.received_at:
            raise ValueError("accepted attempt start precedes request receipt")
        return self

    @property
    def interaction_id(self) -> str:
        """Content identity of the logical interaction this attempt belongs to."""
        return self.identity.interaction_id


class RuntimeAttemptFailedEvent(ContractModel):
    """One completion attempt that ended without a response target."""

    event: Literal["attempt_failed"] = "attempt_failed"
    event_id: str = Field(pattern=r"^runtime-event-[0-9a-f]{20}$")
    ordinal: int = Field(gt=0)
    attempt_ordinal: int = Field(gt=0)
    interaction_id: str = Field(pattern=r"^interaction-[0-9a-f]{20}$")
    failure: StructuredFailure
    retryable: bool
    spend: RoutedSpendLedger
    spend_sha256: Sha256
    attempt_started_at: AwareDatetime
    failed_at: AwareDatetime

    @model_validator(mode="after")
    def _require_matching_failure(self) -> RuntimeAttemptFailedEvent:
        if self.retryable != self.failure.retryable:
            raise ValueError("attempt retryability differs from its structured failure")
        if self.failed_at < self.attempt_started_at:
            raise ValueError("attempt failure precedes its start")
        if self.spend_sha256 != sha256_json(self.spend):
            raise ValueError("attempt failure spend digest differs from canonical spend")
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
    economics: RoutedCompletionEconomics
    economics_sha256: Sha256
    completed_at: AwareDatetime

    @model_validator(mode="after")
    def _require_response_digest(self) -> RuntimeCompletedEvent:
        if self.response_sha256 != sha256_json(self.response):
            raise ValueError("completed response digest differs from canonical response")
        if self.economics_sha256 != sha256_json(self.economics):
            raise ValueError("completed economics digest differs from canonical economics")
        return self


RuntimeJournalEvent = Annotated[
    RuntimeAcceptedEvent | RuntimeAttemptFailedEvent | RuntimeCompletedEvent,
    Field(discriminator="event"),
]
_EVENT_ADAPTER: TypeAdapter[RuntimeJournalEvent] = TypeAdapter(RuntimeJournalEvent)


@dataclass(frozen=True)
class _InteractionState:
    """Validated latest state for one interaction."""

    accepted: RuntimeAcceptedEvent
    terminal: RuntimeAttemptFailedEvent | RuntimeCompletedEvent | None


class RuntimeInteractionJournal:
    """Strict append-only JSONL state for routed interactions in one project."""

    def __init__(self, paths: ProjectPaths) -> None:
        """Bind the journal to one project's canonical mutable runtime path."""
        self.path = paths.runtime_journal
        self.spend_path = self.path.with_name("provider-spend.jsonl")
        self.project_id = paths.project_id

    def read_events(self) -> tuple[RuntimeJournalEvent, ...]:
        """Read and fully validate all durable records, ignoring only a torn final line."""
        with file_write_lock(self.path, what="the routed-interaction journal"):
            events = self._read_unlocked()
            spend_events = self._read_spend_unlocked()
            _validate_combined_spend(events, spend_events)
            return events

    def read_spend_events(self) -> tuple[RuntimeSpendCheckpointEvent, ...]:
        """Read and fully validate the durable pre-dispatch spend checkpoints."""
        with file_write_lock(self.path, what="the routed-interaction journal"):
            events = self._read_unlocked()
            spend_events = self._read_spend_unlocked()
            _validate_combined_spend(events, spend_events)
            return spend_events

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
        validate_events(events)
        return tuple(events)

    def _read_spend_unlocked(self) -> tuple[RuntimeSpendCheckpointEvent, ...]:
        """Read strict spend checkpoints while tolerating only one torn final line."""
        try:
            payload = self.spend_path.read_bytes()
        except FileNotFoundError:
            return ()
        except OSError as exc:
            raise RuntimeJournalError(
                f"cannot read runtime spend journal {self.spend_path}"
            ) from exc
        lines = payload.splitlines(keepends=True)
        if lines and not lines[-1].endswith(b"\n"):
            lines.pop()
        events: list[RuntimeSpendCheckpointEvent] = []
        for index, line in enumerate(lines, start=1):
            if not line.strip():
                raise RuntimeJournalError(
                    f"runtime spend journal has a blank interior line {index}"
                )
            try:
                events.append(parse_spend_event(line))
            except (UnicodeDecodeError, ValidationError, ValueError) as exc:
                raise RuntimeJournalError(
                    f"runtime spend journal has invalid interior line {index}"
                ) from exc
        try:
            validate_spend_events(events)
        except ValueError as exc:
            raise RuntimeJournalError(str(exc)) from exc
        return tuple(events)


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
    spend: RoutedSpendLedger,
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
        identity=identity,
        acceptance=acceptance,
        spend=spend,
        received_at=received_at,
        attempt_started_at=attempt_started_at,
    )
    return provisional.model_copy(update={"event_id": _event_content_id(provisional)})


def _failed_event(
    accepted: RuntimeAcceptedEvent,
    failure: StructuredFailure,
    *,
    spend: RoutedSpendLedger,
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
        spend=spend,
        spend_sha256=sha256_json(spend),
        attempt_started_at=accepted.attempt_started_at,
        failed_at=failed_at,
    )
    return provisional.model_copy(update={"event_id": _event_content_id(provisional)})


def _completed_event(
    accepted: RuntimeAcceptedEvent,
    response: ModelResponse,
    economics: RoutedCompletionEconomics,
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
        economics=economics,
        economics_sha256=sha256_json(economics),
        completed_at=completed_at,
    )
    return provisional.model_copy(update={"event_id": _event_content_id(provisional)})


def _event_content_id(event: RuntimeJournalEvent) -> str:
    """Return the stable content identity for one event, excluding its own ID."""
    material = event.model_dump(mode="json")
    del material["event_id"]
    return stable_id("runtime-event", material)
