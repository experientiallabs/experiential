"""Durable routed-interaction journal and local idempotency boundary.

The journal guarantees one local logical interaction and one durable target for each project and
idempotency key. Provider dispatch can still be at-least-once if the process crashes after the
provider succeeds but before the completion record reaches disk. Remote exactly-once behavior is
available only when the selected provider honors the explicitly forwarded idempotency key.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Annotated, Literal

from pydantic import AwareDatetime, Field, TypeAdapter, ValidationError, model_validator

from exp.common.core.artifacts import (
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
from exp.common.core.locks import file_write_lock
from exp.common.models import BillingSource, ModelRequest, ModelResponse, ModelSnapshot
from exp.common.project import ProjectPaths
from exp.common.routing import RoutingDecision
from exp.runtime.router.economics import (
    BillingSourceEconomics,
    RoutedCompletionEconomics,
    RoutedProviderComponent,
    RoutedProviderOperation,
    RoutedSpendDisposition,
    RoutedSpendLedger,
    routed_completion_economics,
    routed_spend_ledger,
    zero_operation_economics,
)
from exp.runtime.router.journal_io import (
    RuntimeJournalError,
)
from exp.runtime.router.journal_io import (
    fsync_directories as _fsync_directories,
)
from exp.runtime.router.journal_io import (
    prepare_runtime_directory as _prepare_runtime_directory,
)
from exp.runtime.router.journal_io import (
    truncate_torn_tail as _truncate_torn_tail,
)
from exp.runtime.router.journal_spend import (
    RuntimeSpendCheckpointEvent,
    direct_not_incurred_operation,
    parse_spend_event,
    rebind_settlement,
    reserve_operation,
    settle_operation,
    spend_event_content_id,
    validate_spend_events,
)
from exp.runtime.router.journal_spend import (
    live_reservation as _live_reservation,
)
from exp.runtime.router.journal_spend import (
    next_operation_ordinal as _next_operation_ordinal,
)
from exp.runtime.router.journal_spend import (
    settled_operations as _settled_sidecar_operations,
)
from exp.runtime.router.journal_validation import (
    acceptance_pins as acceptance_pins,
)
from exp.runtime.router.journal_validation import (
    candidate_reservation as _candidate_reservation,
)
from exp.runtime.router.journal_validation import (
    claim_for_existing_state as _claim_for_existing_state,
)
from exp.runtime.router.journal_validation import (
    failure_spend as _failure_spend,
)
from exp.runtime.router.journal_validation import (
    require_identity as _require_identity,
)
from exp.runtime.router.journal_validation import (
    require_interaction_spend_identity as _require_interaction_spend_identity,
)
from exp.runtime.router.journal_validation import (
    require_spend_identity as _require_spend_identity,
)
from exp.runtime.router.journal_validation import (
    validate_combined_spend as _validate_combined_spend,
)
from exp.runtime.router.journal_validation import (
    validate_events,
)


class RuntimeIdempotencyConflictError(ValueError):
    """An idempotency key was reused for different request or lineage content."""


class RuntimeInteractionInProgressError(RuntimeError):
    """Another process still owns the live provider attempt for this interaction."""

    retryable = True


class RuntimeInteractionFailedError(RuntimeError):
    """A durable provider attempt ended without a completed response."""

    def __init__(self, failure: StructuredFailure, spend: RoutedSpendLedger) -> None:
        """Retain safe failure meaning and exact alias-free cumulative spend.

        Args:
            failure: Durable redacted provider failure.
            spend: Source-attributed cumulative accounting through the failed attempt.
        """
        super().__init__(failure.message)
        self.failure = failure
        self.spend = spend
        self.retryable = failure.retryable


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


@dataclass(frozen=True)
class JournalClaim:
    """Result of one atomic attempt to claim an interaction."""

    status: Literal["needs_selection", "dispatch", "live", "completed", "failed"]
    accepted: RuntimeAcceptedEvent | None = None
    completed: RuntimeCompletedEvent | None = None
    failure: RuntimeAttemptFailedEvent | None = None


@dataclass(frozen=True)
class SpendReservationClaim:
    """Result of atomically reserving one provider-backed runtime operation."""

    status: Literal["dispatch", "live", "superseded"]
    reservation: RuntimeSpendCheckpointEvent | None = None


class RuntimeInteractionJournal:
    """Strict append-only JSONL state for routed interactions in one project."""

    def __init__(self, paths: ProjectPaths) -> None:
        """Bind the journal to one project's canonical mutable runtime path."""
        self.path = paths.runtime_journal
        self.spend_path = self.path.with_name("provider-spend.jsonl")
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

    def reserve_selection(
        self,
        identity: RuntimeInteractionIdentity,
        reservation: BillingSourceEconomics,
        *,
        now: datetime,
        stale_after: timedelta,
    ) -> SpendReservationClaim:
        """Persist an embedding reservation before any request-time provider dispatch.

        Args:
            identity: Canonical secret-free request and lineage identity.
            reservation: Alias-free embedding ceiling and billing source.
            now: Current timezone-aware time.
            stale_after: Age after which an unclosed reservation becomes ambiguous.

        Returns:
            A dispatch claim for the new reservation or a live/superseded disposition.
        """
        _require_timezone(now)
        if stale_after.total_seconds() <= 0:
            raise ValueError("stale_after must be positive")
        with file_write_lock(self.path, what="the routed-interaction journal"):
            events = list(self._read_unlocked())
            spend_events = list(self._read_spend_unlocked())
            _validate_combined_spend(events, spend_events)
            _require_interaction_spend_identity(spend_events, identity)
            state = validate_events(events).get(identity.interaction_id)
            if state is not None:
                _require_identity(state.accepted, identity)
                return SpendReservationClaim("superseded")
            live = _live_reservation(
                spend_events,
                interaction_id=identity.interaction_id,
                component=RoutedProviderComponent.ROUTER_EMBEDDING,
            )
            if live is not None:
                _require_spend_identity(live, identity)
                if now - live.recorded_at < stale_after:
                    return SpendReservationClaim("live", reservation=live)
                ambiguous = settle_operation(
                    live,
                    ordinal=len(spend_events) + 1,
                    disposition=RoutedSpendDisposition.RESERVED_AMBIGUOUS,
                    recorded_at=now,
                )
                self._append_spend_unlocked(ambiguous)
                spend_events.append(ambiguous)
            prior_sources = {
                event.operation.billing_source
                for event in spend_events
                if event.interaction_id == identity.interaction_id
                and event.operation.component == RoutedProviderComponent.ROUTER_EMBEDDING
            }
            if prior_sources and prior_sources != {reservation.billing_source}:
                raise RuntimeJournalError("router embedding billing source changed across retries")
            checkpoint = reserve_operation(
                interaction_id=identity.interaction_id,
                identity_sha256=sha256_json(identity),
                ordinal=len(spend_events) + 1,
                operation_ordinal=_next_operation_ordinal(spend_events, identity.interaction_id),
                component=RoutedProviderComponent.ROUTER_EMBEDDING,
                reservation=reservation,
                recorded_at=now,
            )
            self._append_spend_unlocked(checkpoint)
            return SpendReservationClaim("dispatch", reservation=checkpoint)

    def record_acceptance(
        self,
        identity: RuntimeInteractionIdentity,
        acceptance: RuntimeAcceptance,
        reservation: RuntimeSpendCheckpointEvent,
        settlement: RoutedProviderOperation,
        *,
        accepted_at: datetime,
    ) -> JournalClaim:
        """Atomically settle selection and persist immutable route pins.

        Args:
            identity: Canonical secret-free request and lineage identity.
            acceptance: Exact selected policy and model pins.
            reservation: Durable embedding reservation written before selection.
            settlement: Runtime-reported embedding outcome.
            accepted_at: Time selection finished and route pins became durable.

        Returns:
            A dispatch claim or a concurrently established canonical state.
        """
        _require_timezone(accepted_at)
        with file_write_lock(self.path, what="the routed-interaction journal"):
            events = list(self._read_unlocked())
            spend_events = list(self._read_spend_unlocked())
            _validate_combined_spend(events, spend_events)
            state = validate_events(events).get(identity.interaction_id)
            if state is not None:
                _require_identity(state.accepted, identity)
                return _claim_for_existing_state(state)
            _require_spend_identity(reservation, identity)
            live = _live_reservation(
                spend_events,
                interaction_id=identity.interaction_id,
                component=RoutedProviderComponent.ROUTER_EMBEDDING,
            )
            if live != reservation:
                return JournalClaim("live")
            settled = rebind_settlement(reservation, settlement)
            prior = _settled_sidecar_operations(spend_events, identity.interaction_id)
            operations = (*prior, settled)
            accepted = _accepted_event(
                identity,
                acceptance,
                spend=routed_spend_ledger(operations),
                ordinal=len(events) + 1,
                attempt_ordinal=1,
                received_at=reservation.recorded_at,
                attempt_started_at=accepted_at,
            )
            self._append_unlocked(accepted)
            return JournalClaim("dispatch", accepted=accepted)

    def record_selection_failure(
        self,
        identity: RuntimeInteractionIdentity,
        reservation: RuntimeSpendCheckpointEvent,
        *,
        failed_at: datetime,
    ) -> RuntimeSpendCheckpointEvent:
        """Settle a selection crash or error as reserved-ambiguous spend.

        Args:
            identity: Canonical request and lineage identity.
            reservation: Pre-embedding checkpoint written before provider dispatch.
            failed_at: Time the selection failed without accepted route pins.

        Returns:
            Durable ambiguous embedding settlement.
        """
        _require_timezone(failed_at)
        with file_write_lock(self.path, what="the routed-interaction journal"):
            events = list(self._read_unlocked())
            spend_events = list(self._read_spend_unlocked())
            _validate_combined_spend(events, spend_events)
            if validate_events(events).get(identity.interaction_id) is not None:
                raise RuntimeJournalError("cannot fail selection after route acceptance")
            _require_spend_identity(reservation, identity)
            live = _live_reservation(
                spend_events,
                interaction_id=identity.interaction_id,
                component=RoutedProviderComponent.ROUTER_EMBEDDING,
            )
            if live != reservation:
                latest = validate_spend_events(spend_events).get(reservation.operation.operation_id)
                if latest is None:
                    raise RuntimeJournalError("selection reservation is absent from spend journal")
                return latest
            failed = settle_operation(
                reservation,
                ordinal=len(spend_events) + 1,
                disposition=RoutedSpendDisposition.RESERVED_AMBIGUOUS,
                recorded_at=failed_at,
            )
            self._append_spend_unlocked(failed)
            return failed

    def reserve_candidate(
        self,
        accepted: RuntimeAcceptedEvent,
        reservation: BillingSourceEconomics,
        *,
        now: datetime,
    ) -> SpendReservationClaim:
        """Persist selected-candidate reservation before provider dispatch.

        Args:
            accepted: Exact live accepted attempt.
            reservation: Candidate ceiling and immutable billing source.
            now: Current timezone-aware reservation time.

        Returns:
            Dispatch ownership or a live/superseded disposition.
        """
        _require_timezone(now)
        with file_write_lock(self.path, what="the routed-interaction journal"):
            events = list(self._read_unlocked())
            spend_events = list(self._read_spend_unlocked())
            _validate_combined_spend(events, spend_events)
            state = validate_events(events).get(accepted.interaction_id)
            if (
                state is None
                or not _same_accepted_event(state.accepted, accepted)
                or state.terminal is not None
            ):
                return SpendReservationClaim("superseded")
            expected_source = accepted.acceptance.selected_model.billing_source
            if reservation.billing_source != expected_source:
                raise RuntimeJournalError("candidate billing source differs from accepted model")
            live = _candidate_reservation(spend_events, accepted)
            if live is not None:
                return SpendReservationClaim("live", reservation=live)
            checkpoint = reserve_operation(
                interaction_id=accepted.interaction_id,
                identity_sha256=sha256_json(accepted.identity),
                ordinal=len(spend_events) + 1,
                operation_ordinal=max(
                    _next_operation_ordinal(spend_events, accepted.interaction_id),
                    len(accepted.spend.operations) + 1,
                ),
                component=RoutedProviderComponent.SELECTED_CANDIDATE,
                reservation=reservation,
                recorded_at=now,
                accepted_attempt_ordinal=accepted.attempt_ordinal,
            )
            self._append_spend_unlocked(checkpoint)
            return SpendReservationClaim("dispatch", reservation=checkpoint)

    def claim(
        self,
        identity: RuntimeInteractionIdentity,
        *,
        now: datetime,
        stale_after: timedelta,
    ) -> JournalClaim:
        """Atomically inspect, create, or retry one provider attempt.

        Args:
            identity: Secret-free interaction identity derived from caller input.
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
            spend_events = list(self._read_spend_unlocked())
            _validate_combined_spend(events, spend_events)
            _require_interaction_spend_identity(spend_events, identity)
            states = validate_events(events)
            state = states.get(identity.interaction_id)
            if state is None:
                live_selection = _live_reservation(
                    spend_events,
                    interaction_id=identity.interaction_id,
                    component=RoutedProviderComponent.ROUTER_EMBEDDING,
                )
                if live_selection is None:
                    return JournalClaim("needs_selection")
                _require_spend_identity(live_selection, identity)
                if now - live_selection.recorded_at < stale_after:
                    return JournalClaim("live")
                ambiguous = settle_operation(
                    live_selection,
                    ordinal=len(spend_events) + 1,
                    disposition=RoutedSpendDisposition.RESERVED_AMBIGUOUS,
                    recorded_at=now,
                )
                self._append_spend_unlocked(ambiguous)
                return JournalClaim("needs_selection")
            _require_identity(state.accepted, identity)
            terminal = state.terminal
            if isinstance(terminal, RuntimeCompletedEvent):
                return JournalClaim("completed", accepted=state.accepted, completed=terminal)
            if isinstance(terminal, RuntimeAttemptFailedEvent) and not terminal.retryable:
                return JournalClaim("failed", accepted=state.accepted, failure=terminal)
            if terminal is None:
                candidate_reservation = _candidate_reservation(spend_events, state.accepted)
                if candidate_reservation is None:
                    return JournalClaim("dispatch", accepted=state.accepted)
                if now - candidate_reservation.recorded_at < stale_after:
                    return JournalClaim("live", accepted=state.accepted)
                stale_failure = StructuredFailure(
                    code=FailureCode.TIMEOUT,
                    message="prior routed model attempt became stale",
                    retryable=True,
                    attribution=FailureAttribution.MODEL,
                )
                failed = _failed_event(
                    state.accepted,
                    stale_failure,
                    spend=_failure_spend(
                        state.accepted,
                        candidate_reservation,
                        disposition=RoutedSpendDisposition.RESERVED_AMBIGUOUS,
                    ),
                    ordinal=len(events) + 1,
                    failed_at=now,
                )
                self._append_unlocked(failed)
                events.append(failed)
            accepted = _accepted_event(
                identity,
                state.accepted.acceptance,
                spend=(
                    terminal.spend
                    if isinstance(terminal, RuntimeAttemptFailedEvent)
                    else failed.spend
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
            spend_events = list(self._read_spend_unlocked())
            _validate_combined_spend(events, spend_events)
            state = validate_events(events).get(accepted.interaction_id)
            if state is None or not any(
                isinstance(event, RuntimeAcceptedEvent) and _same_accepted_event(event, accepted)
                for event in events
            ):
                raise RuntimeJournalError(
                    "cannot fail an interaction attempt that was not accepted"
                )
            if isinstance(state.terminal, RuntimeCompletedEvent):
                return state.terminal
            if not _same_accepted_event(state.accepted, accepted):
                prior = _attempt_failure(events, accepted)
                if prior is None:
                    raise RuntimeJournalError("superseded attempt has no durable failure")
                return prior
            if state.terminal is not None:
                if isinstance(state.terminal, RuntimeAttemptFailedEvent):
                    return state.terminal
                raise RuntimeJournalError("cannot fail an interaction after completion")
            reservation = _candidate_reservation(spend_events, accepted)
            if reservation is None:
                not_incurred = direct_not_incurred_operation(
                    interaction_id=accepted.interaction_id,
                    operation_ordinal=len(accepted.spend.operations) + 1,
                    component=RoutedProviderComponent.SELECTED_CANDIDATE,
                    billing=BillingSourceEconomics(
                        billing_source=accepted.acceptance.selected_model.billing_source,
                        economics=zero_operation_economics(),
                    ),
                )
                spend = routed_spend_ledger((*accepted.spend.operations, not_incurred))
            else:
                spend = _failure_spend(
                    accepted,
                    reservation,
                    disposition=RoutedSpendDisposition.RESERVED_AMBIGUOUS,
                )
            event = _failed_event(
                accepted,
                failure,
                spend=spend,
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
        candidate_operation: RoutedProviderOperation,
        completed_at: datetime,
    ) -> RuntimeAttemptFailedEvent | RuntimeCompletedEvent:
        """Commit the first response or observe an earlier permanent failure."""
        _require_timezone(completed_at)
        with file_write_lock(self.path, what="the routed-interaction journal"):
            events = list(self._read_unlocked())
            spend_events = list(self._read_spend_unlocked())
            _validate_combined_spend(events, spend_events)
            state = validate_events(events).get(accepted.interaction_id)
            if state is None or not any(
                isinstance(event, RuntimeAcceptedEvent) and _same_accepted_event(event, accepted)
                for event in events
            ):
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
            if not _same_accepted_event(state.accepted, accepted):
                prior = _attempt_failure(events, accepted)
                if prior is None or not prior.retryable:
                    raise RuntimeJournalError(
                        "superseded attempt lacks a retryable durable failure"
                    )
            elif state.terminal is not None:
                raise RuntimeJournalError("cannot complete a terminal interaction attempt")
            reservation = _candidate_reservation(spend_events, accepted)
            if reservation is None:
                raise RuntimeJournalError("completed candidate attempt has no durable reservation")
            settled_candidate = rebind_settlement(reservation, candidate_operation)
            operations = list(state.accepted.spend.operations)
            existing_index = next(
                (
                    index
                    for index, item in enumerate(operations)
                    if item.operation_id == settled_candidate.operation_id
                ),
                None,
            )
            if existing_index is None:
                operations.append(settled_candidate)
            else:
                operations[existing_index] = settled_candidate
            current_reservation = _candidate_reservation(spend_events, state.accepted)
            if (
                current_reservation is not None
                and current_reservation.operation.operation_id != settled_candidate.operation_id
            ):
                operations.append(
                    settle_operation(
                        current_reservation,
                        ordinal=len(spend_events) + 1,
                        disposition=RoutedSpendDisposition.RESERVED_AMBIGUOUS,
                        recorded_at=completed_at,
                    ).operation
                )
            operations.sort(key=lambda item: item.operation_ordinal)
            economics = routed_completion_economics(tuple(operations))
            event = _completed_event(
                accepted,
                response,
                economics,
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

    def _append_spend_unlocked(self, event: RuntimeSpendCheckpointEvent) -> None:
        """Append and fsync one pre-dispatch spend checkpoint."""
        if event.event_id != spend_event_content_id(event):
            raise RuntimeJournalError("runtime spend event ID differs from its canonical content")
        _prepare_runtime_directory(self.spend_path)
        _truncate_torn_tail(self.spend_path)
        flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(self.spend_path, flags, 0o600)
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "ab", closefd=True) as handle:
                handle.write(canonical_json_bytes(event) + b"\n")
                handle.flush()
                os.fsync(handle.fileno())
            _fsync_directories(self._durability_directories)
        except OSError as exc:
            raise RuntimeJournalError(
                f"cannot append runtime spend journal {self.spend_path}"
            ) from exc


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


def _same_accepted_event(first: RuntimeAcceptedEvent, second: RuntimeAcceptedEvent) -> bool:
    """Compare durable accepted-attempt content without execution-only model fields."""
    return first.model_dump(mode="json") == second.model_dump(mode="json")


def _require_timezone(value: datetime) -> None:
    """Require an aware timestamp at the journal boundary."""
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("runtime journal timestamps must include a timezone")
