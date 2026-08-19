"""Cross-file state validation for durable routed interaction and spend journals."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

from wmo.common.core.artifacts import sha256_json, stable_id
from wmo.common.routing import RoutingDecision
from wmo.runtime.router.economics import (
    RoutedProviderComponent,
    RoutedProviderOperation,
    RoutedSpendDisposition,
    RoutedSpendLedger,
    routed_spend_ledger,
)
from wmo.runtime.router.journal_spend import (
    RuntimeSpendCheckpointEvent,
    live_reservation,
    settle_operation,
    validate_spend_events,
)
from wmo.runtime.router.journal_spend import (
    require_identity as require_sidecar_identity,
)

if TYPE_CHECKING:
    from wmo.runtime.router.journal import (
        JournalClaim,
        RuntimeAcceptedEvent,
        RuntimeInteractionIdentity,
        RuntimeJournalEvent,
        _InteractionState,
    )


def claim_for_existing_state(state: _InteractionState) -> JournalClaim:
    """Return the non-mutating claim represented by one validated interaction state."""
    from wmo.runtime.router.journal import (
        JournalClaim,
        RuntimeAttemptFailedEvent,
        RuntimeCompletedEvent,
    )

    terminal = state.terminal
    if isinstance(terminal, RuntimeCompletedEvent):
        return JournalClaim("completed", accepted=state.accepted, completed=terminal)
    if isinstance(terminal, RuntimeAttemptFailedEvent) and not terminal.retryable:
        return JournalClaim("failed", accepted=state.accepted, failure=terminal)
    if terminal is None:
        return JournalClaim("live", accepted=state.accepted)
    return JournalClaim("live", accepted=state.accepted, failure=terminal)


def candidate_reservation(
    spend_events: list[RuntimeSpendCheckpointEvent] | tuple[RuntimeSpendCheckpointEvent, ...],
    accepted: RuntimeAcceptedEvent,
) -> RuntimeSpendCheckpointEvent | None:
    """Return the live reservation for one exact accepted candidate attempt."""
    from wmo.runtime.router.journal import RuntimeJournalError

    try:
        return live_reservation(
            spend_events,
            interaction_id=accepted.interaction_id,
            component=RoutedProviderComponent.SELECTED_CANDIDATE,
            accepted_attempt_ordinal=accepted.attempt_ordinal,
        )
    except ValueError as exc:
        raise RuntimeJournalError(str(exc)) from exc


def failure_spend(
    accepted: RuntimeAcceptedEvent,
    reservation: RuntimeSpendCheckpointEvent,
    *,
    disposition: RoutedSpendDisposition,
) -> RoutedSpendLedger:
    """Append one conservative candidate failure disposition to accepted spend."""
    settlement = settle_operation(
        reservation,
        ordinal=reservation.ordinal + 1,
        disposition=disposition,
        recorded_at=reservation.recorded_at,
    ).operation
    return routed_spend_ledger((*accepted.spend.operations, settlement))


def require_spend_identity(
    event: RuntimeSpendCheckpointEvent,
    identity: RuntimeInteractionIdentity,
) -> None:
    """Reject a live reservation reused for different request or lineage content."""
    from wmo.runtime.router.journal import RuntimeIdempotencyConflictError

    try:
        require_sidecar_identity(event, sha256_json(identity))
    except ValueError as exc:
        raise RuntimeIdempotencyConflictError(str(exc)) from exc


def require_interaction_spend_identity(
    events: tuple[RuntimeSpendCheckpointEvent, ...] | list[RuntimeSpendCheckpointEvent],
    identity: RuntimeInteractionIdentity,
) -> None:
    """Reject a changed request or lineage against any prior sidecar history.

    Args:
        events: Validated spend checkpoints for the project journal.
        identity: Current caller request and lineage identity.

    Raises:
        RuntimeIdempotencyConflictError: The interaction already names another identity.
    """
    for event in events:
        if event.interaction_id == identity.interaction_id:
            require_spend_identity(event, identity)


def validate_combined_spend(
    events: tuple[RuntimeJournalEvent, ...] | list[RuntimeJournalEvent],
    spend_events: tuple[RuntimeSpendCheckpointEvent, ...] | list[RuntimeSpendCheckpointEvent],
) -> None:
    """Cross-bind sidecar reservations to immutable route and terminal records."""
    from wmo.runtime.router.journal import (
        RuntimeAcceptedEvent,
        RuntimeAttemptFailedEvent,
        RuntimeCompletedEvent,
        RuntimeJournalError,
    )

    try:
        validate_spend_events(spend_events)
    except ValueError as exc:
        raise RuntimeJournalError(str(exc)) from exc
    states = validate_events(events)
    accepted_by_attempt = {
        (event.interaction_id, event.attempt_ordinal): event
        for event in events
        if isinstance(event, RuntimeAcceptedEvent)
    }
    main_operations: list[tuple[str, RoutedProviderOperation]] = []
    for event in events:
        if isinstance(event, RuntimeAcceptedEvent):
            operations = event.spend.operations
        elif isinstance(event, RuntimeAttemptFailedEvent):
            operations = event.spend.operations
        else:
            operations = event.economics.operations
        main_operations.extend((event.interaction_id, operation) for operation in operations)
    reservations = {
        event.operation.operation_id: event
        for event in spend_events
        if event.operation.disposition == RoutedSpendDisposition.RESERVED
    }
    for checkpoint in spend_events:
        state = states.get(checkpoint.interaction_id)
        if state is not None and checkpoint.identity_sha256 != sha256_json(state.accepted.identity):
            raise RuntimeJournalError("runtime spend checkpoint differs from accepted identity")
        if checkpoint.accepted_attempt_ordinal is not None:
            accepted = accepted_by_attempt.get(
                (checkpoint.interaction_id, checkpoint.accepted_attempt_ordinal)
            )
            if accepted is None:
                raise RuntimeJournalError("candidate spend checkpoint names an unknown attempt")
            if (
                checkpoint.operation.billing_source
                != accepted.acceptance.selected_model.billing_source
            ):
                raise RuntimeJournalError("candidate spend checkpoint changes billing source")
        if checkpoint.operation.disposition != RoutedSpendDisposition.RESERVED:
            if (
                not any(
                    interaction_id == checkpoint.interaction_id
                    and operation == checkpoint.operation
                    for interaction_id, operation in main_operations
                )
                and state is not None
            ):
                raise RuntimeJournalError("settled spend checkpoint is absent from accepted spend")
    for interaction_id, operation in main_operations:
        reservation = reservations.get(operation.operation_id)
        if (
            operation.disposition == RoutedSpendDisposition.DEFINITELY_NOT_INCURRED
            and operation.component == RoutedProviderComponent.SELECTED_CANDIDATE
        ):
            if reservation is not None:
                raise RuntimeJournalError(
                    "not-incurred candidate spend cannot follow a dispatch reservation"
                )
            continue
        if reservation is None or reservation.interaction_id != interaction_id:
            raise RuntimeJournalError("incurred runtime spend lacks a pre-dispatch reservation")
        stable_reservation = reservation.operation.model_dump(
            mode="json", exclude={"disposition", "operation_count", "economics"}
        )
        stable_operation = operation.model_dump(
            mode="json", exclude={"disposition", "operation_count", "economics"}
        )
        if stable_reservation != stable_operation:
            raise RuntimeJournalError("settled runtime spend drifted from its reservation")
    latest_checkpoints = validate_spend_events(spend_events)
    interaction_ids = set(states) | {
        checkpoint.interaction_id for checkpoint in latest_checkpoints.values()
    }
    for interaction_id in interaction_ids:
        state = states.get(interaction_id)
        if state is None:
            main: tuple[RoutedProviderOperation, ...] = ()
        elif isinstance(state.terminal, RuntimeCompletedEvent):
            main = state.terminal.economics.operations
        elif isinstance(state.terminal, RuntimeAttemptFailedEvent):
            main = state.terminal.spend.operations
        else:
            main = state.accepted.spend.operations
        by_ordinal = {item.operation_ordinal: item.operation_id for item in main}
        for checkpoint in latest_checkpoints.values():
            if checkpoint.interaction_id != interaction_id:
                continue
            ordinal = checkpoint.operation.operation_ordinal
            owner = by_ordinal.setdefault(ordinal, checkpoint.operation.operation_id)
            if owner != checkpoint.operation.operation_id:
                raise RuntimeJournalError("runtime spend operation ordinal has multiple owners")
        if tuple(sorted(by_ordinal)) != tuple(range(1, len(by_ordinal) + 1)):
            raise RuntimeJournalError(
                "combined runtime spend operation ordinals are not contiguous"
            )


def validate_events(
    events: list[RuntimeJournalEvent] | tuple[RuntimeJournalEvent, ...],
) -> dict[str, _InteractionState]:
    """Validate global order, content digests, and every interaction transition."""
    from wmo.runtime.router.journal import (
        RuntimeAcceptedEvent,
        RuntimeAttemptFailedEvent,
        RuntimeCompletedEvent,
        RuntimeJournalError,
        _event_content_id,
        _InteractionState,
    )

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
            identity = event.identity
            decision = event.acceptance.decision
            embedding_operations = tuple(
                item
                for item in event.spend.operations
                if item.component == RoutedProviderComponent.ROUTER_EMBEDDING
            )
            candidate_operations = tuple(
                item
                for item in event.spend.operations
                if item.component == RoutedProviderComponent.SELECTED_CANDIDATE
            )
            if not embedding_operations:
                raise RuntimeJournalError("accepted route omits router embedding accounting")
            if any(
                item.billing_source != event.acceptance.router_embedding_billing_source
                for item in embedding_operations
            ):
                raise RuntimeJournalError("accepted embedding spend changes billing source")
            if any(
                item.billing_source != event.acceptance.selected_model.billing_source
                for item in candidate_operations
            ):
                raise RuntimeJournalError("accepted candidate spend changes billing source")
            expected_interaction_id = stable_id(
                "interaction",
                {
                    "project_id": identity.project_id,
                    "idempotency_key_sha256": identity.idempotency_key_sha256,
                },
            )
            if event.interaction_id != expected_interaction_id:
                raise RuntimeJournalError("interaction ID differs from project and key digest")
            expected_episode_sha256 = hashlib.sha256(
                identity.lineage_id.encode("utf-8"), usedforsecurity=False
            ).hexdigest()
            if decision.episode_id_sha256 != expected_episode_sha256:
                raise RuntimeJournalError("routing decision differs from accepted lineage")
            if decision.decision_id != routing_decision_content_id(decision):
                raise RuntimeJournalError("routing decision ID differs from its canonical content")
            key = (identity.project_id, identity.idempotency_key_sha256)
            owner = key_owners.setdefault(key, event.interaction_id)
            if owner != event.interaction_id:
                raise RuntimeJournalError("idempotency key digest maps to multiple interactions")
            if state is None:
                if event.attempt_ordinal != 1:
                    raise RuntimeJournalError("first interaction attempt must have ordinal one")
                if candidate_operations:
                    raise RuntimeJournalError("first acceptance cannot precede candidate spend")
            else:
                if not isinstance(state.terminal, RuntimeAttemptFailedEvent):
                    raise RuntimeJournalError("accepted retry does not follow a failed attempt")
                if not state.terminal.retryable:
                    raise RuntimeJournalError("accepted retry follows a permanent failure")
                if event.attempt_ordinal != state.accepted.attempt_ordinal + 1:
                    raise RuntimeJournalError("interaction attempt ordinals are not contiguous")
                if acceptance_pins(event) != acceptance_pins(state.accepted):
                    raise RuntimeJournalError("retry drifted from the original accepted pins")
                if event.spend != state.terminal.spend:
                    raise RuntimeJournalError("retry dropped or changed prior provider spend")
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
            _require_terminal_spend_extension(accepted, event.spend)
            failed_attempts[(event.interaction_id, event.attempt_ordinal)] = event
            states[event.interaction_id] = _InteractionState(state.accepted, event)
        elif event.completed_at < accepted.attempt_started_at:
            raise RuntimeJournalError("completion precedes its accepted attempt")
        elif not _spend_contains_exact_prefix(
            event.economics.operations, accepted.spend.operations
        ):
            raise RuntimeJournalError("completion dropped or changed accepted provider spend")
        elif not any(
            item.component == RoutedProviderComponent.SELECTED_CANDIDATE
            and item.billing_source == accepted.acceptance.selected_model.billing_source
            for item in event.economics.operations
        ):
            raise RuntimeJournalError("completion omits selected-candidate spend accounting")
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


def acceptance_pins(event: RuntimeAcceptedEvent) -> tuple[object, ...]:
    """Return fields that retries must preserve exactly."""
    return (event.identity, event.acceptance, event.received_at)


def require_identity(
    accepted: RuntimeAcceptedEvent,
    identity: RuntimeInteractionIdentity,
) -> None:
    """Reject reuse of one key with a different request, project, or lineage."""
    from wmo.runtime.router.journal import RuntimeIdempotencyConflictError

    if accepted.identity.model_dump(mode="json") != identity.model_dump(mode="json"):
        raise RuntimeIdempotencyConflictError(
            "idempotency key was already used for different request or conversation content"
        )


def routing_decision_content_id(decision: RoutingDecision) -> str:
    """Return the canonical identity required by durable routing decisions."""
    material = decision.model_dump(mode="json")
    del material["decision_id"]
    return stable_id("routing-decision", material)


def _spend_contains_exact_prefix(
    operations: tuple[RoutedProviderOperation, ...],
    prefix: tuple[RoutedProviderOperation, ...],
) -> bool:
    """Return whether cumulative operations preserve an exact settled prefix."""
    return len(operations) >= len(prefix) and operations[: len(prefix)] == prefix


def _require_terminal_spend_extension(
    accepted: RuntimeAcceptedEvent,
    spend: RoutedSpendLedger,
) -> None:
    """Require a terminal attempt to append one source-bound candidate disposition."""
    from wmo.runtime.router.journal import RuntimeJournalError

    if not _spend_contains_exact_prefix(spend.operations, accepted.spend.operations):
        raise RuntimeJournalError("attempt failure dropped or changed accepted provider spend")
    additions = spend.operations[len(accepted.spend.operations) :]
    if len(additions) != 1:
        raise RuntimeJournalError("attempt failure must settle exactly one candidate operation")
    operation = additions[0]
    if (
        operation.component != RoutedProviderComponent.SELECTED_CANDIDATE
        or operation.billing_source != accepted.acceptance.selected_model.billing_source
    ):
        raise RuntimeJournalError("attempt failure candidate spend differs from accepted model")
