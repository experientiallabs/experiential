"""Append-only request-time spend checkpoints for the routed runtime journal."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import AwareDatetime, Field, TypeAdapter, model_validator

from exp.common.core.artifacts import ContractModel, Sha256, stable_id
from exp.runtime.router.economics import (
    BillingSourceEconomics,
    RoutedProviderComponent,
    RoutedProviderOperation,
    RoutedSpendDisposition,
)


class RuntimeSpendCheckpointEvent(ContractModel):
    """One durable reservation or settlement for an alias-free provider operation."""

    event: Literal["provider_spend_checkpoint"] = "provider_spend_checkpoint"
    event_id: str = Field(pattern=r"^runtime-spend-event-[0-9a-f]{20}$")
    ordinal: int = Field(gt=0)
    interaction_id: str = Field(pattern=r"^interaction-[0-9a-f]{20}$")
    identity_sha256: Sha256
    accepted_attempt_ordinal: int | None = Field(default=None, gt=0)
    operation: RoutedProviderOperation
    recorded_at: AwareDatetime

    @model_validator(mode="after")
    def _require_component_attempt_scope(self) -> RuntimeSpendCheckpointEvent:
        """Bind candidate checkpoints to an accepted attempt and embeddings to selection."""
        if self.operation.component == RoutedProviderComponent.SELECTED_CANDIDATE:
            if self.accepted_attempt_ordinal is None:
                raise ValueError("candidate spend checkpoints require an accepted attempt")
        elif self.accepted_attempt_ordinal is not None:
            raise ValueError("embedding spend checkpoints cannot name a candidate attempt")
        return self


_SPEND_EVENT_ADAPTER: TypeAdapter[RuntimeSpendCheckpointEvent] = TypeAdapter(
    RuntimeSpendCheckpointEvent
)


def parse_spend_event(payload: bytes) -> RuntimeSpendCheckpointEvent:
    """Parse one strict spend checkpoint JSON record.

    Args:
        payload: One newline-free canonical JSON record.

    Returns:
        Validated spend checkpoint event.
    """
    return _SPEND_EVENT_ADAPTER.validate_json(payload)


def reserve_operation(
    *,
    interaction_id: str,
    identity_sha256: Sha256,
    ordinal: int,
    operation_ordinal: int,
    component: RoutedProviderComponent,
    reservation: BillingSourceEconomics,
    recorded_at: datetime,
    accepted_attempt_ordinal: int | None = None,
) -> RuntimeSpendCheckpointEvent:
    """Create one content-addressed pre-dispatch provider reservation.

    Args:
        interaction_id: Secret-free logical interaction identity.
        identity_sha256: Canonical request and lineage identity digest.
        ordinal: Global spend-journal event ordinal.
        operation_ordinal: Interaction-local possible-operation ordinal.
        component: Router embedding or selected candidate.
        reservation: Conservative source-attributed economics.
        recorded_at: Durable reservation timestamp.
        accepted_attempt_ordinal: Candidate attempt whose dispatch is reserved.

    Returns:
        Content-addressed reservation checkpoint.
    """
    operation_id = stable_id(
        "routed-operation",
        {
            "interaction_id": interaction_id,
            "operation_ordinal": operation_ordinal,
            "component": component.value,
        },
    )
    operation = RoutedProviderOperation(
        operation_id=operation_id,
        operation_ordinal=operation_ordinal,
        component=component,
        billing_source=reservation.billing_source,
        disposition=RoutedSpendDisposition.RESERVED,
        operation_count=1,
        economics=reservation.economics,
    )
    return _checkpoint_event(
        interaction_id=interaction_id,
        identity_sha256=identity_sha256,
        ordinal=ordinal,
        accepted_attempt_ordinal=accepted_attempt_ordinal,
        operation=operation,
        recorded_at=recorded_at,
    )


def settle_operation(
    reservation: RuntimeSpendCheckpointEvent,
    *,
    ordinal: int,
    disposition: RoutedSpendDisposition,
    economics: object | None = None,
    recorded_at: datetime,
) -> RuntimeSpendCheckpointEvent:
    """Settle one reservation without changing its identity or billing source.

    Args:
        reservation: Exact live reservation being settled.
        ordinal: Global spend-journal event ordinal.
        disposition: Final observed, locally-priced, ambiguous, or not-incurred state.
        economics: Optional exact ``OperationEconomics`` replacement for successful settlement.
        recorded_at: Settlement timestamp.

    Returns:
        Content-addressed settlement checkpoint.
    """
    from exp.common.models import OperationEconomics

    if disposition == RoutedSpendDisposition.RESERVED:
        raise ValueError("a spend settlement cannot remain reserved")
    settled_economics = (
        reservation.operation.economics
        if economics is None
        else OperationEconomics.model_validate(economics)
    )
    operation = reservation.operation.model_copy(
        update={
            "disposition": disposition,
            "operation_count": (
                0 if disposition == RoutedSpendDisposition.DEFINITELY_NOT_INCURRED else 1
            ),
            "economics": settled_economics,
        }
    )
    return _checkpoint_event(
        interaction_id=reservation.interaction_id,
        identity_sha256=reservation.identity_sha256,
        ordinal=ordinal,
        accepted_attempt_ordinal=reservation.accepted_attempt_ordinal,
        operation=operation,
        recorded_at=recorded_at,
    )


def direct_not_incurred_operation(
    *,
    interaction_id: str,
    operation_ordinal: int,
    component: RoutedProviderComponent,
    billing: BillingSourceEconomics,
) -> RoutedProviderOperation:
    """Create a zero-count operation when provider dispatch was proven absent."""
    from exp.runtime.router.economics import zero_operation_economics

    operation_id = stable_id(
        "routed-operation",
        {
            "interaction_id": interaction_id,
            "operation_ordinal": operation_ordinal,
            "component": component.value,
        },
    )
    return RoutedProviderOperation(
        operation_id=operation_id,
        operation_ordinal=operation_ordinal,
        component=component,
        billing_source=billing.billing_source,
        disposition=RoutedSpendDisposition.DEFINITELY_NOT_INCURRED,
        operation_count=0,
        economics=zero_operation_economics(),
    )


def rebind_settlement(
    reservation: RuntimeSpendCheckpointEvent,
    settled: RoutedProviderOperation,
) -> RoutedProviderOperation:
    """Bind runtime settlement evidence to its durable pre-dispatch reservation."""
    if settled.component != reservation.operation.component:
        raise ValueError("runtime settlement component differs from its reservation")
    if settled.billing_source != reservation.operation.billing_source:
        raise ValueError("runtime settlement billing source differs from its reservation")
    if settled.disposition == RoutedSpendDisposition.RESERVED:
        raise ValueError("runtime settlement evidence remains reserved")
    return reservation.operation.model_copy(
        update={
            "disposition": settled.disposition,
            "operation_count": settled.operation_count,
            "economics": settled.economics,
        }
    )


def validate_spend_events(
    events: tuple[RuntimeSpendCheckpointEvent, ...] | list[RuntimeSpendCheckpointEvent],
) -> dict[str, RuntimeSpendCheckpointEvent]:
    """Validate global ordering and every reservation-to-settlement transition."""
    latest: dict[str, RuntimeSpendCheckpointEvent] = {}
    next_operation: dict[str, int] = {}
    interaction_identities: dict[str, Sha256] = {}
    seen_event_ids: set[str] = set()
    for expected_ordinal, event in enumerate(events, start=1):
        if event.ordinal != expected_ordinal:
            raise ValueError("runtime spend journal ordinals are not contiguous")
        if event.event_id in seen_event_ids:
            raise ValueError("runtime spend journal repeats an event ID")
        if event.event_id != spend_event_content_id(event):
            raise ValueError("runtime spend event ID differs from its canonical content")
        seen_event_ids.add(event.event_id)
        prior_identity = interaction_identities.setdefault(
            event.interaction_id, event.identity_sha256
        )
        if prior_identity != event.identity_sha256:
            raise ValueError("runtime spend interaction changes request or lineage identity")
        prior = latest.get(event.operation.operation_id)
        if prior is None:
            minimum_operation = next_operation.get(event.interaction_id, 1)
            if event.operation.operation_ordinal < minimum_operation:
                raise ValueError("runtime spend operation ordinals do not increase")
            next_operation[event.interaction_id] = event.operation.operation_ordinal + 1
        else:
            if prior.operation.disposition != RoutedSpendDisposition.RESERVED:
                raise ValueError("runtime spend operation has more than one settlement")
            stable_prior = prior.model_dump(
                mode="json", exclude={"event_id", "ordinal", "operation", "recorded_at"}
            )
            stable_event = event.model_dump(
                mode="json", exclude={"event_id", "ordinal", "operation", "recorded_at"}
            )
            if stable_prior != stable_event:
                raise ValueError("runtime spend settlement drifted from its reservation scope")
            prior_operation = prior.operation.model_dump(
                mode="json", exclude={"disposition", "operation_count", "economics"}
            )
            settled_operation = event.operation.model_dump(
                mode="json", exclude={"disposition", "operation_count", "economics"}
            )
            if prior_operation != settled_operation:
                raise ValueError("runtime spend settlement drifted from its reservation identity")
            if event.operation.disposition == RoutedSpendDisposition.RESERVED:
                raise ValueError("runtime spend reservation is duplicated without settlement")
            if event.operation.disposition == RoutedSpendDisposition.RESERVED_AMBIGUOUS and (
                event.operation.economics != prior.operation.economics
            ):
                raise ValueError("ambiguous runtime spend must retain its exact reservation")
        latest[event.operation.operation_id] = event
    return latest


def spend_event_content_id(event: RuntimeSpendCheckpointEvent) -> str:
    """Return the canonical identity for one spend checkpoint event."""
    material = event.model_dump(mode="json")
    del material["event_id"]
    return stable_id("runtime-spend-event", material)


def live_reservation(
    events: tuple[RuntimeSpendCheckpointEvent, ...] | list[RuntimeSpendCheckpointEvent],
    *,
    interaction_id: str,
    component: RoutedProviderComponent,
    accepted_attempt_ordinal: int | None = None,
) -> RuntimeSpendCheckpointEvent | None:
    """Return the matching unsettled reservation, if one exists."""
    latest = validate_spend_events(events)
    matches = tuple(
        event
        for event in latest.values()
        if event.interaction_id == interaction_id
        and event.operation.component == component
        and event.accepted_attempt_ordinal == accepted_attempt_ordinal
        and event.operation.disposition == RoutedSpendDisposition.RESERVED
    )
    if len(matches) > 1:
        raise ValueError("runtime spend scope has multiple live reservations")
    return matches[0] if matches else None


def next_operation_ordinal(
    events: tuple[RuntimeSpendCheckpointEvent, ...] | list[RuntimeSpendCheckpointEvent],
    interaction_id: str,
) -> int:
    """Return the next interaction-local operation ordinal."""
    ordinals = {
        event.operation.operation_ordinal
        for event in events
        if event.interaction_id == interaction_id
    }
    return max(ordinals, default=0) + 1


def settled_operations(
    events: tuple[RuntimeSpendCheckpointEvent, ...] | list[RuntimeSpendCheckpointEvent],
    interaction_id: str,
) -> tuple[RoutedProviderOperation, ...]:
    """Return settled sidecar operations in interaction order."""
    latest = validate_spend_events(events)
    operations = tuple(
        event.operation
        for event in latest.values()
        if event.interaction_id == interaction_id
        and event.operation.disposition != RoutedSpendDisposition.RESERVED
    )
    return tuple(sorted(operations, key=lambda item: item.operation_ordinal))


def require_identity(
    event: RuntimeSpendCheckpointEvent,
    identity_sha256: Sha256,
) -> None:
    """Reject a spend reservation reused for different request or lineage content."""
    if event.identity_sha256 != identity_sha256:
        raise ValueError("idempotency key was already used for different request content")


def _checkpoint_event(
    *,
    interaction_id: str,
    identity_sha256: Sha256,
    ordinal: int,
    accepted_attempt_ordinal: int | None,
    operation: RoutedProviderOperation,
    recorded_at: datetime,
) -> RuntimeSpendCheckpointEvent:
    """Build one content-addressed spend checkpoint."""
    provisional = RuntimeSpendCheckpointEvent(
        event_id="runtime-spend-event-00000000000000000000",
        ordinal=ordinal,
        interaction_id=interaction_id,
        identity_sha256=identity_sha256,
        accepted_attempt_ordinal=accepted_attempt_ordinal,
        operation=operation,
        recorded_at=recorded_at,
    )
    return provisional.model_copy(update={"event_id": spend_event_content_id(provisional)})
