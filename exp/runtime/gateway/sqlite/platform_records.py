"""SQLite row conversion and replay checks for the neutral gateway adapter."""

from __future__ import annotations

import sqlite3
from datetime import datetime

from exp.common.models import BillingSource
from exp.runtime.gateway.auth import utc_text
from exp.runtime.gateway.contracts import DirectTarget, GatewayFailureClass, ProjectTarget
from exp.runtime.gateway.platform import (
    AliasRevisionRecord,
    AttemptReservationRecord,
    AttemptReservationRequest,
    AttemptSettlementRecord,
    AttemptSettlementRequest,
    AttemptTerminalState,
    AttemptUsageSource,
    ProviderRevisionBinding,
    VirtualKeyRecord,
)
from exp.runtime.gateway.sqlite.provider_authority import ProviderConnectionBinding


def key_record(
    row: sqlite3.Row,
    *,
    organization_id: str,
    now: datetime,
) -> VirtualKeyRecord:
    """Decode one durable key row without touching fingerprint material."""
    expires_at = optional_datetime(row["expires_at"])
    revoked_at = optional_datetime(row["revoked_at"])
    return VirtualKeyRecord(
        organization_id=organization_id,
        identity_id=str(row["identity_id"]),
        key_id=str(row["key_id"]),
        prefix=str(row["prefix"]),
        active=revoked_at is None and (expires_at is None or expires_at > now),
        expires_at=expires_at,
        revoked_at=revoked_at,
        created_at=required_datetime(row["created_at"]),
        last_used_at=optional_datetime(row["last_used_at"]),
    )


def provider_binding(binding: ProviderRevisionBinding) -> ProviderConnectionBinding:
    """Convert one neutral provider binding to the existing SQLite contract."""
    return ProviderConnectionBinding(
        connection_id=binding.connection_id,
        connection_revision_id=binding.connection_revision_id,
        connection_sha256=binding.connection_sha256,
    )


def require_settlement_replay(
    settlement: AttemptSettlementRecord,
    *,
    request: AttemptSettlementRequest,
) -> None:
    """Reject a replay whose accounting evidence differs from durable settlement."""
    event_failure = None if request.terminal_event is None else request.terminal_event.failure
    failure = request.failure or event_failure
    usage = None if request.terminal_event is None else request.terminal_event.usage
    if failure is not None:
        state = (
            AttemptTerminalState.CANCELLED
            if failure.failure_class is GatewayFailureClass.CANCELLED
            else AttemptTerminalState.FAILED
        )
        failure_class = failure.failure_class
    else:
        if request.terminal_event is None:
            raise ValueError("attempt settlement needs a terminal event or failure")
        state = AttemptTerminalState(request.terminal_event.kind.value)
        failure_class = None
    usage_source = AttemptUsageSource.OBSERVED if usage is not None else AttemptUsageSource.UNKNOWN
    if (
        settlement.state is not state
        or settlement.failure_class is not failure_class
        or settlement.usage != usage
        or settlement.usage_source is not usage_source
    ):
        raise ValueError("attempt settlement replay differs from durable accounting evidence")


def alias_record(row: sqlite3.Row, *, organization_id: str) -> AliasRevisionRecord:
    """Decode one immutable alias revision and discriminated target."""
    target = (
        DirectTarget(pool_id=str(row["pool_id"]))
        if str(row["target_kind"]) == "direct"
        else ProjectTarget(
            project_ref=str(row["project_ref"]),
            activation_ref=str(row["activation_ref"]),
            catalog_sha256=str(row["catalog_sha256"]),
        )
    )
    return AliasRevisionRecord(
        organization_id=organization_id,
        alias_id=str(row["alias_id"]),
        alias_name=str(row["alias_name"]),
        revision_id=str(row["revision_id"]),
        revision_number=int(row["revision_number"]),
        target=target,
        snapshot_ref=str(row["snapshot_ref"]),
        catalog_sha256=str(row["catalog_sha256"]),
        refusal_failover=bool(row["refusal_failover"]),
        active=bool(row["active"]) and str(row["active_revision_id"]) == str(row["revision_id"]),
        created_at=required_datetime(row["created_at"]),
    )


def reservation_record(
    row: sqlite3.Row,
    *,
    organization_id: str,
) -> AttemptReservationRecord:
    """Decode one reservation from the existing atomic attempt row."""
    return AttemptReservationRecord(
        organization_id=organization_id,
        attempt_id=str(row["attempt_id"]),
        request_id=str(row["request_id"]),
        identity_id=str(row["identity_id"]),
        alias_id=str(row["alias_id"]),
        alias_revision_id=str(row["alias_revision_id"]),
        catalog_sha256=str(row["catalog_sha256"]),
        pool_id=str(row["pool_id"]),
        exact_model_id=str(row["exact_model_id"]),
        deployment_id=str(row["deployment_id"]),
        provider=str(row["provider"]),
        billing_source=BillingSource(str(row["billing_source"])),
        input_rate=optional_int(row["input_rate"]),
        cached_input_rate=optional_int(row["cached_input_rate"]),
        cache_creation_input_rate=optional_int(row["cache_creation_input_rate"]),
        output_rate=optional_int(row["output_rate"]),
        reasoning_rate=optional_int(row["reasoning_rate"]),
        long_context_threshold_tokens=optional_int(row["long_context_threshold_tokens"]),
        long_context_input_rate=optional_int(row["long_context_input_rate"]),
        long_context_cached_input_rate=optional_int(row["long_context_cached_input_rate"]),
        long_context_cache_creation_input_rate=optional_int(
            row["long_context_cache_creation_input_rate"]
        ),
        long_context_output_rate=optional_int(row["long_context_output_rate"]),
        long_context_reasoning_rate=optional_int(row["long_context_reasoning_rate"]),
        attempt_ordinal=int(row["attempt_ordinal"]),
        route_depth=int(row["route_depth"]),
        period=str(row["budget_period_start"])[:7],
        reserved_micro_usd=optional_int(row["budget_reserved_micro_usd"]),
        started_at=required_datetime(row["started_at"]),
    )


def require_reservation_replay(
    row: sqlite3.Row,
    *,
    request: AttemptReservationRequest,
) -> AttemptReservationRecord:
    """Return an exact natural-key replay or reject changed accounting input."""
    record = reservation_record(row, organization_id=request.organization_id)
    authorization = request.snapshot.authorization
    prices = request.deployment.gateway.prices
    expected = (
        authorization.request_id,
        authorization.identity_id,
        authorization.virtual_key_id,
        authorization.alias_revision_id,
        authorization.surface.value,
        authorization.canonical_request_sha256,
        authorization.caller_operation_sha256,
        authorization.catalog_sha256,
        request.snapshot.pool_id,
        request.snapshot.exact_model_id,
        request.deployment.deployment_id,
        request.deployment.provider,
        request.deployment.billing_source,
        request.deployment.gateway.pricing_source,
        (
            None
            if request.deployment.gateway.pricing_effective_at is None
            else utc_text(request.deployment.gateway.pricing_effective_at)
        ),
        prices.input_micro_usd_per_million_tokens,
        prices.cached_input_micro_usd_per_million_tokens,
        prices.output_micro_usd_per_million_tokens,
        prices.reasoning_micro_usd_per_million_tokens,
        (None if prices.long_context is None else prices.long_context.input_threshold_tokens),
        (
            None
            if prices.long_context is None
            else prices.long_context.input_micro_usd_per_million_tokens
        ),
        (
            None
            if prices.long_context is None
            else prices.long_context.cached_input_micro_usd_per_million_tokens
        ),
        (
            None
            if prices.long_context is None
            else prices.long_context.output_micro_usd_per_million_tokens
        ),
        (
            None
            if prices.long_context is None
            else prices.long_context.reasoning_micro_usd_per_million_tokens
        ),
        request.attempt_ordinal,
        request.route_depth,
        request.maximum_cost_micro_usd,
    )
    actual = (
        record.request_id,
        record.identity_id,
        str(row["key_id"]),
        record.alias_revision_id,
        str(row["api_surface"]),
        str(row["canonical_request_sha256"]),
        (None if row["caller_operation_sha256"] is None else str(row["caller_operation_sha256"])),
        record.catalog_sha256,
        record.pool_id,
        record.exact_model_id,
        record.deployment_id,
        record.provider,
        record.billing_source,
        None if row["pricing_source"] is None else str(row["pricing_source"]),
        (None if row["pricing_effective_at"] is None else str(row["pricing_effective_at"])),
        record.input_rate,
        record.cached_input_rate,
        record.output_rate,
        record.reasoning_rate,
        record.long_context_threshold_tokens,
        record.long_context_input_rate,
        record.long_context_cached_input_rate,
        record.long_context_output_rate,
        record.long_context_reasoning_rate,
        record.attempt_ordinal,
        record.route_depth,
        record.reserved_micro_usd,
    )
    if actual != expected:
        raise ValueError("attempt reservation replay differs from durable accounting input")
    return record


def required_datetime(value: object) -> datetime:
    """Parse one required SQLite timestamp."""
    if value is None:
        raise RuntimeError("required platform timestamp is missing")
    return datetime.fromisoformat(str(value))


def optional_datetime(value: object) -> datetime | None:
    """Parse one optional SQLite timestamp."""
    return None if value is None else datetime.fromisoformat(str(value))


def optional_int(value: object) -> int | None:
    """Decode one optional SQLite integer."""
    return None if value is None else int(str(value))
