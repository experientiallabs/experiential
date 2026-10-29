"""Frozen video terminal evidence and per-job delivery ownership."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from exp.common.models.catalog_prices import BilledUnitKind
from exp.runtime.gateway.stream_contracts import (
    BilledUnits,
    GatewayEvent,
    GatewayEventKind,
    GatewayFailure,
    GatewayFailureClass,
    GatewayUsage,
)
from exp.runtime.gateway.video.contracts import VideoJob, VideoStatus
from exp.runtime.gateway.video.providers import VideoProviderSnapshot


@dataclass
class PendingSettlement:
    """Retain the first terminal observation until both durable writes complete.

    Attributes:
        job: Open durable job associated with this observation.
        closed: Exact terminal job to persist after accounting succeeds.
        event: Original terminal meter, or None for an unknown provider outcome.
        failure: Original provider failure, if any.
        finalize_request: Whether this attempt closes the request, true by default.
        ledger_written: Whether accounting acknowledged the terminal meter.
        recovery_polled_at: Last bounded read after a superseded local waiver.
        require_ledger_replay: Whether recovered evidence still needs exact replay acknowledgment.
    """

    job: VideoJob
    closed: VideoJob
    event: GatewayEvent | None
    failure: GatewayFailure | None
    finalize_request: bool = True
    ledger_written: bool = False
    recovery_polled_at: datetime | None = None
    require_ledger_replay: bool = False


@dataclass
class JobGate:
    """Serialize one video while retaining the gate across waiting callers.

    Attributes:
        lock: Serializes provider refresh and settlement delivery for this job.
        users: Holders and waiters retaining this gate, initially zero.
    """

    lock: threading.Lock = field(default_factory=threading.Lock)
    users: int = 0


def zero_units(variant: str) -> GatewayUsage:
    """Observed usage for a render that produced nothing: zero video seconds."""
    return GatewayUsage(
        billed_units=BilledUnits(
            kind=BilledUnitKind.VIDEO_SECOND, variant=variant, quantity_milli=0
        )
    )


def closed_job(
    job: VideoJob,
    failure: GatewayFailure,
    *,
    settled_milli: int | None,
    now: datetime,
    code: str | None = None,
) -> VideoJob:
    """Return the settled failed form of one job without inventing unknown usage."""
    return job.model_copy(
        update={
            "status": VideoStatus.FAILED,
            "error_code": code or failure.failure_class.value,
            "error_message": failure.provider_detail or failure.safe_message,
            "billed_seconds_milli": settled_milli,
            "completed_at": now,
            "last_polled_at": now,
            "settled": True,
        }
    )


def require_same_job(current: VideoJob, expected: VideoJob, *, frozen: bool = False) -> None:
    """Check delivery ownership, requiring frozen provider facts only before a recovery poll."""
    fields = {"video_id", "organization_id", "request_id", "attempt_id"}
    if frozen:
        fields |= {
            "identity_id",
            "deployment",
            "alias_revision_id",
            "catalog_sha256",
            "dialect",
            "provider_job_id",
            "seconds",
            "variant",
            "rate_nano_usd",
            "reserved_nano_usd",
            "output_stored",
        }
    if current.model_dump(include=fields) != expected.model_dump(include=fields):
        raise RuntimeError("video settlement winner changed frozen job identity")


def terminal_settlement(
    job: VideoJob,
    snapshot: VideoProviderSnapshot | None,
    *,
    failure: GatewayFailure | None,
    observed_zero: bool,
    release_reservation: bool = False,
    now: datetime,
) -> PendingSettlement:
    """Freeze one provider terminal or trusted local recovery for durable delivery.

    Completed renders bill at most the requested seconds. Provider failures have
    observed zero usage; local timeouts release customer reservations without
    inventing provider usage. This constructor performs no writes.
    """
    completed = snapshot is not None and snapshot.status is VideoStatus.COMPLETED
    if completed:
        assert snapshot is not None
        requested = job.seconds * 1000
        billed = min(snapshot.duration_milli or requested, requested)
        usage = GatewayUsage(
            billed_units=BilledUnits(
                kind=BilledUnitKind.VIDEO_SECOND, variant=job.variant, quantity_milli=billed
            )
        )
        event = GatewayEvent(kind=GatewayEventKind.COMPLETED, sequence_number=0, usage=usage)
        failure = None
    else:
        if failure is None:
            assert snapshot is not None
            failure = GatewayFailure(
                failure_class=GatewayFailureClass.REFUSAL
                if snapshot.error_code == "content_filtered"
                else GatewayFailureClass.PROVIDER_INTERNAL,
                safe_message=snapshot.error_message or "the video could not be generated",
            )
        billed = 0 if observed_zero else None
        event = (
            GatewayEvent.for_accounting_recovery(
                failure=failure, release_reservation=release_reservation
            )
            if snapshot is None and not observed_zero
            else GatewayEvent(
                kind=GatewayEventKind.FAILED,
                sequence_number=0,
                failure=failure,
                usage=zero_units(job.variant) if observed_zero else None,
            )
        )
    if completed:
        assert snapshot is not None
        lifetime = snapshot.asset_lifetime_seconds
        closed = job.model_copy(
            update={
                "status": VideoStatus.COMPLETED,
                "progress": 100,
                "content_url": snapshot.content_url,
                "billed_seconds_milli": billed,
                "completed_at": now,
                "expires_at": None if lifetime is None else now + timedelta(seconds=lifetime),
                "last_polled_at": now,
                "settled": True,
            }
        )
    else:
        assert failure is not None
        code = None if snapshot is None else snapshot.error_code
        closed = closed_job(job, failure, settled_milli=billed, now=now, code=code)
    return PendingSettlement(job, closed, event, failure)
