"""Frozen video terminal evidence and per-job delivery ownership."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime

from exp.common.models.catalog_prices import BilledUnitKind
from exp.runtime.gateway.stream_contracts import (
    BilledUnits,
    GatewayEvent,
    GatewayFailure,
    GatewayUsage,
)
from exp.runtime.gateway.video.contracts import VideoJob, VideoStatus


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
    """

    job: VideoJob
    closed: VideoJob
    event: GatewayEvent | None
    failure: GatewayFailure | None
    finalize_request: bool = True
    ledger_written: bool = False


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
