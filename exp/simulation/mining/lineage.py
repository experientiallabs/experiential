"""Stable source-lineage assignment before duplicate union and partitioning."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC

from exp.common.core.artifacts import stable_id
from exp.common.core.text import normalize_durable_text
from exp.common.traces import Trace

DEFAULT_LINEAGE_TIME_BUCKET_SECONDS = 86_400


@dataclass(frozen=True)
class LineageAssignment:
    """Auditable initial leakage lineage for one normalized source trace.

    Args:
        trace_id: Canonical trace identity.
        lineage_group_id: Stable initial lineage identity before duplicate unions.
        customer_id: Stable customer boundary from source extensions or source identity.
        conversation_id: Captured conversation boundary, when supplied.
        time_bucket: UTC bucket used when no conversation identity was supplied.
    """

    trace_id: str
    lineage_group_id: str
    customer_id: str
    conversation_id: str | None
    time_bucket: int | None


def assign_source_lineages(
    traces: Sequence[Trace],
    *,
    time_bucket_seconds: int = DEFAULT_LINEAGE_TIME_BUCKET_SECONDS,
) -> tuple[LineageAssignment, ...]:
    """Assign stable initial lineages from customer, conversation, and time boundaries.

    A captured conversation keeps all of its traces together. When a source has no conversation
    ID, a customer-specific UTC time bucket is deliberately conservative, preventing adjacent
    anonymous activity from being split before semantic duplicate checks run.

    Args:
        traces: Canonical traces with ordered source timestamps.
        time_bucket_seconds: Positive UTC fallback boundary for traces lacking conversation IDs.

    Returns:
        One stable, auditable initial lineage assignment per trace in input order.

    Raises:
        ValueError: Trace IDs repeat, source boundary extensions conflict, or the bucket is invalid.
    """
    if time_bucket_seconds <= 0:
        raise ValueError("lineage time bucket must be positive")
    trace_ids = tuple(trace.trace_id for trace in traces)
    if len(set(trace_ids)) != len(trace_ids):
        raise ValueError("source lineage assignment needs unique trace IDs")
    assignments: list[LineageAssignment] = []
    for trace in traces:
        customer_id = _customer_id(trace)
        conversation_id = _conversation_id(trace)
        if conversation_id is not None:
            time_bucket = None
            material = {
                "version": "source-lineage-v1",
                "customer_id": customer_id,
                "conversation_id": conversation_id,
            }
        else:
            started_at = min(span.started_at for span in trace.spans).astimezone(UTC)
            time_bucket = int(started_at.timestamp()) // time_bucket_seconds
            material = {
                "version": "source-lineage-v1",
                "customer_id": customer_id,
                "time_bucket_seconds": time_bucket_seconds,
                "time_bucket": time_bucket,
            }
        assignments.append(
            LineageAssignment(
                trace_id=trace.trace_id,
                lineage_group_id=stable_id("lineage", material),
                customer_id=customer_id,
                conversation_id=conversation_id,
                time_bucket=time_bucket,
            )
        )
    return tuple(assignments)


def _customer_id(trace: Trace) -> str:
    """Resolve a stable customer boundary without inferring it from later outcomes."""
    extension = _consistent_span_text(trace, ("exp.customer.id", "enduser.id"))
    if extension is not None:
        return extension
    return f"source:{trace.source.identity.source_id}"


def _conversation_id(trace: Trace) -> str | None:
    """Reconcile canonical and source-extension conversation identities."""
    from_spans = _consistent_span_text(trace, ("exp.conversation.id", "gen_ai.conversation.id"))
    if trace.conversation_id is not None and from_spans is not None:
        if trace.conversation_id != from_spans:
            raise ValueError(f"trace {trace.trace_id} has conflicting conversation identities")
    return trace.conversation_id or from_spans


def _consistent_span_text(trace: Trace, keys: tuple[str, ...]) -> str | None:
    """Return one non-empty source extension value or reject conflicting values."""
    values: list[str] = []
    for span in trace.spans:
        for key in keys:
            raw = span.attributes.get(key)
            if raw is None:
                continue
            if not isinstance(raw, str) or not raw.strip():
                raise ValueError(f"trace {trace.trace_id} has invalid {key}")
            values.append(normalize_durable_text(raw.strip()))
            break
    if not values:
        return None
    if any(value != values[0] for value in values[1:]):
        raise ValueError(f"trace {trace.trace_id} has conflicting {keys[0]} values")
    return values[0]
