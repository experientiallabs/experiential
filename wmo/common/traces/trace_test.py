"""Tests for canonical normalized production-trace contracts."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from wmo.common.core.artifacts import FailureCode, SourceIdentity, StructuredFailure
from wmo.common.models import BillingSource, ModelSnapshot
from wmo.common.traces import Trace, TraceDataset, TraceOutcome, TraceSource, TraceSpan

_DIGEST = "a" * 64


def _trace() -> Trace:
    started_at = datetime(2026, 8, 11, tzinfo=UTC)
    return Trace(
        trace_id="0123456789abcdef0123456789abcdef",
        conversation_id="conversation-12",
        task="Help the customer request a refund.",
        spans=(
            TraceSpan(
                span_id="span-1",
                name="agent.model_call",
                started_at=started_at,
                ended_at=started_at + timedelta(seconds=1),
                model=ModelSnapshot(
                    billing_source=BillingSource.CUSTOMER_MANAGED,
                    provider="openai",
                    model_id="gpt-5.4",
                    capabilities_sha256=_DIGEST,
                    connection_sha256=_DIGEST,
                ),
            ),
        ),
        outcome=TraceOutcome(status="success", outcome_name="refund_requested"),
        source=TraceSource(
            identity=SourceIdentity(kind="otlp", source_id="upload-1", sha256=_DIGEST),
            semantic_convention_version="1.37.0",
        ),
    )


def test_trace_requires_ordered_unique_spans_and_structured_failure() -> None:
    """Corrupt timing, repeated IDs, and failed outcomes do not silently normalize."""
    trace = _trace()
    span = trace.spans[0]
    with pytest.raises(ValidationError, match="cannot be before"):
        TraceSpan(
            span_id="bad-span",
            name="agent.model_call",
            started_at=span.ended_at,
            ended_at=span.started_at,
        )
    with pytest.raises(ValidationError, match="unique"):
        Trace.model_validate(
            {**trace.model_dump(), "spans": [span.model_dump(), span.model_dump()]}
        )
    with pytest.raises(ValidationError, match="require a structured failure"):
        TraceOutcome(status="failure")
    assert (
        TraceOutcome(
            status="failure",
            failure=StructuredFailure(
                code=FailureCode.PROVIDER,
                message="timed out",
                retryable=True,
            ),
        ).failure
        is not None
    )
    with pytest.raises(ValidationError, match="relative POSIX"):
        TraceDataset(
            schema_version=1,
            created_at=datetime(2026, 8, 11, tzinfo=UTC),
            code_revision="e7aad17",
            dataset_id="traces-20260811",
            semantic_convention_version="1.37.0",
            traces_path="../traces.jsonl",
            traces_sha256=_DIGEST,
            trace_ids=(trace.trace_id,),
        )
    with pytest.raises(ValidationError, match="requires an immutable issues report"):
        TraceDataset(
            schema_version=1,
            created_at=datetime(2026, 8, 11, tzinfo=UTC),
            code_revision="e7aad17",
            dataset_id="traces-20260811",
            semantic_convention_version="1.37.0",
            traces_path="traces.jsonl",
            traces_sha256=_DIGEST,
            invalid_trace_count=1,
            trace_ids=(trace.trace_id,),
        )
