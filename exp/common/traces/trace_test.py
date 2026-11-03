"""Tests for canonical normalized production-trace contracts."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from exp.common.core.artifacts import (
    FailureCode,
    JsonObject,
    SourceIdentity,
    StructuredFailure,
    canonical_json_bytes,
)
from exp.common.models import BillingSource, ModelSnapshot
from exp.common.traces import (
    INPUT_HISTORY_PREFIX_KEY,
    INPUT_MESSAGES_KEY,
    Trace,
    TraceDataset,
    TraceOutcome,
    TraceSource,
    TraceSpan,
)

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


_HISTORY: tuple[JsonObject, ...] = (
    {"role": "user", "content": "Refund order 7."},
    {"role": "assistant", "content": "Checking."},
    {"role": "user", "content": "Thanks."},
)


def _with_input(attributes: JsonObject, history: tuple[JsonObject, ...] = _HISTORY) -> JsonObject:
    """Return the fixture trace as JSON with one span's input attributes and a history.

    Args:
        attributes: Input attributes for the single span.
        history: Shared message history stored on the trace.

    Returns:
        A trace payload ready for validation.
    """
    trace = _trace()
    span = trace.spans[0].model_copy(update={"attributes": attributes})
    return {**trace.model_dump(), "spans": [span.model_dump()], "message_history": list(history)}


def test_input_messages_resolves_a_history_prefix_or_the_inline_value() -> None:
    """A history reference reads its prefix; an inline value is returned as declared."""
    shared = Trace.model_validate(_with_input({INPUT_HISTORY_PREFIX_KEY: 3}))
    assert shared.input_messages(shared.spans[0]) == list(_HISTORY)
    inline = Trace.model_validate(_with_input({INPUT_MESSAGES_KEY: '[{"role":"user"}]'}, ()))
    assert inline.input_messages(inline.spans[0]) == '[{"role":"user"}]'
    assert _trace().input_messages(_trace().spans[0]) is None


def test_trace_without_history_keeps_its_canonical_bytes() -> None:
    """An empty history is omitted, so traces of every other source keep their identity."""
    trace = _trace()
    assert "message_history" not in trace.model_dump(mode="json")
    assert canonical_json_bytes(Trace.model_validate(trace.model_dump())) == canonical_json_bytes(
        trace
    )


@pytest.mark.parametrize(
    ("attributes", "history", "message"),
    [
        ({INPUT_HISTORY_PREFIX_KEY: 4}, _HISTORY, "references 4 history messages"),
        ({INPUT_HISTORY_PREFIX_KEY: -1}, _HISTORY, "must be a count"),
        ({INPUT_HISTORY_PREFIX_KEY: True}, _HISTORY, "must be a count"),
        ({INPUT_HISTORY_PREFIX_KEY: "3"}, _HISTORY, "must be a count"),
        ({INPUT_HISTORY_PREFIX_KEY: 2}, _HISTORY, "reference only 2"),
        ({}, _HISTORY, "reference only 0"),
        ({INPUT_HISTORY_PREFIX_KEY: 3, INPUT_MESSAGES_KEY: []}, _HISTORY, "both inline"),
    ],
)
def test_trace_rejects_unresolvable_or_unreferenced_history(
    attributes: JsonObject, history: tuple[JsonObject, ...], message: str
) -> None:
    """A dangling, malformed, doubled or partly unreferenced history never validates."""
    with pytest.raises(ValidationError, match=message):
        Trace.model_validate(_with_input(attributes, history))
