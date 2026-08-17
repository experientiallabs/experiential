"""Tests for shared vendor observation to canonical trace conversion."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

from wmo.common.core.artifacts import SourceIdentity
from wmo.simulation.ingest.vendor_observations import (
    VendorModelIdentity,
    VendorObservation,
    VendorTokenUsage,
    VendorToolCall,
)
from wmo.simulation.ingest.vendor_trace import (
    SOURCE_SPAN_ATTRIBUTE,
    SOURCE_TRACE_ATTRIBUTE,
    VENDOR_ATTRIBUTE,
    approved_extensions,
    build_vendor_traces,
)

_START = datetime(2024, 5, 1, tzinfo=UTC)
_SOURCE = SourceIdentity(kind="file", source_id="vendor:export.json", sha256="a" * 64)


_MODEL_OBSERVATION = VendorObservation(
    source_trace_id="trace-1",
    source_span_id="span-1",
    ordinal=1,
    started_at=_START,
    ended_at=_START + timedelta(seconds=1),
    kind="model",
    request_text="What is the weather in Paris?",
    tool_calls=(
        VendorToolCall(name="get_weather", arguments='{"city": "Paris"}', call_id="call-1"),
    ),
    model=VendorModelIdentity(provider="openai", model_id="gpt-4o"),
    usage=VendorTokenUsage(input_tokens=11, output_tokens=5),
)
_TOOL_OBSERVATION = VendorObservation(
    source_trace_id="trace-1",
    source_span_id="span-2",
    ordinal=2,
    started_at=_START + timedelta(seconds=1),
    ended_at=_START + timedelta(seconds=2),
    kind="tool_result",
    tool_name="get_weather",
    tool_arguments='{"city": "Paris"}',
    tool_message="18C",
    tool_call_id="call-1",
)


def test_build_vendor_traces_pairs_calls_and_retains_source_provenance() -> None:
    """A model call and its tool result become one trace that keeps the vendor identities."""
    result = build_vendor_traces(
        [_MODEL_OBSERVATION, _TOOL_OBSERVATION], vendor="langfuse", source=_SOURCE
    )

    assert result.issues == ()
    assert len(result.traces) == 1
    trace = result.traces[0]
    assert trace.task == "What is the weather in Paris?"
    call, tool_result = trace.spans
    assert call.attributes[VENDOR_ATTRIBUTE] == "langfuse"
    assert call.attributes[SOURCE_TRACE_ATTRIBUTE] == "trace-1"
    assert call.attributes[SOURCE_SPAN_ATTRIBUTE] == "span-1"
    assert tool_result.attributes[SOURCE_SPAN_ATTRIBUTE] == "span-2"
    assert tool_result.parent_span_id == call.span_id
    assert call.model is not None
    assert call.model.model_id == "gpt-4o"
    assert call.usage is not None
    assert call.usage.input_tokens == 11


def test_build_vendor_traces_pairs_by_tool_name_without_call_ids() -> None:
    """A result without a call id pairs conservatively with the matching requested tool."""
    result = build_vendor_traces(
        [
            replace(
                _MODEL_OBSERVATION,
                tool_calls=(VendorToolCall(name="get_weather", arguments="{}"),),
            ),
            replace(_TOOL_OBSERVATION, tool_call_id=None),
        ],
        vendor="langfuse",
        source=_SOURCE,
    )

    assert result.issues == ()
    call, tool_result = result.traces[0].spans
    assert tool_result.parent_span_id == call.span_id


def test_build_vendor_traces_requires_a_model_snapshot_provider() -> None:
    """Model evidence without a provider is retained without resolving a model snapshot."""
    result = build_vendor_traces(
        [
            replace(
                _MODEL_OBSERVATION,
                model=None,
                declared_attributes={"gen_ai.request.model": "gpt-4o"},
            ),
            _TOOL_OBSERVATION,
        ],
        vendor="langfuse",
        source=_SOURCE,
    )

    call = result.traces[0].spans[0]
    assert call.model is None
    assert call.attributes["gen_ai.request.model"] == "gpt-4o"


def test_build_vendor_traces_excludes_traces_without_a_user_request() -> None:
    """A trace with no declared user request is excluded with its vendor trace key."""
    result = build_vendor_traces(
        [replace(_MODEL_OBSERVATION, request_text=None), _TOOL_OBSERVATION],
        vendor="langfuse",
        source=_SOURCE,
    )

    assert result.traces == ()
    assert [issue.source_record for issue in result.issues] == ["trace-trace-1"]


def test_build_vendor_traces_maps_declared_failures_to_the_trace_outcome() -> None:
    """A declared span failure becomes the canonical failed trace outcome."""
    result = build_vendor_traces(
        [replace(_MODEL_OBSERVATION, failure_message="upstream 500"), _TOOL_OBSERVATION],
        vendor="langfuse",
        source=_SOURCE,
    )

    outcome = result.traces[0].outcome
    assert outcome is not None
    assert outcome.status == "failure"
    assert outcome.failure is not None
    assert "upstream 500" in outcome.failure.message


def test_build_vendor_traces_strict_pairing_rejects_unpaired_calls_and_results() -> None:
    """Strict pairing excludes traces whose declared calls or explicit results never pair."""
    unpaired_call = build_vendor_traces(
        [_MODEL_OBSERVATION], vendor="langfuse", source=_SOURCE, strict_tool_pairing=True
    )
    mismatched_result = build_vendor_traces(
        [_MODEL_OBSERVATION, replace(_TOOL_OBSERVATION, tool_call_id="call-9")],
        vendor="langfuse",
        source=_SOURCE,
        strict_tool_pairing=True,
    )

    assert unpaired_call.traces == ()
    assert "unmatched generated langfuse tool calls: get_weather:call-1" in (
        unpaired_call.issues[0].message
    )
    assert mismatched_result.traces == ()
    assert "unmatched explicit langfuse tool result: get_weather:call-9" in (
        mismatched_result.issues[0].message
    )


def test_approved_extensions_copies_only_approved_keys() -> None:
    """Vendor metadata reaches canonical spans only through approved WMO extensions."""
    extensions = approved_extensions(
        {"wmo.conversation.id": "thread-1", "vendor.internal": "drop me"}
    )

    assert extensions == {"wmo.conversation.id": "thread-1"}
