"""Tests for the shared normalizer helpers (the pieces every adapter relies on)."""

from __future__ import annotations

from wmo.core.types import ErrorClass, StepAttribution
from wmo.ingest.normalize import (
    SpanEmitter,
    SpanRecord,
    iso_to_ordinal,
    openai_call_name_args,
    openai_tool_calls,
    spans_to_traces,
)


def test_iso_to_ordinal_naive_timestamp_is_utc_not_local() -> None:
    """A naive timestamp (no tz) must be read as UTC, so ordinals are machine-independent.

    Regression: `datetime.fromisoformat("...").timestamp()` on a naive value uses the machine's
    local timezone, which would reorder spans differently across machines. `iso_to_ordinal` pins
    naive timestamps to UTC, so a naive value and its explicit-UTC form map to the same ordinal.
    """
    naive = iso_to_ordinal("2026-01-01T00:00:00", fallback=-1)
    explicit_utc = iso_to_ordinal("2026-01-01T00:00:00+00:00", fallback=-1)
    zulu = iso_to_ordinal("2026-01-01T00:00:00Z", fallback=-1)
    assert naive == explicit_utc == zulu
    assert naive > 0


def test_iso_to_ordinal_orders_and_falls_back() -> None:
    earlier = iso_to_ordinal("2026-01-01T00:00:00Z", fallback=0)
    later = iso_to_ordinal("2026-01-01T00:00:01Z", fallback=0)
    assert earlier < later
    # Absent/unparseable/non-string -> the caller's fallback (a list index), not a crash.
    assert iso_to_ordinal(None, fallback=7) == 7
    assert iso_to_ordinal("not-a-date", fallback=9) == 9
    assert iso_to_ordinal("", fallback=3) == 3


def test_openai_tool_calls_from_object_and_list() -> None:
    call = {"function": {"name": "f", "arguments": "{}"}}
    assert openai_tool_calls({"tool_calls": [call, "junk"]}) == [call]
    assert openai_tool_calls([{"tool_calls": [call]}, {"role": "user"}]) == [call]
    assert openai_tool_calls("nope") == []
    assert openai_tool_calls({"content": "hi"}) == []


def test_openai_call_name_args_nested_and_flattened() -> None:
    assert openai_call_name_args({"function": {"name": "search", "arguments": '{"q": 1}'}}) == (
        "search",
        '{"q": 1}',
    )
    # Flattened shape, and an object (non-string) arguments value serialized to compact JSON.
    assert openai_call_name_args({"name": "run", "arguments": {"a": 1}}) == ("run", '{"a": 1}')
    # Missing/typeless fields degrade to empty strings, never a crash.
    assert openai_call_name_args({}) == ("", "")


def test_step_attribution_derived_from_semconv_attributes() -> None:
    """Model/provider/tokens/latency come off the LLM span; a tool error is environmental."""
    spans = [
        SpanRecord(
            trace_id="t1",
            span_id="a",
            start_nano=1_753_000_000_000_000_000,
            end_nano=1_753_000_000_250_000_000,
            attributes={
                "gen_ai.operation.name": "chat",
                "gen_ai.tool.name": "get_user",
                "gen_ai.tool.call.arguments": '{"id": "u1"}',
                "gen_ai.response.model": "claude-haiku-4-5",
                "gen_ai.system": "anthropic",
                "gen_ai.usage.input_tokens": 120,
                "gen_ai.usage.output_tokens": 30,
            },
        ),
        SpanRecord(
            trace_id="t1",
            span_id="b",
            start_nano=1_753_000_001_000_000_000,
            status_error=True,
            attributes={
                "gen_ai.operation.name": "execute_tool",
                "gen_ai.tool.message": "boom",
            },
        ),
    ]
    (trace,) = spans_to_traces(spans, source="test")
    attribution = trace.steps[0].attribution
    assert attribution is not None
    assert attribution.model == "claude-haiku-4-5"
    assert attribution.provider == "anthropic"
    assert attribution.input_tokens == 120
    assert attribution.output_tokens == 30
    assert attribution.latency_ms == 250.0
    assert attribution.error_class == ErrorClass.ENVIRONMENTAL


def test_step_attribution_errored_llm_span_is_controllable() -> None:
    spans = [
        SpanRecord(
            trace_id="t1",
            span_id="a",
            status_error=True,
            attributes={"gen_ai.operation.name": "chat", "gen_ai.completion": "refused"},
        ),
    ]
    (trace,) = spans_to_traces(spans, source="test")
    attribution = trace.steps[0].attribution
    assert attribution is not None
    assert attribution.error_class == ErrorClass.CONTROLLABLE


def test_step_attribution_explicit_wmo_attribute_wins() -> None:
    """A `wmo.attribution` JSON attr round-trips verbatim, beating semconv derivation."""
    explicit = StepAttribution(model="gpt-5.5", cost_usd=0.0123, provenance="capture-v2")
    spans = [
        SpanRecord(
            trace_id="t1",
            span_id="a",
            attributes={
                "gen_ai.operation.name": "chat",
                "gen_ai.completion": "hi",
                "gen_ai.response.model": "ignored-when-explicit",
                "wmo.attribution": explicit.model_dump_json(exclude_none=True),
            },
        ),
    ]
    (trace,) = spans_to_traces(spans, source="test")
    assert trace.steps[0].attribution == explicit


def test_step_attribution_absent_when_nothing_known() -> None:
    spans = [
        SpanRecord(
            trace_id="t1",
            span_id="a",
            attributes={"gen_ai.operation.name": "chat", "gen_ai.completion": "hi"},
        ),
    ]
    (trace,) = spans_to_traces(spans, source="test")
    assert trace.steps[0].attribution is None


def test_step_attribution_latency_ignores_synthetic_ordinals_and_scales_microseconds() -> None:
    """Writer/list-index ordinals must not become fake latencies; iso ordinals are µs, not ns.

    Regression: re-ingesting a corpus written by `otel_writer` (timestamps i*10, i*10+1) stamped
    every step with `latency_ms=1e-6` junk.
    """
    synthetic = SpanRecord(
        trace_id="t1",
        span_id="a",
        start_nano=10,
        end_nano=11,
        attributes={"gen_ai.operation.name": "chat", "gen_ai.response.model": "m"},
    )
    (trace,) = spans_to_traces([synthetic], source="test")
    attribution = trace.steps[0].attribution
    assert attribution is not None and attribution.latency_ms is None

    micros = synthetic.model_copy(
        update={"start_nano": 1_753_000_000_000_000, "end_nano": 1_753_000_000_250_000}
    )
    (trace,) = spans_to_traces([micros], source="test")
    attribution = trace.steps[0].attribution
    assert attribution is not None
    assert attribution.latency_ms == 250.0


def test_step_attribution_captures_request_config() -> None:
    """`gen_ai.request.*` sampling params land in attribution.config (policy-relevant knobs)."""
    spans = [
        SpanRecord(
            trace_id="t1",
            span_id="a",
            attributes={
                "gen_ai.operation.name": "chat",
                "gen_ai.completion": "hi",
                "gen_ai.request.model": "m",
                "gen_ai.request.temperature": 0.2,
                "gen_ai.request.max_tokens": 512,
                "gen_ai.request.top_p": 0.9,
            },
        ),
    ]
    (trace,) = spans_to_traces(spans, source="test")
    attribution = trace.steps[0].attribution
    assert attribution is not None
    assert attribution.config == {"temperature": 0.2, "max_tokens": 512, "top_p": 0.9}
    # The model key is attribution.model, never duplicated into config.
    assert "model" not in attribution.config


def test_span_emitter_orders_spans_and_seeds_only_the_first() -> None:
    """The shape the row adapters share: ordinal ids/sort keys, first-span-only trace attributes."""
    emitter = SpanEmitter("trace-abcdefghijkl", {"gen_ai.prompt": "do it"})
    emitter.emit({"gen_ai.tool.name": "search"}, tool=False)
    emitter.emit({"gen_ai.tool.message": "found"}, tool=True, error=True)

    action, observation = emitter.spans
    assert action.name == "chat"
    assert action.attributes["gen_ai.operation.name"] == "chat"
    assert action.attributes["gen_ai.prompt"] == "do it"
    assert (action.span_id, action.start_nano) == ("trace-abcdef000000a", 0)
    assert not action.status_error
    assert observation.name == "execute_tool"
    assert observation.attributes["gen_ai.operation.name"] == "execute_tool"
    # Trace-level attributes are seeded once, onto the first span only.
    assert "gen_ai.prompt" not in observation.attributes
    assert (observation.span_id, observation.start_nano) == ("trace-abcdef000001t", 1)
    assert observation.status_error


def test_span_emitter_keeps_a_span_own_value_and_needs_no_trace_attributes() -> None:
    emitter = SpanEmitter("t1", {"gen_ai.prompt": "seeded"})
    emitter.emit({"gen_ai.prompt": "own"}, tool=False)
    assert emitter.spans[0].attributes["gen_ai.prompt"] == "own"

    bare = SpanEmitter("t2")
    bare.emit({"gen_ai.completion": "hi"}, tool=False)
    assert bare.spans[0].attributes == {"gen_ai.operation.name": "chat", "gen_ai.completion": "hi"}
