"""Behavior tests for focused PostHog-to-canonical-trace normalization."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

from pydantic import JsonValue

from wmo.common.core.artifacts import JsonObject, SourceIdentity
from wmo.common.traces import Trace
from wmo.simulation.ingest.otlp import normalize_otlp_payload
from wmo.simulation.ingest.posthog import (
    PostHogPullRequest,
    load_posthog_file,
    normalize_posthog_payload,
    pull_posthog_traces,
)
from wmo.simulation.mining.descriptors import routing_descriptor

_TRACE_ID = "1" * 32
_CALL_ID = "call-1"


def _source() -> SourceIdentity:
    return SourceIdentity(kind="production", source_id="posthog-fixture", sha256="b" * 64)


def _posthog_events(*, errored: bool = False) -> list[dict[str, JsonValue]]:
    """Return one ordered normalized PostHog generation, tool, and trace-root fixture."""
    outcome: dict[str, JsonValue] = {
        "wmo.request.context": {"tier": "gold"},
        "wmo.request.tags": ["domain:travel"],
        "wmo.request.tools": [
            {
                "name": "cancel_reservation",
                "description": "Cancel one reservation.",
                "input_schema": {"type": "object"},
            }
        ],
        "wmo.customer.id": "customer-7",
        "wmo.conversation.id": "conversation-9",
        "wmo.outcome.status": "failure" if errored else "success",
    }
    if not errored:
        outcome["wmo.outcome.name"] = "reservation_cancelled"
    return [
        {
            "event": "$ai_generation",
            "timestamp": "2025-10-09T08:53:20Z",
            "properties": {
                "$ai_trace_id": _TRACE_ID,
                "$ai_span_id": "generation-1",
                "$ai_provider": "openai",
                "$ai_model": "gpt-test",
                "$ai_input": [{"role": "user", "content": "Cancel reservation R-17"}],
                "$ai_output_choices": [
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": _CALL_ID,
                                "function": {
                                    "name": "cancel_reservation",
                                    "arguments": {"reservation_id": "R-17"},
                                },
                            }
                        ],
                    }
                ],
                **outcome,
            },
        },
        {
            "event": "$ai_span",
            "timestamp": "2025-10-09T08:53:21Z",
            "properties": {
                "$ai_trace_id": _TRACE_ID,
                "$ai_span_id": "tool-1",
                "$ai_parent_id": "generation-1",
                "$ai_span_name": "cancel_reservation",
                "$ai_tool_call_id": _CALL_ID,
                "$ai_input_state": {"reservation_id": "R-17"},
                "$ai_output_state": "Reservation cancelled",
                "$ai_is_error": errored,
            },
        },
        {
            "event": "$ai_trace",
            "timestamp": "2025-10-09T08:53:22Z",
            "properties": {
                "$ai_trace_id": _TRACE_ID,
                "$ai_span_id": "root-1",
                "$ai_output_state": "Reservation cancelled",
            },
        },
    ]


def _event_properties(event: dict[str, JsonValue]) -> dict[str, JsonValue]:
    """Return one mutable PostHog properties fixture object."""
    properties = event["properties"]
    assert isinstance(properties, dict)
    return properties


def _attribute(key: str, value: JsonValue) -> dict[str, JsonValue]:
    """Encode a fixture value as the OpenTelemetry AnyValue shape."""
    if isinstance(value, bool):
        encoded: dict[str, JsonValue] = {"boolValue": value}
    elif isinstance(value, int):
        encoded = {"intValue": str(value)}
    else:
        encoded = {"stringValue": value if isinstance(value, str) else json.dumps(value)}
    return {"key": key, "value": encoded}


def _otlp_equivalent() -> dict[str, JsonValue]:
    """Return the canonical GenAI evidence equivalent of the normal PostHog fixture."""
    shared = [
        _attribute("wmo.request.context", {"tier": "gold"}),
        _attribute("wmo.request.tags", ["domain:travel"]),
        _attribute("wmo.customer.id", "customer-7"),
        _attribute("wmo.conversation.id", "conversation-9"),
        _attribute("wmo.outcome.status", "success"),
        _attribute("wmo.outcome.name", "reservation_cancelled"),
    ]
    return {
        "resourceSpans": [
            {
                "scopeSpans": [
                    {
                        "spans": [
                            {
                                "traceId": _TRACE_ID,
                                "spanId": "2" * 16,
                                "name": "agent.model_call",
                                "startTimeUnixNano": "1760000000000000000",
                                "endTimeUnixNano": "1760000001000000000",
                                "attributes": [
                                    _attribute("gen_ai.operation.name", "chat"),
                                    _attribute("gen_ai.provider.name", "openai"),
                                    _attribute("gen_ai.request.model", "gpt-test"),
                                    _attribute(
                                        "gen_ai.input.messages",
                                        [{"role": "user", "content": "Cancel reservation R-17"}],
                                    ),
                                    _attribute("gen_ai.tool.name", "cancel_reservation"),
                                    _attribute("gen_ai.tool.call.id", _CALL_ID),
                                    _attribute(
                                        "gen_ai.tool.definitions",
                                        [
                                            {
                                                "name": "cancel_reservation",
                                                "description": "Cancel one reservation.",
                                                "input_schema": {"type": "object"},
                                            }
                                        ],
                                    ),
                                    *shared,
                                ],
                            },
                            {
                                "traceId": _TRACE_ID,
                                "spanId": "3" * 16,
                                "parentSpanId": "2" * 16,
                                "name": "agent.tool_call",
                                "startTimeUnixNano": "1760000001000000000",
                                "endTimeUnixNano": "1760000002000000000",
                                "attributes": [
                                    _attribute("gen_ai.operation.name", "execute_tool"),
                                    _attribute("gen_ai.tool.name", "cancel_reservation"),
                                    _attribute("gen_ai.tool.call.id", _CALL_ID),
                                    _attribute("gen_ai.tool.message", "Reservation cancelled"),
                                ],
                            },
                        ]
                    }
                ]
            }
        ]
    }


def _visible_evidence(trace: Trace) -> tuple[str, ...]:
    """Return the source-independent canonical fields expected to agree across formats."""
    return (
        trace.task,
        json.dumps(trace.initial_context, sort_keys=True),
        ",".join(tool.name for tool in trace.tools),
        "" if trace.outcome is None else trace.outcome.status,
        _outcome_name(trace),
        json.dumps(
            [
                [
                    span.name,
                    span.attributes.get("gen_ai.operation.name"),
                    span.attributes.get("gen_ai.tool.name"),
                    span.attributes.get("gen_ai.tool.call.id"),
                ]
                for span in trace.spans
                if span.name != "agent.trace"
            ],
            sort_keys=True,
        ),
    )


def _outcome_name(trace: Trace) -> str:
    """Return an outcome name as a comparison-safe string."""
    if trace.outcome is None or trace.outcome.outcome_name is None:
        return ""
    return trace.outcome.outcome_name


def test_posthog_and_otlp_equivalent_fixtures_produce_equivalent_visible_evidence() -> None:
    posthog = normalize_posthog_payload(list(reversed(_posthog_events())), source=_source())
    otlp = normalize_otlp_payload(
        _otlp_equivalent(),
        source=SourceIdentity(kind="otlp", source_id="otlp-fixture", sha256="c" * 64),
    )

    assert posthog.issues == ()
    assert otlp.issues == ()
    assert _visible_evidence(posthog.traces[0]) == _visible_evidence(otlp.traces[0])
    assert posthog.traces[0].trace_id == _TRACE_ID
    assert all(len(span.span_id) == 16 for span in posthog.traces[0].spans)
    assert posthog.traces[0].spans[0].started_at < posthog.traces[0].spans[1].started_at


def test_posthog_error_and_jsonl_export_are_retained_as_canonical_failure_evidence(
    tmp_path: Path,
) -> None:
    path = tmp_path / "posthog.jsonl"
    events = _posthog_events(errored=True)
    path.write_text(
        "\n".join((json.dumps(events[0]), json.dumps(events[1]), "{not-json")) + "\n",
        encoding="utf-8",
    )

    result = load_posthog_file(path)

    assert len(result.traces) == 1
    assert result.invalid_trace_count == 1
    assert result.traces[0].outcome is not None
    assert result.traces[0].outcome.status == "failure"
    assert any(span.failure is not None for span in result.traces[0].spans)


def test_posthog_rejects_unmatched_generated_tool_calls() -> None:
    result = normalize_posthog_payload(_posthog_events()[:1], source=_source())

    assert result.traces == ()
    assert len(result.issues) == 1
    assert "unmatched generated PostHog tool calls" in result.issues[0].message


def test_posthog_rejects_unmatched_explicit_tool_results() -> None:
    events = _posthog_events()
    generation_properties = _event_properties(events[0])
    generation_properties["$ai_output_choices"] = [
        {"role": "assistant", "content": "I will check that for you."}
    ]

    result = normalize_posthog_payload(events, source=_source())

    assert result.traces == ()
    assert len(result.issues) == 1
    assert "unmatched explicit PostHog tool result: cancel_reservation:call-1" in (
        result.issues[0].message
    )


def test_posthog_same_tool_multicall_without_result_ids_pairs_fifo() -> None:
    events = _posthog_events()
    generation_properties = _event_properties(events[0])
    generation_properties["$ai_output_choices"] = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call-first",
                    "function": {
                        "name": "cancel_reservation",
                        "arguments": {"reservation_id": "R-17"},
                    },
                },
                {
                    "id": "call-second",
                    "function": {
                        "name": "cancel_reservation",
                        "arguments": {"reservation_id": "R-18"},
                    },
                },
            ],
        }
    ]
    first_result_properties = _event_properties(events[1])
    first_result_properties.pop("$ai_tool_call_id")
    events.insert(
        2,
        {
            "event": "$ai_span",
            "timestamp": "2025-10-09T08:53:22Z",
            "properties": {
                "$ai_trace_id": _TRACE_ID,
                "$ai_span_id": "tool-2",
                "$ai_parent_id": "generation-1",
                "$ai_span_name": "cancel_reservation",
                "$ai_input_state": {"reservation_id": "R-18"},
                "$ai_output_state": "Reservation cancelled",
            },
        },
    )

    result = normalize_posthog_payload(events, source=_source())

    assert result.issues == ()
    model_spans = [span for span in result.traces[0].spans if span.name == "agent.model_call"]
    tool_spans = [span for span in result.traces[0].spans if span.name == "agent.tool_call"]
    assert [span.attributes["gen_ai.tool.call.id"] for span in tool_spans] == [
        "call-first",
        "call-second",
    ]
    assert [span.parent_span_id for span in tool_spans] == [span.span_id for span in model_spans]


def test_posthog_generation_failures_become_trace_failure_evidence() -> None:
    events = _posthog_events()
    generation_properties = _event_properties(events[0])
    generation_properties["$ai_is_error"] = True
    generation_properties.pop("wmo.outcome.status")
    generation_properties.pop("wmo.outcome.name")

    result = normalize_posthog_payload(events, source=_source())

    assert result.issues == ()
    assert result.traces[0].outcome is not None
    assert result.traces[0].outcome.status == "failure"
    assert result.traces[0].spans[0].failure is not None


def test_posthog_timestamp_ties_preserve_source_ordinal_and_valid_parent_links() -> None:
    events = _posthog_events()
    for event in events:
        event["timestamp"] = "2025-10-09T08:53:20Z"

    result = normalize_posthog_payload(events, source=_source())

    assert result.issues == ()
    spans = result.traces[0].spans
    assert [span.name for span in spans] == [
        "agent.model_call",
        "agent.tool_call",
        "agent.trace",
    ]
    emitted_span_ids = {span.span_id for span in spans}
    assert spans[1].parent_span_id == spans[0].span_id
    assert all(
        span.parent_span_id is None or span.parent_span_id in emitted_span_ids for span in spans
    )


def test_posthog_preserves_an_emitted_source_parent_link() -> None:
    events = _posthog_events()
    _event_properties(events[1])["$ai_parent_id"] = "root-1"

    result = normalize_posthog_payload(events, source=_source())

    assert result.issues == ()
    spans = result.traces[0].spans
    model_span = next(span for span in spans if span.name == "agent.model_call")
    tool_span = next(span for span in spans if span.name == "agent.tool_call")
    root_span = next(span for span in spans if span.name == "agent.trace")
    assert tool_span.parent_span_id == root_span.span_id
    assert tool_span.parent_span_id != model_span.span_id


def test_posthog_ignores_late_request_visible_extensions() -> None:
    events = _posthog_events()
    late_properties = _event_properties(events[-1])
    late_properties.update(
        {
            "wmo.request.context": {"tier": "secret"},
            "wmo.request.tags": ["domain:secret"],
            "wmo.request.tools": [
                {
                    "name": "delete_reservation",
                    "description": "Delete one reservation.",
                    "input_schema": {"type": "object"},
                }
            ],
        }
    )

    result = normalize_posthog_payload(events, source=_source())

    assert result.issues == ()
    trace = result.traces[0]
    assert trace.initial_context == {"tier": "gold"}
    assert tuple(tool.name for tool in trace.tools) == ("cancel_reservation",)
    assert routing_descriptor(trace).tags == ("domain:travel",)


class _FakeResponse:
    """Deterministic successful HTTP response for the authorized pull seam."""

    def __init__(self, events: tuple[dict[str, JsonValue], ...] | None = None) -> None:
        self._events = events

    def raise_for_status(self) -> None:
        """Model a successful PostHog query response."""

    def json(self) -> JsonValue:
        """Return matched HogQL AI-event rows in the official result-array shape."""
        events = self._events or tuple(_posthog_events())
        return {
            "results": [
                [
                    event["event"],
                    event["properties"],
                    event["timestamp"],
                    event.get("uuid", f"event-{index:04d}"),
                ]
                for index, event in enumerate(events)
            ],
            "columns": ["event", "properties", "timestamp", "uuid"],
        }


class _FakeClient:
    """Captures the request without making a network call."""

    def __init__(self, response: _FakeResponse | None = None) -> None:
        self.url = ""
        self.headers: Mapping[str, str] = {}
        self.body: JsonObject = {}
        self.timeout = 0.0
        self._response = response or _FakeResponse()

    def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        json: JsonObject,
        timeout: float,
    ) -> _FakeResponse:
        """Capture the bounded HogQL request and return deterministic source rows."""
        self.url = url
        self.headers = headers
        self.body = json
        self.timeout = timeout
        return self._response


def test_authorized_hogql_pull_uses_injected_client_and_source_converter() -> None:
    client = _FakeClient()

    result = pull_posthog_traces(
        PostHogPullRequest(
            project_id="42",
            api_key="fixture-key",
            host="https://eu.posthog.com",
            since=datetime(2025, 10, 9, tzinfo=UTC),
        ),
        client=client,
    )

    assert result.issues == ()
    assert len(result.traces) == 1
    assert client.url == "https://eu.posthog.com/api/projects/42/query/"
    assert dict(client.headers) == {"Authorization": "Bearer fixture-key"}
    assert client.timeout == 60.0
    query = client.body["query"]
    assert isinstance(query, dict)
    query_text = query["query"]
    assert isinstance(query_text, str)
    assert "select event, properties, timestamp, uuid" in query_text
    assert "event like '$ai_%'" in query_text
    assert "order by timestamp asc, uuid asc" in query_text


def test_authorized_hogql_pull_keeps_uuid_timestamp_ties_fifo_deterministic() -> None:
    events = _posthog_events()
    generation_properties = _event_properties(events[0])
    generation_properties["$ai_output_choices"] = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "cancel_reservation",
                        "arguments": {"reservation_id": "R-17"},
                    }
                },
                {
                    "function": {
                        "name": "cancel_reservation",
                        "arguments": {"reservation_id": "R-18"},
                    }
                },
            ],
        }
    ]
    first_result_properties = _event_properties(events[1])
    first_result_properties.pop("$ai_tool_call_id")
    events.insert(
        2,
        {
            "event": "$ai_span",
            "timestamp": "2025-10-09T08:53:20Z",
            "properties": {
                "$ai_trace_id": _TRACE_ID,
                "$ai_span_id": "tool-2",
                "$ai_parent_id": "generation-1",
                "$ai_span_name": "cancel_reservation",
                "$ai_input_state": {"reservation_id": "R-18"},
                "$ai_output_state": "Reservation cancelled",
            },
        },
    )
    for index, event in enumerate(events, start=1):
        event["timestamp"] = "2025-10-09T08:53:20Z"
        event["uuid"] = f"event-{index:04d}"
    response = _FakeResponse(tuple(reversed(events)))
    request = PostHogPullRequest(
        project_id="42",
        api_key="fixture-key",
        host="https://eu.posthog.com",
    )

    first = pull_posthog_traces(request, client=_FakeClient(response))
    second = pull_posthog_traces(request, client=_FakeClient(response))

    assert first == second
    trace = first.traces[0]
    assert [span.name for span in trace.spans] == [
        "agent.model_call",
        "agent.model_call",
        "agent.tool_call",
        "agent.tool_call",
        "agent.trace",
    ]
    tool_spans = [span for span in trace.spans if span.name == "agent.tool_call"]
    assert [span.attributes["gen_ai.tool.call.id"] for span in tool_spans] == [
        "posthog-call-3-0",
        "posthog-call-3-1",
    ]
