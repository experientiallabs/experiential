"""Tests for the PostHog LLM-observability events -> Trace adapter (posthog).

Fixture-based, no network: hand-authored $ai_generation / $ai_span events sharing one $ai_trace_id —
a generation that issues a tool call, the sibling $ai_span carrying the result, and an errored span.

The primary fixture uses PostHog's NORMALIZED output shape (what its LLM-observability integrations
actually emit): a tool call is a `content` PART `{"type": "function", "function": {...}}` with
`arguments` as a JSON object, and assistant text is a `[{"type": "text", "text": ...}]` parts list —
NOT a top-level `tool_calls` array. `test_raw_openai_tool_calls_passthrough` covers the raw-OpenAI
shape we ALSO accept.
"""

from __future__ import annotations

import json
from pathlib import Path

from wmh.core.types import ActionKind
from wmh.ingest import get_adapter
from wmh.ingest.adapter import VendorPull
from wmh.ingest.posthog import PostHogAdapter

_EVENTS = [
    {
        "event": "$ai_generation",
        "timestamp": "2026-01-01T00:00:00.000Z",
        "properties": {
            "$ai_trace_id": "t1",
            "$ai_input": [{"role": "user", "content": "what's the weather in Paris?"}],
            # Normalized shape: the tool call is a content PART, arguments is an object.
            "$ai_output_choices": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": {"city": "Paris"}},
                        }
                    ],
                }
            ],
        },
    },
    {
        "event": "$ai_span",
        "timestamp": "2026-01-01T00:00:01.000Z",
        "properties": {
            "$ai_trace_id": "t1",
            "$ai_span_name": "get_weather",
            "$ai_input_state": {"city": "Paris"},
            "$ai_output_state": "18C and sunny",
        },
    },
    {
        "event": "$ai_generation",
        "timestamp": "2026-01-01T00:00:02.000Z",
        "properties": {
            "$ai_trace_id": "t1",
            # Normalized shape: assistant text is a list of `text` parts.
            "$ai_output_choices": [
                {"role": "assistant", "content": [{"type": "text", "text": "It's 18C and sunny."}]}
            ],
        },
    },
]


def test_posthog_adapter_is_registered() -> None:
    assert get_adapter("posthog").name == "posthog"


def test_converts_generation_tool_call_and_result(tmp_path: Path) -> None:
    path = tmp_path / "events.json"
    path.write_text(json.dumps(_EVENTS), encoding="utf-8")

    traces = PostHogAdapter().from_file(str(path))

    assert len(traces) == 1
    trace = traces[0]
    # get_weather tool call (paired with its result span) + the final assistant message.
    assert len(trace.steps) == 2

    call = trace.steps[0]
    assert call.action.kind == ActionKind.TOOL_CALL
    assert call.action.name == "get_weather"
    assert call.action.arguments == {"city": "Paris"}
    assert call.observation.content == "18C and sunny"
    assert call.task == "what's the weather in Paris?"

    final = trace.steps[1]
    assert final.action.kind == ActionKind.MESSAGE
    assert final.action.content == "It's 18C and sunny."


def test_raw_openai_tool_calls_passthrough(tmp_path: Path) -> None:
    """We ALSO accept the raw-OpenAI shape: a top-level `tool_calls` array with a string-JSON
    `arguments` (some setups forward the provider payload unnormalized)."""
    events = [
        {
            "event": "$ai_generation",
            "timestamp": "2026-01-01T00:00:00Z",
            "properties": {
                "$ai_trace_id": "t3",
                "$ai_output_choices": [
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {"function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}}
                        ],
                    }
                ],
            },
        }
    ]
    path = tmp_path / "raw.json"
    path.write_text(json.dumps(events), encoding="utf-8")

    step = PostHogAdapter().from_file(str(path))[0].steps[0]
    assert step.action.kind == ActionKind.TOOL_CALL
    assert step.action.name == "get_weather"
    assert step.action.arguments == {"city": "Paris"}


def test_query_results_wrapper_and_error_flag(tmp_path: Path) -> None:
    events = [
        {
            "event": "$ai_span",
            "timestamp": "2026-01-01T00:00:00Z",
            "properties": {
                "$ai_trace_id": "t2",
                "$ai_span_name": "charge",
                "$ai_output_state": "declined",
                "$ai_is_error": True,
            },
        }
    ]
    path = tmp_path / "q.json"
    path.write_text(json.dumps({"results": events}), encoding="utf-8")

    trace = PostHogAdapter().from_file(str(path))[0]
    assert trace.steps[0].action.name == "charge"
    assert trace.steps[0].observation.is_error is True


def test_pull_without_key_is_friendly(monkeypatch) -> None:  # noqa: ANN001 - fixture
    monkeypatch.delenv("POSTHOG_API_KEY", raising=False)
    try:
        PostHogAdapter().from_vendor(VendorPull(project="1"))
    except ValueError as exc:
        assert "API key" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


def test_pull_queries_hogql_and_normalizes(monkeypatch) -> None:  # noqa: ANN001 - fixture
    """Live-pull path with httpx mocked: HogQL results (row tuples) -> events -> Trace."""
    import wmh.ingest.posthog as ph

    def fake_post(url, headers=None, json=None, timeout=None):  # noqa: ANN001, ANN202, A002 - stub
        class _Resp:
            def raise_for_status(self) -> None: ...

            def json(self) -> dict:
                # HogQL returns rows as [event, properties, timestamp].
                return {
                    "results": [
                        [
                            "$ai_span",
                            {
                                "$ai_trace_id": "t9",
                                "$ai_span_name": "lookup",
                                "$ai_output_state": "ok",
                            },
                            "2026-01-01T00:00:00Z",
                        ]
                    ],
                    "columns": ["event", "properties", "timestamp"],
                }

        # Trailing slash is required (PostHog 301-redirects `/query` and drops the POST body).
        assert url.endswith("/api/projects/42/query/")
        assert headers == {"Authorization": "Bearer k"}
        return _Resp()

    monkeypatch.setattr(ph.httpx, "post", fake_post)
    traces = PostHogAdapter().from_vendor(VendorPull(api_key="k", project="42"))
    assert len(traces) == 1
    assert traces[0].steps[0].action.name == "lookup"
    assert traces[0].steps[0].observation.content == "ok"
