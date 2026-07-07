"""Tests for the Mastra AI-tracing -> Trace adapter (mastra).

Fixture-based, no network. The primary fixture uses Mastra's CURRENT `ExportedSpan` shape (post the
2025-11 "model" rename): span id field `id`, span kind `type`, `model_generation` for LLM turns,
`startTime` for ordering, and AI-SDK-v5 tool calls (`{toolName, input}`). A `model_generation` span
issues the tool call, the sibling `tool_call` span carries the result, and a final model span holds
the assistant message — all under one `traceId`.
`test_legacy_prerename_span_shape` covers the pre-rename aliases we still accept.
"""

from __future__ import annotations

import json
from pathlib import Path

from wmh.core.types import ActionKind
from wmh.ingest import get_adapter
from wmh.ingest.adapter import VendorPull
from wmh.ingest.mastra import MastraAdapter

_SPANS = [
    {
        "traceId": "t1",
        "id": "s1",
        "type": "agent_run",
        "name": "weatherAgent",
        "input": [{"role": "user", "content": "what's the weather in Paris?"}],
        "startTime": "2026-01-01T00:00:00.000Z",
    },
    {
        "traceId": "t1",
        "id": "s2",
        "parentSpanId": "s1",
        "type": "model_generation",
        "name": "modelGeneration",
        "output": {
            "role": "assistant",
            # AI SDK v5 tool call: `toolName` + `input` (not v4's `args`).
            "toolCalls": [
                {"toolCallId": "c1", "toolName": "getWeather", "input": {"city": "Paris"}}
            ],
        },
        "attributes": {"model": "gpt-4o"},
        "startTime": "2026-01-01T00:00:01.000Z",
    },
    {
        "traceId": "t1",
        "id": "s3",
        "parentSpanId": "s1",
        "type": "tool_call",
        "name": "getWeather",
        "input": {"city": "Paris"},
        "output": "18C and sunny",
        "startTime": "2026-01-01T00:00:02.000Z",
    },
    {
        "traceId": "t1",
        "id": "s4",
        "parentSpanId": "s1",
        "type": "model_generation",
        "name": "modelGeneration",
        "output": {"text": "It's 18C and sunny in Paris."},
        "startTime": "2026-01-01T00:00:03.000Z",
    },
]


def test_mastra_adapter_is_registered() -> None:
    assert get_adapter("mastra").name == "mastra"


def test_converts_llm_tool_call_and_result(tmp_path: Path) -> None:
    path = tmp_path / "spans.json"
    path.write_text(json.dumps(_SPANS), encoding="utf-8")

    traces = MastraAdapter().from_file(str(path))

    assert len(traces) == 1
    trace = traces[0]
    # getWeather tool call (paired with its tool_call span result) + the final assistant message.
    assert len(trace.steps) == 2

    call = trace.steps[0]
    assert call.action.kind == ActionKind.TOOL_CALL
    assert call.action.name == "getWeather"
    assert call.action.arguments == {"city": "Paris"}
    assert call.observation.content == "18C and sunny"
    assert call.task == "what's the weather in Paris?"  # from the agent_run input

    final = trace.steps[1]
    assert final.action.kind == ActionKind.MESSAGE
    assert final.action.content == "It's 18C and sunny in Paris."


def test_openai_tool_call_shape_and_error(tmp_path: Path) -> None:
    spans = [
        {
            "traceId": "t2",
            "spanId": "a",
            "spanType": "llm_generation",
            "output": {
                "tool_calls": [{"id": "x", "function": {"name": "charge", "arguments": "{}"}}]
            },
            "startedAt": "2026-01-01T00:00:00Z",
        },
        {
            "traceId": "t2",
            "spanId": "b",
            "spanType": "tool_call",
            "name": "charge",
            "output": "declined",
            "errorInfo": {"message": "card declined"},
            "startedAt": "2026-01-01T00:00:01Z",
        },
    ]
    path = tmp_path / "e.json"
    path.write_text(json.dumps(spans), encoding="utf-8")

    step = MastraAdapter().from_file(str(path))[0].steps[0]
    assert step.action.name == "charge"  # OpenAI-shaped tool call parsed
    assert step.observation.content == "declined"
    assert step.observation.is_error is True


def test_legacy_prerename_span_shape(tmp_path: Path) -> None:
    """Pre-2025-11 exports (still accepted): `spanId`/`spanType`/`startedAt`, the `llm_generation`
    kind, and an AI-SDK-v4 `args` tool call."""
    spans = [
        {
            "traceId": "t4",
            "spanId": "s1",
            "spanType": "llm_generation",
            "output": {
                "toolCalls": [{"toolCallId": "c", "toolName": "getWeather", "args": {"city": "SF"}}]
            },
            "startedAt": "2026-01-01T00:00:00Z",
        },
        {
            "traceId": "t4",
            "spanId": "s2",
            "spanType": "tool_call",
            "name": "getWeather",
            "output": "60F and foggy",
            "startedAt": "2026-01-01T00:00:01Z",
        },
    ]
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(spans), encoding="utf-8")

    step = MastraAdapter().from_file(str(path))[0].steps[0]
    assert step.action.kind == ActionKind.TOOL_CALL
    assert step.action.name == "getWeather"
    assert step.action.arguments == {"city": "SF"}
    assert step.observation.content == "60F and foggy"


def test_standalone_tool_span_is_self_contained(tmp_path: Path) -> None:
    # A tool_call span with no preceding llm_generation still becomes a complete step.
    spans = [
        {
            "traceId": "t3",
            "spanId": "only",
            "spanType": "mcp_tool_call",
            "name": "search",
            "input": {"q": "otel"},
            "output": "3 results",
        }
    ]
    path = tmp_path / "s.json"
    path.write_text(json.dumps(spans), encoding="utf-8")

    step = MastraAdapter().from_file(str(path))[0].steps[0]
    assert step.action.name == "search"
    assert step.action.arguments == {"q": "otel"}
    assert step.observation.content == "3 results"


def test_spans_wrapper_and_jsonl(tmp_path: Path) -> None:
    # A {"spans": [...]} wrapper works, and so does one span per JSONL line (same trace grouped).
    path = tmp_path / "w.jsonl"
    lines = [json.dumps({"spans": _SPANS[:2]}), json.dumps(_SPANS[2]), json.dumps(_SPANS[3])]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    traces = MastraAdapter().from_file(str(path))
    assert len(traces) == 1
    assert traces[0].steps[0].action.name == "getWeather"
    assert traces[0].steps[0].observation.content == "18C and sunny"


def test_pull_without_url_is_friendly(monkeypatch) -> None:  # noqa: ANN001 - fixture
    monkeypatch.delenv("MASTRA_URL", raising=False)
    try:
        MastraAdapter().from_vendor(VendorPull())
    except ValueError as exc:
        assert "server URL" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


def test_pull_fetches_observability_traces(monkeypatch) -> None:  # noqa: ANN001 - fixture
    """Live-pull path with httpx mocked: fetch the observability API, normalize the spans."""
    import wmh.ingest.mastra as ms

    def fake_get(url, headers=None, params=None, timeout=None):  # noqa: ANN001, ANN202 - stub
        class _Resp:
            def raise_for_status(self) -> None: ...

            def json(self) -> dict:
                return {"traces": [{"spans": _SPANS}]}

        assert url == "http://localhost:4111/api/observability/traces"
        return _Resp()

    monkeypatch.setattr(ms.httpx, "get", fake_get)
    traces = MastraAdapter().from_vendor(VendorPull(project="http://localhost:4111/"))
    assert len(traces) == 1
    assert traces[0].steps[0].action.name == "getWeather"
