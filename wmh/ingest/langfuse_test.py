"""Tests for the Langfuse observation-tree -> Trace adapter (langfuse).

Fixture-based, no network: a hand-authored Langfuse trace export with a GENERATION that issues a
tool call, the sibling TOOL observation carrying the result, and an ERROR tool observation.
"""

from __future__ import annotations

import json
from pathlib import Path

from wmh.core.types import ActionKind
from wmh.ingest import get_adapter
from wmh.ingest.langfuse import LangfuseAdapter

# A realistic Langfuse `GET /api/public/traces/{id}` export: a trace with nested observations.
_TRACE = {
    "id": "lf-trace-abc123",
    "name": "weather-agent",
    "input": "what's the weather in Paris?",
    "output": "It's 18C and sunny in Paris.",
    "metadata": {"benchmark": "demo", "env": "prod"},
    "observations": [
        {
            "id": "o1",
            "type": "GENERATION",
            "name": "llm",
            "startTime": "2026-01-01T00:00:01.000Z",
            "model": "gpt-4o",
            "input": [{"role": "user", "content": "what's the weather in Paris?"}],
            "output": {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "c1",
                        "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                    }
                ],
            },
            "level": "DEFAULT",
        },
        {
            "id": "o2",
            "type": "TOOL",
            "name": "get_weather",
            "startTime": "2026-01-01T00:00:02.000Z",
            "input": {"city": "Paris"},
            "output": "18C and sunny",
            "level": "DEFAULT",
        },
        {
            "id": "o3",
            "type": "TOOL",
            "name": "get_forecast",
            "startTime": "2026-01-01T00:00:03.000Z",
            "input": {"city": "Paris"},
            "output": "forecast service unavailable",
            "level": "ERROR",
        },
    ],
}


def test_langfuse_adapter_is_registered() -> None:
    assert get_adapter("langfuse").name == "langfuse"


def test_generation_tool_call_pairs_with_tool_observation(tmp_path: Path) -> None:
    path = tmp_path / "trace.json"
    path.write_text(json.dumps(_TRACE), encoding="utf-8")

    traces = LangfuseAdapter().from_file(str(path))

    assert len(traces) == 1
    trace = traces[0]
    assert trace.trace_id == "lf-trace-abc123"
    assert trace.metadata == {"benchmark": "demo", "env": "prod"}
    # The GENERATION's tool call pairs with the o2 TOOL result; the o3 ERROR TOOL pairs alone.
    assert len(trace.steps) == 2

    call = trace.steps[0]
    assert call.action.kind == ActionKind.TOOL_CALL
    assert call.action.name == "get_weather"
    assert call.action.arguments == {"city": "Paris"}
    assert call.observation.content == "18C and sunny"
    assert call.observation.is_error is False
    assert call.task == "what's the weather in Paris?"

    err = trace.steps[1]
    assert err.action.kind == ActionKind.TOOL_CALL
    assert err.action.name == "get_forecast"
    assert err.action.arguments == {"city": "Paris"}
    assert err.observation.content == "forecast service unavailable"
    assert err.observation.is_error is True


def test_plain_generation_becomes_message_step(tmp_path: Path) -> None:
    trace = {
        "id": "lf-2",
        "input": "say hi",
        "observations": [
            {
                "id": "g1",
                "type": "GENERATION",
                "startTime": "2026-01-01T00:00:01.000Z",
                "output": {"role": "assistant", "content": "hello there"},
            }
        ],
    }
    path = tmp_path / "msg.json"
    path.write_text(json.dumps(trace), encoding="utf-8")

    step = LangfuseAdapter().from_file(str(path))[0].steps[0]
    assert step.action.kind == ActionKind.MESSAGE
    assert step.action.content is not None
    assert "hello there" in step.action.content
    assert step.observation.content == ""


def test_api_list_page_and_ordering_by_start_time(tmp_path: Path) -> None:
    # API list shape `{"data": [...]}`; observations are deliberately out of start-time order.
    page = {
        "data": [
            {
                "id": "lf-3",
                "input": "list files",
                "observations": [
                    {
                        "id": "t1",
                        "type": "TOOL",
                        "name": "ls",
                        "startTime": "2026-01-01T00:00:05.000Z",
                        "input": {},
                        "output": "a.txt\nb.txt",
                    },
                    {
                        "id": "gen",
                        "type": "GENERATION",
                        "startTime": "2026-01-01T00:00:04.000Z",
                        "output": {
                            "tool_calls": [
                                {"id": "x", "function": {"name": "ls", "arguments": "{}"}}
                            ]
                        },
                    },
                ],
            }
        ]
    }
    path = tmp_path / "page.json"
    path.write_text(json.dumps(page), encoding="utf-8")

    traces = LangfuseAdapter().from_file(str(path))
    assert len(traces) == 1
    step = traces[0].steps[0]
    # Ordered by startTime: the GENERATION (04s) precedes the TOOL result (05s) and they pair.
    assert step.action.name == "ls"
    assert step.observation.content == "a.txt\nb.txt"


def test_jsonl_multiple_traces(tmp_path: Path) -> None:
    other = {
        "id": "lf-4",
        "input": "ping",
        "observations": [
            {
                "id": "g",
                "type": "GENERATION",
                "startTime": "2026-01-01T00:00:01.000Z",
                "output": "pong",
            }
        ],
    }
    path = tmp_path / "traces.jsonl"
    path.write_text(json.dumps(_TRACE) + "\n" + json.dumps(other) + "\n", encoding="utf-8")

    traces = LangfuseAdapter().from_file(str(path))
    assert len(traces) == 2


def test_vendor_pull_unsupported_is_friendly() -> None:
    from wmh.ingest.adapter import VendorPull

    try:
        LangfuseAdapter().from_vendor(VendorPull())
    except ValueError as exc:
        assert "does not support live vendor pulls" in str(exc)
        assert "langfuse" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")
