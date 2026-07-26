"""Tests for the Langfuse observation-tree -> Trace adapter (langfuse).

Fixture-based, no network: a hand-authored Langfuse trace export with a GENERATION that issues a
tool call, the sibling TOOL observation carrying the result, and an ERROR tool observation.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from wmh.core.types import ActionKind
from wmh.ingest import get_adapter
from wmh.ingest.adapter import VendorPull
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


def test_vendor_pull_without_keys_is_friendly(monkeypatch) -> None:  # noqa: ANN001 - fixture
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    with pytest.raises(ValueError, match="LANGFUSE_PUBLIC_KEY"):
        LangfuseAdapter().from_vendor(VendorPull())


def test_vendor_pull_lists_then_fetches_each_trace(monkeypatch) -> None:  # noqa: ANN001 - fixture
    """Live-pull path with httpx mocked: page the trace list, fetch each id in full, normalize.

    The list endpoint returns observation-id STRINGS only (no steps), so the pull must re-fetch
    each trace by id; a pull that normalized list-page traces directly would yield zero steps.
    """
    import wmh.ingest.langfuse as lf

    calls: list[str] = []

    def fake_get(url, auth=None, params=None, timeout=None):  # noqa: ANN001, ANN202 - test stub
        class _Resp:
            def __init__(self, payload) -> None:  # noqa: ANN001
                self._payload = payload

            def raise_for_status(self) -> None: ...

            def json(self):  # noqa: ANN202
                return self._payload

        calls.append(url)
        assert auth == ("pk-1", "sk-1")
        if url.endswith("/api/public/traces"):
            listed = {**_TRACE, "observations": ["o1", "o2"]}
            return _Resp({"data": [listed], "meta": {"totalPages": 1}})
        assert url.endswith(f"/api/public/traces/{_TRACE['id']}")
        return _Resp(_TRACE)

    monkeypatch.setattr(lf.httpx, "get", fake_get)
    traces = LangfuseAdapter().from_vendor(VendorPull(api_key="pk-1:sk-1"))
    assert len(traces) == 1
    assert traces[0].steps, "full-trace fetch must yield real steps"
    assert any(url.endswith(f"/api/public/traces/{_TRACE['id']}") for url in calls)


def test_vendor_pull_unbounded_pagination_has_a_backstop(monkeypatch) -> None:  # noqa: ANN001
    """A credential-only pull against a huge project must not page forever."""
    import wmh.ingest.langfuse as lf

    list_calls = {"count": 0}

    def fake_get(url, auth=None, params=None, timeout=None):  # noqa: ANN001, ANN202 - test stub
        class _Resp:
            def __init__(self, payload) -> None:  # noqa: ANN001
                self._payload = payload

            def raise_for_status(self) -> None: ...

            def json(self):  # noqa: ANN202
                return self._payload

        if url.endswith("/api/public/traces"):
            assert isinstance(params, dict)
            list_calls["count"] += 1
            page = int(params["page"])
            data = [
                {"id": f"lf-{page}-{i}", "observations": ["o"]} for i in range(int(params["limit"]))
            ]
            return _Resp({"data": data})  # always a full page: pagination never ends naturally
        return _Resp({**_TRACE, "id": url.rsplit("/", 1)[-1]})

    monkeypatch.setattr(lf.httpx, "get", fake_get)
    LangfuseAdapter().from_vendor(VendorPull(api_key="pk:sk"))
    assert list_calls["count"] == lf._MAX_PAGES
