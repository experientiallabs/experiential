"""Tests for the Braintrust span-row -> Trace adapter (braintrust).

Fixture-based, no network: a hand-authored Braintrust export with an `llm` span that issues a tool
call, the sibling `tool` span carrying the result, and a second `tool` span in an error state — all
sharing one `root_span_id` so they group into a single trace.
"""

from __future__ import annotations

import json
from pathlib import Path

from wmh.core.types import ActionKind
from wmh.ingest import get_adapter
from wmh.ingest.adapter import VendorPull
from wmh.ingest.braintrust import BraintrustAdapter

# A realistic Braintrust fetch export: rows sharing `root_span_id` "r1" form one trace. An llm row
# emits a tool call; a `tool` row returns the result; a second `tool` row errors.
_ROWS = [
    {
        "span_id": "s1",
        "root_span_id": "r1",
        "span_parents": [],
        "span_attributes": {"name": "agent", "type": "llm"},
        "input": [{"role": "user", "content": "what's the weather in Paris?"}],
        "output": {
            "role": "assistant",
            "tool_calls": [
                {"id": "c1", "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}}
            ],
        },
        "metadata": {"model": "gpt-4o", "benchmark": "demo"},
        "created": "2026-01-01T00:00:01.000Z",
        "error": None,
    },
    {
        "span_id": "s2",
        "root_span_id": "r1",
        "span_parents": ["s1"],
        "span_attributes": {"name": "get_weather", "type": "tool"},
        "input": {"city": "Paris"},
        "output": "18C and sunny",
        "created": "2026-01-01T00:00:02.000Z",
        "error": None,
    },
    {
        "span_id": "s3",
        "root_span_id": "r1",
        "span_parents": ["s1"],
        "span_attributes": {"name": "get_forecast", "type": "tool"},
        "input": {"city": "Paris"},
        "output": "forecast service unavailable",
        "created": "2026-01-01T00:00:03.000Z",
        "error": "ServiceUnavailable: 503",
    },
]


def test_braintrust_adapter_is_registered() -> None:
    assert get_adapter("braintrust").name == "braintrust"


def test_llm_tool_call_pairs_with_tool_row_and_groups_into_one_trace(tmp_path: Path) -> None:
    path = tmp_path / "export.json"
    path.write_text(json.dumps(_ROWS), encoding="utf-8")

    traces = BraintrustAdapter().from_file(str(path))

    # All three rows share root_span_id "r1" -> exactly ONE trace.
    assert len(traces) == 1
    trace = traces[0]
    assert trace.trace_id == "r1"
    assert trace.metadata == {"model": "gpt-4o", "benchmark": "demo"}
    # The llm tool call pairs with the s2 tool result; the s3 error tool row pairs alone.
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


def test_plain_llm_row_becomes_message_step(tmp_path: Path) -> None:
    rows = [
        {
            "span_id": "s1",
            "root_span_id": "r2",
            "span_attributes": {"name": "agent", "type": "llm"},
            "input": [{"role": "user", "content": "say hi"}],
            "output": {"role": "assistant", "content": "hello there"},
            "created": "2026-01-01T00:00:01.000Z",
            "error": None,
        }
    ]
    path = tmp_path / "msg.json"
    path.write_text(json.dumps(rows), encoding="utf-8")

    step = BraintrustAdapter().from_file(str(path))[0].steps[0]
    assert step.action.kind == ActionKind.MESSAGE
    assert step.action.content is not None
    assert "hello there" in step.action.content
    assert step.observation.content == ""


def test_events_wrapper_and_ordering_by_created(tmp_path: Path) -> None:
    # `{"events": [...]}` page wrapper; rows deliberately out of `created` order.
    page = {
        "events": [
            {
                "span_id": "t1",
                "root_span_id": "r3",
                "span_attributes": {"name": "ls", "type": "tool"},
                "input": {},
                "output": "a.txt\nb.txt",
                "created": "2026-01-01T00:00:05.000Z",
            },
            {
                "span_id": "g1",
                "root_span_id": "r3",
                "span_attributes": {"name": "agent", "type": "llm"},
                "input": [{"role": "user", "content": "list files"}],
                "output": {
                    "tool_calls": [{"id": "x", "function": {"name": "ls", "arguments": "{}"}}]
                },
                "created": "2026-01-01T00:00:04.000Z",
            },
        ]
    }
    path = tmp_path / "page.json"
    path.write_text(json.dumps(page), encoding="utf-8")

    traces = BraintrustAdapter().from_file(str(path))
    assert len(traces) == 1
    step = traces[0].steps[0]
    # Ordered by `created`: the llm row (04s) precedes the tool result (05s) and they pair.
    assert step.action.name == "ls"
    assert step.observation.content == "a.txt\nb.txt"


def test_jsonl_multiple_traces(tmp_path: Path) -> None:
    other = {
        "span_id": "z1",
        "root_span_id": "r4",
        "span_attributes": {"name": "agent", "type": "llm"},
        "input": [{"role": "user", "content": "ping"}],
        "output": {"content": "pong"},
        "created": "2026-01-01T00:00:01.000Z",
    }
    path = tmp_path / "rows.jsonl"
    # One JSON array line (the r1 trace) + one bare-row line (the r4 trace).
    path.write_text(json.dumps(_ROWS) + "\n" + json.dumps(other) + "\n", encoding="utf-8")

    traces = BraintrustAdapter().from_file(str(path))
    assert len(traces) == 2


def test_vendor_pull_without_key_is_friendly(monkeypatch) -> None:  # noqa: ANN001 - fixture
    monkeypatch.delenv("BRAINTRUST_API_KEY", raising=False)
    try:
        BraintrustAdapter().from_vendor(VendorPull(project="p"))
    except ValueError as exc:
        assert "API key" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


def test_vendor_pull_without_project_is_friendly() -> None:
    try:
        BraintrustAdapter().from_vendor(VendorPull(api_key="k"))
    except ValueError as exc:
        assert "project" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


def test_vendor_pull_resolves_project_and_fetches(monkeypatch) -> None:  # noqa: ANN001 - fixture
    """Live-pull path with httpx mocked: resolve project name -> id, fetch logs, normalize."""
    import wmh.ingest.braintrust as bt

    def fake_get(url, headers=None, params=None, timeout=None):  # noqa: ANN001, ANN202 - test stub
        class _Resp:
            def raise_for_status(self) -> None: ...

            def __init__(self, payload: dict) -> None:
                self._payload = payload

            def json(self) -> dict:
                return self._payload

        if url.endswith("/v1/project"):
            return _Resp({"objects": [{"id": "pid-1", "name": "Demo"}]})
        assert "/v1/project_logs/pid-1/fetch" in url  # resolved the name to the id
        assert headers == {"Authorization": "Bearer k"}
        return _Resp({"events": _ROWS})

    monkeypatch.setattr(bt.httpx, "get", fake_get)
    traces = BraintrustAdapter().from_vendor(VendorPull(api_key="k", project="Demo"))
    assert len(traces) == 1
    assert traces[0].steps[0].action.name == "get_weather"
