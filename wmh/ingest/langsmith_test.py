"""Tests for the LangSmith run-tree -> Trace adapter (langsmith).

Fixture-based, no network: a hand-authored set of LangSmith runs sharing one `trace_id` — an `llm`
run whose `outputs` carry a tool call (in the LCEL `generations[].message.kwargs.tool_calls`
location), a `tool` run carrying the result, and an `llm` run that errored. Asserts the action
name/args, the observation content, the error flag, and that everything groups into ONE trace.
"""

from __future__ import annotations

import json
from pathlib import Path

from wmh.core.types import ActionKind
from wmh.ingest import get_adapter
from wmh.ingest.langsmith import LangSmithAdapter

# A realistic `Client.list_runs` / `GET /api/v1/runs` export: a flat list of runs in one trace.
_RUNS = [
    {
        "id": "11111111-1111-1111-1111-111111111111",
        "trace_id": "tttttttt-tttt-tttt-tttt-tttttttttttt",
        "parent_run_id": None,
        "run_type": "chain",
        "name": "AgentExecutor",
        "inputs": {"input": "what's the weather in Paris?"},
        "outputs": {"output": "It's 18C and sunny in Paris."},
        "start_time": "2026-01-01T00:00:00.000000",
        "end_time": "2026-01-01T00:00:05.000000",
        "error": None,
    },
    {
        "id": "22222222-2222-2222-2222-222222222222",
        "trace_id": "tttttttt-tttt-tttt-tttt-tttttttttttt",
        "parent_run_id": "11111111-1111-1111-1111-111111111111",
        "run_type": "llm",
        "name": "ChatOpenAI",
        "inputs": {
            "messages": [
                [
                    {
                        "id": ["langchain", "schema", "HumanMessage"],
                        "kwargs": {"content": "what's the weather in Paris?"},
                    },
                ]
            ]
        },
        "outputs": {
            "generations": [
                {
                    "text": "",
                    "message": {
                        "kwargs": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "name": "get_weather",
                                    "args": {"city": "Paris"},
                                }
                            ],
                        }
                    },
                }
            ]
        },
        "start_time": "2026-01-01T00:00:01.000000",
        "end_time": "2026-01-01T00:00:02.000000",
        "error": None,
    },
    {
        "id": "33333333-3333-3333-3333-333333333333",
        "trace_id": "tttttttt-tttt-tttt-tttt-tttttttttttt",
        "parent_run_id": "11111111-1111-1111-1111-111111111111",
        "run_type": "tool",
        "name": "get_weather",
        "inputs": {"city": "Paris"},
        "outputs": {"output": "18C and sunny"},
        "start_time": "2026-01-01T00:00:03.000000",
        "end_time": "2026-01-01T00:00:03.500000",
        "error": None,
    },
    {
        "id": "44444444-4444-4444-4444-444444444444",
        "trace_id": "tttttttt-tttt-tttt-tttt-tttttttttttt",
        "parent_run_id": "11111111-1111-1111-1111-111111111111",
        "run_type": "tool",
        "name": "get_forecast",
        "inputs": {"city": "Paris"},
        "outputs": {"output": "forecast service unavailable"},
        "start_time": "2026-01-01T00:00:04.000000",
        "end_time": "2026-01-01T00:00:04.500000",
        "error": "ToolException: service unavailable",
    },
]


def test_langsmith_adapter_is_registered() -> None:
    assert get_adapter("langsmith").name == "langsmith"


def test_llm_tool_call_pairs_with_tool_run(tmp_path: Path) -> None:
    path = tmp_path / "runs.json"
    path.write_text(json.dumps(_RUNS), encoding="utf-8")

    traces = LangSmithAdapter().from_file(str(path))

    # All four runs share one trace_id -> ONE trace.
    assert len(traces) == 1
    trace = traces[0]
    assert trace.trace_id == "tttttttt-tttt-tttt-tttt-tttttttttttt"

    # The llm tool call pairs with the tool run's result.
    call = trace.steps[0]
    assert call.action.kind == ActionKind.TOOL_CALL
    assert call.action.name == "get_weather"
    assert call.action.arguments == {"city": "Paris"}
    assert call.observation.content == "18C and sunny"
    assert call.observation.is_error is False
    # The first human input becomes the task.
    assert call.task == "what's the weather in Paris?"

    # The errored tool run becomes a step whose observation is flagged as an error.
    err = trace.steps[-1]
    assert err.action.name == "get_forecast"
    assert err.observation.content == "forecast service unavailable"
    assert err.observation.is_error is True


def test_openai_shaped_tool_call_and_runs_wrapper(tmp_path: Path) -> None:
    # The {"runs": [...]} wrapper + the OpenAI tool-call shape (function.name/arguments str)
    # under additional_kwargs.
    payload = {
        "runs": [
            {
                "id": "a",
                "trace_id": "tr-openai",
                "run_type": "llm",
                "inputs": {"messages": [{"role": "user", "content": "list files"}]},
                "outputs": {
                    "generations": [
                        {
                            "message": {
                                "kwargs": {
                                    "additional_kwargs": {
                                        "tool_calls": [
                                            {
                                                "id": "c1",
                                                "function": {"name": "ls", "arguments": "{}"},
                                            }
                                        ]
                                    }
                                }
                            }
                        }
                    ]
                },
                "start_time": "2026-01-01T00:00:01.000000",
                "error": None,
            },
            {
                "id": "b",
                "trace_id": "tr-openai",
                "run_type": "tool",
                "name": "ls",
                "outputs": {"output": "a.txt\nb.txt"},
                "start_time": "2026-01-01T00:00:02.000000",
                "error": None,
            },
        ]
    }
    path = tmp_path / "wrap.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    traces = LangSmithAdapter().from_file(str(path))
    assert len(traces) == 1
    step = traces[0].steps[0]
    assert step.action.name == "ls"
    assert step.action.arguments == {}
    assert step.observation.content == "a.txt\nb.txt"
    assert step.task == "list files"


def test_plain_llm_run_becomes_message_step(tmp_path: Path) -> None:
    run = {
        "id": "solo",
        "trace_id": "tr-msg",
        "run_type": "llm",
        "inputs": {"messages": [{"role": "user", "content": "say hi"}]},
        "outputs": {"generations": [{"text": "hello there"}]},
        "start_time": "2026-01-01T00:00:01.000000",
        "error": None,
    }
    path = tmp_path / "msg.json"
    path.write_text(json.dumps(run), encoding="utf-8")

    step = LangSmithAdapter().from_file(str(path))[0].steps[0]
    assert step.action.kind == ActionKind.MESSAGE
    assert step.action.content is not None
    assert "hello there" in step.action.content
    assert step.observation.content == ""


def test_jsonl_and_chain_runs_skipped(tmp_path: Path) -> None:
    # JSONL (one run per line); the lone chain run produces no actionable step.
    chain = {
        "id": "c",
        "trace_id": "tr-chain",
        "run_type": "chain",
        "inputs": {"input": "noop"},
        "outputs": {"output": "noop"},
        "start_time": "2026-01-01T00:00:01.000000",
        "error": None,
    }
    llm = {
        "id": "l",
        "trace_id": "tr-llm",
        "run_type": "llm",
        "inputs": {"messages": [{"role": "user", "content": "ping"}]},
        "outputs": {"generations": [{"text": "pong"}]},
        "start_time": "2026-01-01T00:00:01.000000",
        "error": None,
    }
    path = tmp_path / "runs.jsonl"
    path.write_text(json.dumps(chain) + "\n" + json.dumps(llm) + "\n", encoding="utf-8")

    traces = LangSmithAdapter().from_file(str(path))
    by_id = {t.trace_id: t for t in traces}
    # The chain-only trace yields no actionable spans, so it produces no Trace at all; the llm
    # trace has one message step.
    assert "tr-chain" not in by_id
    assert by_id["tr-llm"].steps[0].action.content is not None


def test_vendor_pull_unsupported_is_friendly() -> None:
    from wmh.ingest.adapter import VendorPull

    try:
        LangSmithAdapter().from_vendor(VendorPull())
    except ValueError as exc:
        assert "does not support live vendor pulls" in str(exc)
        assert "langsmith" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


def test_empty_string_error_is_not_an_error(tmp_path: Path) -> None:
    """Some LangSmith dumps set error="" on a successful run; it must NOT mark the step errored."""
    run = {
        "id": "r1",
        "trace_id": "t-empty-err",
        "run_type": "tool",
        "name": "lookup",
        "outputs": {"output": "ok"},
        "error": "",  # empty string, not a real error
        "start_time": "2026-01-01T00:00:00",
    }
    path = tmp_path / "run.json"
    path.write_text(json.dumps(run), encoding="utf-8")

    trace = LangSmithAdapter().from_file(str(path))[0]
    assert trace.steps[0].observation.is_error is False
