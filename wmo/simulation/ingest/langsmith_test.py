"""Tests for LangSmith run export normalization."""

from __future__ import annotations

import json
from pathlib import Path

from wmo.simulation.ingest.langsmith import load_langsmith_file


def _runs() -> list[dict[str, object]]:
    """Return one LangSmith trace as an llm run, a tool run, and a chain run."""
    return [
        {
            "id": "run-1",
            "trace_id": "trace-1",
            "run_type": "llm",
            "name": "ChatOpenAI",
            "session_id": "session-4",
            "start_time": "2026-03-01T00:00:01Z",
            "end_time": "2026-03-01T00:00:02Z",
            "inputs": {"messages": [{"role": "user", "content": "Cancel my subscription"}]},
            "outputs": {
                "generations": [
                    [
                        {
                            "text": "",
                            "message": {
                                "kwargs": {
                                    "content": "",
                                    "tool_calls": [
                                        {
                                            "id": "call-9",
                                            "name": "cancel_subscription",
                                            "args": {"plan": "pro"},
                                        }
                                    ],
                                }
                            },
                        }
                    ]
                ],
                "llm_output": {"token_usage": {"prompt_tokens": 12, "completion_tokens": 3}},
            },
            "extra": {"metadata": {"ls_provider": "openai", "ls_model_name": "gpt-4o"}},
        },
        {
            "id": "run-2",
            "trace_id": "trace-1",
            "parent_run_id": "run-1",
            "run_type": "tool",
            "name": "cancel_subscription",
            "start_time": "2026-03-01T00:00:03Z",
            "end_time": "2026-03-01T00:00:04Z",
            "inputs": {"plan": "pro"},
            "outputs": {"output": "cancelled"},
        },
        {
            "id": "run-3",
            "trace_id": "trace-1",
            "run_type": "chain",
            "name": "AgentExecutor",
            "start_time": "2026-03-01T00:00:00Z",
            "inputs": {"input": "Cancel my subscription"},
        },
    ]


def test_load_langsmith_file_converts_llm_and_tool_runs(tmp_path: Path) -> None:
    """LangSmith llm and tool runs become paired canonical spans with retained identity."""
    path = tmp_path / "langsmith.json"
    path.write_text(json.dumps(_runs()), encoding="utf-8")

    result = load_langsmith_file(path)

    assert result.issues == ()
    trace = result.traces[0]
    assert trace.task == "Cancel my subscription"
    assert trace.conversation_id == "session-4"
    assert [span.name for span in trace.spans] == ["agent.model_call", "agent.tool_call"]
    call, tool_result = trace.spans
    assert call.attributes["gen_ai.tool.name"] == "cancel_subscription"
    assert call.attributes["gen_ai.tool.call.arguments"] == '{"plan":"pro"}'
    assert call.model is not None
    assert (call.model.provider, call.model.model_id) == ("openai", "gpt-4o")
    assert call.usage is not None
    assert (call.usage.input_tokens, call.usage.output_tokens) == (12, 3)
    assert tool_result.attributes["gen_ai.tool.call.id"] == "call-9"
    assert tool_result.attributes["gen_ai.tool.message"] == "cancelled"
    assert tool_result.parent_span_id == call.span_id


def test_load_langsmith_file_keeps_model_name_without_provider(tmp_path: Path) -> None:
    """A run naming only a model keeps it as evidence and resolves no model identity."""
    runs = _runs()
    runs[0]["extra"] = {"metadata": {"ls_model_name": "gpt-4o"}}
    path = tmp_path / "langsmith.jsonl"
    path.write_text("\n".join(json.dumps(run) for run in runs), encoding="utf-8")

    result = load_langsmith_file(path)

    span = result.traces[0].spans[0]
    assert span.model is None
    assert span.attributes["gen_ai.request.model"] == "gpt-4o"


def test_load_langsmith_file_retains_run_errors(tmp_path: Path) -> None:
    """A run error becomes a structured span failure and a failure trace outcome."""
    runs = _runs()
    runs[1]["error"] = "tool timed out"
    path = tmp_path / "langsmith.json"
    path.write_text(json.dumps({"runs": runs}), encoding="utf-8")

    result = load_langsmith_file(path)

    trace = result.traces[0]
    assert trace.spans[1].failure is not None
    assert trace.spans[1].failure.message == "tool timed out"
    assert trace.outcome is not None
    assert trace.outcome.status == "failure"


def test_load_langsmith_file_excludes_run_without_start_time(tmp_path: Path) -> None:
    """A run with no start time excludes its trace with an explicit issue."""
    runs = _runs()
    del runs[0]["start_time"]
    path = tmp_path / "langsmith.json"
    path.write_text(json.dumps(runs), encoding="utf-8")

    result = load_langsmith_file(path)

    assert [issue.source_record for issue in result.issues] == ["record-1", "trace-trace-1"]
    assert result.traces == ()


def test_load_langsmith_file_ignores_orchestration_only_traces(tmp_path: Path) -> None:
    """A trace of chain runs alone yields no canonical trace and no invented evidence."""
    path = tmp_path / "langsmith.json"
    path.write_text(json.dumps([_runs()[2]]), encoding="utf-8")

    result = load_langsmith_file(path)

    assert result.traces == ()
    assert result.issues == ()
