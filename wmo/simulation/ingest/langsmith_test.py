"""Tests for LangSmith run export normalization."""

from __future__ import annotations

import json
from pathlib import Path

from wmo.simulation.ingest.langsmith import LANGSMITH_SOURCE


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


def test_load_langsmith_file_ignores_orchestration_only_traces(tmp_path: Path) -> None:
    """A trace of chain runs alone yields no canonical trace and no invented evidence."""
    path = tmp_path / "langsmith.json"
    path.write_text(json.dumps([_runs()[2]]), encoding="utf-8")

    result = LANGSMITH_SOURCE.load(path)

    assert result.traces == ()
    assert result.issues == ()
