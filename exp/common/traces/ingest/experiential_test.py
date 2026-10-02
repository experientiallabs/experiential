"""Native chat-capture ingestion and prompt/tool fidelity."""

import json
from pathlib import Path

from exp.common.traces.ingest.sources import load_trace_source


def test_capture_preserves_instructions_tools_and_unnamed_tool_results(tmp_path: Path) -> None:
    """Standard chat tool results retain their call IDs without requiring a nonstandard name."""
    record = {
        "request": {
            "request_id": "capture-a",
            "protocol": "chat_completions",
            "context": {
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "description": "Lookup",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
                "messages": [
                    {"role": "developer", "content": "Cite sources."},
                    {"role": "user", "content": "Research Acme"},
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "a",
                                "type": "function",
                                "function": {"name": "lookup", "arguments": "{}"},
                            }
                        ],
                    },
                    {"role": "tool", "tool_call_id": "a", "content": "Acme is a company."},
                ],
            },
        },
        "response": {
            "kind": "json",
            "status": 200,
            "body": {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "Company found."},
                    }
                ]
            },
        },
    }
    path = tmp_path / "capture.jsonl"
    path.write_text(json.dumps(record) + "\n")
    loaded = load_trace_source("experiential", path)
    assert not loaded.issues
    trace = loaded.traces[0]
    assert trace.tools[0].name == "lookup"
    assert trace.initial_context["instruction_messages"] == [
        {"role": "developer", "content": "Cite sources."}
    ]
    record["response"]["kind"] = "sse"
    path.write_text(json.dumps(record) + "\n")
    assert load_trace_source("experiential", path).issues
