"""Tests for Braintrust log export normalization."""

from __future__ import annotations

import json
from pathlib import Path

from wmo.simulation.ingest.braintrust import load_braintrust_file


def _rows() -> list[dict[str, object]]:
    """Return one Braintrust trace as an llm row, a tool row, and a task row."""
    return [
        {
            "id": "row-1",
            "span_id": "span-1",
            "root_span_id": "root-1",
            "span_attributes": {"type": "llm", "name": "chat"},
            "metrics": {
                "start": 1_772_000_000.0,
                "end": 1_772_000_001.0,
                "prompt_tokens": 30,
                "completion_tokens": 4,
            },
            "metadata": {"provider": "anthropic", "model": "claude-sonnet-4"},
            "input": [{"role": "user", "content": "Book a table for two"}],
            "output": {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "call-7",
                        "name": "book_table",
                        "input": {"seats": 2},
                    }
                ],
            },
        },
        {
            "id": "row-2",
            "span_id": "span-2",
            "root_span_id": "root-1",
            "span_parents": ["span-1"],
            "span_attributes": {"type": "tool", "name": "book_table"},
            "metrics": {"start": 1_772_000_002.0, "end": 1_772_000_003.0},
            "input": {"seats": 2},
            "output": "table booked",
        },
        {
            "id": "row-3",
            "span_id": "span-3",
            "root_span_id": "root-1",
            "span_attributes": {"type": "task", "name": "agent"},
            "metrics": {"start": 1_772_000_000.0},
            "input": [{"role": "user", "content": "Book a table for two"}],
        },
    ]


def test_load_braintrust_file_converts_llm_and_tool_rows(tmp_path: Path) -> None:
    """Braintrust llm and tool rows become paired canonical spans."""
    path = tmp_path / "braintrust.json"
    path.write_text(json.dumps(_rows()), encoding="utf-8")

    result = load_braintrust_file(path)

    assert result.issues == ()
    trace = result.traces[0]
    assert trace.task == "Book a table for two"
    assert [span.name for span in trace.spans] == ["agent.model_call", "agent.tool_call"]
    call, tool_result = trace.spans
    assert call.attributes["gen_ai.tool.name"] == "book_table"
    assert call.attributes["gen_ai.tool.call.id"] == "call-7"
    assert call.model is not None
    assert (call.model.provider, call.model.model_id) == ("anthropic", "claude-sonnet-4")
    assert call.usage is not None
    assert (call.usage.input_tokens, call.usage.output_tokens) == (30, 4)
    assert tool_result.attributes["gen_ai.tool.call.id"] == "call-7"
    assert tool_result.attributes["gen_ai.tool.message"] == "table booked"
    assert tool_result.parent_span_id == call.span_id


def test_load_braintrust_file_accepts_jsonl_envelope_rows(tmp_path: Path) -> None:
    """Rows wrapped in an events envelope per JSONL line normalize identically."""
    path = tmp_path / "braintrust.jsonl"
    lines = [json.dumps({"events": [row]}) for row in _rows()]
    path.write_text("\n".join(lines), encoding="utf-8")

    result = load_braintrust_file(path)

    assert len(result.traces) == 1
    assert len(result.traces[0].spans) == 2


def test_load_braintrust_file_retains_row_errors(tmp_path: Path) -> None:
    """A row error becomes a structured span failure and a failure trace outcome."""
    rows = _rows()
    rows[0]["error"] = {"message": "overloaded"}
    path = tmp_path / "braintrust.json"
    path.write_text(json.dumps(rows), encoding="utf-8")

    result = load_braintrust_file(path)

    trace = result.traces[0]
    assert trace.spans[0].failure is not None
    assert trace.outcome is not None
    assert trace.outcome.status == "failure"


def test_load_braintrust_file_excludes_row_without_timing(tmp_path: Path) -> None:
    """A row with no start metric is excluded with an explicit issue."""
    rows = _rows()
    rows[0]["metrics"] = {"prompt_tokens": 1, "completion_tokens": 1}
    path = tmp_path / "braintrust.json"
    path.write_text(json.dumps(rows), encoding="utf-8")

    result = load_braintrust_file(path)

    assert [issue.source_record for issue in result.issues] == ["record-1", "trace-root-1"]
    assert result.traces == ()


def test_load_braintrust_file_keeps_model_name_without_provider(tmp_path: Path) -> None:
    """A row naming only a model keeps it as evidence and resolves no model identity."""
    rows = _rows()
    rows[0]["metadata"] = {"model": "claude-sonnet-4"}
    path = tmp_path / "braintrust.json"
    path.write_text(json.dumps(rows), encoding="utf-8")

    result = load_braintrust_file(path)

    span = result.traces[0].spans[0]
    assert span.model is None
    assert span.attributes["gen_ai.request.model"] == "claude-sonnet-4"
