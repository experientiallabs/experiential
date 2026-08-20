"""Tests for Braintrust log export normalization."""

from __future__ import annotations

import json
from pathlib import Path

from exp.simulation.ingest.braintrust import BRAINTRUST_SOURCE


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


def test_load_braintrust_file_accepts_jsonl_envelope_rows(tmp_path: Path) -> None:
    """Rows wrapped in an events envelope per JSONL line normalize identically."""
    path = tmp_path / "braintrust.jsonl"
    lines = [json.dumps({"events": [row]}) for row in _rows()]
    path.write_text("\n".join(lines), encoding="utf-8")

    result = BRAINTRUST_SOURCE.load(path)

    assert len(result.traces) == 1
    assert len(result.traces[0].spans) == 2
