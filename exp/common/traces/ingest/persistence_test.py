"""Persist normalized tools, issues, source bytes and model identity without rebuilding."""

import hashlib
import json
from pathlib import Path

import pytest

from exp.common.core.artifacts import JsonObject
from exp.common.traces.ingest.persistence import ingest_traces, read_ingested_traces
from exp.common.traces.ingest.sources import load_trace_source


def _source(tmp_path: Path, count: int = 20) -> Path:
    """Write a multistep chat corpus with exact prompt and tool contracts plus an exclusion."""
    records: list[JsonObject] = [
        {
            "trace_id": f"research-{index}",
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "description": "Look up a company",
                        "parameters": {
                            "type": "object",
                            "properties": {"name": {"type": "string"}},
                        },
                    },
                }
            ],
            "messages": [
                {"role": "system", "content": "Research companies."},
                {"role": "developer", "content": "Cite sources."},
                {"role": "user", "content": f"Research company {index}"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call-a",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": '{"name":"Acme"}'},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call-a", "content": "Acme is a company."},
                {"role": "assistant", "content": "Company found."},
            ],
        }
        for index in range(count)
    ]
    path = tmp_path / "research.jsonl"
    path.write_text("\n".join(json.dumps(record) for record in records) + '\n{"broken":true}\n')
    return path


def test_complete_normalization_roundtrip_after_source_removed(tmp_path: Path) -> None:
    """Immutable imports keep prompts, tools, observations, identity evidence and exclusions."""
    root = tmp_path / "state"
    path = _source(tmp_path)
    original = load_trace_source("chat-json", path)
    summary, receipt = ingest_traces("powerset", root=root, source_format="chat-json", path=path)
    assert summary.trace_count == len(original.traces)
    assert summary.source == original.source and summary.issues == original.issues
    assert receipt is not None and receipt.trace_count == 20
    assert original.source is not None
    assert original.source.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert len(original.issues) == 1
    assert original.identity_evidence is not None
    path.unlink()
    restored = read_ingested_traces(root, receipt.import_id)
    assert restored == original
    assert all(trace.tools[0].name == "lookup" for trace in restored.traces)
    assert restored.traces[0].initial_context["instruction_messages"] == [
        {"role": "system", "content": "Research companies."},
        {"role": "developer", "content": "Cite sources."},
    ]
    assert "Acme is a company." in restored.traces[0].model_dump_json()
    assert not (root / "projects").exists()


@pytest.mark.parametrize("source_format", ["chat-json", "posthog", "otel-genai"])
def test_exclusion_only_import_retains_original_source_identity(
    tmp_path: Path, source_format: str
) -> None:
    """Two rejected files with the same error remain distinct source evidence."""
    path = tmp_path / "invalid.json"
    root = tmp_path / "state"
    imports: list[str] = []
    for value in (1, 2):
        path.write_text(json.dumps({"broken": value}))
        expected = load_trace_source(source_format, path)
        result, receipt = ingest_traces(
            "powerset", root=root, source_format=source_format, path=path
        )
        assert receipt is not None and result.trace_count == 0 and result.issues
        assert result.source is not None
        assert result.source.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
        assert read_ingested_traces(root, receipt.import_id) == expected
        imports.append(receipt.import_id)
    assert imports[0] != imports[1]
