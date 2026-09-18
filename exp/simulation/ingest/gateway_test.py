"""Durable local traffic supplies scoped canonical build evidence."""

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from exp.common.claas import ClaasScope, Experience, ExperienceProvenance
from exp.runtime.gateway.capture_context import capture_request_context
from exp.runtime.openai_protocol.requests import decode_chat
from exp.simulation.ingest.gateway import load_gateway_capture
from exp.simulation.ingest.sources import TraceSourceError, load_trace_source


def _experience(identity: str = "developer") -> Experience:
    """Build one tool-using exchange with explicit matching call/result IDs."""
    request = decode_chat(
        {
            "model": "coding",
            "messages": [
                {"role": "user", "content": "Find record A"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": '{"id":"A"}'},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call-1", "content": "Record A"},
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "description": "Lookup the record",
                        "parameters": {"type": "object", "properties": {"id": {"type": "string"}}},
                    },
                }
            ],
        }
    ).request
    return Experience(
        experience_id=f"experience-{identity}",
        response_id=f"response-{identity}",
        scope=ClaasScope(user_id=identity, application_id="gateway"),
        protocol="chat_completions",
        captured_at=datetime.now(UTC),
        request={"exp_context": capture_request_context(request)},
        response={
            "choices": [
                {
                    "message": {"role": "assistant", "content": "Found record A"},
                    "finish_reason": "stop",
                }
            ]
        },
        provenance=ExperienceProvenance(
            source_kind="traffic", source_id="request", model_id="worker"
        ),
    )


def _database(path: Path, experiences: tuple[Experience, ...]) -> None:
    """Persist exact native-compatible rows, then close before independent ingestion."""
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE claas_experiences (sequence INTEGER PRIMARY KEY, user_id TEXT, "
            "application_id TEXT, expires_at INTEGER, payload TEXT)"
        )
        for index, experience in enumerate(experiences, start=1):
            connection.execute(
                "INSERT INTO claas_experiences VALUES (?, ?, ?, ?, ?)",
                (
                    index,
                    experience.scope.user_id,
                    experience.scope.application_id,
                    9999999999,
                    experience.model_dump_json(),
                ),
            )


def test_reopened_capture_preserves_tools_and_pairs_results_without_claiming_success(
    tmp_path: Path,
) -> None:
    """Identity isolation survives restart and tool context reaches canonical evidence."""
    path = tmp_path / "traffic.db"
    _database(path, (_experience(), _experience("other")))
    result = load_trace_source("gateway", path, identity_id="developer")
    assert not result.issues
    assert len(result.traces) == 1
    trace = result.traces[0]
    assert trace.task == "Find record A"
    assert trace.tools[0].name == "lookup"
    assert trace.tools[0].input_schema["properties"] == {"id": {"type": "string"}}
    assert trace.outcome is None
    assert trace.conversation_id is None
    assert trace.initial_context["identity_id"] == "developer"
    assert any(span.attributes.get("gen_ai.tool.message") == "Record A" for span in trace.spans)
    assert load_gateway_capture(path, identity_id="absent").traces == ()


def test_missing_effective_context_is_excluded_not_guessed(tmp_path: Path) -> None:
    """Older or incomplete captures cannot masquerade as reproducible input."""
    path = tmp_path / "traffic.db"
    _database(path, (_experience().model_copy(update={"request": {}}),))
    result = load_gateway_capture(path, identity_id="developer")
    assert not result.traces
    assert len(result.issues) == 1


def test_identity_is_required_and_not_accepted_for_unscoped_sources(tmp_path: Path) -> None:
    """An omitted scope never means all local identities."""
    with pytest.raises(TraceSourceError, match="explicit --identity"):
        load_trace_source("gateway", tmp_path / "unused")
    with pytest.raises(TraceSourceError, match="only with"):
        load_trace_source("chat-json", tmp_path / "unused", identity_id="developer")
