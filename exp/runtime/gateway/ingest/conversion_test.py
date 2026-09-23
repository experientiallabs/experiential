"""Durable local traffic supplies scoped canonical build evidence."""

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from exp.common.traces.ingest.sources import TraceSourceError
from exp.runtime.anthropic_protocol.requests import decode_messages
from exp.runtime.gateway.capture_context import capture_request_context
from exp.runtime.gateway.ingest.conversion import load_gateway_capture
from exp.runtime.gateway.local_capture_contracts import (
    CapturedExchange,
    CaptureProvenance,
    LocalCaptureScope,
)
from exp.runtime.openai_protocol.requests import decode_chat, decode_responses


def _experience(identity: str = "developer") -> CapturedExchange:
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
    return CapturedExchange(
        experience_id=f"experience-{identity}",
        response_id=f"response-{identity}",
        scope=LocalCaptureScope(user_id=identity, application_id="gateway"),
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
        provenance=CaptureProvenance(source_id="request", model_id="worker"),
    )


def _database(path: Path, experiences: tuple[CapturedExchange, ...]) -> None:
    """Persist exact native-compatible rows, then close before independent ingestion."""
    path.touch(mode=0o600)
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE gateway_captures (sequence INTEGER PRIMARY KEY, user_id TEXT, "
            "application_id TEXT, expires_at INTEGER, payload TEXT)"
        )
        for index, experience in enumerate(experiences, start=1):
            connection.execute(
                "INSERT INTO gateway_captures VALUES (?, ?, ?, ?, ?)",
                (
                    index,
                    experience.scope.user_id,
                    experience.scope.application_id,
                    9999999999,
                    experience.model_dump_json(),
                ),
            )


def test_gateway_capture_exports_only_current_turn_metrics(tmp_path: Path) -> None:
    """The public loader exposes measured timing, tokens and opaque Gemini evidence."""
    experience = _experience()
    metrics = {
        "started_at": 1_800_000_000,
        "first_token_at": 1_800_000_001,
        "terminal_at": 1_800_000_002,
        "duration_ms": 2000,
        "usage_complete": True,
        "usage": {
            "input_tokens": 80,
            "output_tokens": 9,
            "cached_input_tokens": None,
            "reasoning_tokens": 3,
        },
    }
    parts = [{"thoughtSignature": "opaque==", "text": "answer"}]
    request = dict(experience.request)
    request["exp_capture_output"] = {"metrics": metrics, "gemini_thought_parts": parts}
    experience = experience.model_copy(update={"request": request})
    path = tmp_path / "traffic.db"
    _database(path, (experience,))
    result = load_gateway_capture(path, identity_id="developer")
    assert not result.issues
    trace = result.traces[0]
    measured = [span for span in trace.spans if span.usage is not None]
    assert len(measured) == 1
    span = measured[0]
    assert span.attributes["gen_ai.completion"] == "Found record A"
    assert span.started_at.timestamp() == metrics["started_at"]
    assert (span.ended_at - span.started_at).total_seconds() == 2
    assert span.usage is not None and span.usage.input_tokens == 80
    assert span.usage.cached_input_tokens is None
    assert span.attributes["gen_ai.usage.reasoning_tokens"] == 3
    assert trace.initial_context["capture_output"] == request["exp_capture_output"]
    historical = [other for other in trace.spans if other.span_id != span.span_id]
    assert all(other.usage is None for other in historical)
    assert all(other.attributes["exp.source.time.synthetic"] for other in historical)


@pytest.mark.parametrize("winner", ["worker", "child-model", None])
@pytest.mark.parametrize("truncated", [True, False, None])
def test_gateway_capture_preserves_root_winner_and_nullable_omission(
    tmp_path: Path, winner: str | None, truncated: bool | None
) -> None:
    """Scoped persisted evidence never infers a winner or an absent truncation fact."""
    experience = _experience()
    output = {"canonical_model_id": winner, "gemini_thought_parts_truncated": truncated}
    request = {**experience.request, "exp_capture_output": output}
    path = tmp_path / "traffic.db"
    _database(path, (experience.model_copy(update={"request": request}),))
    result = load_gateway_capture(path, identity_id="developer")
    assert not result.issues and len(result.traces) == 1
    context = result.traces[0].initial_context
    assert context["model_id"] == "worker"
    assert context["canonical_model_id"] == winner
    assert context["capture_output"] == output
    assert load_gateway_capture(path, identity_id="another").traces == ()


@pytest.mark.parametrize(
    "output",
    [
        {"canonical_model_id": 1},
        {"canonical_model_id": " "},
        {"canonical_model_id": "x" * 513},
        {"gemini_thought_parts_truncated": 1},
        {"gemini_thought_parts_truncated": "false"},
    ],
)
def test_gateway_capture_rejects_untyped_winner_or_omission(
    tmp_path: Path, output: dict[str, str | int]
) -> None:
    """Malformed optional evidence excludes the record instead of coercing its facts."""
    experience = _experience()
    path = tmp_path / "traffic.db"
    _database(
        path,
        (
            experience.model_copy(
                update={"request": {**experience.request, "exp_capture_output": output}}
            ),
        ),
    )
    result = load_gateway_capture(path, identity_id="developer")
    assert not result.traces
    assert len(result.issues) == 1


def test_reopened_capture_preserves_tools_and_pairs_results_without_claiming_success(
    tmp_path: Path,
) -> None:
    """Identity isolation survives restart and tool context reaches canonical evidence."""
    path = tmp_path / "traffic.db"
    _database(path, (_experience(), _experience("other")))
    result = load_gateway_capture(path, identity_id="developer")
    assert not result.issues
    assert len(result.traces) == 1
    trace = result.traces[0]
    assert trace.task == "Find record A"
    assert trace.tools[0].name == "lookup"
    assert trace.tools[0].input_schema["properties"] == {"id": {"type": "string"}}
    assert trace.outcome is None
    assert trace.conversation_id is None
    assert trace.initial_context["identity_id"] == "developer"
    assert trace.initial_context["model_id"] == "worker"
    assert trace.initial_context["canonical_model_id"] is None
    assert trace.initial_context["capture_output"] is None
    assert any(span.attributes.get("gen_ai.tool.message") == "Record A" for span in trace.spans)
    assert load_gateway_capture(path, identity_id="absent").traces == ()


def test_missing_effective_context_is_excluded_not_guessed(tmp_path: Path) -> None:
    """Older or incomplete captures cannot masquerade as reproducible input."""
    path = tmp_path / "traffic.db"
    _database(path, (_experience().model_copy(update={"request": {}}),))
    result = load_gateway_capture(path, identity_id="developer")
    assert not result.traces
    assert len(result.issues) == 1


def test_same_episode_label_never_exposes_another_identity(tmp_path: Path) -> None:
    """Caller session labels cannot override the authenticated storage scope."""
    path = tmp_path / "traffic.db"
    _database(
        path,
        tuple(
            _experience(identity).model_copy(update={"episode_id": "same-harness-session"})
            for identity in ("developer", "other")
        ),
    )
    for identity in ("developer", "other"):
        result = load_gateway_capture(path, identity_id=identity)
        assert not result.issues
        assert len(result.traces) == 1
        assert result.traces[0].conversation_id == "same-harness-session"
        assert result.traces[0].initial_context["identity_id"] == identity
        assert result.traces[0].initial_context["response_id"] == f"response-{identity}"


def test_missing_capture_database_has_an_actionable_error(tmp_path: Path) -> None:
    """An absent capture database never becomes an empty successful import."""
    with pytest.raises(TraceSourceError, match="Collect fresh traffic"):
        load_gateway_capture(tmp_path / "absent.db", identity_id="developer")
    assert not (tmp_path / "absent.db").exists()


def test_reasoning_raw_arguments_and_environment_bytes_reach_semantic_spans(tmp_path: Path) -> None:
    """Full submitted history survives storage projection and canonical ingestion."""
    environment = "  ENV\r\nalpha\0beta\t雪✓\r\nEND  \n"
    arguments = '{ "id" : "A" }'
    request = decode_chat(
        {
            "model": "coding",
            "messages": [
                {"role": "system", "content": "exact system\n"},
                {"role": "user", "content": "Find record A"},
                {
                    "role": "assistant",
                    "content": None,
                    "reasoning_content": "first reasoning\n",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": arguments},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call-1", "content": environment},
                {"role": "user", "content": "Now summarize"},
            ],
        }
    ).request
    experience = _experience().model_copy(
        update={
            "request": {
                "exp_context": capture_request_context(request),
                "exp_capture_output": {"provider_reasoning": "final reasoning\n"},
            }
        }
    )
    path = tmp_path / "traffic.db"
    _database(path, (experience,))
    result = load_gateway_capture(path, identity_id="developer")
    assert not result.issues
    trace = result.traces[0]
    assert any(span.attributes.get("gen_ai.tool.message") == environment for span in trace.spans)
    outputs = [span.attributes.get("gen_ai.output.messages") for span in trace.spans]
    assert any(
        isinstance(messages, list)
        and any(
            isinstance(message, dict) and message.get("reasoning_content") == "first reasoning\n"
            for message in messages
        )
        for messages in outputs
    )
    assert any(
        isinstance(messages, list)
        and any(
            isinstance(message, dict) and message.get("reasoning_content") == "final reasoning\n"
            for message in messages
        )
        for messages in outputs
    )
    assert any(
        span.attributes.get("gen_ai.tool.call.arguments") == arguments for span in trace.spans
    )


def test_messages_tool_error_and_thinking_are_preserved(tmp_path: Path) -> None:
    """Messages tool failures and signed thinking remain evidence, not success claims."""
    request = decode_messages(
        {
            "model": "coding",
            "max_tokens": 128,
            "messages": [
                {"role": "user", "content": "Find record A"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": "call-1", "name": "lookup", "input": {"id": "A"}}
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "call-1",
                            "is_error": True,
                            "content": "  tool failed\n",
                        }
                    ],
                },
            ],
        }
    ).request
    experience = _experience().model_copy(
        update={
            "protocol": "messages",
            "request": {"exp_context": capture_request_context(request)},
            "response": {
                "id": "msg",
                "type": "message",
                "role": "assistant",
                "stop_reason": "end_turn",
                "content": [
                    {"type": "thinking", "thinking": "exact thinking\n", "signature": "signed"},
                    {"type": "text", "text": "The lookup failed"},
                ],
            },
        }
    )
    path = tmp_path / "traffic.db"
    _database(path, (experience,))
    result = load_gateway_capture(path, identity_id="developer")
    assert not result.issues
    trace = result.traces[0]
    errors = [span for span in trace.spans if span.attributes.get("gen_ai.tool.is_error")]
    assert errors and errors[0].attributes["gen_ai.tool.message"] == "  tool failed\n"
    outputs = [span.attributes.get("gen_ai.output.messages") for span in trace.spans]
    assert "exact thinking\\n" in str(outputs) and "signed" in str(outputs)
    assert trace.outcome is None


@pytest.mark.parametrize("new_user_turn", [False, True])
def test_messages_visible_reasoning_with_sealed_replay_survives_standalone_ingestion(
    tmp_path: Path, new_user_turn: bool
) -> None:
    """A single retained exchange includes visible reasoning from its full submitted history."""
    visible = "Observed reasoning α\nexact\x00text\t"
    carrier = "x-experiential-hunyuan-reasoning-v1:ZGVwbG95bWVudC0x:c2VhbGVkLWVudmVsb3Bl"
    request = decode_messages(
        {
            "model": "coding",
            "max_tokens": 128,
            "messages": [
                {"role": "user", "content": "Find record A"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": visible, "signature": ""},
                        {
                            "type": "tool_use",
                            "id": "call-1",
                            "name": "lookup",
                            "input": {"id": "A"},
                        },
                        {"type": "redacted_thinking", "data": carrier},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "call-1", "content": "Record A"}
                    ],
                },
                *([{"role": "user", "content": "Now summarize"}] if new_user_turn else []),
            ],
        }
    ).request
    experience = _experience().model_copy(
        update={
            "protocol": "messages",
            "request": {"exp_context": capture_request_context(request)},
            "response": {
                "id": "msg-final",
                "type": "message",
                "role": "assistant",
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": "Found record A"}],
            },
        }
    )
    path = tmp_path / "traffic.db"
    _database(path, (experience,))
    result = load_gateway_capture(path, identity_id="developer")
    assert not result.issues
    assert len(result.traces) == 1
    outputs = [span.attributes.get("gen_ai.output.messages") for span in result.traces[0].spans]
    assert any(
        isinstance(messages, list)
        and any(
            isinstance(message, dict) and message.get("reasoning_content") == visible
            for message in messages
        )
        for messages in outputs
    )
    assert any(
        span.attributes.get("gen_ai.tool.message") == "Record A" for span in result.traces[0].spans
    )


def test_explicit_response_lineage_restores_reasoning_without_prefix_joining(
    tmp_path: Path,
) -> None:
    """A parent link supplies plaintext evidence without decrypting or guessing a session."""
    call = {
        "type": "function_call",
        "id": "fc1",
        "call_id": "call-1",
        "name": "lookup",
        "arguments": '{ "id" : "A" }',
    }
    parent_request = decode_responses({"model": "coding", "input": "Find record A"}).request
    parent = _experience().model_copy(
        update={
            "protocol": "responses",
            "response_id": "parent",
            "request": {
                "exp_context": capture_request_context(parent_request),
                "exp_capture_output": {"provider_reasoning": "observed parent reasoning"},
            },
            "response": {
                "id": "parent",
                "status": "completed",
                "output": [
                    {"type": "reasoning", "id": "rs_empty", "summary": [], "status": "completed"},
                    call,
                ],
            },
        }
    )
    continued_request = decode_responses(
        {
            "model": "coding",
            "input": [
                {"role": "user", "content": "Find record A"},
                call,
                {"type": "function_call_output", "call_id": "call-1", "output": "Record A"},
            ],
        }
    ).request
    child = parent.model_copy(
        update={
            "experience_id": "child",
            "response_id": "child",
            "parent_response_id": "parent",
            "request": {"exp_context": capture_request_context(continued_request)},
            "response": {
                "id": "child",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "id": "msg",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Found", "annotations": []}],
                    }
                ],
            },
        }
    )
    path = tmp_path / "traffic.db"
    _database(path, (parent, child))
    result = load_gateway_capture(path, identity_id="developer")
    assert not result.issues
    assert {trace.conversation_id for trace in result.traces} == {"response:parent"}
    trace = next(
        trace for trace in result.traces if trace.initial_context["response_id"] == "child"
    )
    assert "observed parent reasoning" in str(
        [span.attributes.get("gen_ai.output.messages") for span in trace.spans]
    )
    missing = tmp_path / "missing.db"
    _database(missing, (child,))
    incomplete = load_gateway_capture(missing, identity_id="developer")
    assert incomplete.traces[0].initial_context["missing_parent_response_id"] == "parent"


@pytest.mark.parametrize("text_first", [False, True])
@pytest.mark.parametrize("changed", [False, True])
@pytest.mark.parametrize("leading_text", [False, True])
def test_linked_reasoning_survives_coalesced_text_and_tools(
    tmp_path: Path, text_first: bool, changed: bool, leading_text: bool
) -> None:
    """Retained assistant segments keep parent reasoning unless visible content changed."""
    arguments = '{ "id" : "A" }'
    call = {
        "type": "function_call",
        "id": "fc1",
        "call_id": "call-1",
        "name": "lookup",
        "arguments": arguments,
    }
    text = {
        "type": "message",
        "id": "msg1",
        "role": "assistant",
        "content": [{"type": "output_text", "text": "\n\n", "annotations": []}],
    }
    output = [text, call] if text_first else [call, text]
    if leading_text:
        # Two message items delimit separate retained segments. Put the call
        # after the second text so the first message remains a distinct turn.
        output = [
            dict(text, id="prefix", content=[{"type": "output_text", "text": "First"}]),
            text,
            call,
        ]
    parent_request = decode_responses({"model": "coding", "input": "Find record A"}).request
    parent = _experience().model_copy(
        update={
            "protocol": "responses",
            "response_id": "parent",
            "request": {
                "exp_context": capture_request_context(parent_request),
                "exp_capture_output": {"provider_reasoning": "observed parent reasoning"},
            },
            "response": {
                "id": "parent",
                "status": "completed",
                "output": output,
            },
        }
    )
    continued_request = decode_chat(
        {
            "model": "coding",
            "messages": [
                {"role": "user", "content": "Find record A"},
                *([{"role": "assistant", "content": "First"}] if leading_text else []),
                {
                    "role": "assistant",
                    "content": "changed by guardrail" if changed else "\n\n",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": arguments},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call-1", "content": "Record A"},
            ],
        }
    ).request
    child = parent.model_copy(
        update={
            "experience_id": "child",
            "response_id": "child",
            "parent_response_id": "parent",
            "request": {"exp_context": capture_request_context(continued_request)},
            "response": {
                "id": "child",
                "status": "completed",
                "output": [dict(text, id="child-msg")],
            },
        }
    )
    path = tmp_path / "traffic.db"
    _database(path, (parent, child))
    result = load_gateway_capture(path, identity_id="developer")
    assert not result.issues
    trace = next(t for t in result.traces if t.initial_context["response_id"] == "child")
    outputs = [span.attributes.get("gen_ai.output.messages") for span in trace.spans]
    assert ("observed parent reasoning" in str(outputs)) is not changed


def test_messages_output_keeps_exact_provider_argument_text(tmp_path: Path) -> None:
    """Messages JSON objects do not erase the source provider's raw argument text."""
    request = decode_messages(
        {
            "model": "coding",
            "max_tokens": 128,
            "messages": [{"role": "user", "content": "Find record A"}],
        }
    ).request
    arguments = '{  "id" : "A"  }'
    experience = _experience().model_copy(
        update={
            "protocol": "messages",
            "request": {
                "exp_context": capture_request_context(request),
                "exp_capture_output": {
                    "provider_tool_calls_json": json.dumps(
                        [{"call_id": "call-1", "name": "lookup", "raw_arguments": arguments}]
                    )
                },
            },
            "response": {
                "id": "msg",
                "type": "message",
                "role": "assistant",
                "stop_reason": "tool_use",
                "content": [
                    {"type": "tool_use", "id": "call-1", "name": "lookup", "input": {"id": "A"}}
                ],
            },
        }
    )
    path = tmp_path / "traffic.db"
    _database(path, (experience,))
    result = load_gateway_capture(path, identity_id="developer")
    assert not result.issues
    assert any(
        span.attributes.get("gen_ai.tool.call.arguments") == arguments
        for span in result.traces[0].spans
    )
