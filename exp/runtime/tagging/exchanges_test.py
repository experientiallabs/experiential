"""Tests for building tagging exchanges from local captures and capture record parts.

Requests go through the real protocol decoders and ``capture_context_document``, so the
contexts match what the gateway stores. Response frames follow the native collector tests:
Chat chunks ending in ``[DONE]``, Messages block lifecycles, and the Responses terminal event.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import JsonValue

from exp.common.core.artifacts import JsonObject
from exp.common.tags import opening_key, tag_facts, turn_windows
from exp.common.tags.exchange import TagExchange, TagMessage
from exp.runtime.anthropic_protocol.requests import decode_messages
from exp.runtime.gateway.capture_context import capture_context_document
from exp.runtime.gateway.contracts import GatewayRequest
from exp.runtime.gateway.ingest.conversion import CaptureProtocol
from exp.runtime.gateway.local_capture_contracts import (
    CapturedExchange,
    CaptureProvenance,
    LocalCaptureScope,
)
from exp.runtime.gateway.native_capture import CaptureRecord
from exp.runtime.openai_protocol.requests import decode_chat, decode_responses
from exp.runtime.tagging.exchanges import (
    TagExchangeError,
    completed_response,
    exchange_from_capture_parts,
    exchange_from_capture_record,
    exchange_from_captured,
)
from exp.runtime.tagging.responses_test import CHAT_BODY, CHAT_FRAMES, MESSAGES_FRAMES

LOOKUP_TOOL: JsonObject = {
    "type": "function",
    "function": {
        "name": "lookup",
        "description": "Lookup the record",
        "parameters": {"type": "object", "properties": {"id": {"type": "string"}}},
    },
}


def _chat_request(**extra: JsonValue) -> GatewayRequest:
    """Decode a tool-using Chat request with an image.

    Args:
        **extra: Additional Chat body fields.

    Returns:
        The effective gateway request.
    """
    return decode_chat(
        {
            "model": "coding",
            "messages": [
                {"role": "system", "content": "You look up records."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Find record A"},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                    ],
                },
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
                {"role": "tool", "tool_call_id": "call-1", "content": "not found"},
            ],
            "tools": [LOOKUP_TOOL],
            **extra,
        }
    ).request


def _experience(
    protocol: CaptureProtocol,
    request: GatewayRequest,
    response: JsonObject,
    *,
    session_id: str | None = None,
) -> CapturedExchange:
    """Build a local capture the way the gateway capture store writes it.

    Args:
        protocol: Public surface.
        request: Effective request.
        response: Complete response body.
        session_id: Optional caller session label.

    Returns:
        The local capture.
    """
    return CapturedExchange(
        experience_id="experience-1",
        response_id="response-1",
        episode_id=session_id,
        scope=LocalCaptureScope(user_id="developer", application_id="gateway"),
        protocol=protocol,
        captured_at=datetime.now(UTC),
        request={"exp_context": capture_context_document(request, session_id=session_id)},
        response=response,
        provenance=CaptureProvenance(source_id="request-1", model_id="worker"),
    )


def _record(
    protocol: CaptureProtocol, request: GatewayRequest, response: JsonObject, **context: str
) -> CaptureRecord:
    """Build a complete native collector record.

    Args:
        protocol: Public surface.
        request: Effective request.
        response: Response document.
        **context: Capture context options such as ``session_id``.

    Returns:
        The validated record.
    """
    return CaptureRecord.model_validate(
        {
            "schema_version": 1,
            "request": {
                "request_id": "request-1",
                "scope": {
                    "organization_id": "organization",
                    "identity_id": "identity",
                    "application_id": "application",
                },
                "protocol": protocol,
                "model_id": "worker",
                "context": capture_context_document(request, **context),
            },
            "response": response,
            "deployment_id": None,
            "captured_at": 1.0,
            "metrics": None,
            "gemini_thought_parts": [],
            "gemini_thought_parts_source_json": None,
        }
    )


RESPONSES_BODY: JsonObject = {
    "id": "resp_1",
    "object": "response",
    "status": "incomplete",
    "incomplete_details": {"reason": "max_output_tokens"},
    "error": None,
    "output": [
        {"type": "reasoning", "id": "rs_1", "summary": []},
        {
            "type": "message",
            "id": "msg_1",
            "role": "assistant",
            "status": "incomplete",
            "content": [{"type": "output_text", "text": "Partial forecast", "annotations": []}],
        },
    ],
}


def _messages_request() -> GatewayRequest:
    """Decode a Messages request whose last tool result failed.

    Returns:
        The effective gateway request.
    """
    return decode_messages(
        {
            "model": "coding",
            "max_tokens": 256,
            "system": "You run commands.",
            "messages": [
                {"role": "user", "content": "List files"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_0",
                            "name": "bash",
                            "input": {"cmd": "ls x"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_0",
                            "content": "No such file",
                            "is_error": True,
                        }
                    ],
                },
            ],
            "tools": [{"name": "bash", "description": "Run", "input_schema": {"type": "object"}}],
        }
    ).request


def _responses_request() -> GatewayRequest:
    """Decode a Responses request with instructions and a JSON schema format.

    Returns:
        The effective gateway request.
    """
    return decode_responses(
        {
            "model": "coding",
            "instructions": "Answer as JSON.",
            "input": "Forecast for SF",
            "text": {"format": {"type": "json_schema", "name": "f", "schema": {"type": "object"}}},
        }
    ).request


def test_local_chat_capture_keeps_tools_images_and_tool_results() -> None:
    """A local Chat capture becomes chat-form messages with facts and rendered windows."""
    exchange = exchange_from_captured(
        _experience("chat_completions", _chat_request(), CHAT_BODY, session_id="session-7")
    )
    assert exchange.exchange_id == "experience-1"
    assert exchange.conversation_key == "session-7"
    assert [tool.name for tool in exchange.tools] == ["lookup"]
    assert [message.role for message in exchange.messages] == [
        "system",
        "user",
        "assistant",
        "tool",
    ]
    user = exchange.messages[1]
    assert user.text() == "Find record A [image]"
    assert "AAAA" not in exchange.model_dump_json()
    assert exchange.response[0].tool_calls[0].function.arguments == '{"id":"B"}'
    facts = tag_facts(exchange)
    assert (facts.tool_calls, facts.turns, facts.has_images) == (1, 1, True)
    assert not facts.truncated and not facts.structured_output
    (window,) = turn_windows(exchange)
    assert window.text.splitlines() == [
        "[task] Find record A [image]",
        "[tool_result lookup] not found",
        "[assistant] Record A is missing.",
        '[tool_call lookup] {"id":"B"}',
    ]


def test_local_messages_capture_marks_failed_tool_results_and_drops_thinking() -> None:
    """Anthropic tool errors keep their failure flag and thinking blocks are omitted."""
    body: JsonObject = {
        "id": "msg",
        "type": "message",
        "role": "assistant",
        "stop_reason": "max_tokens",
        "content": [
            {"type": "thinking", "thinking": "private", "signature": "sig"},
            {"type": "text", "text": "The path does not exist."},
        ],
    }
    exchange = exchange_from_captured(_experience("messages", _messages_request(), body))
    assert exchange.messages[-1].text() == "[tool error] No such file"
    assert "private" not in exchange.model_dump_json()
    assert exchange.response[0].text() == "The path does not exist."
    assert tag_facts(exchange).truncated


def test_local_responses_capture_reads_length_limit_and_structured_output() -> None:
    """An incomplete Response at its token limit is tagged and marked truncated."""
    exchange = exchange_from_captured(
        _experience("responses", _responses_request(), RESPONSES_BODY)
    )
    assert [message.role for message in exchange.messages] == ["developer", "user"]
    assert [message.text() for message in exchange.response] == ["Partial forecast"]
    facts = tag_facts(exchange)
    assert facts.truncated and facts.structured_output


def test_chat_refusal_text_is_kept_as_the_reply() -> None:
    """A Chat refusal with no content stays readable for the refusal tag."""
    body: JsonObject = {
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": None, "refusal": "I can't help."},
                "finish_reason": "stop",
            }
        ]
    }
    exchange = exchange_from_captured(_experience("chat_completions", _chat_request(), body))
    assert exchange.response[0].text() == "I can't help."


def test_json_object_output_counts_as_structured_output() -> None:
    """The provider-internal JSON mode flag is a structured output request."""
    request = _chat_request(response_format={"type": "json_object"})
    exchange = exchange_from_captured(_experience("chat_completions", request, CHAT_BODY))
    assert exchange.structured_output


def test_local_capture_without_context_is_rejected_with_guidance() -> None:
    """A capture missing its effective request says how to recover."""
    experience = _experience("chat_completions", _chat_request(), CHAT_BODY).model_copy(
        update={"request": {}}
    )
    with pytest.raises(TagExchangeError, match="experience-1 cannot be tagged.*capture enabled"):
        exchange_from_captured(experience)


@pytest.mark.parametrize(
    ("protocol", "request_builder", "frames", "expected"),
    [
        (
            "chat_completions",
            _chat_request,
            CHAT_FRAMES,
            ['[assistant] Record A is missing.\n[tool_call lookup] {"id":"B"}'],
        ),
        ("messages", _messages_request, MESSAGES_FRAMES, ['[tool_call bash] {"cmd":"ls"}']),
        (
            "responses",
            _responses_request,
            [
                {"type": "response.created", "response": {"id": "resp_1", "status": "in_progress"}},
                {"type": "response.incomplete", "response": RESPONSES_BODY},
            ],
            ["[assistant] Partial forecast"],
        ),
    ],
)
def test_streamed_records_reassemble_through_the_native_collector(
    protocol: CaptureProtocol,
    request_builder: Callable[[], GatewayRequest],
    frames: list[JsonValue],
    expected: list[str],
) -> None:
    """SSE frames from a record reassemble into the same response turns for every surface.

    Args:
        protocol: Public surface.
        request_builder: Builds the effective request.
        frames: Stored SSE data payloads.
        expected: Rendered response turns.
    """
    record = _record(
        protocol,
        request_builder(),
        {
            "kind": "sse",
            "status": 200,
            "frames": frames,
            "truncated": False,
            "client_disconnected": False,
        },
    )
    exchange = exchange_from_capture_record(record)
    window = turn_windows(exchange)[-1]
    assert window.text.endswith(expected[0])
    assert exchange.exchange_id == "request-1"


@pytest.mark.parametrize(
    ("protocol", "request_builder", "frames"),
    [
        ("chat_completions", _chat_request, CHAT_FRAMES),
        ("messages", _messages_request, MESSAGES_FRAMES),
    ],
    ids=["chat", "messages"],
)
def test_record_and_local_capture_of_the_same_traffic_agree(
    protocol: CaptureProtocol,
    request_builder: Callable[[], GatewayRequest],
    frames: list[JsonValue],
) -> None:
    """A collector record and the local store's copy of it yield the same exchange.

    The local store keeps the collector's reassembled body, so the local capture here is
    built from the same frames.

    Args:
        protocol: Public surface.
        request_builder: Builds the effective request.
        frames: Stored SSE data payloads.
    """
    request = request_builder()
    document: JsonObject = {
        "kind": "sse",
        "status": 200,
        "frames": frames,
        "truncated": False,
        "client_disconnected": False,
    }
    local = exchange_from_captured(
        _experience(
            protocol, request, completed_response(protocol, document), session_id="session-7"
        )
    )
    from_record = exchange_from_capture_record(
        _record(protocol, request, document, session_id="session-7")
    )
    assert from_record.model_copy(update={"exchange_id": local.exchange_id}) == local
    assert from_record.conversation_key == "session-7"
    assert opening_key(from_record) == opening_key(local)
    if protocol == "messages":
        assert from_record.messages[-1].text() == "[tool error] No such file"


def test_parts_without_an_envelope_still_tag_the_messages() -> None:
    """A store that kept only messages and the response still yields an exchange."""
    context = capture_context_document(_chat_request())
    effective = context["request"]
    assert isinstance(effective, dict)
    messages = effective["messages"]
    assert isinstance(messages, list)
    exchange = exchange_from_capture_parts(
        exchange_id="request-9",
        protocol="chat_completions",
        messages=[
            {"role": "system", "truncated": True, "content": "[3 messages not captured]"},
            *messages,
        ],
        request=None,
        response={"kind": "json", "status": 200, "body": CHAT_BODY, "received_at": 1.0},
        conversation_key="ledger-conversation",
    )
    assert exchange.tools == ()
    assert exchange.conversation_key == "ledger-conversation"
    assert exchange.messages[0].text() == "[3 messages not captured]"


def test_streamed_tool_input_with_a_wide_integer_decodes() -> None:
    """A Messages tool input beyond 64-bit range reassembles and decodes exactly."""
    frames = [dict(frame) if isinstance(frame, dict) else frame for frame in MESSAGES_FRAMES]
    frames[7] = {
        "type": "content_block_delta",
        "index": 1,
        "delta": {"type": "input_json_delta", "partial_json": "18446744073709551617}"},
    }
    body = completed_response(
        "messages",
        {
            "kind": "sse",
            "status": 200,
            "frames": frames,
            "truncated": False,
            "client_disconnected": False,
        },
    )
    content = body["content"]
    assert isinstance(content, list) and isinstance(content[1], dict)
    assert content[1]["input"] == {"cmd": 18446744073709551617}
    assert "capture_input_source_json" not in content[1]
    exchange = exchange_from_capture_parts(
        exchange_id="request-2",
        protocol="messages",
        messages=[{"role": "user", "content": "go"}],
        request=None,
        response={"kind": "json", "status": 200, "body": body},
    )
    assert exchange.response[0].tool_calls[0].function.arguments == '{"cmd":18446744073709551617}'


def test_responses_refusals_and_server_tools_stay_readable() -> None:
    """Refusal parts keep their text and server-side tool calls render as tool calls."""
    body: JsonObject = {
        "status": "completed",
        "output": [
            {
                "type": "web_search_call",
                "id": "ws_1",
                "status": "completed",
                "action": {"type": "search", "query": "forecast"},
            },
            {
                "type": "message",
                "id": "msg_1",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "refusal", "refusal": "I can't share that."}],
            },
        ],
    }
    exchange = exchange_from_captured(_experience("responses", _responses_request(), body))
    (window,) = turn_windows(exchange)
    assert window.text.endswith(
        '[tool_call web_search] {"action":{"type":"search","query":"forecast"}}\n'
        "[assistant] I can't share that."
    )


def test_cited_anthropic_text_and_server_tool_use_stay_readable() -> None:
    """Cited text keeps its text and server tool results keep their readable fields."""
    body: JsonObject = {
        "type": "message",
        "stop_reason": "end_turn",
        "content": [
            {
                "type": "server_tool_use",
                "id": "srv_1",
                "name": "web_search",
                "input": {"query": "q"},
            },
            {
                "type": "web_search_tool_result",
                "tool_use_id": "srv_1",
                "content": [
                    {
                        "type": "web_search_result",
                        "url": "https://a.example",
                        "title": "A",
                        "encrypted_content": "opaque",
                    }
                ],
            },
            {
                "type": "text",
                "text": "Paris is the capital.",
                "citations": [
                    {
                        "type": "web_search_result_location",
                        "url": "https://a.example",
                        "title": "A",
                        "encrypted_index": "e",
                        "cited_text": "Paris",
                    }
                ],
            },
        ],
    }
    exchange = exchange_from_captured(_experience("messages", _messages_request(), body))
    (window,) = turn_windows(exchange)
    assert window.text.splitlines()[-3:] == [
        '[tool_call web_search] {"query":"q"}',
        '[tool_result web_search] [{"type":"web_search_result","url":"https://a.example",'
        '"title":"A"}]',
        "[assistant] Paris is the capital.",
    ]
    assert "opaque" not in exchange.model_dump_json()


def test_long_non_ascii_conversation_labels_fall_back_like_the_local_store() -> None:
    """A label over 512 UTF-8 bytes is ignored in favor of the session id."""
    request = _chat_request(metadata={"conversation_id": "\u96ea" * 200})
    record = _record(
        "chat_completions",
        request,
        {"kind": "json", "status": 200, "body": CHAT_BODY},
        session_id="session-7",
    )
    assert exchange_from_capture_record(record).conversation_key == "session-7"
    short = _record(
        "chat_completions",
        _chat_request(metadata={"conversation_id": "\u96ea" * 100}),
        {"kind": "json", "status": 200, "body": CHAT_BODY},
        session_id="session-7",
    )
    assert exchange_from_capture_record(short).conversation_key == "\u96ea" * 100


def test_cut_off_responses_tool_arguments_keep_their_text() -> None:
    """A call cut off by the length limit keeps its partial arguments and marks truncation."""
    body: JsonObject = {
        "status": "incomplete",
        "incomplete_details": {"reason": "max_output_tokens"},
        "output": [
            {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_9",
                "name": "search",
                "arguments": '{"query":',
                "status": "incomplete",
            }
        ],
    }
    exchange = exchange_from_captured(_experience("responses", _responses_request(), body))
    assert exchange.response[0].tool_calls[0].function.arguments == '{"query":'
    assert tag_facts(exchange).truncated


def test_custom_tools_native_declarations_and_results_stay_readable() -> None:
    """Custom calls, their native results and native tool declarations reach the windows."""
    request = decode_responses(
        {
            "model": "coding",
            "input": [
                {"role": "user", "content": "Patch the file"},
                {
                    "type": "custom_tool_call",
                    "call_id": "c1",
                    "name": "apply_patch",
                    "input": "*** p",
                },
                {
                    "type": "custom_tool_call_output",
                    "call_id": "c1",
                    "output": "Patch failed: file not found",
                },
            ],
            "tools": [
                {
                    "type": "namespace",
                    "name": "files",
                    "description": "File tools",
                    "tools": [
                        {
                            "type": "function",
                            "name": "read",
                            "description": "Read a file",
                            "parameters": {"type": "object"},
                        }
                    ],
                },
                {"type": "custom", "name": "apply_patch", "description": "Apply a patch"},
                {"type": "web_search"},
            ],
        }
    ).request
    body: JsonObject = {
        "status": "completed",
        "output": [
            {
                "type": "custom_tool_call",
                "id": "ct",
                "call_id": "c2",
                "name": "apply_patch",
                "input": "*** retry",
                "status": "completed",
            },
            {
                "type": "image_generation_call",
                "id": "ig",
                "status": "completed",
                "result": "iVBORw0KGgo" + "QUJDRA" * 60,
                "revised_prompt": "a diagram",
            },
        ],
    }
    exchange = exchange_from_captured(_experience("responses", request, body))
    assert [tool.name for tool in exchange.tools] == ["files.read", "apply_patch", "web_search"]
    (window,) = turn_windows(exchange)
    assert window.text.splitlines()[1:] == [
        "[tool_result apply_patch] Patch failed: file not found",
        "[tool_call apply_patch] *** retry",
        "[assistant] [image]",
        '[tool_call image_generation] {"result":"[image]","revised_prompt":"a diagram"}',
    ]
    assert "QUJDRA" not in exchange.model_dump_json()
    assert exchange.response[1].has_image()


def test_chat_custom_tool_calls_count_as_tool_calls() -> None:
    """A Chat custom tool call keeps its name and input."""
    body: JsonObject = {
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "c3",
                            "type": "custom",
                            "custom": {"name": "apply_patch", "input": "*** p"},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ]
    }
    exchange = exchange_from_captured(_experience("chat_completions", _chat_request(), body))
    assert tag_facts(exchange).tool_calls == 1
    assert exchange.response[0].tool_calls[0].function.arguments == "*** p"


def test_messages_schema_output_and_context_exhaustion_are_facts() -> None:
    """A Messages JSON schema format is structured output; context exhaustion is truncation."""
    request = decode_messages(
        {
            "model": "coding",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "List files as JSON"}],
            "output_config": {"format": {"type": "json_schema", "schema": {"type": "object"}}},
        }
    ).request
    body: JsonObject = {
        "type": "message",
        "stop_reason": "model_context_window_exceeded",
        "content": [{"type": "text", "text": '{"files": ['}],
    }
    facts = tag_facts(exchange_from_captured(_experience("messages", request, body)))
    assert facts.structured_output and facts.truncated


def test_native_results_server_errors_namespaces_and_server_tools_are_read() -> None:
    """Shell output, screenshots, server tool errors and namespaced calls stay readable."""
    request = decode_responses(
        {
            "model": "coding",
            "input": [
                {"role": "user", "content": "Run the tests"},
                {
                    "type": "function_call",
                    "call_id": "c1",
                    "name": "read",
                    "namespace": "files",
                    "arguments": "{}",
                },
                {"type": "function_call_output", "call_id": "c1", "output": "data"},
            ],
            "tools": [
                {
                    "type": "namespace",
                    "name": "files",
                    "description": "File tools",
                    "tools": [
                        {"type": "function", "name": "read", "parameters": {"type": "object"}}
                    ],
                }
            ],
        }
    ).request
    body: JsonObject = {
        "status": "completed",
        "output": [
            {
                "type": "message",
                "id": "m",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "Read it.", "annotations": []}],
            }
        ],
    }
    exchange = exchange_from_captured(_experience("responses", request, body))
    assert exchange.messages[1].tool_calls[0].function.name == "files.read"
    assert turn_windows(exchange)[0].text.splitlines()[1] == "[tool_result files.read] data"
    shell = _native_message(
        "tool",
        {
            "type": "shell_call_output",
            "call_id": "s1",
            "output": [{"stdout": "", "stderr": "AssertionError", "outcome": {"exit_code": 1}}],
        },
    )
    assert shell.text() == '[{"stdout":"","stderr":"AssertionError","outcome":{"exit_code":1}}]'
    screen = _native_message(
        "tool",
        {
            "type": "computer_call_output",
            "call_id": "k1",
            "output": {"type": "computer_screenshot", "image_url": "data:image/png;base64,AAAA"},
        },
    )
    assert screen.has_image() and "AAAA" not in screen.model_dump_json()
    error = _native_message(
        "assistant",
        {
            "type": "web_search_tool_result",
            "tool_use_id": "srv_1",
            "content": {"type": "web_search_tool_result_error", "error_code": "max_uses_exceeded"},
        },
    )
    assert (error.role, error.tool_call_id) == ("tool", "srv_1")
    assert "max_uses_exceeded" in error.text()


def test_messages_server_tool_declarations_are_offered_tools() -> None:
    """Anthropic server tools such as web search join the tool list."""
    request = decode_messages(
        {
            "model": "coding",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "Search"}],
            "tools": [{"type": "web_search_20250305", "name": "web_search", "max_uses": 3}],
        }
    ).request
    body: JsonObject = {
        "type": "message",
        "stop_reason": "end_turn",
        "content": [{"type": "text", "text": "ok"}],
    }
    exchange = exchange_from_captured(_experience("messages", request, body))
    assert [tool.name for tool in exchange.tools] == ["web_search"]


def _native_message(role: str, item: JsonObject) -> TagMessage:
    """Convert one native item carried by a canonical request message.

    Args:
        role: Carrying message role.
        item: The native item.

    Returns:
        The converted message.
    """
    return exchange_from_capture_parts(
        exchange_id="request-3",
        protocol="chat_completions",
        messages=[{"role": "user", "content": "go"}, {"role": role, "provider_native_item": item}],
        request=None,
        response={"kind": "json", "status": 200, "body": CHAT_BODY},
    ).messages[1]


def test_native_items_keep_errors_statuses_results_and_code() -> None:
    """Every readable field of a native call or result reaches the window."""
    mcp = _native_message(
        "assistant",
        {
            "type": "mcp_call",
            "id": "mcp_1",
            "name": "delete_row",
            "server_label": "crm",
            "arguments": '{"id":7}',
            "error": "Permission denied",
            "status": "failed",
        },
    )
    arguments = mcp.tool_calls[0].function.arguments
    assert "Permission denied" in arguments and '"status":"failed"' in arguments
    program = _native_message(
        "assistant", {"type": "program", "id": "p1", "code": "console.log(fs.readdirSync('.'))"}
    )
    assert not program.tool_calls and "readdirSync" in program.text()
    result = _native_message(
        "tool", {"type": "program_output", "call_id": "p1", "result": "TypeError: x is undefined"}
    )
    assert "TypeError" in result.text()
    patch = _native_message(
        "tool", {"type": "apply_patch_call_output", "call_id": "a1", "status": "failed"}
    )
    assert patch.text() == '{"status":"failed"}'
    reasoning = exchange_from_capture_parts(
        exchange_id="request-5",
        protocol="chat_completions",
        messages=[
            {"role": "user", "content": "go"},
            {"role": "assistant", "provider_native_item": {"type": "reasoning", "summary": []}},
        ],
        request=None,
        response={"kind": "json", "status": 200, "body": CHAT_BODY},
    )
    assert len(reasoning.messages) == 1


def test_chat_generated_images_become_markers() -> None:
    """Generated Chat images become image markers without their bytes."""
    body: JsonObject = {
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Here it is.",
                    "images": [
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}}
                    ],
                },
                "finish_reason": "stop",
            }
        ]
    }
    drawn = exchange_from_captured(_experience("chat_completions", _chat_request(), body))
    assert drawn.response[0].text() == "Here it is. [image]"
    assert "base64" not in drawn.model_dump_json()


def test_chat_tool_result_names_survive_without_their_call() -> None:
    """A tool result naming its tool keeps that name when its call is outside the request."""
    exchange = exchange_from_capture_parts(
        exchange_id="request-6",
        protocol="chat_completions",
        messages=[
            {"role": "user", "content": "Find the refund policy"},
            {"role": "tool", "tool_call_id": "call_0", "name": "kb_search", "content": "Policy A"},
        ],
        request=None,
        response={"kind": "json", "status": 200, "body": CHAT_BODY},
    )
    assert exchange.messages[1].name == "kb_search"
    assert "[tool_result kb_search] Policy A" in turn_windows(exchange)[0].text


def test_gateway_searches_and_call_images_are_kept() -> None:
    """Searches the gateway ran stay in the reply, and images inside calls count as images."""
    body: JsonObject = {
        "status": "completed",
        "output": [
            {
                "type": "web_search_call",
                "id": "wsgw_" + "0" * 32,
                "status": "completed",
                "action": {"type": "search", "query": "rates"},
            },
            {
                "type": "message",
                "id": "m",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "Rates rose.", "annotations": []}],
            },
        ],
    }
    exchange = exchange_from_captured(_experience("responses", _responses_request(), body))
    assert exchange.response[0].tool_calls[0].function.name == "web_search"
    chart = _native_message(
        "assistant",
        {
            "type": "code_interpreter_call",
            "id": "ci_1",
            "code": "plot()",
            "outputs": [{"type": "image", "url": "data:image/png;base64,AAAA"}],
        },
    )
    assert chart.has_image() and chart.tool_calls[0].function.name == "code_interpreter"


@pytest.mark.parametrize(
    ("protocol", "body"),
    [
        (
            "responses",
            {
                "status": "incomplete",
                "incomplete_details": {"reason": "max_output_tokens"},
                "output": [],
            },
        ),
        ("messages", {"type": "message", "stop_reason": "max_tokens", "content": []}),
        (
            "messages",
            {
                "type": "message",
                "stop_reason": "max_tokens",
                "content": [{"type": "text", "text": "", "citations": None}],
            },
        ),
        (
            "messages",
            {
                "type": "message",
                "stop_reason": "max_tokens",
                "content": [{"type": "thinking", "thinking": "", "signature": ""}],
            },
        ),
    ],
    ids=["responses", "messages", "messages-empty-text", "messages-thinking-only"],
)
def test_empty_replies_at_the_length_limit_are_tagged(
    protocol: CaptureProtocol, body: JsonObject
) -> None:
    """A reply that hit its limit before any output is an empty, truncated turn.

    Args:
        protocol: Public surface.
        body: Terminal body with no output.
    """
    request = _responses_request() if protocol == "responses" else _messages_request()
    exchange = exchange_from_captured(_experience(protocol, request, body))
    assert exchange.response == (TagMessage(role="assistant"),)
    assert tag_facts(exchange).truncated


def test_tools_declared_by_input_items_are_offered_tools() -> None:
    """Tools a request declares in an additional_tools item join the tool list and key."""

    def declaring(name: str) -> TagExchange:
        """Build an exchange whose request declares one namespaced tool inline.

        Args:
            name: The declared function name.

        Returns:
            The exchange.
        """
        request = decode_responses(
            {
                "model": "coding",
                "input": [
                    {
                        "type": "additional_tools",
                        "id": "at_1",
                        "role": "developer",
                        "tools": [
                            {
                                "type": "namespace",
                                "name": "files",
                                "description": "File tools",
                                "tools": [
                                    {
                                        "type": "function",
                                        "name": name,
                                        "parameters": {"type": "object"},
                                    }
                                ],
                            }
                        ],
                    },
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "Go"}],
                    },
                ],
            }
        ).request
        return exchange_from_captured(_experience("responses", request, RESPONSES_BODY))

    reading, deleting = declaring("read"), declaring("delete")
    assert [tool.name for tool in reading.tools] == ["files.read"]
    assert opening_key(reading) != opening_key(deleting)


def test_chat_refusals_keep_the_text_beside_them() -> None:
    """A reply with both text and a refusal keeps both."""
    body: JsonObject = {
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Let me check.", "refusal": "I can't."},
                "finish_reason": "stop",
            }
        ]
    }
    exchange = exchange_from_captured(_experience("chat_completions", _chat_request(), body))
    assert exchange.response[0].text() == "Let me check.\nI can't."


def test_namespaced_tool_results_keep_their_namespace() -> None:
    """A result whose call is not in view keeps its namespace-qualified name."""
    exchange = exchange_from_capture_parts(
        exchange_id="request-8",
        protocol="chat_completions",
        messages=[
            {"role": "user", "content": "Read it"},
            {
                "role": "tool",
                "tool_call_id": "c1",
                "content": "data",
                "provider_tool_name": "read",
                "provider_tool_namespace": "files",
            },
        ],
        request=None,
        response={"kind": "json", "status": 200, "body": CHAT_BODY},
    )
    assert exchange.messages[1].name == "files.read"


def test_dropped_history_ignores_replay_entries_for_the_original_messages() -> None:
    """A capture whose history was replaced by a marker still tags with its marker."""
    exchange = exchange_from_capture_parts(
        exchange_id="request-9",
        protocol="chat_completions",
        messages=[{"role": "system", "truncated": True, "content": "[1101 messages not captured]"}],
        request=None,
        response={"kind": "json", "status": 200, "body": CHAT_BODY},
        capture_context={
            "provider_context": {
                "provider_replay": [{"message_index": 1100, "tool_is_error": True}]
            },
            "capture_limits": {"messages_truncated": True, "messages_dropped": 1101},
        },
    )
    assert exchange.messages[0].text() == "[1101 messages not captured]"
    with pytest.raises(TagExchangeError, match="invalid provider message index"):
        exchange_from_capture_parts(
            exchange_id="request-9",
            protocol="chat_completions",
            messages=[{"role": "user", "content": "go"}],
            request=None,
            response={"kind": "json", "status": 200, "body": CHAT_BODY},
            capture_context={"provider_context": {"provider_replay": [{"message_index": 5}]}},
        )


def test_mcp_listings_are_readable_but_not_tool_calls() -> None:
    """Only invocation items count as tool calls; listings and approvals stay as text."""
    body: JsonObject = {
        "status": "completed",
        "output": [
            {
                "type": "mcp_list_tools",
                "id": "ml_1",
                "server_label": "crm",
                "tools": [{"name": "delete_row", "input_schema": {"type": "object"}}],
            },
            {
                "type": "mcp_call",
                "id": "mc_1",
                "name": "delete_row",
                "server_label": "crm",
                "arguments": "{}",
                "output": "deleted",
            },
        ],
    }
    exchange = exchange_from_captured(_experience("responses", _responses_request(), body))
    assert tag_facts(exchange).tool_calls == 1
    assert exchange.response[0].text().startswith('[mcp_list_tools] {"server_label":"crm"')


def test_text_that_only_starts_with_data_is_kept() -> None:
    """Readable text beginning with data: is not mistaken for an inline payload."""
    call = _native_message(
        "assistant",
        {
            "type": "custom_tool_call",
            "call_id": "c1",
            "name": "edit",
            "input": 'data: dict[str, int] = {"port": 443}',
        },
    )
    assert call.tool_calls[0].function.arguments == 'data: dict[str, int] = {"port": 443}'


def test_approval_requests_name_their_tool_and_capture_evidence_is_unchanged() -> None:
    """Approval text names the tool, and tagging never edits the capture it reads."""
    approval = _native_message(
        "assistant",
        {
            "type": "mcp_approval_request",
            "id": "ap_1",
            "name": "delete_customer",
            "server_label": "crm",
            "arguments": "{}",
        },
    )
    assert not approval.tool_calls
    assert approval.text().startswith("[mcp_approval_request delete_customer] ")
    experience = _experience("chat_completions", _chat_request(), CHAT_BODY)
    request = dict(experience.request)
    request["exp_capture_output"] = {"provider_reasoning": "private thoughts"}
    experience = experience.model_copy(update={"request": request})
    before = experience.model_dump_json()
    exchange_from_captured(experience)
    assert experience.model_dump_json() == before


def test_local_shell_output_links_to_its_call() -> None:
    """A local shell output links to its call through its id."""
    exchange = exchange_from_capture_parts(
        exchange_id="request-10",
        protocol="chat_completions",
        messages=[
            {"role": "user", "content": "List files"},
            {
                "role": "assistant",
                "provider_native_item": {
                    "type": "local_shell_call",
                    "id": "lsh_1",
                    "call_id": "call_1",
                    "action": {"type": "exec", "command": ["ls"]},
                },
            },
            {
                "role": "tool",
                "provider_native_item": {
                    "type": "local_shell_call_output",
                    "id": "call_1",
                    "output": "README.md",
                },
            },
        ],
        request=None,
        response={"kind": "json", "status": 200, "body": CHAT_BODY},
    )
    assert exchange.messages[2].tool_call_id == "call_1"
    assert "[tool_result local_shell] README.md" in turn_windows(exchange)[0].text


def test_binary_content_is_recognized_by_field_not_by_appearance() -> None:
    """Long unbroken text stays readable, while any image generation result is hidden."""
    sequence = "ACGT" * 75
    result = _native_message(
        "assistant",
        {"type": "mcp_call", "id": "m1", "name": "genome", "arguments": "{}", "output": sequence},
    )
    assert sequence in result.tool_calls[0].function.arguments
    small = _native_message(
        "assistant",
        {"type": "image_generation_call", "id": "ig", "result": "iVBORw0KGgoAAAANSUhEUg=="},
    )
    assert small.has_image() and "iVBORw0" not in small.model_dump_json()


def test_serialized_mcp_images_become_markers() -> None:
    """Images inside a JSON-encoded MCP result become markers without their bytes."""
    result = json.dumps(
        {"content": [{"type": "image", "data": "iVBORw0KGgo=", "mimeType": "image/png"}]}
    )
    call = _native_message(
        "assistant",
        {"type": "mcp_call", "id": "m1", "name": "screenshot", "arguments": "{}", "output": result},
    )
    assert call.has_image()
    assert "iVBORw0" not in call.model_dump_json()
    plain = _native_message(
        "assistant",
        {"type": "mcp_call", "id": "m2", "name": "count", "arguments": "{}", "output": '{"n": 3}'},
    )
    assert '"output":"{\\"n\\": 3}"' in plain.tool_calls[0].function.arguments


def test_image_typed_arguments_without_media_stay_readable() -> None:
    """An object typed image but carrying no image is ordinary data."""
    call = _native_message(
        "assistant",
        {
            "type": "mcp_call",
            "id": "m3",
            "name": "search_images",
            "arguments": '{"type":"image","query":"landscape","limit":10}',
        },
    )
    assert not call.has_image()
    assert "landscape" in call.tool_calls[0].function.arguments


def test_serialized_results_with_lone_surrogates_stay_taggable() -> None:
    """A JSON-encoded result escaping a lone surrogate still renders into windows."""
    result = '{"note":"\\ud800","content":[{"type":"image","data":"QUJD","mimeType":"image/png"}]}'
    exchange = exchange_from_capture_parts(
        exchange_id="request-11",
        protocol="chat_completions",
        messages=[
            {"role": "user", "content": "Shoot"},
            {
                "role": "assistant",
                "provider_native_item": {
                    "type": "mcp_call",
                    "id": "m4",
                    "name": "screenshot",
                    "arguments": "{}",
                    "output": result,
                },
            },
        ],
        request=None,
        response={"kind": "json", "status": 200, "body": CHAT_BODY},
    )
    assert "\ufffd" in exchange.messages[1].tool_calls[0].function.arguments
    assert turn_windows(exchange)


def test_messages_tool_input_cut_off_by_the_length_limit_is_kept() -> None:
    """A tool use cut off mid-input keeps its partial text and marks truncation."""
    body: JsonObject = {
        "type": "message",
        "stop_reason": "max_tokens",
        "content": [
            {
                "type": "tool_use",
                "id": "toolu_9",
                "name": "bash",
                "input": {},
                "capture_partial_input": '{"cmd": "ls -',
            }
        ],
    }
    exchange = exchange_from_captured(_experience("messages", _messages_request(), body))
    assert exchange.response[0].tool_calls[0].function.arguments == '{"cmd": "ls -'
    assert tag_facts(exchange).truncated


def test_tool_results_project_serialized_media() -> None:
    """JSON tool results holding images, audio or blobs keep no bytes."""
    image = json.dumps({"content": [{"type": "image", "data": "QUJD", "mimeType": "image/png"}]})
    audio = json.dumps({"content": [{"type": "audio", "data": "QUJD", "mimeType": "audio/wav"}]})
    blob = json.dumps(
        {
            "content": [
                {
                    "type": "resource",
                    "resource": {"uri": "f", "blob": "QUJD", "mimeType": "image/png"},
                }
            ]
        }
    )
    exchange = exchange_from_capture_parts(
        exchange_id="request-12",
        protocol="chat_completions",
        messages=[
            {"role": "user", "content": "Go"},
            {"role": "tool", "tool_call_id": "a", "content": image},
            {"role": "tool", "tool_call_id": "b", "content": audio},
            {"role": "tool", "tool_call_id": "c", "content": blob},
        ],
        request=None,
        response={"kind": "json", "status": 200, "body": CHAT_BODY},
    )
    assert "QUJD" not in exchange.model_dump_json()
    assert exchange.messages[1].has_image() and exchange.messages[3].has_image()
    assert "[audio]" in exchange.messages[2].text()
    assert tag_facts(exchange).has_images


def test_checked_in_anthropic_tool_use_stream_is_tagged() -> None:
    """The native dialect's recorded tool-use stream, with output-only caller metadata, tags."""
    path = (
        Path(__file__).parents[1]
        / "gateway/native/src/dialects/anthropic/testdata/anthropic_stream_tool_use.sse"
    )
    frames: list[JsonValue] = [
        json.loads(line.removeprefix("data:").strip())
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.startswith("data:")
    ]
    exchange = exchange_from_capture_parts(
        exchange_id="request-13",
        protocol="messages",
        messages=[{"role": "user", "content": "List files"}],
        request=None,
        response={
            "kind": "sse",
            "status": 200,
            "frames": frames,
            "truncated": False,
            "client_disconnected": False,
        },
    )
    call = exchange.response[-1].tool_calls[0]
    assert call.function.name == "Bash"
    assert json.loads(call.function.arguments)["command"] == "ls -la"


def test_cut_off_server_tool_input_and_data_url_results_are_kept() -> None:
    """Server tool input cut off by the limit keeps its text; data URL results become markers."""
    body: JsonObject = {
        "type": "message",
        "stop_reason": "max_tokens",
        "content": [
            {
                "type": "server_tool_use",
                "id": "srv_9",
                "name": "web_search",
                "input": {},
                "capture_partial_input": '{"query": "rates in',
            }
        ],
    }
    exchange = exchange_from_captured(_experience("messages", _messages_request(), body))
    assert exchange.response[0].tool_calls[0].function.arguments == '{"query": "rates in'
    screenshot = exchange_from_capture_parts(
        exchange_id="request-14",
        protocol="chat_completions",
        messages=[
            {"role": "user", "content": "Shoot"},
            {"role": "tool", "tool_call_id": "s", "content": "data:image/png;base64,QUJD"},
        ],
        request=None,
        response={"kind": "json", "status": 200, "body": CHAT_BODY},
    )
    assert screenshot.messages[1].has_image() and "QUJD" not in screenshot.model_dump_json()


def test_inline_media_in_ordinary_call_arguments_becomes_markers() -> None:
    """Chat custom inputs and function arguments carrying data URLs keep no bytes."""
    body: JsonObject = {
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "custom",
                            "custom": {"name": "upload", "input": "data:image/png;base64,QUJDRA=="},
                        },
                        {
                            "id": "f1",
                            "type": "function",
                            "function": {
                                "name": "attach",
                                "arguments": '{"file": "data:image/png;base64,QUJDRA=="}',
                            },
                        },
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ]
    }
    exchange = exchange_from_captured(_experience("chat_completions", _chat_request(), body))
    reply = exchange.response[0]
    assert "QUJDRA" not in exchange.model_dump_json()
    assert reply.has_image() and len(reply.tool_calls) == 2
    assert reply.tool_calls[1].function.arguments == '{"file":"[image]"}'


def test_record_without_a_response_is_rejected() -> None:
    """A prompt checkpoint with no response yet cannot be tagged."""
    record = _record(
        "chat_completions", _chat_request(), {"kind": "json", "status": 200, "body": CHAT_BODY}
    )
    pending = record.model_copy(update={"response": None})
    with pytest.raises(TagExchangeError, match="no response yet"):
        exchange_from_capture_record(pending)
