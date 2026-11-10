"""Lone surrogate escapes on the native Chat, Messages, and Responses surfaces.

Node's well-formed ``JSON.stringify`` writes a string sliced inside an emoji as
a lone ``\\udXXX`` escape. Each body is sent as those exact ASCII bytes through
the real native engine: content is served with U+FFFD and disclosed, a name is
a typed 400 before acceptance, and nothing answers 500.
"""

from __future__ import annotations

import json

import httpx
import pytest

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.tests.native_messages_test import (
    _engine,  # noqa: F401 - module fixture `engine`.
    _responses_engine,  # noqa: F401 - module fixture `responses_engine`.
    _ResponsesUpstream,
    _ServingEngine,
    _SseUpstream,
)

pytest.importorskip("exp_gateway_native")

_DISCLOSURES = "x-experiential-ignored-parameters"
_LONE_HIGH = "\\ud83d"
"""A lone high surrogate exactly as JSON text (six ASCII characters)."""


def _send(engine: _ServingEngine, path: str, raw: str, **headers: str) -> httpx.Response:
    """POST ``raw`` verbatim as ASCII bytes with the seeded key."""
    return httpx.post(
        f"{engine.base}{path}",
        headers={
            "authorization": f"Bearer {engine.raw_key}",
            "content-type": "application/json",
            **headers,
        },
        content=raw.encode("ascii"),
        timeout=30.0,
    )


def _escaped(body: JsonObject, marker: str = "SLICE") -> str:
    """Serialize ``body`` as ASCII JSON with every ``marker`` a lone escape."""
    text = json.dumps(body, ensure_ascii=True)
    assert marker in text
    return text.replace(marker, _LONE_HIGH)


def _last_chat_payload() -> JsonObject:
    """Return the most recent body the loopback upstream received."""
    with _SseUpstream.payloads_lock:
        return _SseUpstream.payloads[-1]


def test_chat_content_lone_surrogates_are_served_replaced_and_disclosed(
    engine: _ServingEngine,
) -> None:
    """Message text and tool-call arguments become U+FFFD; the 200 names both."""
    raw = _escaped(
        {
            "model": "coding",
            "messages": [
                {"role": "system", "content": "Be brief SLICE"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": '{"q": "cut SLICE"}'},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "found SLICE"},
                {"role": "user", "content": "hello SLICE"},
            ],
        }
    )
    response = _send(engine, "/v1/chat/completions", raw)
    assert response.status_code == 200, response.text
    assert response.json()[_DISCLOSURES] == [
        "messages.content->replaced(lone_surrogate)",
        "messages.tool_calls.function.arguments->replaced(lone_surrogate)",
    ]
    sent = json.dumps(_last_chat_payload(), ensure_ascii=False)
    assert "Be brief \ufffd" in sent
    assert "found \ufffd" in sent
    assert "hello \ufffd" in sent
    assert "\\ud83d" not in sent


def test_chat_stream_discloses_the_repair_on_its_chunks(engine: _ServingEngine) -> None:
    """A streamed completion carries the same disclosure as the buffered one."""
    raw = _escaped(
        {"model": "coding", "stream": True, "messages": [{"role": "user", "content": "hi SLICE"}]}
    )
    response = _send(engine, "/v1/chat/completions", raw)
    assert response.status_code == 200, response.text
    chunks = [
        json.loads(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith("data: {")
    ]
    assert chunks
    assert chunks[0][_DISCLOSURES] == ["messages.content->replaced(lone_surrogate)"]


@pytest.mark.parametrize(
    ("field", "param"),
    [
        ("model", "model"),
        ("tool_call_id", "messages.1.tool_calls.0.id"),
        ("tool_name", "messages.1.tool_calls.0.function.name"),
    ],
)
def test_chat_identifier_lone_surrogates_are_a_typed_400(
    engine: _ServingEngine, field: str, param: str
) -> None:
    """A model, call ID, or tool name holding a lone surrogate is refused by name."""
    body: JsonObject = {
        "model": "codingSLICE" if field == "model" else "coding",
        "messages": [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1SLICE" if field == "tool_call_id" else "call_1",
                        "type": "function",
                        "function": {
                            "name": "lookupSLICE" if field == "tool_name" else "lookup",
                            "arguments": "{}",
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "found"},
        ],
    }
    response = _send(engine, "/v1/chat/completions", _escaped(body))
    assert response.status_code == 400, response.text
    error = response.json()["error"]
    assert error["code"] == "invalid_parameter"
    assert error["param"] == param
    assert "lone UTF-16 surrogate" in error["message"]


def test_chat_raw_invalid_byte_keeps_the_invalid_json_refusal(engine: _ServingEngine) -> None:
    """Bytes that are not UTF-8 never reach repair; the data plane refuses them."""
    raw = b'{"model":"coding","messages":[{"role":"user","content":"x\xff"}]}'
    response = httpx.post(
        f"{engine.base}/v1/chat/completions",
        headers={"authorization": f"Bearer {engine.raw_key}", "content-type": "application/json"},
        content=raw,
        timeout=30.0,
    )
    assert response.status_code == 400, response.text
    assert response.json()["error"]["code"] == "invalid_json"


def test_messages_content_lone_surrogates_are_served_replaced_and_disclosed(
    engine: _ServingEngine,
) -> None:
    """System text, a tool result, and user text are repaired on the Anthropic surface."""
    raw = _escaped(
        {
            "model": "coding",
            "max_tokens": 64,
            "system": [{"type": "text", "text": "Be brief SLICE"}],
            "messages": [
                {"role": "user", "content": "look it up"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {}}
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "toolu_1", "content": "res SLICE"},
                        {"type": "text", "text": "hello SLICE"},
                    ],
                },
            ],
            "tools": [{"name": "lookup", "input_schema": {"type": "object"}}],
        }
    )
    response = _send(engine, "/v1/messages", raw, **{"anthropic-version": "2023-06-01"})
    assert response.status_code == 200, response.text
    assert response.json()[_DISCLOSURES] == [
        "messages.content.content->replaced(lone_surrogate)",
        "messages.content.text->replaced(lone_surrogate)",
        "system.text->replaced(lone_surrogate)",
    ]
    sent = json.dumps(_last_chat_payload(), ensure_ascii=False)
    assert "Be brief \ufffd" in sent
    assert "res \ufffd" in sent
    assert "hello \ufffd" in sent


def test_messages_tool_use_id_lone_surrogate_is_a_typed_400(engine: _ServingEngine) -> None:
    """A tool_use ID names the call its result answers, so it is refused by name."""
    raw = _escaped(
        {
            "model": "coding",
            "max_tokens": 64,
            "messages": [
                {"role": "user", "content": "look it up"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": "toolu_1SLICE", "name": "lookup", "input": {}}
                    ],
                },
            ],
        }
    )
    response = _send(engine, "/v1/messages", raw, **{"anthropic-version": "2023-06-01"})
    assert response.status_code == 400, response.text
    error = response.json()["error"]
    assert error["type"] == "invalid_request_error"
    assert "messages.1.content.0.id" in error["message"]


def test_responses_content_lone_surrogates_are_served_replaced_and_disclosed(
    responses_engine: _ServingEngine,
) -> None:
    """Input text parts and plain message content are repaired on the Responses surface."""
    raw = _escaped(
        {
            "model": "responses",
            "input": [
                {"type": "message", "role": "user", "content": "Be brief SLICE"},
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "count-only SLICE"}],
                },
            ],
        }
    )
    response = _send(responses_engine, "/v1/responses", raw)
    assert response.status_code == 200, response.text
    assert response.json()[_DISCLOSURES] == [
        "input.content->replaced(lone_surrogate)",
        "input.content.text->replaced(lone_surrogate)",
    ]
    with _ResponsesUpstream.payloads_lock:
        sent = json.dumps(_ResponsesUpstream.payloads[-1], ensure_ascii=False)
    assert "Be brief \ufffd" in sent
    assert "count-only \ufffd" in sent


def test_responses_function_call_id_lone_surrogate_is_a_typed_400(
    responses_engine: _ServingEngine,
) -> None:
    """A function_call call_id names the call its output answers; it is refused by name."""
    raw = _escaped(
        {
            "model": "responses",
            "input": [
                {"type": "message", "role": "user", "content": "count-only"},
                {
                    "type": "function_call",
                    "call_id": "call_1SLICE",
                    "name": "lookup",
                    "arguments": "{}",
                },
            ],
        }
    )
    response = _send(responses_engine, "/v1/responses", raw)
    assert response.status_code == 400, response.text
    error = response.json()["error"]
    assert error["code"] == "invalid_parameter"
    assert error["param"] == "input.1.call_id"
