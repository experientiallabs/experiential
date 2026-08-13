"""Official OpenAI SDK routing endpoint tests."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from openai import OpenAI
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessageParam,
    ChatCompletionToolUnionParam,
)
from openai.types.responses import Response, ResponseStreamEvent

from wmo.runtime.router import create_router_endpoint
from wmo.runtime.router.runtime import RouterRuntime
from wmo.runtime.router.runtime_test import _Client, _runtime


def _clients() -> tuple[OpenAI, TestClient, RouterRuntime, _Client]:
    runtime, model_client = _runtime()
    app = FastAPI()
    app.include_router(create_router_endpoint({"router-a": runtime}))
    http = TestClient(app)
    openai = OpenAI(api_key="local-test", base_url="http://testserver/v1", http_client=http)
    return openai, http, runtime, model_client


def test_official_chat_client_preserves_tools_without_cross_caller_affinity() -> None:
    """Official SDK chat calls need no WMO header and do not join equal transcripts."""
    openai, _http, _runtime_value, model_client = _clients()
    messages: list[ChatCompletionMessageParam] = [
        {"role": "user", "content": "do it"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-in",
                    "type": "function",
                    "function": {"name": "read", "arguments": '{"path":"a"}'},
                }
            ],
        },
        {"role": "tool", "content": "result", "tool_call_id": "call-in"},
    ]
    tools: list[ChatCompletionToolUnionParam] = [
        {
            "type": "function",
            "function": {
                "name": "read",
                "description": "read a file",
                "parameters": {"type": "object"},
            },
        }
    ]

    completion = openai.chat.completions.create(model="router-a", messages=messages, tools=tools)

    assert isinstance(completion, ChatCompletion)
    assert completion.model == "router-a"
    assert completion.choices[0].message.tool_calls is not None
    assert completion.choices[0].message.tool_calls[0].id == "call-out"
    assert completion.model_extra is None or "routing_decision" not in completion.model_extra
    captured = model_client.requests[-1]
    assert [message.role for message in captured.messages] == ["user", "assistant", "tool"]
    assert captured.messages[1].assistant_action is not None
    assert captured.messages[1].assistant_action.tool_calls[0].call_id == "call-in"
    assert captured.messages[2].tool_call_id == "call-in"

    next_turn = openai.chat.completions.create(
        model="router-a",
        messages=[
            *messages,
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-out",
                        "type": "function",
                        "function": {"name": "write", "arguments": '{"x":1}'},
                    }
                ],
            },
            {"role": "tool", "content": "done", "tool_call_id": "call-out"},
            {"role": "user", "content": "next turn"},
        ],
        tools=tools,
    )

    assert next_turn.model == "router-a"
    assert model_client.embed_calls == 2
    assert len(model_client.requests[-1].messages) == 6

    openai.chat.completions.create(model="router-a", messages=messages, tools=tools)
    assert model_client.embed_calls == 3


def test_official_responses_client_continues_with_previous_response_id() -> None:
    """Responses continuation keeps routing affinity through the standard response ID."""
    openai, _http, _runtime_value, model_client = _clients()

    first = openai.responses.create(model="router-a", input="Help me")
    second = openai.responses.create(
        model="router-a", input="Continue", previous_response_id=first.id
    )

    assert isinstance(first, Response)
    assert isinstance(second, Response)
    assert second.previous_response_id == first.id
    assert model_client.embed_calls == 1
    assert [message.role for message in model_client.requests[-1].messages] == [
        "user",
        "assistant",
        "user",
    ]


def test_openai_text_content_parts_are_preserved() -> None:
    """Chat and Responses text parts reach the routed model as complete visible text."""
    openai, _http, _runtime_value, model_client = _clients()

    openai.chat.completions.create(
        model="router-a",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "first "},
                    {"type": "text", "text": "second"},
                ],
            }
        ],
    )
    openai.responses.create(
        model="router-a",
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "third "},
                    {"type": "input_text", "text": "fourth"},
                ],
            }
        ],
    )

    assert model_client.requests[-2].messages[0].content == "first second"
    assert model_client.requests[-1].messages[0].content == "third fourth"


def test_official_clients_parse_buffered_streams() -> None:
    """Both official SDK streaming iterators parse WMO's OpenAI SSE events."""
    openai, _http, _runtime_value, _model_client = _clients()

    chat_stream = openai.chat.completions.create(
        model="router-a", messages=[{"role": "user", "content": "stream"}], stream=True
    )
    chat_chunks: list[ChatCompletionChunk] = list(chat_stream)
    response_stream = openai.responses.create(model="router-a", input="stream", stream=True)
    response_events: list[ResponseStreamEvent] = list(response_stream)

    assert chat_chunks
    assert all(isinstance(item, ChatCompletionChunk) for item in chat_chunks)
    assert chat_chunks[-1].choices[0].finish_reason == "tool_calls"
    assert response_events
    assert response_events[0].type == "response.created"
    assert response_events[-1].type == "response.completed"


def test_idempotency_key_reuses_decision_after_provider_failure() -> None:
    """A standard idempotency key pins the retry decision without a WMO episode header."""
    openai, http, runtime, model_client = _clients()
    model_client.completion_error = RuntimeError("provider unavailable")
    payload = {"model": "router-a", "messages": [{"role": "user", "content": "retry"}]}
    headers = {"Idempotency-Key": "interaction-a"}

    assert http.post("/v1/chat/completions", json=payload, headers=headers).status_code == 502
    cached = next(iter(runtime._request_decisions.values()))  # noqa: SLF001
    response = openai.chat.completions.create(
        model="router-a",
        messages=[{"role": "user", "content": "retry"}],
        extra_headers=headers,
    )

    assert response.id.startswith("chatcmpl-")
    assert next(iter(runtime._request_decisions.values())) == cached  # noqa: SLF001
    assert model_client.embed_calls == 1
    assert model_client.complete_calls == 2


@pytest.mark.parametrize(
    "path,payload",
    [
        (
            "/v1/chat/completions",
            {
                "model": "router-a",
                "messages": [
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "x",
                                "function": {"name": "read", "arguments": "[1]"},
                            }
                        ],
                    }
                ],
            },
        ),
        (
            "/v1/chat/completions",
            {
                "model": "router-a",
                "messages": [{"role": "system", "content": "no user"}],
            },
        ),
        (
            "/v1/responses",
            {"model": "router-a", "input": "next", "previous_response_id": "missing"},
        ),
    ],
)
def test_invalid_official_request_never_reaches_provider(
    path: str, payload: dict[str, object]
) -> None:
    """Malformed OpenAI requests fail as 4xx before embedding or model dispatch."""
    _openai, http, _runtime_value, model_client = _clients()

    response = http.post(path, json=payload)

    assert response.status_code in {400, 422}
    assert model_client.embed_calls == 0
    assert model_client.complete_calls == 0
