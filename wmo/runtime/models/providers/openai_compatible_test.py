"""Contract tests for shared OpenAI-compatible conversion and client behavior."""

from __future__ import annotations

import math
from collections.abc import Mapping

import pytest

from wmo.common.core.artifacts import JsonObject
from wmo.common.models import (
    AssistantAction,
    ModelMessage,
    ModelRequest,
    ModelSnapshot,
    ToolCall,
    ToolChoice,
)
from wmo.common.tasks import ToolSchema
from wmo.runtime.models.providers.openai_compatible import (
    OpenAICompatibleClient,
    OpenAICompatibleResponseError,
    openai_compatible_request,
)
from wmo.runtime.models.providers.retry import RetryPolicy
from wmo.runtime.models.providers.transport import JsonHttpResponse, JsonHttpTransport


class _ScriptedTransport(JsonHttpTransport):
    """Returns frozen responses while retaining every request for assertions."""

    def __init__(self, responses: list[JsonHttpResponse]) -> None:
        self._responses = list(responses)
        self.requests: list[tuple[str, Mapping[str, str], JsonObject]] = []

    def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: JsonObject,
        timeout_seconds: float,
    ) -> JsonHttpResponse:
        del timeout_seconds
        self.requests.append((url, headers, payload))
        if not self._responses:
            raise AssertionError("test made an unexpected provider request")
        return self._responses.pop(0)


def _snapshot() -> ModelSnapshot:
    return ModelSnapshot(
        provider="openai-compatible",
        model_id="fake-model",
        revision="fake-revision",
        capabilities_sha256="a" * 64,
    )


def _request() -> ModelRequest:
    return ModelRequest(
        messages=(
            ModelMessage(role="system", content="You are careful."),
            ModelMessage(role="user", content="Create a ticket."),
            ModelMessage(
                role="assistant",
                assistant_action=AssistantAction(
                    tool_calls=(
                        ToolCall(
                            call_id="call-old",
                            name="create_ticket",
                            arguments={"priority": "high"},
                        ),
                    )
                ),
            ),
            ModelMessage(role="tool", content="created", tool_call_id="call-old"),
        ),
        tools=(
            ToolSchema(
                name="create_ticket",
                description="Create one support ticket.",
                input_schema={"type": "object"},
            ),
        ),
        tool_choice=ToolChoice(name="create_ticket"),
        temperature=0.4,
        maximum_output_tokens=256,
    )


def test_openai_compatible_request_keeps_history_tools_and_non_streaming_cap() -> None:
    """Shared conversion keeps every tool turn and emits no streaming request."""
    payload = openai_compatible_request("fake-model", _request())

    assert payload["stream"] is False
    assert payload["max_tokens"] == 256
    assert payload["tool_choice"] == {
        "type": "function",
        "function": {"name": "create_ticket"},
    }
    assert payload["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "create_ticket",
                "description": "Create one support ticket.",
                "parameters": {"type": "object"},
            },
        }
    ]
    messages = payload["messages"]
    assert isinstance(messages, list)
    assert messages[2]["tool_calls"] == [
        {
            "id": "call-old",
            "type": "function",
            "function": {"name": "create_ticket", "arguments": '{"priority": "high"}'},
        }
    ]


def test_openai_compatible_client_converts_tool_usage_and_resolved_identity() -> None:
    """One frozen tool response produces typed output, normalized usage, and actual model ID."""
    transport = _ScriptedTransport(
        [
            JsonHttpResponse(
                status_code=200,
                body={
                    "model": "served-model-20260811",
                    "choices": [
                        {
                            "message": {
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call-new",
                                        "function": {
                                            "name": "create_ticket",
                                            "arguments": '{"priority":"urgent"}',
                                        },
                                    }
                                ],
                            }
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 12,
                        "completion_tokens": 7,
                        "prompt_tokens_details": {"cached_tokens": 5},
                    },
                },
            )
        ]
    )
    client = OpenAICompatibleClient(
        model=_snapshot(),
        base_url="https://example.test/v1",
        api_key="fake-key",
        transport=transport,
    )

    response = client.complete(_request())

    assert response.model.model_id == "served-model-20260811"
    assert response.output.content is None
    assert response.output.tool_calls == (
        ToolCall(call_id="call-new", name="create_ticket", arguments={"priority": "urgent"}),
    )
    assert response.economics.usage is not None
    assert response.economics.usage.input_tokens == 12
    assert response.economics.usage.cached_input_tokens == 5
    assert response.economics.latency_seconds is not None
    assert transport.requests[0][0] == "https://example.test/v1/chat/completions"
    assert transport.requests[0][1]["Authorization"] == "Bearer fake-key"


def test_openai_compatible_client_retries_only_the_same_endpoint() -> None:
    """A retryable status retries the frozen request without a failover model path."""
    transport = _ScriptedTransport(
        [
            JsonHttpResponse(status_code=503, body={"error": {"message": "busy"}}),
            JsonHttpResponse(
                status_code=200,
                body={
                    "choices": [{"message": {"content": "ok"}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                },
            ),
        ]
    )
    client = OpenAICompatibleClient(
        model=_snapshot(),
        base_url="https://example.test/v1",
        api_key="fake-key",
        transport=transport,
        retry_policy=RetryPolicy(maximum_attempts=2, initial_delay_seconds=0),
    )

    assert client.complete(_request()).output.content == "ok"
    assert [request[0] for request in transport.requests] == [
        "https://example.test/v1/chat/completions",
        "https://example.test/v1/chat/completions",
    ]


def test_openai_compatible_embedding_response_is_ordered_and_normalized() -> None:
    """Embedding conversion restores provider indexes and returns unit-length vectors."""
    transport = _ScriptedTransport(
        [
            JsonHttpResponse(
                status_code=200,
                body={
                    "data": [
                        {"index": 1, "embedding": [0.0, 3.0]},
                        {"index": 0, "embedding": [4.0, 0.0]},
                    ]
                },
            )
        ]
    )
    client = OpenAICompatibleClient(
        model=_snapshot(),
        base_url="https://example.test/v1",
        api_key="fake-key",
        transport=transport,
    )

    embeddings = client.embed(("first", "second"))

    assert embeddings[0].values == (1.0, 0.0)
    assert embeddings[1].values == (0.0, 1.0)
    assert all(
        math.isclose(sum(value * value for value in item.values), 1.0) for item in embeddings
    )
    assert transport.requests[0][0] == "https://example.test/v1/embeddings"


def test_openai_compatible_conversion_rejects_malformed_tool_arguments() -> None:
    """A provider cannot turn malformed tool JSON into an invented empty argument object."""
    transport = _ScriptedTransport(
        [
            JsonHttpResponse(
                status_code=200,
                body={
                    "choices": [
                        {
                            "message": {
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "bad",
                                        "function": {"name": "create_ticket", "arguments": "{"},
                                    }
                                ],
                            }
                        }
                    ]
                },
            )
        ]
    )
    client = OpenAICompatibleClient(
        model=_snapshot(),
        base_url="https://example.test/v1",
        api_key="fake-key",
        transport=transport,
    )

    with pytest.raises(OpenAICompatibleResponseError, match="arguments is not JSON"):
        client.complete(_request())
