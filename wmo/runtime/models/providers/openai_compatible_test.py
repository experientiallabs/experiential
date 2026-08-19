"""Contract tests for shared OpenAI-compatible conversion and client behavior.

This module owns the fixtures shared by the provider suites: `_snapshot` and `_request`
are imported by `azure_test` and `native_test` so every adapter exercises one transcript.
"""

from __future__ import annotations

import math
from typing import Literal

import pytest

from wmo.common.models import (
    AssistantAction,
    BillingSource,
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
from wmo.runtime.models.providers.transport import (
    JsonHttpResponse,
    RetryPolicy,
    ScriptedJsonTransport,
)


def _snapshot(provider: str = "openai-compatible", model_id: str = "fake-model") -> ModelSnapshot:
    """Build an immutable identity fixture for one adapter.

    Args:
        provider: Catalog provider name under test.
        model_id: Exact configured model or deployment identity.

    Returns:
        A frozen snapshot with fixture digests.
    """
    return ModelSnapshot(
        billing_source=BillingSource.CUSTOMER_MANAGED,
        provider=provider,
        model_id=model_id,
        revision="fixture-revision",
        capabilities_sha256="a" * 64,
        connection_sha256="a" * 64,
    )


def _request(
    *,
    tool_choice: ToolChoice | Literal["auto", "none", "required"] | None = None,
) -> ModelRequest:
    """Build a visible transcript containing an earlier tool call and result.

    Args:
        tool_choice: Optional tool-choice constraint forwarded to the request.

    Returns:
        A typed request with system, user, assistant tool-call, and tool-result turns.
    """
    return ModelRequest(
        messages=(
            ModelMessage(role="system", content="You are precise."),
            ModelMessage(role="user", content="Create a ticket."),
            ModelMessage(
                role="assistant",
                assistant_action=AssistantAction(
                    tool_calls=(
                        ToolCall(
                            call_id="call-old",
                            name="create_ticket",
                            arguments={"priority": "normal"},
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
        tool_choice=tool_choice,
        temperature=0.2,
        maximum_output_tokens=128,
    )


def test_openai_compatible_request_keeps_history_tools_and_non_streaming_cap() -> None:
    """Shared conversion keeps every tool turn and emits no streaming request."""
    payload = openai_compatible_request(
        "fake-model", _request(tool_choice=ToolChoice(name="create_ticket"))
    )

    assert payload["stream"] is False
    assert payload["max_tokens"] == 128
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
            "function": {"name": "create_ticket", "arguments": '{"priority": "normal"}'},
        }
    ]


def test_openai_compatible_client_converts_tool_usage_and_resolved_identity() -> None:
    """One frozen tool response produces typed output, normalized usage, and actual model ID."""
    transport = ScriptedJsonTransport(
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
        ToolCall(
            call_id="call-new",
            name="create_ticket",
            arguments={"priority": "urgent"},
            raw_arguments='{"priority":"urgent"}',
        ),
    )
    assert response.economics.usage is not None
    assert response.economics.usage.input_tokens == 12
    assert response.economics.usage.cached_input_tokens == 5
    assert response.economics.latency_seconds is not None
    assert transport.requests[0][0] == "https://example.test/v1/chat/completions"
    assert transport.requests[0][1]["Authorization"] == "Bearer fake-key"


def test_openai_compatible_client_retries_only_the_same_endpoint() -> None:
    """A retryable status retries the frozen request without a failover model path."""
    transport = ScriptedJsonTransport(
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
    idempotency_keys = [request[1]["Idempotency-Key"] for request in transport.requests]
    assert idempotency_keys[0].startswith("wmo-")
    assert idempotency_keys[0] == idempotency_keys[1]


def test_response_without_choices_fails_closed_without_exposing_the_key() -> None:
    """A response with no choices raises a typed error that never includes the credential."""
    secret = "fake-secret-key-value"
    transport = ScriptedJsonTransport([JsonHttpResponse(status_code=200, body={"choices": []})])
    client = OpenAICompatibleClient(
        model=_snapshot(),
        base_url="https://example.test/v1",
        api_key=secret,
        transport=transport,
    )

    with pytest.raises(OpenAICompatibleResponseError, match="no choices") as captured:
        client.complete(_request())
    assert secret not in str(captured.value)
    assert secret not in repr(captured.value)


def test_openai_compatible_embedding_response_is_ordered_and_normalized() -> None:
    """Embedding conversion restores provider indexes and returns unit-length vectors."""
    transport = ScriptedJsonTransport(
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
    transport = ScriptedJsonTransport(
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
