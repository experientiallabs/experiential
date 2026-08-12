"""Frozen native-provider contract fixtures for the focused W3 adapters."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from wmo.common.core.artifacts import JsonObject
from wmo.common.models import (
    AssistantAction,
    EmbeddingClient,
    ModelClient,
    ModelMessage,
    ModelRequest,
    ModelSnapshot,
    ToolCall,
    ToolChoice,
    Usage,
)
from wmo.common.tasks import ToolSchema
from wmo.runtime.models.providers.anthropic import (
    AnthropicClient,
    anthropic_messages_request,
)
from wmo.runtime.models.providers.gemini import GeminiClient
from wmo.runtime.models.providers.openai import OpenAIClient
from wmo.runtime.models.providers.openrouter import OpenRouterClient
from wmo.runtime.models.providers.tinker_sampling import (
    TinkerSample,
    TinkerSampler,
    TinkerSamplingClient,
)
from wmo.runtime.models.providers.transport import JsonHttpResponse, JsonHttpTransport


class _ScriptedTransport(JsonHttpTransport):
    """Returns frozen JSON responses while retaining calls for wire assertions."""

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
        """Record one fake call and return the next frozen response."""
        del timeout_seconds
        self.requests.append((url, headers, payload))
        if not self._responses:
            raise AssertionError("test made an unexpected provider request")
        return self._responses.pop(0)


class _FakeTinkerSampler:
    """Represents a completed trained handle without importing training code."""

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    def sample(self, request: ModelRequest) -> TinkerSample:
        """Return one frozen action and preserve the typed request."""
        self.requests.append(request)
        return TinkerSample(
            output=AssistantAction(content="sampled from completed handle"),
            usage=Usage(input_tokens=8, output_tokens=4),
            served_model_id="tinker://completed-handle-v2",
        )


def _snapshot(provider: str, model_id: str) -> ModelSnapshot:
    """Build an immutable identity fixture for one adapter."""
    return ModelSnapshot(
        provider=provider,
        model_id=model_id,
        revision="fixture-revision",
        capabilities_sha256="a" * 64,
    )


def _request(
    *,
    tool_choice: ToolChoice | Literal["auto", "none", "required"] | None = None,
) -> ModelRequest:
    """Build a visible transcript containing an earlier tool call and result."""
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


def test_openai_responses_client_preserves_native_tool_wire_usage_and_identity() -> None:
    """Direct OpenAI uses Responses, not the compatible chat-completions shape."""
    transport = _ScriptedTransport(
        [
            JsonHttpResponse(
                status_code=200,
                body={
                    "status": "completed",
                    "model": "gpt-5.4-2026-08-11",
                    "output": [
                        {
                            "type": "function_call",
                            "call_id": "call-new",
                            "name": "create_ticket",
                            "arguments": '{"priority":"urgent"}',
                        }
                    ],
                    "usage": {
                        "input_tokens": 13,
                        "output_tokens": 5,
                        "input_tokens_details": {"cached_tokens": 4},
                    },
                },
            )
        ]
    )
    client = OpenAIClient(
        model=_snapshot("openai", "gpt-5.4"),
        api_key="fixture-openai-key",
        base_url="https://openai.fixture/v1",
        transport=transport,
    )

    response = client.complete(_request(tool_choice=ToolChoice(name="create_ticket")))

    assert isinstance(client, ModelClient)
    assert isinstance(client, EmbeddingClient)
    assert response.model.model_id == "gpt-5.4-2026-08-11"
    assert response.output.tool_calls == (
        ToolCall(call_id="call-new", name="create_ticket", arguments={"priority": "urgent"}),
    )
    assert response.economics.usage == Usage(
        input_tokens=13,
        output_tokens=5,
        cached_input_tokens=4,
    )
    url, headers, payload = transport.requests[0]
    assert url == "https://openai.fixture/v1/responses"
    assert headers["Authorization"] == "Bearer fixture-openai-key"
    assert payload["store"] is False
    assert payload["stream"] is False
    assert payload["tool_choice"] == {"type": "function", "name": "create_ticket"}
    inputs = payload["input"]
    assert isinstance(inputs, list)
    assert inputs[1] == {
        "type": "function_call",
        "call_id": "call-old",
        "name": "create_ticket",
        "arguments": '{"priority": "normal"}',
    }


def test_openai_embeddings_use_the_shared_normalized_response_contract() -> None:
    """Direct OpenAI reuses only the common non-streaming embedding conversion."""
    transport = _ScriptedTransport(
        [
            JsonHttpResponse(
                status_code=200,
                body={
                    "data": [
                        {"index": 0, "embedding": [3.0, 4.0]},
                        {"index": 1, "embedding": [0.0, 2.0]},
                    ]
                },
            )
        ]
    )
    client = OpenAIClient(
        model=_snapshot("openai", "text-embedding-3-small"),
        api_key="fixture-openai-key",
        base_url="https://openai.fixture/v1",
        transport=transport,
    )

    embeddings = client.embed(("first", "second"))

    assert tuple(item.values for item in embeddings) == ((0.6, 0.8), (0.0, 1.0))
    assert transport.requests[0][0] == "https://openai.fixture/v1/embeddings"


def test_openrouter_uses_one_compatible_endpoint_without_failover() -> None:
    """OpenRouter decorates the shared request without adding a provider chain."""
    transport = _ScriptedTransport(
        [
            JsonHttpResponse(
                status_code=200,
                body={"model": "served/router", "choices": [{"message": {"content": "ok"}}]},
            )
        ]
    )
    client = OpenRouterClient(
        model=_snapshot("openrouter", "vendor/model"),
        api_key="fixture-router-key",
        base_url="https://router.fixture/v1",
        transport=transport,
    )

    response = client.complete(_request())

    assert isinstance(client, ModelClient)
    assert response.model.model_id == "served/router"
    url, headers, payload = transport.requests[0]
    assert url == "https://router.fixture/v1/chat/completions"
    assert headers["HTTP-Referer"] == "https://github.com/experientiallabs/world-model-optimizer"
    assert headers["X-Title"] == "world-model-optimizer"
    assert payload["stream"] is False


def test_anthropic_uses_native_tool_blocks_and_normalizes_cache_usage() -> None:
    """Anthropic tool and cache fields stay native until the shared response boundary."""
    transport = _ScriptedTransport(
        [
            JsonHttpResponse(
                status_code=200,
                body={
                    "model": "claude-fixture-20260811",
                    "content": [
                        {"type": "text", "text": "Working."},
                        {
                            "type": "tool_use",
                            "id": "call-new",
                            "name": "create_ticket",
                            "input": {"priority": "urgent"},
                        },
                    ],
                    "usage": {
                        "input_tokens": 5,
                        "cache_read_input_tokens": 3,
                        "cache_creation_input_tokens": 2,
                        "output_tokens": 4,
                    },
                },
            )
        ]
    )
    client = AnthropicClient(
        model=_snapshot("anthropic", "claude-fixture"),
        api_key="fixture-anthropic-key",
        base_url="https://anthropic.fixture/v1",
        transport=transport,
    )

    response = client.complete(_request(tool_choice="required"))

    assert isinstance(client, ModelClient)
    assert not isinstance(client, EmbeddingClient)
    assert response.model.model_id == "claude-fixture-20260811"
    assert response.output.content == "Working."
    assert response.output.tool_calls[0].call_id == "call-new"
    assert response.economics.usage == Usage(
        input_tokens=10,
        output_tokens=4,
        cached_input_tokens=3,
    )
    url, headers, payload = transport.requests[0]
    assert url == "https://anthropic.fixture/v1/messages"
    assert headers["x-api-key"] == "fixture-anthropic-key"
    assert payload["tool_choice"] == {"type": "any"}
    messages = payload["messages"]
    assert isinstance(messages, list)
    assert messages[1]["content"][0] == {
        "type": "tool_use",
        "id": "call-old",
        "name": "create_ticket",
        "input": {"priority": "normal"},
    }


def test_anthropic_tool_none_keeps_history_schemas_and_uses_native_none() -> None:
    """The closed none choice retains schemas needed by native historical tool blocks."""
    payload = anthropic_messages_request("claude-fixture", _request(tool_choice="none"))

    assert payload["tools"] == [
        {
            "name": "create_ticket",
            "description": "Create one support ticket.",
            "input_schema": {"type": "object"},
        }
    ]
    assert payload["tool_choice"] == {"type": "none"}


def test_gemini_uses_native_function_calls_usage_identity_and_embeddings() -> None:
    """Gemini retains its content parts, model version, and batch embedding shape."""
    transport = _ScriptedTransport(
        [
            JsonHttpResponse(
                status_code=200,
                body={
                    "modelVersion": "gemini-2.5-pro-001",
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {"text": "Working."},
                                    {
                                        "functionCall": {
                                            "id": "call-new",
                                            "name": "create_ticket",
                                            "args": {"priority": "urgent"},
                                        }
                                    },
                                ]
                            }
                        }
                    ],
                    "usageMetadata": {
                        "promptTokenCount": 12,
                        "candidatesTokenCount": 6,
                        "cachedContentTokenCount": 4,
                    },
                },
            ),
            JsonHttpResponse(
                status_code=200,
                body={
                    "embeddings": [
                        {"values": [3.0, 4.0]},
                        {"values": [0.0, 2.0]},
                    ]
                },
            ),
        ]
    )
    client = GeminiClient(
        model=_snapshot("gemini", "gemini-2.5-pro"),
        api_key="fixture-gemini-key",
        base_url="https://gemini.fixture/v1beta",
        transport=transport,
    )

    response = client.complete(_request(tool_choice=ToolChoice(name="create_ticket")))
    embeddings = client.embed(("first", "second"))

    assert isinstance(client, ModelClient)
    assert isinstance(client, EmbeddingClient)
    assert response.model.model_id == "gemini-2.5-pro-001"
    assert response.output.content == "Working."
    assert response.output.tool_calls == (
        ToolCall(call_id="call-new", name="create_ticket", arguments={"priority": "urgent"}),
    )
    assert response.economics.usage == Usage(
        input_tokens=12,
        output_tokens=6,
        cached_input_tokens=4,
    )
    assert tuple(item.values for item in embeddings) == ((0.6, 0.8), (0.0, 1.0))
    generate_url, headers, generate_payload = transport.requests[0]
    assert generate_url == "https://gemini.fixture/v1beta/models/gemini-2.5-pro:generateContent"
    assert headers["x-goog-api-key"] == "fixture-gemini-key"
    assert generate_payload["toolConfig"] == {
        "functionCallingConfig": {"mode": "ANY", "allowedFunctionNames": ["create_ticket"]}
    }
    assert transport.requests[1][0].endswith(":batchEmbedContents")


def test_tinker_sampling_client_requires_only_a_completed_handle_sampler() -> None:
    """The Tinker client adapts a sampler without importing or exposing training behavior."""
    sampler = _FakeTinkerSampler()
    client = TinkerSamplingClient(
        model=_snapshot("tinker", "tinker://completed-handle-v1"),
        sampler=sampler,
    )

    response = client.complete(_request())

    assert isinstance(sampler, TinkerSampler)
    assert isinstance(client, ModelClient)
    assert not isinstance(client, EmbeddingClient)
    assert sampler.requests == [_request()]
    assert response.model.model_id == "tinker://completed-handle-v2"
    assert response.economics.usage == Usage(input_tokens=8, output_tokens=4)
