"""Frozen native-provider contract fixtures for the focused W3 adapters."""

from __future__ import annotations

import pytest

from wmo.common.models import (
    AssistantAction,
    EmbeddingClient,
    ModelClient,
    ModelRequest,
    ToolCall,
    ToolChoice,
    Usage,
)
from wmo.runtime.models.providers.anthropic import (
    AnthropicClient,
    anthropic_messages_request,
)
from wmo.runtime.models.providers.gemini import GeminiClient
from wmo.runtime.models.providers.openai import OpenAIClient
from wmo.runtime.models.providers.openai_compatible import OpenRouterClient
from wmo.runtime.models.providers.openai_compatible_test import _request, _snapshot
from wmo.runtime.models.providers.tinker_sampling import (
    TinkerSample,
    TinkerSampler,
    TinkerSamplingClient,
    TinkerSdkSampler,
    create_tinker_sampler,
)
from wmo.runtime.models.providers.transport import JsonHttpResponse, ScriptedJsonTransport


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


def test_default_tinker_factory_constructs_a_lazy_sdk_sampler_without_sampling() -> None:
    """The runtime-owned factory uses the installed dependency but creates no provider session."""
    pytest.importorskip("tinker")
    pytest.importorskip("tinker_cookbook")

    sampler = create_tinker_sampler(
        model=_snapshot("tinker", "tinker://completed-handle-v2"),
        api_key="fixture-tinker-key",
        base_url="https://tinker.fixture",
    )

    assert isinstance(sampler, TinkerSdkSampler)


def test_openai_responses_client_preserves_native_tool_wire_usage_and_identity() -> None:
    """Direct OpenAI uses Responses, not the compatible chat-completions shape."""
    transport = ScriptedJsonTransport(
        [
            JsonHttpResponse(
                status_code=200,
                body={
                    "id": "resp_native",
                    "object": "response",
                    "created_at": 1.0,
                    "status": "completed",
                    "model": "gpt-5.4-2026-08-11",
                    "parallel_tool_calls": True,
                    "tool_choice": "auto",
                    "tools": [],
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
                        "total_tokens": 18,
                        "input_tokens_details": {"cached_tokens": 4, "cache_write_tokens": 0},
                        "output_tokens_details": {"reasoning_tokens": 0},
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


def test_openai_reasoning_model_declarations_shape_the_wire_payload() -> None:
    """A no-temperature declaration drops the parameter and a pinned effort is sent verbatim."""
    transport = ScriptedJsonTransport(
        [
            JsonHttpResponse(
                status_code=200,
                body={
                    "id": "resp_reasoning",
                    "object": "response",
                    "created_at": 1.0,
                    "status": "completed",
                    "model": "gpt-5.6-luna",
                    "parallel_tool_calls": True,
                    "tool_choice": "auto",
                    "tools": [],
                    "output": [
                        {
                            "type": "message",
                            "id": "msg_reasoning",
                            "role": "assistant",
                            "status": "completed",
                            "content": [{"type": "output_text", "text": "ok", "annotations": []}],
                        }
                    ],
                },
            )
        ]
    )
    client = OpenAIClient(
        model=_snapshot("openai", "gpt-5.6-luna"),
        api_key="fixture-openai-key",
        base_url="https://openai.fixture/v1",
        transport=transport,
        supports_temperature=False,
        reasoning_effort="xhigh",
    )

    client.complete(_request())

    payload = transport.requests[0][2]
    assert "temperature" not in payload
    assert payload["reasoning"] == {"effort": "xhigh"}


def test_openai_embeddings_use_the_shared_normalized_response_contract() -> None:
    """Direct OpenAI reuses only the common non-streaming embedding conversion."""
    transport = ScriptedJsonTransport(
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
    transport = ScriptedJsonTransport(
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
    transport = ScriptedJsonTransport(
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
        cache_write_input_tokens=2,
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
    transport = ScriptedJsonTransport(
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
