"""Vertex Anthropic (Claude on Vertex AI) routing, envelope quirks, and OAuth tests."""

from __future__ import annotations

import pytest

from exp.common.core.artifacts import JsonObject
from exp.common.models import EmbeddingClient, ModelClient
from exp.runtime.gateway.contracts import GatewayApiSurface, GatewayMessage, GatewayRequest
from exp.runtime.models.providers.dialect_dispatch import (
    _apply_vertex_anthropic_envelope,
    dialect_stream_payload,
)
from exp.runtime.models.providers.errors import ProviderCapabilityError
from exp.runtime.models.providers.openai_compatible_test import _request, _snapshot
from exp.runtime.models.providers.transport import JsonHttpResponse, ScriptedJsonTransport
from exp.runtime.models.providers.vertex import (
    VERTEX_ANTHROPIC_VERSION,
    VertexAnthropicClient,
    _vertex_anthropic_model_id,
    vertex_wire_for_model,
)

_BASE_URL = (
    "https://us-east5-aiplatform.googleapis.com/v1/projects/fixture-project/locations/us-east5"
)
_MODEL = "publishers/anthropic/models/claude-opus-4-5@20250514"


def _messages_response() -> JsonObject:
    """Return one minimal completed Anthropic Messages payload."""
    return {
        "id": "msg_fixture",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-4-5",
        "content": [{"type": "text", "text": "Working."}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 12, "output_tokens": 6},
    }


def test_anthropic_publisher_id_selects_the_anthropic_messages_wire() -> None:
    """A publishers/anthropic id picks the Anthropic wire; Google and MaaS ids are unchanged."""
    assert vertex_wire_for_model(_MODEL) == "anthropic_messages"
    assert vertex_wire_for_model("gemini-2.5-pro") == "gemini_generate_content"
    assert (
        vertex_wire_for_model("publishers/google/models/gemini-2.5-pro")
        == "gemini_generate_content"
    )
    assert vertex_wire_for_model("deepseek-ai/deepseek-v3.2-maas") == "openai_compatible"


def test_anthropic_model_id_strips_the_resource_path_prefix() -> None:
    """The URL segment is the bare publisher model id, with or without the prefix."""
    assert _vertex_anthropic_model_id(_MODEL) == "claude-opus-4-5@20250514"
    assert _vertex_anthropic_model_id("claude-opus-4-5@20250514") == "claude-opus-4-5@20250514"


def test_completion_posts_the_anthropic_envelope_to_raw_predict() -> None:
    """Blocking completion hits :rawPredict with OAuth, body anthropic_version, and no model."""
    transport = ScriptedJsonTransport(
        [JsonHttpResponse(status_code=200, body=_messages_response())]
    )
    client = VertexAnthropicClient(
        model=_snapshot("vertex", _MODEL),
        api_key='{"placeholder": true}',
        base_url=_BASE_URL,
        transport=transport,
        token_provider=lambda: "fixture-bearer-token",
    )

    response = client.complete(_request())

    assert isinstance(client, ModelClient)
    assert not isinstance(client, EmbeddingClient)
    assert response.output.content == "Working."
    url, headers, payload = transport.requests[0]
    assert url == (f"{_BASE_URL}/publishers/anthropic/models/claude-opus-4-5@20250514:rawPredict")
    assert headers["authorization"] == "Bearer fixture-bearer-token"
    assert "x-api-key" not in headers
    assert "anthropic-version" not in headers
    # Vertex reads the version from the body and the model from the URL.
    assert payload["anthropic_version"] == VERTEX_ANTHROPIC_VERSION
    assert "model" not in payload
    assert payload["messages"]
    assert payload["max_tokens"]


def test_stream_profile_pins_the_anthropic_sse_route() -> None:
    """The native profile is the Anthropic dialect on the Vertex :streamRawPredict route."""
    client = VertexAnthropicClient(
        model=_snapshot("vertex", _MODEL),
        api_key='{"placeholder": true}',
        base_url=_BASE_URL,
        token_provider=lambda: "fixture-bearer-token",
        supports_reasoning=True,
    )

    profile = client.gateway_wire_profile()

    assert profile.dialect == "anthropic_messages"
    assert profile.url == (
        f"{_BASE_URL}/publishers/anthropic/models/claude-opus-4-5@20250514:streamRawPredict"
    )
    assert profile.headers["authorization"] == "Bearer fixture-bearer-token"
    assert profile.reasoning_wire_format == "anthropic_adaptive"
    assert profile.supports_reasoning is True


def test_dispatch_body_drops_model_and_sets_vertex_anthropic_version() -> None:
    """The ACTUAL gateway path (dialect_stream_payload) emits the Vertex envelope."""
    client = VertexAnthropicClient(
        model=_snapshot("vertex", _MODEL),
        api_key='{"placeholder": true}',
        base_url=_BASE_URL,
        token_provider=lambda: "fixture-bearer-token",
    )
    request = GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(GatewayMessage(role="user", content="Preserve these controls."),),
        maximum_output_tokens=128,
    )

    payload = dialect_stream_payload(client.gateway_wire_profile(), request)

    # Vertex names the model in the URL and reads the version from the body.
    assert "model" not in payload
    assert payload["anthropic_version"] == VERTEX_ANTHROPIC_VERSION
    assert payload["stream"] is True
    assert payload["messages"]


def test_dispatch_refuses_an_anthropic_files_handle() -> None:
    """A Files handle (no Files API on Vertex) fails clearly so admission reroutes."""
    with pytest.raises(ProviderCapabilityError):
        _apply_vertex_anthropic_envelope(
            {
                "model": "claude-opus-4-5@20250514",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "document", "source": {"type": "file", "file_id": "file_x"}}
                        ],
                    }
                ],
            },
            VERTEX_ANTHROPIC_VERSION,
        )


def test_dispatch_allows_tool_calls_whose_args_mention_file_keys() -> None:
    """Caller JSON in tool_use.input / tool_result is NOT a Files handle; allowed."""
    payload: JsonObject = {
        "model": "claude-opus-4-5@20250514",
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tu_1",
                        "name": "lookup",
                        "input": {"file_id": "abc", "nested": {"type": "file"}},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tu_1",
                        "content": [{"type": "text", "text": '{"file_id": "abc"}'}],
                    }
                ],
            },
        ],
    }

    _apply_vertex_anthropic_envelope(payload, VERTEX_ANTHROPIC_VERSION)

    assert "model" not in payload
    assert payload["anthropic_version"] == VERTEX_ANTHROPIC_VERSION


def test_token_provider_is_read_for_every_profile() -> None:
    """Each profile build mints a fresh bearer so the hourly refresh is never stale."""
    tokens = iter(["token-one", "token-two"])
    client = VertexAnthropicClient(
        model=_snapshot("vertex", _MODEL),
        api_key='{"placeholder": true}',
        base_url=_BASE_URL,
        token_provider=lambda: next(tokens),
    )

    assert client.gateway_wire_profile().headers["authorization"] == "Bearer token-one"
    assert client.gateway_wire_profile().headers["authorization"] == "Bearer token-two"
