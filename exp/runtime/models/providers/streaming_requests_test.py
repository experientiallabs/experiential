"""Tests for launch-provider streaming request payload translation."""

import pytest

from exp.runtime.gateway.contracts import (
    GatewayApiSurface,
    GatewayMessage,
    GatewayRequest,
)
from exp.runtime.models.providers.base import GatewayWireProfile
from exp.runtime.models.providers.bedrock_requests import converse_body
from exp.runtime.models.providers.errors import ProviderCapabilityError
from exp.runtime.models.providers.gemini_requests import gemini_generate_request
from exp.runtime.models.providers.streaming_requests import (
    anthropic_messages_stream_payload,
    bedrock_converse_stream_payload,
    dialect_stream_payload,
    gemini_generate_content_stream_payload,
    openai_compatible_stream_payload,
    openai_responses_stream_payload,
)
from exp.runtime.openai_protocol.model_adapter import model_request


def _chat_request(
    *,
    temperature: float | None = None,
    top_p: float | None = None,
) -> GatewayRequest:
    """Build one Chat Completions streaming request with optional sampling fields.

    Args:
        temperature: Optional sampling temperature to include.
        top_p: Optional nucleus-sampling mass to include.

    Returns:
        A streaming Chat request carrying the supplied sampling fields.
    """
    return GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(GatewayMessage(role="user", content="hello"),),
        temperature=temperature,
        top_p=top_p,
        stream=True,
        include_usage=True,
    )


def test_openai_compatible_stream_payload_forwards_top_p_and_usage() -> None:
    """Streaming Chat payloads preserve nucleus sampling and always request usage."""
    payload = openai_compatible_stream_payload(
        "exact-model",
        _chat_request(top_p=1.0, temperature=0.2),
    )

    assert payload["stream"] is True
    assert payload["stream_options"] == {"include_usage": True}
    assert payload["temperature"] == 0.2
    assert payload["top_p"] == 1.0


def test_openai_compatible_stream_payload_omits_absent_top_p() -> None:
    """An omitted nucleus parameter stays off the wire instead of being invented."""
    payload = openai_compatible_stream_payload("exact-model", _chat_request())

    assert "top_p" not in payload
    assert "temperature" not in payload


def test_openai_compatible_stream_payload_honors_token_limit_key() -> None:
    """The output-token ceiling lands on the configured wire field only."""
    request = _chat_request().model_copy(update={"maximum_output_tokens": 64})

    default_payload = openai_compatible_stream_payload("exact-model", request)
    azure_payload = openai_compatible_stream_payload(
        "exact-model", request, token_limit_key="max_completion_tokens"
    )

    assert default_payload["max_tokens"] == 64
    assert "max_completion_tokens" not in default_payload
    assert azure_payload["max_completion_tokens"] == 64
    assert "max_tokens" not in azure_payload


def test_openai_responses_stream_payload_forwards_top_p_when_sampling_is_open() -> None:
    """Native Responses streaming preserves nucleus sampling on models that accept it."""
    payload = openai_responses_stream_payload(
        "exact-model",
        _chat_request(top_p=1.0, temperature=0.2),
        supports_temperature=True,
        reasoning_effort=None,
    )

    assert payload["stream"] is True
    assert payload["temperature"] == 0.2
    assert payload["top_p"] == 1.0


def test_openai_responses_stream_payload_rejects_top_p_when_sampling_is_pinned() -> None:
    """Pinned-sampling Responses models fail closed instead of dropping caller top_p."""
    with pytest.raises(ProviderCapabilityError, match="top_p") as captured:
        openai_responses_stream_payload(
            "exact-model",
            _chat_request(top_p=1.0),
            supports_temperature=False,
            reasoning_effort="xhigh",
        )
    assert captured.value.capability == "top_p"


def test_openai_responses_stream_payload_omits_absent_top_p() -> None:
    """Native Responses streaming does not invent a nucleus-sampling value."""
    payload = openai_responses_stream_payload(
        "exact-model",
        _chat_request(),
        supports_temperature=True,
        reasoning_effort=None,
    )

    assert "top_p" not in payload


def test_anthropic_messages_stream_payload_forwards_top_p() -> None:
    """Native Anthropic streaming preserves nucleus sampling on the Messages wire."""
    payload = anthropic_messages_stream_payload(
        "exact-model",
        _chat_request(top_p=1.0, temperature=0.2),
    )

    assert payload["stream"] is True
    assert payload["temperature"] == 0.2
    assert payload["top_p"] == 1.0


def test_anthropic_messages_stream_payload_omits_absent_top_p() -> None:
    """Native Anthropic streaming does not invent a nucleus-sampling value."""
    payload = anthropic_messages_stream_payload("exact-model", _chat_request())

    assert "top_p" not in payload
    assert "temperature" not in payload


def test_gemini_stream_payload_matches_the_provider_client_builder() -> None:
    """The gemini dialect payload is the exact generateContent body the python
    provider client sends, built through the same shared converter."""
    request = _chat_request(temperature=0.3)
    payload = gemini_generate_content_stream_payload("gemini-2.5-pro", request)

    assert payload == gemini_generate_request("gemini-2.5-pro", model_request(request))
    assert payload["contents"] == [{"role": "user", "parts": [{"text": "hello"}]}]
    generation = payload["generationConfig"]
    assert isinstance(generation, dict)
    assert generation["temperature"] == 0.3
    # Streaming is selected by the streamGenerateContent route, not the body.
    assert "stream" not in payload


def test_dialect_dispatch_builds_the_gemini_payload() -> None:
    """The shared dialect dispatch routes gemini_generate_content correctly."""
    profile = GatewayWireProfile(
        dialect="gemini_generate_content",
        url="https://example.invalid/models/gemini-2.5-pro:streamGenerateContent?alt=sse",
        model_id="gemini-2.5-pro",
    )
    payload = dialect_stream_payload(profile, _chat_request())

    assert payload["contents"] == [{"role": "user", "parts": [{"text": "hello"}]}]


def test_bedrock_stream_payload_matches_the_provider_builder_without_model_id() -> None:
    """The bedrock dialect body is the shared Converse payload with the
    ``modelId`` routing key removed: on the REST route the model travels in
    the URL path, not the body."""
    request = _chat_request(temperature=0.2)
    payload = bedrock_converse_stream_payload("us.anthropic.claude-sonnet-4-5", request)

    assert payload == converse_body(model_request(request))
    assert "modelId" not in payload
    assert payload["messages"] == [{"role": "user", "content": [{"text": "hello"}]}]
    inference = payload["inferenceConfig"]
    assert isinstance(inference, dict)
    assert inference["temperature"] == 0.2


def test_dialect_dispatch_builds_the_bedrock_payload() -> None:
    """The shared dialect dispatch routes bedrock_converse_stream correctly."""
    profile = GatewayWireProfile(
        dialect="bedrock_converse_stream",
        url="https://bedrock-runtime.us-east-1.amazonaws.com/model/m/converse-stream",
        model_id="us.anthropic.claude-sonnet-4-5",
        signs_request_body=True,
    )
    payload = dialect_stream_payload(profile, _chat_request())

    assert "modelId" not in payload
    assert payload["messages"] == [{"role": "user", "content": [{"text": "hello"}]}]
