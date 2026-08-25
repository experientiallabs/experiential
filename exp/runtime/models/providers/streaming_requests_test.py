"""Tests for launch-provider streaming request payload translation."""

from typing import cast

import pytest

from exp.runtime.gateway.contracts import (
    GatewayApiSurface,
    GatewayMessage,
    GatewayRequest,
)
from exp.runtime.models.providers.base import GatewayWireProfile
from exp.runtime.models.providers.bedrock_requests import converse_body
from exp.runtime.models.providers.gemini_requests import gemini_generate_request
from exp.runtime.models.providers.streaming_requests import (
    anthropic_messages_stream_payload,
    bedrock_converse_stream_payload,
    dialect_stream_payload,
    gemini_generate_content_stream_payload,
    openai_compatible_stream_payload,
    openai_responses_stream_payload,
    route_generation_parameter_requests,
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


def test_openai_compatible_stream_payload_omits_unproven_controls() -> None:
    """Unknown compatible routes drop top-k and logprobs instead of guessing wire support."""
    request = _chat_request().model_copy(update={"top_k": 40, "logprobs": True, "top_logprobs": 5})
    payload = openai_compatible_stream_payload("exact-model", request)
    assert "top_k" not in payload
    assert "logprobs" not in payload
    assert "top_logprobs" not in payload


def test_openai_compatible_stream_payload_ignores_logprobs_even_when_flagged() -> None:
    """Logprob controls stay off the wire until normalized output can preserve them."""
    request = _chat_request().model_copy(update={"top_k": 40, "logprobs": True, "top_logprobs": 5})
    payload = openai_compatible_stream_payload(
        "exact-model",
        request,
        supports_top_k=True,
        supports_logprobs=True,
    )
    assert payload["top_k"] == 40
    assert "logprobs" not in payload
    assert "top_logprobs" not in payload


def test_anthropic_stream_payload_omits_logprobs_even_when_flagged() -> None:
    """Anthropic's native Messages lane never receives an OpenAI logprob field."""
    request = _chat_request().model_copy(update={"logprobs": True, "top_logprobs": 5})
    payload = anthropic_messages_stream_payload(
        "claude-sonnet-5",
        request,
        supports_logprobs=True,
    )
    assert "logprobs" not in payload
    assert "top_logprobs" not in payload


def test_openai_compatible_stream_payload_omits_reasoning_without_route_capability() -> None:
    """A compatible route never receives a reasoning field without explicit capability proof."""
    request = _chat_request().model_copy(update={"reasoning_effort": "high"})

    payload = openai_compatible_stream_payload(
        "cloud-opus-5",
        request,
        reasoning_effort="medium",
    )

    assert "reasoning_effort" not in payload


def test_openai_compatible_stream_payload_forwards_reasoning_with_route_capability() -> None:
    """A compatible route receives the requested effort only when its row opts in."""
    request = _chat_request().model_copy(update={"reasoning_effort": "high"})

    payload = openai_compatible_stream_payload(
        "reasoning-model",
        request,
        supports_reasoning=True,
        reasoning_effort="medium",
    )

    assert payload["reasoning_effort"] == "high"


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
        supports_reasoning=False,
        reasoning_effort=None,
    )

    assert payload["stream"] is True
    assert payload["temperature"] == 0.2
    assert payload["top_p"] == 1.0


def test_openai_responses_stream_payload_omits_top_p_when_sampling_is_pinned() -> None:
    """Pinned-sampling Responses models omit caller top_p instead of failing the request."""
    payload = openai_responses_stream_payload(
        "exact-model",
        _chat_request(top_p=1.0),
        supports_temperature=False,
        supports_reasoning=True,
        reasoning_effort="xhigh",
    )
    assert "top_p" not in payload


def test_openai_responses_stream_payload_omits_absent_top_p() -> None:
    """Native Responses streaming does not invent a nucleus-sampling value."""
    payload = openai_responses_stream_payload(
        "exact-model",
        _chat_request(),
        supports_temperature=True,
        supports_reasoning=False,
        reasoning_effort=None,
    )

    assert "top_p" not in payload


def test_openai_responses_stream_payload_omits_reasoning_without_route_capability() -> None:
    """An unsupported native route drops a caller reasoning request before upstream dispatch."""
    request = _chat_request().model_copy(update={"reasoning_effort": "high"})

    payload = openai_responses_stream_payload(
        "cloud-opus-5",
        request,
        supports_temperature=True,
        reasoning_effort="medium",
    )

    assert "reasoning" not in payload


def test_openai_responses_stream_payload_forwards_reasoning_with_route_capability() -> None:
    """A verified native Responses route receives the caller's requested effort."""
    request = _chat_request().model_copy(update={"reasoning_effort": "high"})

    payload = openai_responses_stream_payload(
        "gpt-5.6-luna",
        request,
        supports_temperature=False,
        supports_reasoning=True,
        reasoning_effort="medium",
    )

    assert payload["reasoning"] == {"effort": "high"}


def test_openai_responses_stream_payload_forwards_reasoning_summary() -> None:
    """A native reasoning route receives the current Responses summary selector."""
    request = GatewayRequest(
        surface=GatewayApiSurface.RESPONSES,
        messages=(GatewayMessage(role="user", content="hello"),),
        reasoning_summary="detailed",
        reasoning_summary_parameters=("reasoning.generate_summary",),
    )

    payload = openai_responses_stream_payload(
        "gpt-5.6-luna",
        request,
        supports_temperature=False,
        supports_reasoning=True,
    )

    assert payload["reasoning"] == {"summary": "detailed"}


def test_route_shaping_omits_reasoning_summary_outside_native_responses() -> None:
    """A fallback without summary output support removes and discloses the exact aliases."""
    request = GatewayRequest(
        surface=GatewayApiSurface.RESPONSES,
        messages=(GatewayMessage(role="user", content="hello"),),
        reasoning_summary="concise",
        reasoning_summary_parameters=(
            "reasoning.generate_summary",
            "reasoning.summary",
        ),
    )
    profiles = (
        GatewayWireProfile(
            dialect="openai_responses",
            url="https://openai.test",
            supports_reasoning=True,
        ),
        GatewayWireProfile(
            dialect="openai_compatible",
            url="https://fallback.test",
            supports_reasoning=True,
            reasoning_wire_format="reasoning",
        ),
    )

    public_request, provider_request = route_generation_parameter_requests(profiles, request)

    assert public_request.ignored_parameters == (
        "reasoning.generate_summary",
        "reasoning.summary",
    )
    assert public_request.reasoning_summary == "concise"
    assert provider_request.reasoning_summary is None


def test_route_generation_controls_use_the_whole_waterfall_intersection() -> None:
    """One incompatible fallback removes safe optional controls before any dispatch."""
    request = _chat_request(temperature=0.2, top_p=0.8).model_copy(
        update={
            "top_k": 20,
            "logprobs": True,
            "top_logprobs": 5,
            "reasoning_effort": "high",
        }
    )
    profiles = (
        GatewayWireProfile(
            dialect="openai_compatible",
            url="https://first.test",
            supports_temperature=True,
            supports_top_p=True,
            supports_top_k=True,
            supports_reasoning=True,
            reasoning_wire_format="reasoning",
        ),
        GatewayWireProfile(
            dialect="anthropic_messages",
            url="https://fallback.test",
            supports_temperature=False,
            supports_top_p=False,
            supports_top_k=False,
            supports_reasoning=True,
            reasoning_wire_format="anthropic_adaptive",
        ),
    )

    public_request, provider_request = route_generation_parameter_requests(profiles, request)

    assert public_request.ignored_parameters == (
        "temperature",
        "top_p",
        "top_k",
        "logprobs",
        "top_logprobs",
    )
    assert provider_request.temperature is None
    assert provider_request.top_p is None
    assert provider_request.top_k is None
    assert provider_request.logprobs is None
    assert provider_request.top_logprobs is None
    assert provider_request.reasoning_effort == "high"


def test_route_shaping_omits_tool_controls_when_no_tools_exist() -> None:
    """No-op tool selectors never reach providers that reject them without schemas."""
    request = _chat_request().model_copy(
        update={"tool_choice": "none", "parallel_tool_calls": False}
    )

    public_request, provider_request = route_generation_parameter_requests(
        (GatewayWireProfile(dialect="openai_compatible", url="https://provider.test"),),
        request,
    )

    assert public_request.ignored_parameters == ("tool_choice", "parallel_tool_calls")
    assert provider_request.tool_choice is None
    assert provider_request.parallel_tool_calls is None


def test_route_shaping_omits_out_of_range_provider_controls() -> None:
    """Portable public ranges never become provider-side range errors."""
    request = _chat_request(temperature=1.5).model_copy(update={"top_k": 101})

    public_request, provider_request = route_generation_parameter_requests(
        (
            GatewayWireProfile(
                dialect="gemini_generate_content",
                url="https://provider.test",
                maximum_temperature=1.0,
                supports_top_k=True,
                maximum_top_k=100,
            ),
        ),
        request,
    )

    assert public_request.ignored_parameters == ("temperature", "top_k")
    assert provider_request.temperature is None
    assert provider_request.top_k is None


def test_responses_logprob_shaping_discloses_only_the_caller_field() -> None:
    """A synthetic canonical logprobs flag never leaks into ignored-field disclosure."""
    request = GatewayRequest(
        surface=GatewayApiSurface.RESPONSES,
        messages=(GatewayMessage(role="user", content="hello"),),
        logprobs=True,
        top_logprobs=5,
        ignored_parameters=("top_logprobs",),
    )

    public_request, provider_request = route_generation_parameter_requests(
        (GatewayWireProfile(dialect="openai_responses", url="https://provider.test"),),
        request,
    )

    assert public_request.ignored_parameters == ("top_logprobs",)
    assert provider_request.logprobs is None
    assert provider_request.top_logprobs is None


def test_route_shaping_omits_temperature_outside_any_provider_range() -> None:
    """A cross-provider route never forwards a value one wire rejects by range."""
    request = _chat_request(temperature=1.5)
    profiles = (
        GatewayWireProfile(dialect="openai_responses", url="https://openai.test"),
        GatewayWireProfile(
            dialect="bedrock_converse_stream",
            url="https://bedrock.test",
            maximum_temperature=1.0,
        ),
    )

    public_request, provider_request = route_generation_parameter_requests(profiles, request)

    assert public_request.ignored_parameters == ("temperature",)
    assert provider_request.temperature is None


def test_reasoning_effort_uses_each_provider_native_wire_shape() -> None:
    """The same public control translates without leaking an OpenAI field across providers."""
    request = _chat_request().model_copy(update={"reasoning_effort": "xhigh"})

    openrouter = dialect_stream_payload(
        GatewayWireProfile(
            dialect="openai_compatible",
            url="https://openrouter.test",
            model_id="anthropic/claude-opus-5",
            supports_reasoning=True,
            reasoning_wire_format="reasoning",
        ),
        request,
    )
    anthropic = dialect_stream_payload(
        GatewayWireProfile(
            dialect="anthropic_messages",
            url="https://anthropic.test",
            model_id="claude-sonnet-4-6",
            supports_reasoning=True,
            reasoning_wire_format="anthropic_adaptive",
        ),
        request,
    )
    gemini = dialect_stream_payload(
        GatewayWireProfile(
            dialect="gemini_generate_content",
            url="https://gemini.test",
            model_id="gemini-3.7-flash",
            supports_reasoning=True,
            reasoning_wire_format="gemini_thinking",
        ),
        request.model_copy(update={"reasoning_effort": "minimal"}),
    )

    assert openrouter["reasoning"] == {"effort": "xhigh"}
    assert "reasoning_effort" not in openrouter
    assert anthropic["thinking"] == {"type": "adaptive"}
    assert anthropic["output_config"] == {"effort": "high"}
    assert "reasoning" not in anthropic
    generation = cast("dict[str, object]", gemini["generationConfig"])
    assert generation["thinkingConfig"] == {"thinkingLevel": "LOW"}


@pytest.mark.parametrize(
    "profile",
    (
        GatewayWireProfile(dialect="anthropic_messages", url="https://anthropic.test"),
        GatewayWireProfile(dialect="gemini_generate_content", url="https://gemini.test"),
        GatewayWireProfile(dialect="bedrock_converse_stream", url="https://bedrock.test"),
        GatewayWireProfile(dialect="openai_compatible", url="https://compatible.test"),
    ),
)
def test_non_native_reasoning_profiles_never_emit_openai_reasoning_fields(
    profile: GatewayWireProfile,
) -> None:
    """Every non-Responses provider path drops OpenAI reasoning controls by default."""
    request = _chat_request().model_copy(update={"reasoning_effort": "high"})

    payload = dialect_stream_payload(profile, request)

    assert "reasoning" not in payload
    assert "reasoning_effort" not in payload


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


def test_anthropic_messages_stream_payload_round_trips_tool_error_state() -> None:
    """A failed tool result keeps is_error on the Anthropic wire only when set."""
    request = GatewayRequest(
        surface=GatewayApiSurface.MESSAGES,
        messages=(
            GatewayMessage(role="tool", content="boom", tool_call_id="call-1", tool_is_error=True),
            GatewayMessage(role="tool", content="fine", tool_call_id="call-2"),
        ),
        stream=True,
        include_usage=True,
    )
    payload = anthropic_messages_stream_payload("exact-model", request)
    messages = cast("list[dict[str, object]]", payload["messages"])
    blocks = cast("list[dict[str, object]]", messages[0]["content"])
    assert blocks[0] == {
        "type": "tool_result",
        "tool_use_id": "call-1",
        "content": "boom",
        "is_error": True,
    }
    # An ordinary result stays byte-identical to the pre-existing payload shape.
    assert blocks[1] == {"type": "tool_result", "tool_use_id": "call-2", "content": "fine"}


def test_gemini_stream_payload_matches_the_provider_client_builder() -> None:
    """The gemini dialect payload is the exact generateContent body the python
    provider client sends, built through the same shared converter."""
    request = _chat_request(temperature=0.3)
    payload = gemini_generate_content_stream_payload("gemini-2.5-pro", request)

    assert payload == gemini_generate_request("gemini-2.5-pro", model_request(request))
    assert payload["contents"] == [{"role": "user", "parts": [{"text": "hello"}]}]
    generation = cast("dict[str, object]", payload["generationConfig"])
    assert isinstance(generation, dict)
    assert generation["temperature"] == 0.3
    # Streaming is selected by the streamGenerateContent route, not the body.
    assert "stream" not in payload


def test_gemini_generation_forwards_top_p_and_top_k_only_when_declared() -> None:
    """Gemini's camel-case sampling names are gated independently."""
    request = _chat_request().model_copy(update={"top_p": 0.8, "top_k": 20})
    payload = gemini_generate_content_stream_payload(
        "gemini-2.5-pro",
        request,
        supports_top_k=True,
    )
    generation = cast("dict[str, object]", payload["generationConfig"])
    assert isinstance(generation, dict)
    assert generation["topP"] == 0.8
    assert generation["topK"] == 20


def test_openai_responses_stream_payload_ignores_logprobs_even_when_flagged() -> None:
    """Responses logprob controls are accepted but not sent without output projection."""
    request = _chat_request().model_copy(update={"top_logprobs": 5})
    payload = openai_responses_stream_payload(
        "exact-model",
        request,
        supports_temperature=True,
        supports_logprobs=True,
    )

    assert "top_logprobs" not in payload


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
