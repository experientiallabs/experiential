"""Tests for shared Chat Completions and Responses request decoding."""

from __future__ import annotations

import base64
from collections.abc import Callable
from typing import cast

import pytest

from exp.common.core.artifacts import JsonObject, sha256_json
from exp.common.models.content import (
    GEMINI_FILE_URI_PREFIX,
    MAXIMUM_DOCUMENTS_PER_REQUEST,
    MediaHandle,
    VideoContentPart,
)
from exp.runtime.gateway.contracts import (
    EncryptedReasoningBlock,
    GatewayApiSurface,
    GatewayMessage,
    GatewayNamedToolChoice,
    GatewayRequest,
)
from exp.runtime.gateway.reasoning_carrier import FIREWORKS_REASONING_CONTENT_PREFIX
from exp.runtime.models.providers.streaming_requests import openai_responses_stream_payload
from exp.runtime.openai_protocol.errors import OpenAIProtocolError
from exp.runtime.openai_protocol.model_adapter import model_request
from exp.runtime.openai_protocol.requests import (
    DecodedGatewayRequest,
    decode_chat,
    decode_embeddings,
    decode_responses,
)


def test_chat_decoder_preserves_every_supported_semantic_field() -> None:
    """Chat conversion retains roles, raw tools, strict schema, controls, usage, and metadata."""
    decoded = decode_chat(
        {
            "model": "coding",
            "messages": [
                {"role": "developer", "content": "Follow policy."},
                {"role": "user", "content": [{"type": "text", "text": "Call weather."}]},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-one",
                            "type": "function",
                            "function": {
                                "name": "weather",
                                "arguments": '{ "city" : "Zürich" }',
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call-one", "content": "sunny"},
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "weather",
                        "description": "Read weather",
                        "parameters": {"type": "object"},
                        "strict": True,
                    },
                }
            ],
            "tool_choice": {"type": "function", "function": {"name": "weather"}},
            "parallel_tool_calls": True,
            "max_completion_tokens": 123,
            "stop": ["END", "STOP"],
            "temperature": 0.2,
            "top_p": 1,
            "reasoning_effort": "high",
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "answer",
                    "schema": {"type": "object"},
                    "strict": True,
                },
            },
            "stream": True,
            "stream_options": {"include_usage": True},
            "metadata": {"cohort": "test"},
        },
        idempotency_key="operation-one",
        client_request_id="operation-one",
    )

    request = decoded.request
    assert decoded.alias == "coding"
    assert request.surface == GatewayApiSurface.CHAT_COMPLETIONS
    assert tuple(message.role for message in request.messages) == (
        "developer",
        "user",
        "assistant",
        "tool",
    )
    assert request.messages[2].tool_calls[0].raw_arguments == '{ "city" : "Zürich" }'
    assert request.tools[0].strict
    assert isinstance(request.tool_choice, GatewayNamedToolChoice)
    assert request.tool_choice.name == "weather"
    assert request.maximum_output_tokens == 123
    assert request.maximum_output_tokens_parameter == "max_completion_tokens"
    assert request.stop == ("END", "STOP")
    assert request.temperature == 0.2
    assert request.top_p == 1.0
    assert request.reasoning_effort == "high"
    assert request.structured_text is not None and request.structured_text.strict
    assert request.include_usage
    assert request.metadata == {"cohort": "test"}


def test_chat_legacy_max_tokens_reaches_native_responses_as_max_output_tokens() -> None:
    """A Chat request using legacy max_tokens serves a native Responses max_output_tokens.

    Chat clients (playground and agents) commonly send the legacy max_tokens field. On a
    direct OpenAI deployment the native Responses API rejects max_tokens and wants
    max_output_tokens, so the canonical request must translate the field and the native
    payload must never carry max_tokens.
    """
    decoded = decode_chat(
        {
            "model": "coding",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 256,
            "stream": True,
        }
    )

    assert decoded.request.maximum_output_tokens == 256
    assert decoded.request.maximum_output_tokens_parameter == "max_tokens"
    payload = openai_responses_stream_payload(
        "gpt-fixture",
        decoded.request,
        supports_temperature=True,
        reasoning_effort=None,
    )
    assert payload["max_output_tokens"] == 256
    assert "max_tokens" not in payload


def test_responses_decoder_preserves_continuation_and_distinct_wire_shapes() -> None:
    """Responses conversion keeps instructions, item history, named tools, and structured text."""
    decoded = decode_responses(
        {
            "model": "coding",
            "instructions": "Use tools.",
            "input": [
                {"type": "message", "role": "user", "content": "Weather?"},
                {
                    "type": "function_call",
                    "call_id": "call-one",
                    "name": "weather",
                    "arguments": '{"city":"Paris"}',
                },
                {"type": "function_call_output", "call_id": "call-one", "output": "sunny"},
            ],
            "previous_response_id": "resp_previous",
            "tools": [
                {
                    "type": "function",
                    "name": "weather",
                    "parameters": {"type": "object"},
                    "strict": True,
                }
            ],
            "tool_choice": {"type": "function", "name": "weather"},
            "parallel_tool_calls": False,
            "max_output_tokens": 321,
            "temperature": 0.4,
            "reasoning": {
                "effort": "high",
                "generate_summary": "concise",
                "summary": "concise",
            },
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "answer",
                    "schema": {"type": "object"},
                    "strict": True,
                }
            },
            "stream": True,
            "metadata": {"cohort": "test"},
        },
        client_request_id="operation-two",
    )

    request = decoded.request
    assert decoded.developer_messages_param == "instructions"
    assert request.surface == GatewayApiSurface.RESPONSES
    assert tuple(message.role for message in request.messages) == (
        "developer",
        "user",
        "assistant",
        "tool",
    )
    assert request.previous_response_id == "resp_previous"
    assert request.reasoning_effort == "high"
    assert request.reasoning_summary == "concise"
    assert request.reasoning_summary_parameters == (
        "reasoning.generate_summary",
        "reasoning.summary",
    )
    assert request.ignored_parameters == ()
    assert request.messages[2].tool_calls[0].raw_arguments == '{"city":"Paris"}'
    assert request.parallel_tool_calls is False
    assert request.maximum_output_tokens == 321
    assert request.maximum_output_tokens_parameter == "max_output_tokens"
    assert request.structured_text is not None
    assert request.client_request_id == "operation-two"


def test_responses_decoder_tracks_developer_input_origin() -> None:
    """Capability errors can identify an input developer role without inventing instructions."""
    decoded = decode_responses(
        {
            "model": "coding",
            "input": [
                {"type": "message", "role": "user", "content": "hello"},
                {"type": "message", "role": "developer", "content": "follow policy"},
            ],
        }
    )

    assert decoded.developer_messages_param == "input.1.role"


def test_responses_decoder_rejects_conflicting_reasoning_summary_aliases() -> None:
    """Current and deprecated summary selectors cannot request different outputs."""
    with pytest.raises(OpenAIProtocolError) as captured:
        decode_responses(
            {
                "model": "coding",
                "input": "hello",
                "reasoning": {"summary": "concise", "generate_summary": "detailed"},
            }
        )

    assert captured.value.detail.code == "invalid_parameter"


@pytest.mark.parametrize(
    ("decoder", "payload", "param"),
    (
        (
            decode_chat,
            {"model": "coding", "messages": [{"role": "user", "content": "x"}], "n": 2},
            "n",
        ),
        (
            decode_responses,
            {"model": "coding", "input": "x", "background": True},
            "background",
        ),
        (
            decode_chat,
            {"model": "coding", "messages": [{"role": "user", "content": "x"}], "future": 1},
            "future",
        ),
    ),
)
def test_unknown_and_excluded_fields_fail_with_exact_param(
    decoder: Callable[[JsonObject], DecodedGatewayRequest], payload: JsonObject, param: str
) -> None:
    """Closed manifests reject excluded and future SDK fields before dispatch."""
    with pytest.raises(OpenAIProtocolError) as captured:
        decoder(payload)
    assert captured.value.detail.code == "unsupported_parameter"
    assert captured.value.detail.param == param


def test_chat_decoder_accepts_echoed_assistant_message_with_empty_sdk_fields() -> None:
    """Assistant messages echoed verbatim from a prior gateway response must decode.

    The gateway's own Chat responses and official SDK message dumps carry
    refusal, annotations, audio, function_call, and a possibly null tool_calls
    key; a tool-call continuation sends that message back unchanged.
    """
    decoded = decode_chat(
        {
            "model": "coding",
            "messages": [
                {"role": "user", "content": "Weather in Paris?"},
                {
                    "role": "assistant",
                    "content": None,
                    "refusal": None,
                    "annotations": [],
                    "audio": None,
                    "function_call": None,
                    "tool_calls": [
                        {
                            "id": "call-one",
                            "type": "function",
                            "function": {"name": "weather", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call-one", "content": "sunny"},
                {
                    "role": "assistant",
                    "content": "It is sunny.",
                    "refusal": None,
                    "tool_calls": None,
                },
                {"role": "user", "content": "Thanks."},
            ],
        }
    )
    assert decoded.request.messages[1].tool_calls[0].name == "weather"
    assert decoded.request.messages[3].content == "It is sunny."
    assert decoded.request.messages[3].tool_calls == ()


def test_chat_decoder_preserves_only_a_gateway_issued_reasoning_carrier() -> None:
    """A Fireworks continuation stays encrypted until authorized admission."""
    deployment = base64.urlsafe_b64encode(b"fireworks-rung").rstrip(b"=").decode()
    envelope = base64.urlsafe_b64encode(b"x" * 32).rstrip(b"=").decode()
    carrier = f"{FIREWORKS_REASONING_CONTENT_PREFIX}{deployment}:{envelope}"
    decoded = decode_chat(
        {
            "model": "coding",
            "messages": [
                {"role": "user", "content": "Use a tool"},
                {
                    "role": "assistant",
                    "content": None,
                    "reasoning_content": carrier,
                    "tool_calls": [
                        {
                            "id": "call-one",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call-one", "content": "done"},
            ],
        }
    )

    block = decoded.request.messages[1].provider_reasoning[0]
    assert block.kind == "sealed_reasoning_content"
    assert block.carrier == carrier
    assert block.deployment_hint == "fireworks-rung"
    adapted = model_request(decoded.request)
    assert adapted.messages[1].assistant_action is not None
    assert "provider_reasoning" not in adapted.messages[1].assistant_action.model_dump()


def test_chat_decoder_rejects_duplicate_assistant_tool_call_ids() -> None:
    """Two active calls cannot share the result-linkage identity."""
    with pytest.raises(OpenAIProtocolError) as captured:
        decode_chat(
            {
                "model": "coding",
                "messages": [
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call-one",
                                "type": "function",
                                "function": {"name": "first", "arguments": "{}"},
                            },
                            {
                                "id": "call-one",
                                "type": "function",
                                "function": {"name": "second", "arguments": "{}"},
                            },
                        ],
                    }
                ],
            }
        )

    assert captured.value.detail.param == "messages.0"


@pytest.mark.parametrize(
    "reasoning_content",
    (
        "raw provider reasoning",
        FIREWORKS_REASONING_CONTENT_PREFIX,
        f"{FIREWORKS_REASONING_CONTENT_PREFIX}not-base64:payload",
    ),
)
def test_chat_decoder_rejects_unbound_or_malformed_reasoning_content(
    reasoning_content: str,
) -> None:
    """Public Chat input accepts only a bounded gateway-issued carrier."""
    with pytest.raises(OpenAIProtocolError) as raised:
        decode_chat(
            {
                "model": "coding",
                "messages": [
                    {
                        "role": "assistant",
                        "content": "x",
                        "reasoning_content": reasoning_content,
                    }
                ],
            }
        )
    assert raised.value.detail.param == "messages.0.reasoning_content"


def test_chat_decoder_still_rejects_populated_unsupported_message_fields() -> None:
    """A populated refusal or annotation in request history stays rejected."""
    for extra in ({"refusal": "no"}, {"annotations": [{"type": "url_citation"}]}):
        with pytest.raises(OpenAIProtocolError) as captured:
            decode_chat(
                {
                    "model": "coding",
                    "messages": [{"role": "assistant", "content": "x", **extra}],
                }
            )
        param = captured.value.detail.param
        assert param is not None and param.startswith("messages.0.")


def test_invalid_tool_arguments_and_divergent_operation_headers_are_specific() -> None:
    """Malformed history names its field; independent identity headers both decode."""
    with pytest.raises(OpenAIProtocolError) as arguments:
        decode_chat(
            {
                "model": "coding",
                "messages": [
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call-one",
                                "type": "function",
                                "function": {"name": "tool", "arguments": "{"},
                            }
                        ],
                    }
                ],
            }
        )
    assert arguments.value.detail.param == "messages.0.tool_calls.0.function.arguments"

    # Idempotency-Key names one retriable operation; X-Client-Request-Id is
    # session correlation identity (Codex sends its session id on every
    # request of a session), so divergent values decode side by side.
    decoded = decode_responses(
        {"model": "coding", "input": "x"},
        idempotency_key="one",
        client_request_id="two",
    )
    assert decoded.request.idempotency_key == "one"
    assert decoded.request.client_request_id == "two"


def _assert_no_cache_control(decoded: DecodedGatewayRequest) -> None:
    """Require cache_control to be absent from canonical and provider-bound messages."""
    for message in decoded.request.messages:
        assert "cache_control" not in message.model_dump(mode="json")
    adapted = model_request(decoded.request)
    for message in adapted.messages:
        assert "cache_control" not in message.model_dump(mode="json")


def test_chat_decoder_drops_opencode_message_cache_control() -> None:
    """OpenCode Chat Completions annotate messages with Anthropic cache_control.

    The live failure is Invalid value for 'messages.0.cache_control' because the
    closed Chat wire model forbids that nested field before routing. Supported
    ephemeral forms must decode and never reach the canonical or provider-bound
    message.
    """
    decoded = decode_chat(
        {
            "model": "coding",
            "messages": [
                {
                    "role": "system",
                    "content": "You are concise.",
                    "cache_control": {"type": "ephemeral"},
                },
                {
                    "role": "user",
                    "content": "hello",
                    "cache_control": {"type": "ephemeral", "ttl": "5m"},
                },
                {
                    "role": "assistant",
                    "content": "hi",
                    "cache_control": {"type": "ephemeral", "ttl": "1h"},
                },
                {"role": "user", "content": "again", "cache_control": None},
            ],
            "top_p": 1,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    )

    assert tuple(message.content for message in decoded.request.messages) == (
        "You are concise.",
        "hello",
        "hi",
        "again",
    )
    assert decoded.request.top_p == 1.0
    assert decoded.request.stream
    assert decoded.request.include_usage
    _assert_no_cache_control(decoded)


def test_chat_decoder_drops_opencode_text_part_cache_control() -> None:
    """OpenCode openai-compatible conversion can put cache_control on text parts.

    applyCaching marks the last content part, and @ai-sdk/openai-compatible
    keeps that annotation on the text block when the user message has more
    than one part.
    """
    decoded = decode_chat(
        {
            "model": "coding",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "prefix "},
                        {
                            "type": "text",
                            "text": "cached suffix",
                            "cache_control": {"type": "ephemeral"},
                        },
                    ],
                }
            ],
        }
    )

    assert decoded.request.messages[0].content == "prefix cached suffix"
    _assert_no_cache_control(decoded)


@pytest.mark.parametrize(
    "cache_control",
    (
        {"type": "persistent"},
        {"type": "ephemeral", "ttl": "2h"},
        {"type": "ephemeral", "ttl": None},
        {"type": "ephemeral", "extra": True},
        "ephemeral",
        1,
        [],
    ),
)
def test_chat_decoder_rejects_malformed_message_cache_control(cache_control: object) -> None:
    """Unsupported cache_control shapes fail at messages.<index>.cache_control."""
    with pytest.raises(OpenAIProtocolError) as captured:
        decode_chat(
            {
                "model": "coding",
                "messages": [{"role": "user", "content": "hello", "cache_control": cache_control}],
            }
        )
    assert captured.value.detail.code == "invalid_parameter"
    assert captured.value.detail.param == "messages.0.cache_control"
    assert captured.value.detail.message == "Invalid value for 'messages.0.cache_control'."


def test_chat_decoder_rejects_malformed_text_part_cache_control() -> None:
    """Unsupported text-part cache_control stays a field-specific invalid value."""
    with pytest.raises(OpenAIProtocolError) as captured:
        decode_chat(
            {
                "model": "coding",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "hello",
                                "cache_control": {"type": "persistent"},
                            }
                        ],
                    }
                ],
            }
        )
    assert captured.value.detail.code == "invalid_parameter"
    assert captured.value.detail.param == "messages.0.content.0.cache_control"


def test_chat_decoder_still_rejects_unknown_nested_message_fields() -> None:
    """Dropping cache_control must not weaken unrelated unknown nested fields."""
    with pytest.raises(OpenAIProtocolError) as captured:
        decode_chat(
            {
                "model": "coding",
                "messages": [
                    {
                        "role": "user",
                        "content": "hello",
                        "cache_control": {"type": "ephemeral"},
                        "providerOptions": {},
                    }
                ],
            }
        )
    param = captured.value.detail.param
    assert param == "messages.0.providerOptions"


def test_chat_decoder_still_rejects_unknown_text_part_fields_with_cache_control() -> None:
    """A valid text-part cache_control drop still rejects other extra part keys."""
    with pytest.raises(OpenAIProtocolError) as captured:
        decode_chat(
            {
                "model": "coding",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "hello",
                                "cache_control": {"type": "ephemeral"},
                                "future": 1,
                            }
                        ],
                    }
                ],
            }
        )
    param = captured.value.detail.param
    assert param is not None and param.startswith("messages.0.content")


def test_chat_decoder_accepts_opencode_nucleus_and_usage_stream_shape() -> None:
    """OpenCode Chat Completions send top_p=1 with streamed usage and must decode losslessly."""
    decoded = decode_chat(
        {
            "model": "coding",
            "messages": [{"role": "user", "content": "hello"}],
            "top_p": 1,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    )

    assert decoded.request.top_p == 1.0
    assert decoded.request.stream
    assert decoded.request.include_usage


def test_chat_decoder_rejects_out_of_range_top_p() -> None:
    """Nucleus sampling stays inside the official [0, 1] interval."""
    with pytest.raises(OpenAIProtocolError) as captured:
        decode_chat(
            {
                "model": "coding",
                "messages": [{"role": "user", "content": "hello"}],
                "top_p": 1.5,
            }
        )
    assert captured.value.detail.code == "invalid_parameter"
    assert captured.value.detail.param == "top_p"


def test_responses_decoder_preserves_top_p() -> None:
    """Responses accepts the provider-supported nucleus-sampling control."""
    decoded = decode_responses({"model": "coding", "input": "hello", "top_p": 1})
    assert decoded.request.top_p == 1


def test_chat_decoder_rejects_unprojectable_top_logprobs() -> None:
    """Alternate-token probability output is rejected at the public boundary."""
    with pytest.raises(OpenAIProtocolError) as raised:
        decode_chat(
            {
                "model": "coding",
                "messages": [{"role": "user", "content": "hello"}],
                "logprobs": True,
                "top_logprobs": 5,
            }
        )
    assert raised.value.detail.code == "unsupported_parameter"
    assert raised.value.detail.param == "top_logprobs"


def test_chat_decoder_preserves_logprobs_for_route_validation() -> None:
    """The route gate distinguishes a semantic true request from a false no-op."""
    for value in (True, False):
        decoded = decode_chat(
            {
                "model": "coding",
                "messages": [{"role": "user", "content": "hello"}],
                "logprobs": value,
            }
        )
        assert decoded.request.logprobs is value


def test_chat_decoder_preserves_gateway_top_k_extension() -> None:
    """The provider-neutral top-k extension survives official SDK validation."""
    decoded = decode_chat(
        {
            "model": "coding",
            "messages": [{"role": "user", "content": "hello"}],
            "top_k": 40,
        }
    )
    assert decoded.request.top_k == 40


def test_empty_responses_input_is_a_public_protocol_error() -> None:
    """Canonical validation failures do not leak internal Pydantic exceptions."""
    with pytest.raises(OpenAIProtocolError) as captured:
        decode_responses({"model": "coding", "input": []})
    assert captured.value.detail.code == "invalid_parameter"
    assert captured.value.detail.param == "messages"


def test_chat_decoder_captures_end_user_attribution_and_cache_hint() -> None:
    """safety_identifier/user/prompt_cache_key are captured; the label prefers safety_identifier."""
    decoded = decode_chat(
        {
            "model": "coding",
            "messages": [{"role": "user", "content": "hi"}],
            "safety_identifier": "sha256:abc",
            "user": "legacy-user",
            "prompt_cache_key": "prompt_v1:sess",
        }
    )
    request = decoded.request
    assert request.safety_identifier == "sha256:abc"
    assert request.user == "legacy-user"
    assert request.prompt_cache_key == "prompt_v1:sess"
    # The current safety_identifier wins over the deprecated user for attribution.
    assert request.attribution_label == "sha256:abc"


def test_chat_attribution_label_falls_back_to_deprecated_user() -> None:
    """With no safety_identifier, the deprecated user field still labels the request."""
    decoded = decode_chat(
        {"model": "coding", "messages": [{"role": "user", "content": "hi"}], "user": "u-1"}
    )
    assert decoded.request.safety_identifier is None
    assert decoded.request.attribution_label == "u-1"


def test_prompt_cache_key_is_never_an_attribution_label() -> None:
    """prompt_cache_key is a cache-routing hint, not an end-user identity."""
    decoded = decode_chat(
        {
            "model": "coding",
            "messages": [{"role": "user", "content": "hi"}],
            "prompt_cache_key": "pck",
        }
    )
    assert decoded.request.prompt_cache_key == "pck"
    assert decoded.request.attribution_label is None


def test_responses_decoder_captures_end_user_attribution() -> None:
    """The Responses surface captures the same attribution/cache fields as Chat."""
    decoded = decode_responses(
        {
            "model": "coding",
            "input": "hi",
            "safety_identifier": "sid-9",
            "prompt_cache_key": "pck-1",
        }
    )
    request = decoded.request
    assert request.safety_identifier == "sid-9"
    assert request.prompt_cache_key == "pck-1"
    assert request.attribution_label == "sid-9"


def test_responses_decoder_accepts_the_codex_request_shape() -> None:
    """store:false, include, ultra effort, and replayed reasoning all decode."""
    decoded = decode_responses(
        {
            "model": "coding",
            "store": False,
            "include": ["reasoning.encrypted_content"],
            "reasoning": {"effort": "ultra", "summary": "auto"},
            "prompt_cache_key": "codex-session-1",
            "input": [
                {"type": "message", "role": "user", "content": "run the tool"},
                {
                    "type": "reasoning",
                    "id": "rs_1",
                    "summary": [{"type": "summary_text", "text": "planned"}],
                    "encrypted_content": "blob==",
                },
                {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "call-1",
                    "name": "search",
                    "arguments": "{}",
                },
                {
                    "type": "reasoning",
                    "id": "rs_2",
                    "summary": [],
                    "encrypted_content": "second-blob==",
                },
                {"type": "function_call_output", "call_id": "call-1", "output": "found"},
            ],
        }
    )
    request = decoded.request
    assert request.response_store is False
    assert request.include_encrypted_reasoning is True
    assert request.reasoning_effort == "ultra"
    assert request.reasoning_summary == "auto"
    assert request.prompt_cache_key == "codex-session-1"
    assistant = request.messages[1]
    assert assistant.role == "assistant"
    assert assistant.tool_calls[0].call_id == "call-1"
    blocks = assistant.provider_reasoning
    assert len(blocks) == 1
    assert blocks[0].kind == "encrypted_reasoning"
    assert blocks[0].id == "rs_1"
    assert blocks[0].encrypted_content == "blob=="
    trailing_block = request.messages[2].provider_reasoning[0]
    assert trailing_block.kind == "encrypted_reasoning"
    assert trailing_block.id == "rs_2"
    assert trailing_block.encrypted_content == "second-blob=="
    payload = openai_responses_stream_payload(
        "gpt-fixture",
        request,
        supports_temperature=False,
        supports_reasoning=True,
    )
    payload_input = cast(list[JsonObject], payload["input"])
    assert payload_input[1:] == [
        {
            "type": "reasoning",
            "summary": [],
            "encrypted_content": "blob==",
        },
        {
            "type": "function_call",
            "id": "fc_1",
            "call_id": "call-1",
            "name": "search",
            "arguments": "{}",
        },
        {
            "type": "reasoning",
            "summary": [],
            "encrypted_content": "second-blob==",
        },
        {"type": "function_call_output", "call_id": "call-1", "output": "found"},
    ]


def test_responses_decoder_rejects_reasoning_without_item_id() -> None:
    """Opaque reasoning replay requires the provider-issued item identity."""
    with pytest.raises(OpenAIProtocolError) as raised:
        decode_responses(
            {
                "model": "coding",
                "input": [{"type": "reasoning", "summary": [], "encrypted_content": "blob=="}],
            }
        )

    assert raised.value.status_code == 400
    assert raised.value.detail.param == "input.0.id"


def test_responses_decoder_replays_output_message_in_provider_order() -> None:
    """A stateless output transcript keeps reasoning, message, and call order."""
    decoded = decode_responses(
        {
            "model": "coding",
            "input": [
                {
                    "type": "reasoning",
                    "id": "rs_0",
                    "status": "completed",
                    "summary": [],
                    "encrypted_content": "opaque",
                },
                {
                    "type": "message",
                    "id": "msg_1",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "I will look that up.",
                            "annotations": [],
                            "logprobs": [],
                        }
                    ],
                },
                {
                    "type": "function_call",
                    "id": "fc_2",
                    "call_id": "call-2",
                    "name": "lookup",
                    "arguments": '{ "q" : "x" }',
                    "status": "completed",
                },
                {"type": "function_call_output", "call_id": "call-2", "output": "found"},
            ],
        }
    )
    payload = openai_responses_stream_payload(
        "gpt-fixture",
        decoded.request,
        supports_temperature=False,
        supports_reasoning=True,
    )
    payload_input = cast(list[JsonObject], payload["input"])
    assert [(item["type"], item.get("id")) for item in payload_input[:3]] == [
        ("reasoning", None),
        ("message", "msg_1"),
        ("function_call", "fc_2"),
    ]
    message_content = cast(list[JsonObject], payload_input[1]["content"])
    assert message_content[0]["text"] == "I will look that up."
    # A replayed reasoning item never carries status (the provider rejects
    # it: "Unknown parameter", verified live 2026-08-29); other item types
    # keep it.
    assert "status" not in payload_input[0]
    assert payload_input[2]["arguments"] == '{ "q" : "x" }'
    assert payload_input[2]["status"] == "completed"


def test_responses_decoder_preserves_multiple_official_output_message_phases() -> None:
    """Official SDK output messages keep distinct identity, status, phase, and order."""
    from openai.types.responses.response_output_message import ResponseOutputMessage

    commentary = ResponseOutputMessage.model_validate(
        {
            "type": "message",
            "id": "msg_commentary",
            "role": "assistant",
            "status": "incomplete",
            "phase": "commentary",
            "content": [
                {
                    "type": "output_text",
                    "text": "I am checking.",
                    "annotations": [],
                }
            ],
        }
    ).model_dump(mode="json", exclude_none=True)
    final = ResponseOutputMessage.model_validate(
        {
            "type": "message",
            "id": "msg_final",
            "role": "assistant",
            "status": "completed",
            "phase": "final_answer",
            "content": [
                {
                    "type": "output_text",
                    "text": "Done.",
                    "annotations": [],
                }
            ],
        }
    ).model_dump(mode="json", exclude_none=True)

    decoded = decode_responses(
        {
            "model": "coding",
            "input": [
                commentary,
                {
                    "type": "function_call",
                    "call_id": "call-between",
                    "name": "lookup",
                    "arguments": "{}",
                    "status": "incomplete",
                },
                final,
            ],
        }
    )

    assistant_messages = tuple(
        message for message in decoded.request.messages if message.role == "assistant"
    )
    assert [message.provider_item_id for message in assistant_messages] == [
        "msg_commentary",
        None,
        "msg_final",
    ]
    assert [message.provider_phase for message in assistant_messages] == [
        "commentary",
        None,
        "final_answer",
    ]
    call = assistant_messages[1].tool_calls[0]
    assert call.call_id == "call-between"
    assert call.provider_item_id is None
    assert call.provider_output_index == 1
    assert call.provider_status == "incomplete"

    payload = openai_responses_stream_payload(
        "gpt-fixture",
        decoded.request,
        supports_temperature=False,
        supports_reasoning=True,
    )
    replay = cast(list[JsonObject], payload["input"])
    assert [(item["type"], item.get("id")) for item in replay] == [
        ("message", "msg_commentary"),
        ("function_call", None),
        ("message", "msg_final"),
    ]
    assert replay[0]["phase"] == "commentary"
    assert replay[0]["status"] == "incomplete"
    assert replay[1]["call_id"] == "call-between"
    assert replay[1]["status"] == "incomplete"
    assert replay[2]["phase"] == "final_answer"


@pytest.mark.parametrize(
    ("item", "param"),
    (
        (
            {"type": "reasoning", "id": "rs_1", "summary": []},
            "input.0.encrypted_content",
        ),
        (
            {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_1",
                "name": "",
                "arguments": "{}",
            },
            "input.0.name",
        ),
    ),
)
def test_responses_decoder_reports_the_specific_malformed_output_item_field(
    item: JsonObject,
    param: str,
) -> None:
    """Union validation never leaks implementation branches such as input.str."""
    with pytest.raises(OpenAIProtocolError) as raised:
        decode_responses({"model": "coding", "input": [item]})

    assert raised.value.status_code == 400
    assert raised.value.detail.param == param


def test_responses_decoder_orders_function_calls_without_optional_item_ids() -> None:
    """A legacy ID-less call keeps its provider output index beside identified calls."""
    decoded = decode_responses(
        {
            "model": "coding",
            "input": [
                {
                    "type": "reasoning",
                    "id": "rs_0",
                    "summary": [],
                    "encrypted_content": "opaque",
                },
                {
                    "type": "function_call",
                    "call_id": "call-1",
                    "name": "first",
                    "arguments": '{ "position" : 1 }',
                },
                {
                    "type": "function_call",
                    "id": "fc_2",
                    "call_id": "call-2",
                    "name": "second",
                    "arguments": '{"position":2}',
                },
            ],
        }
    )
    calls = decoded.request.messages[0].tool_calls
    assert [(call.provider_item_id, call.provider_output_index) for call in calls] == [
        (None, 1),
        ("fc_2", 2),
    ]
    payload = openai_responses_stream_payload(
        "gpt-fixture",
        decoded.request,
        supports_temperature=False,
        supports_reasoning=True,
    )
    payload_input = cast(list[JsonObject], payload["input"])
    assert [(item["type"], item.get("id")) for item in payload_input] == [
        # The reasoning id never crosses upstream: encrypted_content is
        # cryptographically bound to the provider's original item id, and an
        # id-less item verifies against the id embedded in the payload.
        ("reasoning", None),
        ("function_call", None),
        ("function_call", "fc_2"),
    ]
    assert [item["arguments"] for item in payload_input[1:]] == [
        '{ "position" : 1 }',
        '{"position":2}',
    ]


@pytest.mark.parametrize("reasoning_first", [True, False])
def test_responses_decoder_groups_fireworks_carrier_with_all_tool_calls(
    reasoning_first: bool,
) -> None:
    """One carrier and contiguous calls reconstruct their exact assistant turn in any order."""
    deployment = base64.urlsafe_b64encode(b"fireworks-rung").rstrip(b"=").decode()
    envelope = base64.urlsafe_b64encode(b"x" * 32).rstrip(b"=").decode()
    carrier = f"{FIREWORKS_REASONING_CONTENT_PREFIX}{deployment}:{envelope}"

    reasoning = {
        "type": "reasoning",
        "id": "rs_fireworks",
        "summary": [],
        "encrypted_content": carrier,
    }
    message = {
        "type": "message",
        "role": "assistant",
        "content": "I will check.",
    }
    assistant_items = [reasoning, message] if reasoning_first else [message, reasoning]
    decoded = decode_responses(
        {
            "model": "coding",
            "store": False,
            "include": ["reasoning.encrypted_content"],
            "input": [
                *assistant_items,
                {
                    "type": "function_call",
                    "call_id": "call-1",
                    "name": "first",
                    "arguments": "{}",
                },
                {
                    "type": "function_call",
                    "call_id": "call-2",
                    "name": "second",
                    "arguments": '{"value":2}',
                },
                {"type": "function_call_output", "call_id": "call-1", "output": "one"},
                {"type": "function_call_output", "call_id": "call-2", "output": "two"},
            ],
        }
    )

    assistant = decoded.request.messages[0]
    assert assistant.content == "I will check."
    assert tuple(call.call_id for call in assistant.tool_calls) == ("call-1", "call-2")
    assert assistant.provider_reasoning[0].kind == "sealed_reasoning_content"
    assert assistant.provider_reasoning[0].carrier == carrier


def test_responses_decoder_rejects_malformed_gateway_carrier() -> None:
    """A carrier-prefixed item cannot fall back to native opaque replay."""
    with pytest.raises(OpenAIProtocolError) as raised:
        decode_responses(
            {
                "model": "coding",
                "input": [
                    {
                        "type": "reasoning",
                        "id": "rs_malformed",
                        "summary": [],
                        "encrypted_content": f"{FIREWORKS_REASONING_CONTENT_PREFIX}broken",
                    }
                ],
            }
        )

    assert raised.value.detail.param == "input.0.encrypted_content"


def test_responses_decoder_keeps_orphaned_reasoning_as_its_own_turn() -> None:
    """Trailing reasoning with no assistant successor stays a standalone turn."""
    decoded = decode_responses(
        {
            "model": "coding",
            "input": [
                {"type": "message", "role": "user", "content": "go"},
                {
                    "type": "reasoning",
                    "id": "rs_1",
                    "summary": [],
                    "encrypted_content": "blob==",
                },
            ],
        }
    )
    trailing = decoded.request.messages[-1]
    assert trailing.role == "assistant"
    assert trailing.content is None
    block = trailing.provider_reasoning[0]
    assert block.kind == "encrypted_reasoning"
    assert block.encrypted_content == "blob=="


def test_responses_decoder_rejects_unknown_include_paths() -> None:
    """Only the encrypted reasoning include selector is honored."""
    with pytest.raises(OpenAIProtocolError) as raised:
        decode_responses(
            {
                "model": "coding",
                "include": ["message.output_text.logprobs"],
                "input": "hi",
            }
        )
    assert raised.value.detail.param == "include"

    with pytest.raises(OpenAIProtocolError):
        decode_responses(
            {
                "model": "coding",
                "input": [{"type": "reasoning", "id": "rs_1", "summary": []}],
            }
        )


def test_chat_decoder_accepts_the_ultra_reasoning_effort() -> None:
    """The wire model owns effort validation ahead of the installed SDK literal."""
    decoded = decode_chat(
        {
            "model": "coding",
            "messages": [{"role": "user", "content": "hi"}],
            "reasoning_effort": "ultra",
        }
    )
    assert decoded.request.reasoning_effort == "ultra"

    with pytest.raises(OpenAIProtocolError) as raised:
        decode_chat(
            {
                "model": "coding",
                "messages": [{"role": "user", "content": "hi"}],
                "reasoning_effort": "extreme",
            }
        )
    assert raised.value.detail.param == "reasoning_effort"


def test_responses_decoder_accepts_reasoning_context_and_names_rejections() -> None:
    """reasoning.context decodes verbatim; unknown values and fields 400 by name.

    Customer repro: pydantic_ai's OpenAIResponsesModel sends
    reasoning={"effort": ..., "context": "all_turns"} on the gpt-5.6 family
    and OpenAI-direct accepts it, so the gateway must too.
    """
    for value in ("auto", "current_turn", "all_turns"):
        decoded = decode_responses(
            {
                "model": "coding",
                "input": "hi",
                "reasoning": {"effort": "high", "context": value},
            }
        )
        assert decoded.request.reasoning_context == value
    assert decode_responses({"model": "coding", "input": "hi"}).request.reasoning_context is None

    with pytest.raises(OpenAIProtocolError) as invalid_value:
        decode_responses(
            {
                "model": "coding",
                "input": "hi",
                "reasoning": {"context": "every_turn"},
            }
        )
    assert invalid_value.value.detail.param == "reasoning.context"

    # The consciously rejected reasoning.mode field keeps its named 400.
    with pytest.raises(OpenAIProtocolError) as rejected_field:
        decode_responses(
            {
                "model": "coding",
                "input": "hi",
                "reasoning": {"mode": "pro"},
            }
        )
    assert rejected_field.value.detail.param == "reasoning.mode"


def test_responses_decoder_accepts_verbatim_echoes_of_prior_output_items() -> None:
    """Turn-2 input echoing turn-1 output items verbatim must decode.

    Customer repro (Codex continuation model, 2026-08-28): a function_call
    output item carries {arguments, call_id, id, name, status, type}; echoing
    it with its function_call_output failed 400 on the echo-only ``id`` and
    ``status`` markers. Every item shape below mirrors what this gateway (and
    OpenAI-direct) actually emit.
    """
    decoded = decode_responses(
        {
            "model": "coding",
            "input": [
                {"type": "message", "role": "user", "content": "Weather in Paris?"},
                {
                    "type": "reasoning",
                    "id": "rs_1",
                    "summary": [{"type": "summary_text", "text": "plan"}],
                    "content": [{"type": "reasoning_text", "text": "thinking"}],
                    "encrypted_content": "blob==",
                    "status": "completed",
                },
                {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "call-1",
                    "name": "get_weather",
                    "arguments": "{}",
                    "status": "completed",
                },
                {
                    "type": "function_call_output",
                    "id": "fco_1",
                    "call_id": "call-1",
                    "output": "sunny",
                    "status": "completed",
                },
                {
                    "type": "message",
                    "id": "msg_1",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "It is sunny.",
                            "annotations": [],
                            "logprobs": [],
                        }
                    ],
                },
                {"type": "message", "role": "user", "content": "And tomorrow?"},
            ],
        }
    )
    roles = [message.role for message in decoded.request.messages]
    assert roles == ["user", "assistant", "tool", "assistant", "user"]
    assistant_call = decoded.request.messages[1]
    assert assistant_call.tool_calls[0].call_id == "call-1"
    assert assistant_call.provider_reasoning[0].kind == "encrypted_reasoning"
    assert decoded.request.messages[3].content == "It is sunny."


def test_responses_union_errors_name_the_item_field_not_the_branch() -> None:
    """A bad echoed item names its field, never a union branch like input.str."""
    with pytest.raises(OpenAIProtocolError) as raised:
        decode_responses(
            {
                "model": "coding",
                "input": [
                    {"type": "message", "role": "user", "content": "hi"},
                    {
                        "type": "function_call",
                        "call_id": "call-1",
                        "name": "get_weather",
                        "arguments": "{}",
                        "caller": {"type": "direct"},
                    },
                ],
            }
        )
    assert raised.value.detail.param == "input.1.caller"


def test_every_chat_cache_control_placement_follows_its_classified_decision() -> None:
    """Each classified cache_control placement decodes per the manifest table.

    Customer repro (opencode, 2026-08-28): the @ai-sdk stack lands the hint
    inside a ``tool_calls`` entry when the message's last content part is a
    tool call, and the closed wire model 400ed every Claude-family multi-turn
    tool session with no client-side workaround.
    """
    from exp.runtime.openai_protocol.manifest import CHAT_CACHE_CONTROL_PLACEMENTS

    assert CHAT_CACHE_CONTROL_PLACEMENTS == {
        "messages": "validated_and_dropped",
        "messages.content": "validated_and_dropped",
        "messages.tool_calls": "validated_and_forwarded_to_anthropic_tool_use",
    }
    decoded = decode_chat(
        {
            "model": "coding",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "read both files",
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                    "cache_control": {"type": "ephemeral", "ttl": "5m"},
                },
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": "read_file", "arguments": '{"path":"a.txt"}'},
                        },
                        {
                            "id": "call-2",
                            "type": "function",
                            "function": {"name": "read_file", "arguments": '{"path":"b.txt"}'},
                            "cache_control": {"type": "ephemeral"},
                        },
                    ],
                },
                {"role": "tool", "tool_call_id": "call-1", "content": "a"},
                {"role": "tool", "tool_call_id": "call-2", "content": "b"},
            ],
        }
    )
    calls = decoded.request.messages[1].tool_calls
    assert calls[0].cache_control is None
    assert calls[1].cache_control == {"type": "ephemeral"}
    # Message- and part-level hints stay validated-and-dropped.
    assert "cache_control" not in decoded.request.messages[0].model_dump(mode="json")

    with pytest.raises(OpenAIProtocolError) as raised:
        decode_chat(
            {
                "model": "coding",
                "messages": [
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {"name": "read_file", "arguments": "{}"},
                                "cache_control": {"type": "persistent"},
                            }
                        ],
                    }
                ],
            }
        )
    assert "cache_control" in str(raised.value.detail.param)


def test_empty_tool_call_arguments_decode_as_the_canonical_empty_object() -> None:
    """A zero-argument echo with arguments '' decodes as {} on both surfaces.

    Mirrors the streaming completion seed: no provider wire accepts empty
    argument bytes, and the @ai-sdk stack normally sends "{}", so the empty
    string is normalized instead of 400ing the continuation.
    """
    chat = decode_chat(
        {
            "model": "coding",
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": "get_time", "arguments": ""},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call-1", "content": "noon"},
            ],
        }
    )
    call = chat.request.messages[0].tool_calls[0]
    assert call.arguments == {}
    assert call.raw_arguments == "{}"

    responses = decode_responses(
        {
            "model": "coding",
            "input": [
                {"type": "function_call", "call_id": "call-1", "name": "get_time", "arguments": ""},
                {"type": "function_call_output", "call_id": "call-1", "output": "noon"},
            ],
        }
    )
    assert responses.request.messages[0].tool_calls[0].raw_arguments == "{}"


def test_the_captured_codex_request_shape_decodes_losslessly() -> None:
    """Regression fixture: the field shapes real Codex (0.151.0) sends by
    default, trimmed from a live capture (2026-08-29). Native items carry
    byte-for-byte; non-assistant message ids are accepted and dropped;
    assistant echoes carry id+phase without status."""
    additional_tools = {
        "type": "additional_tools",
        "id": "at_fixture",
        "role": "developer",
        "tools": [
            {
                "type": "namespace",
                "name": "functions",
                "description": "",
                "tools": [
                    {"type": "custom", "name": "exec", "description": "Run JavaScript"},
                    {
                        "type": "function",
                        "name": "followup_task",
                        "description": "Send a follow-up task",
                        "parameters": {"type": "object", "properties": {}},
                    },
                ],
            }
        ],
    }
    custom_call = {
        "type": "custom_tool_call",
        "id": "ctc_fixture",
        "status": "completed",
        "call_id": "call_fixture",
        "name": "exec",
        "input": 'const r = await tools.exec_command({cmd:"ls"});',
    }
    custom_output = {
        "type": "custom_tool_call_output",
        "id": "ctco_fixture",
        "call_id": "call_fixture",
        "output": "[{'type': 'input_text', 'text': 'file_a.txt'}]",
    }
    decoded = decode_responses(
        {
            "model": "gpt-5.6-sol",
            "store": False,
            "stream": True,
            "include": ["reasoning.encrypted_content"],
            "reasoning": {"effort": "low", "context": "all_turns"},
            "text": {"verbosity": "low"},
            "tool_choice": "auto",
            "parallel_tool_calls": False,
            "prompt_cache_key": "session-fixture",
            "client_metadata": {"thread_id": "thread-fixture"},
            "input": [
                additional_tools,
                {
                    "type": "message",
                    "id": "msg_dev_fixture",
                    "role": "developer",
                    "content": [{"type": "input_text", "text": "You are Codex."}],
                },
                {
                    "type": "message",
                    "id": "msg_user_fixture",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Run ls."}],
                },
                {
                    "type": "message",
                    "id": "msg_echo_fixture",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Listing now."}],
                    "phase": "commentary",
                },
                custom_call,
                custom_output,
            ],
        }
    )
    request = decoded.request
    assert request.ignored_parameters == ()
    assert request.text_verbosity == "low"
    assert request.client_metadata == {"thread_id": "thread-fixture"}
    assert request.reasoning_effort == "low"
    roles = [message.role for message in request.messages]
    assert roles == ["developer", "developer", "user", "assistant", "assistant", "tool"]
    natives = [
        message.provider_native_item
        for message in request.messages
        if message.provider_native_item is not None
    ]
    assert natives == [additional_tools, custom_call, custom_output]
    # Non-assistant ids drop; the assistant echo retains identity with
    # status OPTIONAL.
    developer = request.messages[1]
    assert developer.provider_item_id is None
    echo = request.messages[3]
    assert echo.provider_item_id == "msg_echo_fixture"
    assert echo.provider_status is None
    assert echo.provider_phase == "commentary"


def test_the_captured_codex_reasoning_echo_with_null_content_decodes() -> None:
    """Regression fixture: the third request of a real Codex (0.151.0)
    session, trimmed from a live capture (2026-08-29). After a
    custom_tool_call round Codex echoes the reasoning output item with an
    explicit ``content: null``; the provider accepts that request, so the
    gateway must decode it instead of rejecting ``input.N.content``."""
    reasoning_echo = {
        "type": "reasoning",
        "id": "rs_fixture",
        "summary": [],
        "content": None,
        "encrypted_content": "gAAAAABfixture",
    }
    decoded = decode_responses(
        {
            "model": "gpt-5.6-sol",
            "store": False,
            "stream": True,
            "include": ["reasoning.encrypted_content"],
            "reasoning": {"effort": "max", "context": "all_turns"},
            "tool_choice": "auto",
            "parallel_tool_calls": False,
            "prompt_cache_key": "session-fixture",
            "client_metadata": {"thread_id": "thread-fixture"},
            "input": [
                {
                    "type": "message",
                    "id": "msg_user_fixture",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Read data.txt."}],
                },
                reasoning_echo,
                {
                    "type": "custom_tool_call",
                    "id": "ctc_fixture",
                    "status": "completed",
                    "call_id": "call_fixture",
                    "name": "exec",
                    "input": 'const r = await tools.exec_command({cmd:"cat data.txt"});',
                },
                {
                    "type": "custom_tool_call_output",
                    "id": "ctco_fixture",
                    "call_id": "call_fixture",
                    "output": [
                        {"type": "input_text", "text": "Script completed\n"},
                        {"type": "input_text", "text": "gateway test file\n"},
                    ],
                },
            ],
        }
    )
    request = decoded.request
    roles = [message.role for message in request.messages]
    assert roles == ["user", "assistant", "assistant", "tool"]
    carrier = request.messages[1].provider_reasoning
    assert len(carrier) == 1
    assert isinstance(carrier[0], EncryptedReasoningBlock)
    assert carrier[0].encrypted_content == "gAAAAABfixture"


def test_decode_errors_name_the_expected_shape_against_the_arriving_type() -> None:
    """Union rejections say what shape the field expected and what arrived,
    at type level only, matching the provider's own error style."""
    with pytest.raises(OpenAIProtocolError) as content:
        decode_responses(
            {
                "model": "coding",
                "input": [
                    {
                        "type": "reasoning",
                        "id": "rs_bad",
                        "encrypted_content": "gAAAAABfixture",
                        "content": 7,
                    }
                ],
            }
        )
    assert content.value.detail.param == "input.0.content"
    assert content.value.detail.message == (
        "Invalid value for 'input.0.content': expected an array, but got an integer instead."
    )

    with pytest.raises(OpenAIProtocolError) as summary:
        decode_responses(
            {
                "model": "coding",
                "input": [
                    {
                        "type": "reasoning",
                        "id": "rs_bad",
                        "encrypted_content": "gAAAAABfixture",
                        "summary": None,
                    }
                ],
            }
        )
    assert summary.value.detail.param == "input.0.summary"
    assert summary.value.detail.message == (
        "Invalid value for 'input.0.summary': expected an array, but got null instead."
    )


_PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8"
    "z8DwHwAFAAH/q842iQAAAABJRU5ErkJggg=="
)
"""One valid single-pixel PNG, base64 encoded."""


def test_chat_decoder_retains_image_parts_in_caller_order() -> None:
    """A chat image part is kept beside its text in the order the caller sent."""
    decoded = decode_chat(
        {
            "model": "coding",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "what is this"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{_PNG_BASE64}",
                                "detail": "high",
                            },
                        },
                        {"type": "text", "text": "be brief"},
                    ],
                }
            ],
        }
    )
    message = decoded.request.messages[0]
    assert message.content == "what is thisbe brief"
    assert [part.kind for part in message.content_parts] == ["text", "image", "text"]
    image = decoded.request.images[0]
    assert image.data == _PNG_BASE64
    assert image.media_type == "image/png"
    assert image.detail == "high"


def test_an_empty_text_part_beside_an_image_drops() -> None:
    """A client's empty text part never reaches a wire that rejects one."""
    decoded = decode_chat(
        {
            "model": "coding",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": ""},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{_PNG_BASE64}"},
                        },
                        {"type": "text", "text": "read it"},
                    ],
                }
            ],
        }
    )
    message = decoded.request.messages[0]
    assert message.content == "read it"
    assert [part.kind for part in message.content_parts] == ["image", "text"]


def test_responses_decoder_retains_input_image_parts() -> None:
    """A Responses ``input_image`` survives decoding as a canonical image."""
    decoded = decode_responses(
        {
            "model": "coding",
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "describe"},
                        {
                            "type": "input_image",
                            "image_url": "https://example.com/cat.png",
                            "detail": "auto",
                        },
                    ],
                }
            ],
        }
    )
    assert [part.kind for part in decoded.request.messages[0].content_parts] == ["text", "image"]
    assert decoded.request.images[0].url == "https://example.com/cat.png"


def test_text_only_chat_messages_keep_no_content_parts() -> None:
    """Text-only requests decode exactly as before, with no retained parts."""
    decoded = decode_chat(
        {
            "model": "coding",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        }
    )
    assert decoded.request.messages[0].content_parts == ()
    assert decoded.request.images == ()


def test_images_change_the_canonical_request_digest() -> None:
    """An image changes what the model is asked, so it changes replay identity."""
    text_only = decode_chat(
        {"model": "coding", "messages": [{"role": "user", "content": "what is this"}]}
    )
    with_image = decode_chat(
        {
            "model": "coding",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "what is this"},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{_PNG_BASE64}"},
                        },
                    ],
                }
            ],
        }
    )
    assert sha256_json(text_only.request) != sha256_json(with_image.request)


def test_malformed_chat_image_url_is_rejected_with_its_field() -> None:
    """An unusable image carrier names the exact offending request field."""
    with pytest.raises(OpenAIProtocolError) as error:
        decode_chat(
            {
                "model": "coding",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": "ftp://example.com/a.png"}}
                        ],
                    }
                ],
            }
        )
    assert error.value.detail.param == "messages.0.content.0.image_url"


def test_assistant_image_parts_are_rejected() -> None:
    """Only a caller message may carry an image."""
    with pytest.raises(OpenAIProtocolError):
        decode_chat(
            {
                "model": "coding",
                "messages": [
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{_PNG_BASE64}"},
                            }
                        ],
                    }
                ],
            }
        )


_PDF_BASE64 = "JVBERi0xLjQKJSBtaW5pbWFsIHBkZgo="
"""One short PDF header, base64 encoded."""

_PDF_DATA_URL = f"data:application/pdf;base64,{_PDF_BASE64}"
"""The same PDF as an OpenAI ``file_data`` value."""


def _chat_file(file_data: str = _PDF_DATA_URL, filename: str | None = "brief.pdf") -> JsonObject:
    """Build one Chat Completions ``file`` content part."""
    file: JsonObject = {"file_data": file_data}
    if filename is not None:
        file["filename"] = filename
    return {"type": "file", "file": file}


def test_chat_decoder_retains_file_parts_interleaved_with_text() -> None:
    """Chat ``file`` parts keep their positions among the caller's text."""
    decoded = decode_chat(
        {
            "model": "coding",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "first: "},
                        _chat_file(),
                        {"type": "text", "text": " second: "},
                        _chat_file("JVBERi0xLjcK", filename=None),
                        {"type": "text", "text": " compare"},
                    ],
                }
            ],
        }
    )
    message = decoded.request.messages[0]
    assert message.content == "first:  second:  compare"
    assert [part.kind for part in message.content_parts] == [
        "text",
        "document",
        "text",
        "document",
        "text",
    ]
    documents = decoded.request.documents
    assert [document.data for document in documents] == [_PDF_BASE64, "JVBERi0xLjcK"]
    assert [document.name for document in documents] == ["brief.pdf", None]
    assert all(document.media_type == "application/pdf" for document in documents)


def test_chat_file_sent_once_survives_a_multi_turn_thread() -> None:
    """A PDF in an earlier user turn is retained when later turns reference it."""
    decoded = decode_chat(
        {
            "model": "coding",
            "messages": [
                {"role": "user", "content": [_chat_file(), {"type": "text", "text": "title?"}]},
                {"role": "assistant", "content": "Minimal PDF."},
                {"role": "user", "content": "page count?"},
            ],
        }
    )
    assert [len(message.documents) for message in decoded.request.messages] == [1, 0, 0]


def test_responses_decoder_retains_input_file_parts() -> None:
    """Responses ``input_file`` decodes inline data and remote URLs separately."""
    decoded = decode_responses(
        {
            "model": "coding",
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "compare"},
                        {"type": "input_file", "filename": "a.pdf", "file_data": _PDF_DATA_URL},
                        {"type": "input_file", "file_url": "https://example.com/b.pdf"},
                    ],
                }
            ],
        }
    )
    message = decoded.request.messages[0]
    assert [part.kind for part in message.content_parts] == ["text", "document", "document"]
    inline, remote = decoded.request.documents
    assert (inline.data, inline.name, inline.url) == (_PDF_BASE64, "a.pdf", None)
    assert (remote.data, remote.url) == (None, "https://example.com/b.pdf")


def test_documents_change_the_canonical_request_digest() -> None:
    """Document bytes, names, and order all change what the model is asked."""

    def digest(*parts: JsonObject) -> str:
        """Digest one chat request whose user turn carries the given parts."""
        decoded = decode_chat(
            {
                "model": "coding",
                "messages": [{"role": "user", "content": [{"type": "text", "text": "q"}, *parts]}],
            }
        )
        return sha256_json(decoded.request)

    text_only = digest()
    one = digest(_chat_file())
    other_bytes = digest(_chat_file("JVBERi0xLjcK"))
    renamed = digest(_chat_file(filename="other.pdf"))
    assert len({text_only, one, other_bytes, renamed}) == 4
    assert digest(_chat_file(), _chat_file("JVBERi0xLjcK")) != digest(
        _chat_file("JVBERi0xLjcK"), _chat_file()
    )


@pytest.mark.parametrize(
    ("part", "param"),
    [
        (_chat_file("data:text/plain;base64,aGk="), "messages.0.content.0.file.file_data"),
        (_chat_file("!!not base64"), "messages.0.content.0.file.file_data"),
        (
            {"type": "file", "file": {"file_id": "not-a-file-id"}},
            "messages.0.content.0.file.file_id",
        ),
        ({"type": "image_url", "image_url": {}}, "messages.0.content.0.image_url.url"),
    ],
)
def test_unservable_chat_file_parts_are_rejected_at_their_field(
    part: JsonObject, param: str
) -> None:
    """A part the gateway cannot forward names its real field (the union tag
    that doubles as the payload key is reported once), never drops."""
    with pytest.raises(OpenAIProtocolError) as error:
        decode_chat({"model": "coding", "messages": [{"role": "user", "content": [part]}]})
    assert error.value.detail.param == param


@pytest.mark.parametrize(
    "part",
    [
        {"type": "input_file", "file_url": "ftp://example.com/a.pdf"},
        {"type": "input_file", "file_data": _PDF_DATA_URL, "file_url": "https://example.com/a.pdf"},
        {"type": "input_file", "filename": "a.pdf"},
        {"type": "input_file", "file_id": "file_1"},
    ],
)
def test_unservable_responses_input_file_parts_are_rejected(part: JsonObject) -> None:
    """Responses files need exactly one servable carrier."""
    with pytest.raises(OpenAIProtocolError):
        decode_responses({"model": "coding", "input": [{"role": "user", "content": [part]}]})


def test_assistant_file_parts_are_rejected() -> None:
    """Only a caller message may carry a document."""
    with pytest.raises(OpenAIProtocolError):
        decode_chat(
            {"model": "coding", "messages": [{"role": "assistant", "content": [_chat_file()]}]}
        )


def test_too_many_chat_files_are_rejected() -> None:
    """The per-request document ceiling fails closed with the ceiling named."""
    with pytest.raises(OpenAIProtocolError, match="at most 5 documents"):
        decode_chat(
            {
                "model": "coding",
                "messages": [
                    {
                        "role": "user",
                        "content": [_chat_file() for _ in range(MAXIMUM_DOCUMENTS_PER_REQUEST + 1)],
                    }
                ],
            }
        )


def test_non_function_tool_declarations_carry_verbatim_at_their_positions() -> None:
    """Regression fixture: the top-level tools array real Codex (0.151.0)
    sends by default on gpt-5.x models, trimmed from a live capture
    (2026-09-01). Every non-function declaration type api.openai.com accepts
    with a plain key (custom, namespace, web_search, tool_search; each
    verified live 2026-09-01) carries byte-for-byte with its position;
    function declarations keep the strict typed profile."""
    function_tool = {
        "type": "function",
        "name": "exec_command",
        "description": "Execute shell commands",
        "strict": False,
        "parameters": {"type": "object", "properties": {}},
    }
    custom_tool = {
        "type": "custom",
        "name": "apply_patch",
        "description": "Use the `apply_patch` tool to edit files.",
        "format": {
            "type": "grammar",
            "syntax": "lark",
            "definition": 'start: begin_patch hunk+ end_patch\nbegin_patch: "*** Begin Patch"',
        },
    }
    namespace_tool = {
        "type": "namespace",
        "name": "multi_agent_v1",
        "description": "Tools for spawning and managing sub-agents.",
        "tools": [
            {
                "type": "function",
                "name": "close_agent",
                "description": "Close an agent.",
                "strict": False,
                "parameters": {"type": "object", "properties": {}},
            }
        ],
    }
    web_search_tool = {"type": "web_search", "external_web_access": False}
    tool_search_tool = {
        "type": "tool_search",
        "description": "Search for additional tools.",
        "parameters": {"type": "object", "properties": {}},
        "execution": {"type": "server"},
    }
    decoded = decode_responses(
        {
            "model": "gpt-5.2",
            "store": False,
            "stream": True,
            "tool_choice": "required",
            "input": "Run ls.",
            "tools": [
                function_tool,
                custom_tool,
                namespace_tool,
                web_search_tool,
                tool_search_tool,
            ],
        }
    )
    request = decoded.request
    assert [tool.name for tool in request.tools] == ["exec_command"]
    assert [(entry.index, entry.tool) for entry in request.provider_native_tools] == [
        (1, custom_tool),
        (2, namespace_tool),
        (3, web_search_tool),
        (4, tool_search_tool),
    ]


def test_a_malformed_function_tool_declaration_still_fails_closed() -> None:
    """The opaque carrier accepts only non-function types; a function
    declaration missing its name is a named validation error, never an
    opaque forward."""
    with pytest.raises(OpenAIProtocolError) as rejection:
        decode_responses(
            {
                "model": "gpt-5.2",
                "input": "hi",
                "tools": [{"type": "function", "description": "nameless"}],
            }
        )
    assert rejection.value.status_code == 400
    assert "tools" in (rejection.value.detail.param or "")


def test_embeddings_decoder_preserves_supported_fields() -> None:
    """Embeddings conversion keeps the alias, inputs, dimensions, encoding, and attribution."""
    decoded = decode_embeddings(
        {
            "model": "text-embedding-3-small",
            "input": "hello world",
            "dimensions": 256,
            "encoding_format": "float",
            "user": "end-user-7",
        }
    )

    assert decoded.alias == "text-embedding-3-small"
    assert decoded.request.surface == GatewayApiSurface.EMBEDDINGS
    assert decoded.request.inputs == ("hello world",)
    assert decoded.request.dimensions == 256
    assert decoded.request.encoding_format == "float"
    assert decoded.request.user == "end-user-7"


def test_embeddings_decoder_accepts_a_list_of_inputs() -> None:
    """An array of texts decodes in order with defaults for the optional fields."""
    decoded = decode_embeddings({"model": "m", "input": ["first", "second"]})

    assert decoded.request.inputs == ("first", "second")
    assert decoded.request.dimensions is None
    assert decoded.request.encoding_format is None
    assert decoded.request.user is None


def test_embeddings_decoder_rejects_unknown_and_streaming_fields() -> None:
    """A field outside the closed embeddings manifest is a named 400, never silently dropped."""
    with pytest.raises(OpenAIProtocolError) as rejection:
        decode_embeddings({"model": "m", "input": "x", "stream": True})
    assert rejection.value.status_code == 400
    assert "stream" in rejection.value.detail.message


def test_embeddings_decoder_rejects_token_array_input() -> None:
    """Pre-tokenized id arrays pass official validation but this text surface rejects them."""
    with pytest.raises(OpenAIProtocolError) as rejection:
        decode_embeddings({"model": "m", "input": [1, 2, 3]})
    assert rejection.value.status_code == 400
    assert "input" in (rejection.value.detail.param or "")


def test_embeddings_decoder_rejects_empty_and_malformed_inputs() -> None:
    """Empty strings, an empty array, a missing input, and bad options each fail with a param."""
    cases: tuple[tuple[JsonObject, str], ...] = (
        ({"model": "m", "input": ""}, "input"),
        ({"model": "m", "input": []}, "input"),
        ({"model": "m"}, "input"),
        ({"model": "m", "input": "x", "dimensions": 0}, "dimensions"),
        ({"model": "m", "input": "x", "encoding_format": "weird"}, "encoding_format"),
    )
    for payload, param in cases:
        with pytest.raises(OpenAIProtocolError) as rejection:
            decode_embeddings(payload)
        assert rejection.value.status_code == 400
        assert param in (rejection.value.detail.param or "")


_MP4_BASE64 = "AAAAIGZ0eXBpc29tAAACAGlzb21pc28yYXZjMW1wNDE="
"""A base64 prefix of an MP4 ``ftyp`` box, enough for a carrier fixture."""


def test_chat_decoder_retains_video_parts_in_caller_order() -> None:
    """A chat ``video_url`` part is kept beside its text and images in caller order."""
    decoded = decode_chat(
        {
            "model": "coding",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "first "},
                        {"type": "video_url", "video_url": {"url": "https://example.com/a.mp4"}},
                        {"type": "text", "text": "then "},
                        {
                            "type": "video_url",
                            "video_url": {"url": f"data:video/webm;base64,{_MP4_BASE64}"},
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{_PNG_BASE64}"},
                        },
                        {"type": "text", "text": "compare"},
                    ],
                }
            ],
        }
    )
    message = decoded.request.messages[0]
    assert message.content == "first then compare"
    assert [part.kind for part in message.content_parts] == [
        "text",
        "video",
        "text",
        "video",
        "image",
        "text",
    ]
    remote, inline = decoded.request.videos
    assert remote.url == "https://example.com/a.mp4"
    assert inline.media_type == "video/webm"
    assert inline.data == _MP4_BASE64
    assert len(decoded.request.images) == 1
    assert [part.kind for part in model_request(decoded.request).messages[0].content_parts] == [
        part.kind for part in message.content_parts
    ]


def test_videos_change_the_canonical_request_digest() -> None:
    """A video changes what the model is asked, so it changes replay identity."""
    text_only = decode_chat(
        {"model": "coding", "messages": [{"role": "user", "content": "what happens"}]}
    )
    with_video = decode_chat(
        {
            "model": "coding",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "what happens"},
                        {
                            "type": "video_url",
                            "video_url": {"url": f"data:video/mp4;base64,{_MP4_BASE64}"},
                        },
                    ],
                }
            ],
        }
    )
    other_video = decode_chat(
        {
            "model": "coding",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "what happens"},
                        {
                            "type": "video_url",
                            "video_url": {"url": f"data:video/mp4;base64,{_MP4_BASE64[:-4]}"},
                        },
                    ],
                }
            ],
        }
    )
    digests = {
        sha256_json(text_only.request),
        sha256_json(with_video.request),
        sha256_json(other_video.request),
    }
    assert len(digests) == 3
    assert "video" in with_video.request.model_dump_json()


def test_malformed_chat_video_url_is_rejected_with_its_field() -> None:
    """An unusable video carrier names the exact offending request field."""
    for url in ("ftp://example.com/a.mp4", f"data:video/x-matroska;base64,{_MP4_BASE64}"):
        with pytest.raises(OpenAIProtocolError) as error:
            decode_chat(
                {
                    "model": "coding",
                    "messages": [
                        {
                            "role": "user",
                            "content": [{"type": "video_url", "video_url": {"url": url}}],
                        }
                    ],
                }
            )
        assert error.value.detail.param == "messages.0.content.0.video_url"


def test_a_request_carries_at_most_the_video_ceiling() -> None:
    """The eleventh video in one request is refused rather than dropped."""
    part: JsonObject = {"type": "video_url", "video_url": {"url": "https://example.com/a.mp4"}}
    decode_chat({"model": "coding", "messages": [{"role": "user", "content": [part] * 10}]})
    with pytest.raises(OpenAIProtocolError):
        decode_chat({"model": "coding", "messages": [{"role": "user", "content": [part] * 11}]})
    with pytest.raises(ValueError, match="at most 10 videos"):
        GatewayRequest(
            surface=GatewayApiSurface.CHAT_COMPLETIONS,
            messages=(
                GatewayMessage(
                    role="user",
                    content="",
                    content_parts=tuple(
                        VideoContentPart(url="https://example.com/a.mp4") for _ in range(11)
                    ),
                ),
            ),
        )


def test_assistant_video_parts_are_rejected() -> None:
    """Only a caller message may carry a video."""
    with pytest.raises(OpenAIProtocolError):
        decode_chat(
            {
                "model": "coding",
                "messages": [
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "video_url", "video_url": {"url": "https://example.com/a.mp4"}}
                        ],
                    }
                ],
            }
        )


def test_responses_surface_defines_no_video_part() -> None:
    """The Responses wire has no video content type, so one is refused, not dropped."""
    with pytest.raises(OpenAIProtocolError):
        decode_responses(
            {
                "model": "coding",
                "input": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "describe"},
                            {
                                "type": "video_url",
                                "video_url": {"url": "https://example.com/a.mp4"},
                            },
                        ],
                    }
                ],
            }
        )
    with pytest.raises(OpenAIProtocolError):
        decode_responses(
            {
                "model": "coding",
                "input": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_video", "video_url": "https://example.com/a.mp4"}
                        ],
                    }
                ],
            }
        )


def test_responses_decoder_wraps_file_ids_as_openai_handles() -> None:
    """``input_image.file_id`` and ``input_file.file_id`` become OpenAI handles in order."""
    decoded = decode_responses(
        {
            "model": "coding",
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_image", "file_id": "file-img", "detail": "low"},
                        {"type": "input_text", "text": "and this"},
                        {"type": "input_file", "file_id": "file-doc"},
                    ],
                }
            ],
        }
    )
    message = decoded.request.messages[0]
    assert [part.kind for part in message.content_parts] == ["image", "text", "document"]
    (image,) = decoded.request.images
    (document,) = decoded.request.documents
    assert image.handle == MediaHandle(provider="openai", reference="file-img")
    assert (image.detail, image.data, image.url) == ("low", None, None)
    assert document.handle == MediaHandle(provider="openai", reference="file-doc")
    assert decoded.request.media_handles == (image.handle, document.handle)


def test_chat_decoder_wraps_file_part_ids_as_openai_handles() -> None:
    """A Chat ``file.file_id`` becomes an OpenAI handle beside inline siblings."""
    decoded = decode_chat(
        {
            "model": "coding",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        _chat_file(),
                        {"type": "file", "file": {"file_id": "file-doc", "filename": "b.pdf"}},
                        {"type": "text", "text": "compare"},
                    ],
                }
            ],
        }
    )
    inline, handled = decoded.request.documents
    assert inline.data == _PDF_BASE64 and inline.handle is None
    assert handled.handle == MediaHandle(provider="openai", reference="file-doc")
    assert handled.name == "b.pdf"


def test_provider_uris_in_url_fields_decode_as_handles() -> None:
    """``s3://``, ``gs://``, and Gemini Files URIs are handles on every URL field."""
    gemini = f"{GEMINI_FILE_URI_PREFIX}abc123"

    def chat(part: JsonObject) -> GatewayRequest:
        """Decode one Chat request carrying a single media part and a text run."""
        return decode_chat(
            {
                "model": "coding",
                "messages": [
                    {"role": "user", "content": [part, {"type": "text", "text": "describe"}]}
                ],
            }
        ).request

    (image,) = chat({"type": "image_url", "image_url": {"url": "s3://bkt/cat.png"}}).images
    assert image.handle == MediaHandle(provider="bedrock", reference="s3://bkt/cat.png")
    assert image.media_type == "image/png" and image.url is None
    (video,) = chat({"type": "video_url", "video_url": {"url": "gs://bkt/clip.mp4"}}).videos
    assert video.handle == MediaHandle(provider="vertex", reference="gs://bkt/clip.mp4")
    assert video.media_type == "video/mp4"
    responses = decode_responses(
        {
            "model": "coding",
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_file", "file_url": gemini},
                        {"type": "input_image", "image_url": gemini},
                    ],
                }
            ],
        }
    )
    assert [handle.provider for handle in responses.request.media_handles] == ["gemini", "gemini"]
    assert responses.request.media_handles[0].reference == gemini


def test_handles_from_two_providers_in_one_request_are_rejected() -> None:
    """No route can resolve both, so the decoder refuses the request as a whole."""
    with pytest.raises(OpenAIProtocolError, match="same provider"):
        decode_chat(
            {
                "model": "coding",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": "s3://bkt/cat.png"}},
                            {"type": "video_url", "video_url": {"url": "gs://bkt/clip.mp4"}},
                        ],
                    }
                ],
            }
        )


@pytest.mark.parametrize(
    ("part", "suffix"),
    [
        (
            {"type": "input_image", "file_id": "file-img", "image_url": "https://x.test/a.png"},
            "content.0.input_image",
        ),
        ({"type": "input_image", "file_id": "not-openai"}, "content.0.file_id"),
        (
            {"type": "input_file", "file_id": "file-a", "file_data": _PDF_DATA_URL},
            "content.0.input_file",
        ),
        ({"type": "input_image", "image_url": "s3://bkt/no-suffix"}, "content.0.image_url"),
    ],
)
def test_malformed_or_doubled_responses_handles_are_rejected(part: JsonObject, suffix: str) -> None:
    """A handle beside another carrier, a foreign id, or a suffixless object fails at its field."""
    with pytest.raises(OpenAIProtocolError) as error:
        decode_responses({"model": "coding", "input": [{"role": "user", "content": [part]}]})
    assert error.value.detail.param is not None
    assert error.value.detail.param.endswith(suffix)
