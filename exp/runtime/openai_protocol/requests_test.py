"""Tests for shared Chat Completions and Responses request decoding."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.contracts import GatewayApiSurface, GatewayNamedToolChoice
from exp.runtime.models.providers.streaming_requests import openai_responses_stream_payload
from exp.runtime.openai_protocol.errors import OpenAIProtocolError
from exp.runtime.openai_protocol.model_adapter import model_request
from exp.runtime.openai_protocol.requests import (
    DecodedGatewayRequest,
    decode_chat,
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


def test_invalid_tool_arguments_and_conflicting_operation_headers_are_specific() -> None:
    """Malformed history and mismatched dedup headers identify the exact public field."""
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

    with pytest.raises(OpenAIProtocolError) as operation:
        decode_responses(
            {"model": "coding", "input": "x"},
            idempotency_key="one",
            client_request_id="two",
        )
    assert operation.value.detail.code == "idempotency_conflict"
    assert operation.value.detail.param == "Idempotency-Key"


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
                    "call_id": "call-1",
                    "name": "search",
                    "arguments": "{}",
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
