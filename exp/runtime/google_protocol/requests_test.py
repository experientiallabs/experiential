"""Round-trip and rejection tests for the Gemini generateContent decoder."""

from __future__ import annotations

import pytest
from pydantic import JsonValue

from exp.common.core.artifacts import JsonObject
from exp.common.models.content import AudioContentPart, DocumentContentPart, TextContentPart
from exp.runtime.gateway.contracts import GatewayApiSurface
from exp.runtime.gateway.tool_contracts import GatewayAllowedToolsChoice, GatewayNamedToolChoice
from exp.runtime.google_protocol.requests import (
    RESPONSE_SCHEMA_DROPPED,
    SAFETY_SETTINGS_DROPPED,
    THOUGHT_DROPPED,
    decode_generate_content,
    decode_generate_content_count_tokens,
)
from exp.runtime.models.providers.base import GatewayWireProfile
from exp.runtime.models.providers.gemini_requests import THOUGHT_SIGNATURE_DROPPED
from exp.runtime.models.providers.streaming_requests import route_generation_parameter_requests
from exp.runtime.openai_protocol.errors import OpenAIProtocolError


def _body(**overrides: JsonValue) -> JsonObject:
    """Return one minimal valid generateContent body with overrides applied."""
    payload: JsonObject = {"contents": [{"role": "user", "parts": [{"text": "hi"}]}]}
    payload.update(overrides)
    return payload


def _error(payload: JsonObject) -> OpenAIProtocolError:
    """Decode an invalid body and return its protocol error."""
    with pytest.raises(OpenAIProtocolError) as caught:
        decode_generate_content(payload, model="coding", stream=False)
    return caught.value


def test_decode_full_sdk_request_is_lossless() -> None:
    """The body the official SDK sends for a configured call lands on the canonical request."""
    decoded = decode_generate_content(
        _body(
            contents=[
                {"parts": [{"text": "hi"}], "role": "user"},
                {
                    "parts": [
                        {
                            "functionCall": {"id": "call-1", "args": {"a": 1}, "name": "search"},
                            "thoughtSignature": "c2tpcA==",
                        }
                    ],
                    "role": "model",
                },
                {
                    "parts": [
                        {
                            "functionResponse": {
                                "id": "call-1",
                                "name": "search",
                                "response": {"result": "ok"},
                            }
                        }
                    ],
                    "role": "user",
                },
            ],
            systemInstruction={"parts": [{"text": "be terse"}], "role": "user"},
            tools=[
                {
                    "functionDeclarations": [
                        {
                            "description": "d",
                            "name": "search",
                            "parameters_json_schema": {"type": "object"},
                        }
                    ]
                }
            ],
            toolConfig={
                "functionCallingConfig": {"allowedFunctionNames": ["search"], "mode": "ANY"}
            },
            generationConfig={
                "temperature": 0.5,
                "topP": 0.9,
                "maxOutputTokens": 64,
                "stopSequences": ["END", "END"],
                "responseMimeType": "application/json",
                "responseJsonSchema": {"type": "object"},
                "thinkingConfig": {"include_thoughts": True, "thinking_level": "LOW"},
            },
        ),
        model="coding",
        stream=True,
    )
    request = decoded.request
    assert decoded.alias == "coding"
    assert request.surface == GatewayApiSurface.GENERATE_CONTENT
    assert [message.role for message in request.messages] == ["system", "user", "assistant", "tool"]
    assert request.messages[0].content == "be terse"
    call = request.messages[2].tool_calls[0]
    assert (call.call_id, call.name, call.arguments, call.provider_thought_signature) == (
        "call-1",
        "search",
        {"a": 1},
        "c2tpcA==",
    )
    assert request.messages[3].tool_call_id == "call-1"
    assert request.messages[3].content == "ok"
    assert request.tools[0].name == "search"
    assert request.tool_choice == GatewayNamedToolChoice(name="search")
    assert request.structured_text is not None
    assert request.structured_text.json_schema == {"type": "object"}
    assert request.maximum_output_tokens == 64
    assert request.maximum_output_tokens_parameter == "generationConfig.maxOutputTokens"
    assert request.stop == ("END",)
    assert (request.temperature, request.top_p) == (0.5, 0.9)
    assert request.reasoning_effort == "low"
    assert request.reasoning_effort_parameter == "generationConfig.thinkingConfig.thinkingLevel"
    assert request.caller_effort_parameter == "generationConfig.thinkingConfig.thinkingLevel"
    assert request.include_thoughts is True
    assert request.stream is True and request.include_usage is True
    assert request.ignored_parameters == ()


def test_snake_case_spellings_and_schema_enums_decode() -> None:
    """Google's protobuf JSON mapping accepts both spellings and upper-case Schema types."""
    decoded = decode_generate_content(
        {
            "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
            "system_instruction": {"parts": [{"text": "a"}, {"text": "b"}]},
            "generation_config": {
                "max_output_tokens": 8,
                "response_mime_type": "application/json",
                "response_schema": {
                    "type": "OBJECT",
                    "properties": {"n": {"type": "INTEGER", "nullable": True}},
                },
            },
            "tool_config": {"function_calling_config": {"mode": "NONE"}},
        },
        model="m",
        stream=False,
    )
    request = decoded.request
    assert request.messages[0].content == "a\n\nb"
    assert request.maximum_output_tokens == 8
    assert request.tool_choice == "none"
    assert request.structured_text is not None
    assert request.structured_text.json_schema == {
        "type": "object",
        "properties": {"n": {"type": "integer", "nullable": True}},
    }


def test_json_mode_without_a_schema_rides_json_object_output() -> None:
    """``responseMimeType`` alone selects schema-free JSON mode."""
    request = decode_generate_content(
        _body(generationConfig={"responseMimeType": "application/json"}), model="m", stream=False
    ).request
    assert request.json_object_output is True
    assert request.structured_text is None


def test_function_responses_pair_by_name_order_without_ids() -> None:
    """Responses without ids answer the oldest unanswered call of their name, as Gemini does."""
    request = decode_generate_content(
        _body(
            contents=[
                {"role": "user", "parts": [{"text": "go"}]},
                {
                    "role": "model",
                    "parts": [
                        {"functionCall": {"name": "a", "args": {}}},
                        {"functionCall": {"name": "b", "args": {"x": 1}}},
                    ],
                },
                {
                    "role": "user",
                    "parts": [
                        {"functionResponse": {"name": "b", "response": {"content": "B"}}},
                        {"functionResponse": {"name": "a", "response": {"k": 1}}},
                        {"text": "and?"},
                    ],
                },
            ]
        ),
        model="m",
        stream=False,
    ).request
    calls = request.messages[1].tool_calls
    assert [call.name for call in calls] == ["a", "b"]
    tools = [message for message in request.messages if message.role == "tool"]
    assert [(message.tool_call_id, message.content) for message in tools] == [
        (calls[1].call_id, "B"),
        (calls[0].call_id, '{"k":1}'),
    ]
    assert request.messages[-1].role == "user" and request.messages[-1].content == "and?"


def test_allowed_function_names_become_an_allowed_tools_choice() -> None:
    """Several allowed names restrict the model without forcing one function."""
    request = decode_generate_content(
        _body(
            tools=[{"functionDeclarations": [{"name": "a"}, {"name": "b"}]}],
            toolConfig={
                "functionCallingConfig": {"mode": "AUTO", "allowedFunctionNames": ["a", "b"]}
            },
        ),
        model="m",
        stream=False,
    ).request
    assert request.tool_choice == GatewayAllowedToolsChoice(mode="auto", names=("a", "b"))


def test_thoughts_and_safety_settings_drop_with_disclosure() -> None:
    """Replayed thought summaries and Google's thresholds never reach a provider silently."""
    request = decode_generate_content(
        _body(
            contents=[
                {"role": "user", "parts": [{"text": "hi"}]},
                {
                    "role": "model",
                    "parts": [{"text": "pondering", "thought": True}, {"text": "yo"}],
                },
                {"role": "user", "parts": [{"text": "more"}]},
            ],
            safetySettings=[{"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"}],
        ),
        model="m",
        stream=False,
    ).request
    assert request.messages[1].content == "yo"
    assert set(request.ignored_parameters) == {THOUGHT_DROPPED, SAFETY_SETTINGS_DROPPED}


def test_gemini_audio_spellings_and_file_urls_land_on_their_own_part_kinds() -> None:
    """``audio/mp3`` is MP3; a PDF fileData URL is a document, never an image; audio URLs refuse."""
    request = decode_generate_content(
        _body(
            contents=[
                {
                    "role": "user",
                    "parts": [
                        {"inlineData": {"mimeType": "audio/mp3", "data": "SUQz"}},
                        {
                            "fileData": {
                                "mimeType": "application/pdf",
                                "fileUri": "https://example.com/paper.pdf",
                            }
                        },
                        {"fileData": {"fileUri": "https://example.com/photo.png"}},
                    ],
                }
            ]
        ),
        model="m",
        stream=False,
    ).request
    parts = request.messages[0].content_parts
    assert [part.kind for part in parts] == ["audio", "document", "image"]
    audio, document = parts[0], parts[1]
    assert isinstance(audio, AudioContentPart)
    assert audio.media_type == "audio/mpeg"
    assert isinstance(document, DocumentContentPart)
    assert document.url == "https://example.com/paper.pdf"
    error = _error(
        _body(
            contents=[
                {
                    "role": "user",
                    "parts": [
                        {"fileData": {"mimeType": "audio/mp3", "fileUri": "https://x.test/a.mp3"}}
                    ],
                }
            ]
        )
    )
    assert error.detail.param == "contents.0.parts.0.fileData.fileUri"


def test_inline_media_becomes_canonical_parts() -> None:
    """Images, audio and PDFs ride the canonical content parts by media type."""
    request = decode_generate_content(
        _body(
            contents=[
                {
                    "role": "user",
                    "parts": [
                        {"text": "see"},
                        {"inlineData": {"mimeType": "image/png", "data": "iVBORw0KGgo="}},
                        {"inline_data": {"mime_type": "application/pdf", "data": "JVBERi0="}},
                    ],
                }
            ]
        ),
        model="m",
        stream=False,
    ).request
    kinds = [part.kind for part in request.messages[0].content_parts]
    assert kinds == ["text", "image", "document"]
    assert request.messages[0].content == "see"


@pytest.mark.parametrize(
    ("payload", "status", "code", "param"),
    [
        (_body(unknown=1), 400, "unsupported_parameter", "unknown"),
        (_body(cachedContent="cachedContents/x"), 400, "unsupported_parameter", "cachedContent"),
        (_body(tools=[{"googleSearch": {}}]), 400, "unsupported_parameter", "tools.0.googleSearch"),
        (
            _body(contents=[{"role": "user", "parts": [{"bogus": 1}]}]),
            400,
            "invalid_parameter",
            "contents.0.parts.0.bogus",
        ),
        (
            _body(generationConfig={"thinkingConfig": {"thinkingLevel": "EXTREME"}}),
            400,
            "invalid_parameter",
            "generationConfig.thinkingConfig.thinkingLevel",
        ),
        (
            _body(generationConfig={"candidateCount": 2}),
            400,
            "unsupported_parameter",
            "generationConfig.candidateCount",
        ),
        (
            _body(generationConfig={"responseModalities": ["IMAGE"]}),
            400,
            "unsupported_parameter",
            "generationConfig.responseModalities",
        ),
        (
            _body(
                contents=[
                    {"role": "user", "parts": [{"functionResponse": {"name": "x", "response": {}}}]}
                ]
            ),
            400,
            "invalid_parameter",
            "contents.0.parts.0.functionResponse",
        ),
        (
            _body(
                contents=[
                    {
                        "role": "user",
                        "parts": [{"inlineData": {"mimeType": "image/bmp", "data": "x"}}],
                    }
                ]
            ),
            400,
            "invalid_parameter",
            "contents.0.parts.0.inlineData.mimeType",
        ),
        # A protobuf oneof set twice is malformed, never a silent pick.
        (
            _body(
                contents=[
                    {
                        "role": "user",
                        "parts": [
                            {"text": "a", "inlineData": {"mimeType": "image/png", "data": "x"}}
                        ],
                    }
                ]
            ),
            400,
            "invalid_parameter",
            "contents.0.parts.0",
        ),
        (
            _body(
                contents=[
                    {"role": "user", "parts": [{"text": "q"}]},
                    {
                        "role": "model",
                        "parts": [
                            {
                                "functionCall": {"name": "f", "args": {}},
                                "functionResponse": {"name": "f", "response": {}},
                            }
                        ],
                    },
                ]
            ),
            400,
            "invalid_parameter",
            "contents.1.parts.0",
        ),
        (
            _body(contents=[{"role": "user", "parts": [{"thought": True}]}]),
            400,
            "invalid_parameter",
            "contents.0.parts.0",
        ),
        (
            _body(systemInstruction={"parts": [{"text": "s", "videoMetadata": {"fps": 1}}]}),
            400,
            "unsupported_parameter",
            "systemInstruction.parts.0.videoMetadata",
        ),
        # One call answered twice by explicit id is one invocation with two results.
        (
            _body(
                contents=[
                    {"role": "user", "parts": [{"text": "q"}]},
                    {"role": "model", "parts": [{"functionCall": {"id": "c", "name": "f"}}]},
                    {
                        "role": "user",
                        "parts": [
                            {"functionResponse": {"id": "c", "name": "f", "response": {}}},
                            {"functionResponse": {"id": "c", "name": "f", "response": {}}},
                        ],
                    },
                ]
            ),
            400,
            "invalid_parameter",
            "contents.2.parts.1.functionResponse.id",
        ),
    ],
)
def test_rejections_name_the_offending_field(
    payload: JsonObject, status: int, code: str, param: str
) -> None:
    """Every refusal is a field-specific 400 the caller can act on."""
    error = _error(payload)
    assert (error.status_code, error.detail.code, error.detail.param) == (status, code, param)


def test_count_tokens_accepts_contents_or_a_wrapped_request() -> None:
    """Both of Google's count shapes decode; a count never streams and may lack a ceiling."""
    bare = decode_generate_content_count_tokens(
        {"contents": [{"role": "user", "parts": [{"text": "hi"}]}]}, model="m"
    )
    assert bare.request.maximum_output_tokens is None and bare.request.stream is False
    wrapped = decode_generate_content_count_tokens(
        {
            "generateContentRequest": {
                "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
                "generationConfig": {"maxOutputTokens": 5},
            }
        },
        model="m",
    )
    assert wrapped.request.maximum_output_tokens == 5
    with pytest.raises(OpenAIProtocolError) as caught:
        decode_generate_content_count_tokens({}, model="m")
    assert caught.value.detail.param == "contents"


def test_interleaved_media_and_captions_keep_the_callers_part_order() -> None:
    """An image, its caption and the next image land on the message in that order."""
    png = {"inlineData": {"mimeType": "image/png", "data": "iVBORw0KGgo="}}
    request = decode_generate_content(
        _body(
            contents=[
                {
                    "role": "user",
                    "parts": [
                        png,
                        {"text": "first"},
                        {"text": " caption"},
                        png,
                        {"text": "second"},
                    ],
                }
            ]
        ),
        model="m",
        stream=False,
    ).request
    message = request.messages[0]
    assert [part.kind for part in message.content_parts] == [
        "image",
        "text",
        "text",
        "image",
        "text",
    ]
    assert message.content_parts[1] == TextContentPart(text="first")
    assert message.content == "first captionsecond"


def test_function_response_schemas_drop_with_disclosure() -> None:
    """A declared output schema has no canonical carrier; the caller learns it was dropped."""
    request = decode_generate_content(
        _body(
            tools=[
                {
                    "functionDeclarations": [
                        {
                            "name": "f",
                            "parameters": {"type": "OBJECT"},
                            "response": {"type": "STRING"},
                        }
                    ]
                }
            ]
        ),
        model="m",
        stream=False,
    ).request
    assert request.tools[0].parameters == {"type": "object"}
    assert RESPONSE_SCHEMA_DROPPED in request.ignored_parameters


def test_an_echoed_signature_is_disclosed_as_dropped_off_gemini_wires() -> None:
    """A generateContent alias routed only to a non-Gemini wire discloses the dropped signature."""
    request = decode_generate_content(
        _body(
            contents=[
                {"role": "user", "parts": [{"text": "q"}]},
                {
                    "role": "model",
                    "parts": [
                        {
                            "functionCall": {"id": "c1", "name": "f", "args": {}},
                            "thoughtSignature": "c2lnbmF0dXJlLWE=",
                        }
                    ],
                },
                {
                    "role": "user",
                    "parts": [{"functionResponse": {"id": "c1", "name": "f", "response": {}}}],
                },
            ]
        ),
        model="m",
        stream=False,
    ).request
    openai = GatewayWireProfile(
        dialect="openai_compatible", url="https://openai.test/v1", model_id="gpt-5.5"
    )
    gemini = GatewayWireProfile(
        dialect="gemini_generate_content", url="https://gemini.test", model_id="gemini-3"
    )
    public, _provider = route_generation_parameter_requests((openai,), request)
    assert THOUGHT_SIGNATURE_DROPPED in public.ignored_parameters
    public, _provider = route_generation_parameter_requests((gemini, openai), request)
    assert THOUGHT_SIGNATURE_DROPPED not in public.ignored_parameters


def test_empty_objects_and_empty_text_parts_survive_as_sent() -> None:
    """``response: {}`` is the text ``{}`` and an empty text part is an empty turn, not a 400."""
    request = decode_generate_content(
        _body(
            contents=[
                {"role": "user", "parts": [{"text": ""}]},
                {"role": "model", "parts": [{"functionCall": {"id": "c", "name": "f"}}]},
                {
                    "role": "user",
                    "parts": [{"functionResponse": {"id": "c", "name": "f", "response": {}}}],
                },
            ]
        ),
        model="m",
        stream=False,
    ).request
    assert request.messages[0].content == ""
    assert request.messages[-1].content == "{}"
