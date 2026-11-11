"""Cache breakpoints survive public decoding and supported provider adapters."""

from __future__ import annotations

import json
from copy import deepcopy

import pytest

from exp.common.core.artifacts import JsonObject, JsonValue
from exp.runtime.anthropic_protocol.requests import decode_messages
from exp.runtime.gateway.cache_write import requests_hour_cache
from exp.runtime.models.providers.base import GatewayWireProfile
from exp.runtime.models.providers.dialect_dispatch import dialect_stream_payload
from exp.runtime.models.providers.streaming_requests import route_generation_parameter_requests
from exp.runtime.openai_protocol.errors import OpenAIProtocolError
from exp.runtime.openai_protocol.requests import decode_chat


def _opencode_tool_cache_request(marker: JsonValue) -> JsonObject:
    """Build the tool, system and message checkpoints emitted by OpenCode.

    Args:
        marker: Annotation on the final function tool definition.

    Returns:
        A Chat request with an unmarked tool before the marked definition.
    """
    return {
        "model": "coding",
        "max_tokens": 32,
        "tools": [
            {"type": "function", "function": {"name": "list_files", "parameters": {}}},
            {
                "type": "function",
                "function": {"name": "read_file", "parameters": {"type": "object"}},
                "cache_control": marker,
            },
        ],
        "messages": [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "Read files.", "cache_control": {"type": "ephemeral"}}
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Read a.py.", "cache_control": {"type": "ephemeral"}}
                ],
            },
        ],
    }


@pytest.mark.parametrize("ttl", [None, "5m", "1h"])
@pytest.mark.parametrize("explicit_type", [False, True])
def test_opencode_tool_cache_checkpoint_reaches_anthropic(
    ttl: str | None, explicit_type: bool
) -> None:
    """Retain every OpenCode checkpoint without mutating its original request.

    Args:
        ttl: Explicit supported cache lifetime, or the provider default.
        explicit_type: Whether function declarations include their default type.
    """
    marker: JsonObject = {"type": "ephemeral"}
    if ttl is not None:
        marker["ttl"] = ttl
    body = _opencode_tool_cache_request(marker)
    if not explicit_type:
        declarations = body["tools"]
        assert isinstance(declarations, list)
        for declaration in declarations:
            assert isinstance(declaration, dict)
            declaration.pop("type")
    original = deepcopy(body)
    request = decode_chat(body).request
    assert body == original
    assert request.tools[0].cache_control is None
    assert request.tools[1].cache_control == marker
    assert requests_hour_cache(request) == (ttl == "1h")
    assert not request.ignored_parameters
    payload = dialect_stream_payload(
        GatewayWireProfile(
            dialect="anthropic_messages",
            model_id="claude-sonnet-5-5",
            url="https://example.invalid",
        ),
        request,
    )
    tools = payload["tools"]
    assert isinstance(tools, list) and isinstance(tools[1], dict)
    assert tools[1]["cache_control"] == marker
    assert '"cache_control"' in json.dumps(payload["system"])
    assert '"cache_control"' in json.dumps(payload["messages"])


@pytest.mark.parametrize(
    "marker",
    [
        True,
        1,
        "ephemeral",
        [],
        {},
        {"type": "persistent"},
        {"type": "ephemeral", "ttl": None},
        {"type": "ephemeral", "ttl": "24h"},
        {"type": "ephemeral", "extra": True},
    ],
)
def test_chat_tool_cache_checkpoint_rejects_invalid_values(marker: JsonValue) -> None:
    """Reject malformed annotations at their original tool index.

    Args:
        marker: Value outside the supported ephemeral cache schema.
    """
    with pytest.raises(OpenAIProtocolError) as raised:
        decode_chat(_opencode_tool_cache_request(marker))
    assert raised.value.detail.param == "tools.1.cache_control"


def test_chat_tool_cache_checkpoint_null_is_unmarked() -> None:
    """An explicit null annotation requests no tool checkpoint."""
    request = decode_chat(_opencode_tool_cache_request(None)).request
    assert all(tool.cache_control is None for tool in request.tools)


def test_chat_tool_cache_checkpoint_keeps_indices_across_native_tools() -> None:
    """Provider-native declarations must not shift function checkpoint ownership."""
    body = _opencode_tool_cache_request({"type": "ephemeral"})
    tools = body["tools"]
    assert isinstance(tools, list)
    tools.insert(1, {"type": "openrouter:tool_search", "max_results": 1})
    request = decode_chat(body).request
    assert [(tool.name, tool.cache_control) for tool in request.tools] == [
        ("list_files", None),
        ("read_file", {"type": "ephemeral"}),
    ]
    assert request.provider_native_tools[0].index == 1


def test_chat_tool_cache_checkpoint_does_not_hide_unknown_tool_fields() -> None:
    """Accepting a cache annotation keeps other tool fields strictly validated."""
    body = _opencode_tool_cache_request({"type": "ephemeral"})
    tools = body["tools"]
    assert isinstance(tools, list) and isinstance(tools[1], dict)
    tools[1]["unknown"] = True
    with pytest.raises(OpenAIProtocolError) as raised:
        decode_chat(body)
    assert raised.value.detail.param == "tools.1.unknown"


def test_chat_tool_cache_checkpoint_is_disclosed_on_unsupported_routes() -> None:
    """Generic Chat routes omit and disclose the unsupported tool checkpoint."""
    request = decode_chat(_opencode_tool_cache_request({"type": "ephemeral"})).request
    profile = GatewayWireProfile(dialect="openai_compatible", url="https://example.invalid")
    public, _ = route_generation_parameter_requests((profile,), request)
    assert any(
        item.startswith("tools.cache_control->not_forwarded(") for item in public.ignored_parameters
    )
    assert "cache_control" not in json.dumps(dialect_stream_payload(profile, request))


@pytest.mark.parametrize(
    "tool",
    [
        {"type": "custom", "custom": {"name": "read_file", "format": {"type": "text"}}},
        {"type": "openrouter:tool_search", "max_results": 1},
    ],
)
def test_chat_tool_cache_checkpoint_refuses_non_function_tools(tool: JsonObject) -> None:
    """Unsupported tool kinds cannot silently discard a supplied checkpoint.

    Args:
        tool: A provider-native declaration without a function cache carrier.
    """
    with pytest.raises(OpenAIProtocolError) as raised:
        decode_chat(
            {
                "model": "coding",
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [{**tool, "cache_control": {"type": "ephemeral"}}],
            }
        )
    assert raised.value.detail.param == "tools.0.cache_control"


@pytest.mark.parametrize("surface", ["chat", "messages"])
@pytest.mark.parametrize("wire", ["anthropic_messages", "bedrock_converse_stream", "openrouter"])
def test_explicit_prefix_reaches_cache_capable_adapters(surface: str, wire: str) -> None:
    """The customer's two API formats retain the same exact prefix breakpoint."""
    marked: JsonObject = {
        "type": "text",
        "text": "a stable prefix",
        "cache_control": {"type": "ephemeral"},
    }
    body: JsonObject = {
        "model": "coding",
        "max_tokens": 32,
        "messages": [{"role": "user", "content": "answer briefly"}],
    }
    if surface == "messages":
        body["system"] = [marked]
        request = decode_messages(body).request
    else:
        body["messages"] = [
            {"role": "system", "content": [marked]},
            {"role": "user", "content": "answer briefly"},
        ]
        request = decode_chat(body).request
    profile = GatewayWireProfile(
        dialect="openai_compatible" if wire == "openrouter" else wire,
        model_id="claude-sonnet-4-6",
        url="https://example.invalid",
        forwards_cache_control=wire == "openrouter",
    )
    payload = dialect_stream_payload(profile, request)
    if wire == "bedrock_converse_stream":
        assert payload["system"] == [
            {"text": "a stable prefix"},
            {"cachePoint": {"type": "default"}},
        ]
    elif wire == "anthropic_messages":
        assert payload["system"] == [marked]
    else:
        messages = payload["messages"]
        assert isinstance(messages, list)
        first = messages[0]
        assert isinstance(first, dict)
        assert first["content"] == [marked]


@pytest.mark.parametrize("surface", ["chat", "messages"])
@pytest.mark.parametrize("wire", ["anthropic_messages", "bedrock_converse_stream", "openrouter"])
@pytest.mark.parametrize("media", ["image", "document"])
@pytest.mark.parametrize("ttl", ["5m", "1h"])
@pytest.mark.parametrize(
    "position", ["leading", "after_text", "after_image", "between_images", "adjacent"]
)
def test_empty_multimodal_checkpoint_keeps_its_ordered_prefix(
    surface: str, wire: str, media: str, ttl: str, position: str
) -> None:
    """Dropping empty text must not move a checkpoint across an image."""
    image: JsonObject = (
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGk="}}
        if surface == "chat"
        else {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": "aGk="},
        }
    )
    if media == "document":
        image = (
            {
                "type": "file",
                "file": {
                    "file_data": "data:application/pdf;base64,JVBERi0xLjcK",
                    "filename": "a.pdf",
                },
            }
            if surface == "chat"
            else {
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": "JVBERi0xLjcK",
                },
            }
        )
    marker: JsonObject = {"type": "ephemeral", "ttl": ttl}
    empty: JsonObject = {"type": "text", "text": "", "cache_control": marker}
    prefix: JsonObject = {"type": "text", "text": "prefix"}
    content: list[JsonObject]
    if position == "leading":
        content = [empty, image, prefix]
        expected = []  # Unrepresentable leading checkpoints must refuse before serialization.
    elif position == "after_text":
        content = [prefix, empty, image]
        expected = ["text", "checkpoint", "image"]
    elif position == "between_images":
        content = [prefix, image, empty, image]
        expected = ["text", "image", "checkpoint", "image"]
    else:
        content = [prefix, image, empty, *([empty] if position == "adjacent" else [])]
        expected = ["text", "image", "checkpoint"]
    body: JsonObject = {
        "model": "coding",
        "max_tokens": 32,
        "messages": [{"role": "user", "content": content}],
    }
    if position == "leading":
        with pytest.raises(OpenAIProtocolError, match="preceding content") as refused:
            (decode_chat(body) if surface == "chat" else decode_messages(body))
        assert refused.value.detail.param is not None
        assert "cache_control" in refused.value.detail.param
        return
    request = (decode_chat(body) if surface == "chat" else decode_messages(body)).request
    profile = GatewayWireProfile(
        dialect="openai_compatible" if wire == "openrouter" else wire,
        model_id="claude-sonnet-4-6",
        url="https://example.invalid",
        forwards_cache_control=wire == "openrouter",
        billing_customer_managed=True,
    )
    payload = dialect_stream_payload(profile, request)
    messages = payload["messages"]
    assert isinstance(messages, list) and isinstance(messages[0], dict)
    blocks = messages[0]["content"]
    assert isinstance(blocks, list)
    observed: list[str] = []
    for block in blocks:
        assert isinstance(block, dict)
        if "cachePoint" in block:
            observed.append("checkpoint")
        else:
            observed.append("text" if "text" in block else "image")
            if "cache_control" in block:
                observed.append("checkpoint")
    assert observed == expected
    assert all(part.kind != "text" or part.text for part in request.messages[0].content_parts)


@pytest.mark.parametrize("surface", ["chat", "messages"])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("media", [False, True])
@pytest.mark.parametrize("customer_managed", [False, True])
@pytest.mark.parametrize("wire", ["anthropic_messages", "bedrock_converse_stream", "openrouter"])
def test_conflicting_empty_checkpoint_ttls_refuse_before_dispatch(
    surface: str, reverse: bool, media: bool, customer_managed: bool, wire: str
) -> None:
    """Colocated distinct durations cannot lose one-hour intent before the pricing guard."""
    markers = [
        {"type": "ephemeral", "ttl": ttl} for ttl in (("1h", "5m") if reverse else ("5m", "1h"))
    ]
    body: JsonObject = {
        "model": "coding",
        "max_tokens": 32,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "prefix"},
                    *(
                        [
                            {
                                "type": "image_url",
                                "image_url": {"url": "data:image/png;base64,aGk="},
                            }
                            if surface == "chat"
                            else {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": "aGk=",
                                },
                            }
                        ]
                        if media
                        else []
                    ),
                    *[{"type": "text", "text": "", "cache_control": marker} for marker in markers],
                ],
            }
        ],
    }
    profile = GatewayWireProfile(
        dialect="openai_compatible" if wire == "openrouter" else wire,
        url="https://example.invalid",
        billing_customer_managed=customer_managed,
        forwards_cache_control=wire == "openrouter",
    )
    with pytest.raises(OpenAIProtocolError, match="conflicting cache"):
        request = (decode_chat(body) if surface == "chat" else decode_messages(body)).request
        dialect_stream_payload(profile, request)


@pytest.mark.parametrize("same", [False, True])
def test_media_checkpoint_and_empty_checkpoint_must_agree(same: bool) -> None:
    """A media marker and trailing empty marker share one boundary, not two TTL slots."""
    body: JsonObject = {
        "model": "coding",
        "max_tokens": 32,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/png", "data": "aGk="},
                        "cache_control": {"type": "ephemeral", "ttl": "5m"},
                    },
                    {
                        "type": "text",
                        "text": "",
                        "cache_control": {"type": "ephemeral", "ttl": "5m" if same else "1h"},
                    },
                ],
            }
        ],
    }
    if not same:
        with pytest.raises(OpenAIProtocolError, match="conflicting cache"):
            decode_messages(body)
    else:
        request = decode_messages(body).request
        part = request.messages[0].content_parts[0]
        assert part.kind == "image"
        assert part.cache_control == {
            "type": "ephemeral",
            "ttl": "5m",
        }


@pytest.mark.parametrize("same", [False, True])
@pytest.mark.parametrize("media", [False, True])
def test_chat_message_marker_cannot_overwrite_empty_checkpoint(same: bool, media: bool) -> None:
    """Message-level and explicit final-block hints name the same immutable boundary."""
    content: list[JsonObject] = [{"type": "text", "text": "prefix"}]
    if media:
        content.append({"type": "image_url", "image_url": {"url": "data:image/png;base64,aGk="}})
    content.append(
        {"type": "text", "text": "", "cache_control": {"type": "ephemeral", "ttl": "1h"}}
    )
    body: JsonObject = {
        "model": "coding",
        "messages": [
            {
                "role": "user",
                "content": content,
                "cache_control": {"type": "ephemeral", "ttl": "1h" if same else "5m"},
            }
        ],
    }
    if not same:
        with pytest.raises(OpenAIProtocolError, match="conflicting cache"):
            decode_chat(body)
    else:
        request = decode_chat(body).request
        assert requests_hour_cache(request)
        payload = dialect_stream_payload(
            GatewayWireProfile(
                dialect="anthropic_messages",
                url="https://example.invalid",
                maximum_output_tokens=128_000,
            ),
            request,
        )
        messages = payload["messages"]
        assert isinstance(messages, list) and isinstance(messages[0], dict)
        blocks = messages[0]["content"]
        assert isinstance(blocks, list) and isinstance(blocks[-1], dict)
        assert blocks[-1]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}


@pytest.mark.parametrize("wire", ["anthropic_messages", "bedrock_converse_stream", "openrouter"])
def test_different_cache_durations_at_distinct_boundaries_remain_distinct(wire: str) -> None:
    """Do not mistake checkpoints separated by actual content for a conflict."""
    request = decode_chat(
        {
            "model": "coding",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "first",
                            "cache_control": {"type": "ephemeral", "ttl": "1h"},
                        },
                        {
                            "type": "text",
                            "text": "second",
                            "cache_control": {"type": "ephemeral", "ttl": "5m"},
                        },
                    ],
                }
            ],
        }
    ).request
    profile = GatewayWireProfile(
        dialect="openai_compatible" if wire == "openrouter" else wire,
        url="https://example.invalid",
        billing_customer_managed=True,
        forwards_cache_control=wire == "openrouter",
        maximum_output_tokens=128_000,
    )
    payload = dialect_stream_payload(profile, request)
    assert '"ttl": "1h"' in json.dumps(payload) and '"ttl": "5m"' in json.dumps(payload)


@pytest.mark.parametrize("surface", ["chat", "messages"])
@pytest.mark.parametrize("ttl", ["5m", "1h"])
def test_leading_empty_text_checkpoint_is_not_widened_to_text(surface: str, ttl: str) -> None:
    """Text-only leading checkpoints get the same refusal as leading media checkpoints."""
    body: JsonObject = {
        "model": "coding",
        "max_tokens": 32,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "",
                        "cache_control": {"type": "ephemeral", "ttl": ttl},
                    },
                    {"type": "text", "text": "dynamic"},
                ],
            }
        ],
    }
    with pytest.raises(OpenAIProtocolError, match="preceding content"):
        (decode_chat(body) if surface == "chat" else decode_messages(body))


@pytest.mark.parametrize("surface", ["chat", "messages"])
@pytest.mark.parametrize("ttl", ["5m", "1h"])
@pytest.mark.parametrize("customer_managed", [False, True])
def test_relocated_multimodal_checkpoint_retains_reservation_ttl(
    surface: str, ttl: str, customer_managed: bool
) -> None:
    """Moving a checkpoint onto its image preserves the TTL used for reservation."""
    image: JsonObject = (
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGk="}}
        if surface == "chat"
        else {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": "aGk="},
        }
    )
    body: JsonObject = {
        "model": "coding",
        "max_tokens": 32,
        "messages": [
            {
                "role": "user",
                "content": [
                    image,
                    {
                        "type": "text",
                        "text": "",
                        "cache_control": {"type": "ephemeral", "ttl": ttl},
                    },
                ],
            }
        ],
    }
    request = (decode_chat(body) if surface == "chat" else decode_messages(body)).request
    profile = GatewayWireProfile(
        dialect="bedrock_converse_stream",
        url="https://example.invalid",
        billing_customer_managed=customer_managed,
    )
    assert requests_hour_cache(request) == (ttl == "1h")
    payload = dialect_stream_payload(profile, request)
    messages = payload["messages"]
    assert isinstance(messages, list) and isinstance(messages[0], dict)
    blocks = messages[0]["content"]
    assert isinstance(blocks, list)
    assert blocks[-1] == {"cachePoint": {"type": "default", "ttl": ttl}}


@pytest.mark.parametrize("leading", [False, True])
def test_empty_checkpoint_beside_unsupported_media_is_disclosed(leading: bool) -> None:
    """An unmarkable boundary is omitted explicitly, never shifted past video."""
    empty: JsonObject = {"type": "text", "text": "", "cache_control": {"type": "ephemeral"}}
    video: JsonObject = {"type": "video_url", "video_url": {"url": "https://example.test/clip.mp4"}}
    parts = [empty, video] if leading else [video, empty]
    body: JsonObject = {"model": "coding", "messages": [{"role": "user", "content": parts}]}
    if leading:
        with pytest.raises(OpenAIProtocolError, match="preceding content"):
            decode_chat(body)
        return
    request = decode_chat(body).request
    assert request.ignored_parameters == (
        "messages.0.content.cache_control->dropped(unsupported_media)",
    )
    assert len(request.messages[0].content_parts) == 1
    assert request.messages[0].provider_text_blocks == ()


def test_generic_wire_discloses_dropped_relocated_checkpoint() -> None:
    """No marker capability is inferred for generic Chat after carrier correction."""
    request = decode_chat(
        {
            "model": "coding",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGk="}},
                        {"type": "text", "text": "", "cache_control": {"type": "ephemeral"}},
                    ],
                }
            ],
        }
    ).request
    profile = GatewayWireProfile(dialect="openai_compatible", url="https://example.invalid")
    public, _ = route_generation_parameter_requests((profile,), request)
    assert any(
        item.startswith("messages.content.cache_control->not_forwarded(")
        for item in public.ignored_parameters
    )
    assert "cache_control" not in json.dumps(dialect_stream_payload(profile, request))


def test_unknown_chat_adapter_does_not_receive_unsupported_markers() -> None:
    """A generic endpoint is not declared cache-capable merely for speaking Chat."""
    request = decode_chat(
        {
            "model": "coding",
            "messages": [
                {"role": "user", "content": "hello", "cache_control": {"type": "ephemeral"}}
            ],
        }
    ).request
    profile = GatewayWireProfile(dialect="openai_compatible", url="https://example.invalid")
    payload = dialect_stream_payload(profile, request)
    assert not profile.preserves_cache_control
    assert payload["messages"] == [{"role": "user", "content": "hello"}]


def test_chat_marker_keeps_exact_text_boundaries() -> None:
    """A breakpoint before a dynamic suffix stays at that prefix boundary."""
    blocks: list[JsonObject] = [
        {"type": "text", "text": "prefix", "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": "dynamic suffix"},
    ]
    request = decode_chat(
        {"model": "coding", "messages": [{"role": "user", "content": blocks}]}
    ).request
    assert request.messages[0].content == "prefixdynamic suffix"
    assert request.messages[0].provider_text_blocks == tuple(blocks)


@pytest.mark.parametrize("wire", ["bedrock_converse_stream", "openrouter"])
def test_folded_instruction_keeps_prior_checkpoint(wire: str) -> None:
    """A dynamic system reminder must not erase or extend the user's cached prefix."""
    request = decode_chat(
        {
            "model": "coding",
            "messages": [
                {
                    "role": "user",
                    "content": "stable prefix",
                    "cache_control": {"type": "ephemeral"},
                },
                {"role": "system", "content": "dynamic reminder"},
            ],
        }
    ).request
    profile = GatewayWireProfile(
        dialect="openai_compatible" if wire == "openrouter" else wire,
        model_id="claude-sonnet-4-6",
        url="https://example.invalid",
        forwards_cache_control=wire == "openrouter",
        system_messages_leading_only=wire == "openrouter",
    )
    payload = dialect_stream_payload(profile, request)
    messages = payload["messages"]
    assert isinstance(messages, list)
    first = messages[0]
    assert isinstance(first, dict)
    if wire == "bedrock_converse_stream":
        assert first["content"] == [
            {"text": "stable prefix"},
            {"cachePoint": {"type": "default"}},
            {"text": "\n\n"},
            {"text": "dynamic reminder"},
        ]
    else:
        assert first["content"] == [
            {"type": "text", "text": "stable prefix", "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": "\n\n"},
            {"type": "text", "text": "dynamic reminder"},
        ]


@pytest.mark.parametrize(
    "media",
    [
        {"type": "video_url", "video_url": {"url": "https://example.test/clip.mp4"}},
        {"type": "input_audio", "input_audio": {"data": "AAAA", "format": "wav"}},
    ],
)
def test_unsupported_media_marker_drops_with_disclosure(media: JsonObject) -> None:
    """A cache hint cannot turn an accepted media request into a 400."""
    request = decode_chat(
        {
            "model": "coding",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "prefix", "cache_control": {"type": "ephemeral"}},
                        media,
                    ],
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        }
    ).request
    assert request.ignored_parameters == ("messages.0.cache_control->dropped(unsupported_media)",)
    assert len(request.messages[0].content_parts) == 2
    assert request.messages[0].provider_text_blocks == (
        {"type": "text", "text": "prefix", "cache_control": {"type": "ephemeral"}},
    )
