"""Tests for the disclosed degrade of unservable inline media in a Messages history."""

from __future__ import annotations

import base64
import copy
from typing import cast

import pytest

from exp.common.core.artifacts import JsonObject
from exp.common.models.content import (
    MAXIMUM_DOCUMENT_BASE64_BYTES,
    MAXIMUM_IMAGE_BASE64_BYTES,
)
from exp.runtime.anthropic_protocol.history_media import degrade_history_media, unservable_reason
from exp.runtime.anthropic_protocol.requests import decode_messages
from exp.runtime.openai_protocol.errors import OpenAIProtocolError
from exp.runtime.openai_protocol.history_media import placeholder_text

_PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8"
    "z8DwHwAFAAH/q842iQAAAABJRU5ErkJggg=="
)
"""One valid single-pixel PNG, base64 encoded."""

_OVERSIZED_IMAGE = "A" * (MAXIMUM_IMAGE_BASE64_BYTES + 4)
"""Well-formed base64 one quantum past the image ceiling."""


def _image(data: str, media_type: str = "image/png", **extra: object) -> JsonObject:
    """Return one base64 image block."""
    block: JsonObject = {
        "type": "image",
        "source": {"type": "base64", "media_type": media_type, "data": data},
    }
    block.update(cast(JsonObject, extra))
    return block


def _pdf(data: str) -> JsonObject:
    """Return one base64 PDF document block."""
    return {
        "type": "document",
        "source": {"type": "base64", "media_type": "application/pdf", "data": data},
    }


def _tool_turns(part: JsonObject, *, latest: str | None = "next") -> list[JsonObject]:
    """Return a Claude Code shaped history whose tool result carries ``part``."""
    turns: list[JsonObject] = [
        {"role": "user", "content": "look at the screen"},
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {}}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_1",
                    "content": [{"type": "text", "text": "screenshot:"}, part],
                }
            ],
        },
    ]
    if latest is not None:
        turns += [
            {"role": "assistant", "content": "I see it."},
            {"role": "user", "content": latest},
        ]
    return turns


def _body(messages: list[JsonObject]) -> JsonObject:
    """Return one minimal Messages body around ``messages``."""
    return {"model": "coding", "max_tokens": 64, "messages": cast(list[object], messages)}


def test_unservable_reason_judges_only_inline_base64_sources() -> None:
    """Size, decodability, and media type are judged; other sources pass through."""
    assert unservable_reason("image", _image(_PNG_BASE64)) is None
    assert unservable_reason("image", _image(_OVERSIZED_IMAGE)) == "oversized"
    assert unservable_reason("image", _image("not base64!")) == "undecodable"
    assert unservable_reason("image", _image(_PNG_BASE64, "image/bmp")) == (
        "unsupported_media_type"
    )
    assert unservable_reason("image", {"type": "image", "source": {"type": "url"}}) is None
    assert unservable_reason("image", {"type": "image"}) is None
    pdf = base64.b64encode(b"%PDF-1.4").decode()
    assert unservable_reason("document", _pdf(pdf)) is None
    assert unservable_reason("document", _pdf("A" * (MAXIMUM_DOCUMENT_BASE64_BYTES + 4))) == (
        "oversized"
    )


def test_placeholder_text_states_the_kind_and_the_limit() -> None:
    """The model reads which block was omitted and why."""
    assert placeholder_text("image", "oversized") == (
        "[image omitted from conversation history: it is larger than the gateway's 20 MiB limit]"
    )
    assert "32 MiB" in placeholder_text("document", "oversized")
    assert "not valid base64" in placeholder_text("image", "undecodable")
    assert "not a PDF" in placeholder_text("document", "unsupported_media_type")


def test_a_servable_history_is_returned_unchanged() -> None:
    """Nothing to degrade means the same payload object and no disclosure."""
    payload = _body(_tool_turns(_image(_PNG_BASE64)))
    assert degrade_history_media(payload) == (payload, ())
    assert degrade_history_media({"messages": "not a list"}) == ({"messages": "not a list"}, ())


def test_an_oversized_tool_result_image_degrades_without_mutating_the_payload() -> None:
    """The bricking case: a screenshot baked into a tool result becomes a placeholder."""
    payload = _body(_tool_turns(_image(_OVERSIZED_IMAGE, cache_control={"type": "ephemeral"})))
    original = copy.deepcopy(payload)
    degraded, disclosures = degrade_history_media(payload)
    assert payload == original
    assert disclosures == ("messages.content.tool_result.image->placeholder(oversized)",)
    tool_result = cast(list[JsonObject], cast(list[JsonObject], degraded["messages"])[2]["content"])
    inner = cast(list[JsonObject], tool_result[0]["content"])
    assert inner[1] == {
        "type": "text",
        "text": placeholder_text("image", "oversized"),
        "cache_control": {"type": "ephemeral"},
    }


def test_a_tool_result_image_degrades_in_the_latest_turn_too() -> None:
    """A tool's own output is history the moment the tool returns it."""
    payload = _body(_tool_turns(_image("@@not-base64@@"), latest=None))
    _, disclosures = degrade_history_media(payload)
    assert disclosures == ("messages.content.tool_result.image->placeholder(undecodable)",)


def test_an_earlier_top_level_image_degrades_and_disclosures_stay_bounded() -> None:
    """Earlier user-turn media degrades once per reason, however many blocks carried it."""
    turns: list[JsonObject] = [
        {"role": "user", "content": [_image(_OVERSIZED_IMAGE), _image(_OVERSIZED_IMAGE)]},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": [_pdf("A" * (MAXIMUM_DOCUMENT_BASE64_BYTES + 4))]},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "and now?"},
    ]
    degraded, disclosures = degrade_history_media(_body(turns))
    assert disclosures == (
        "messages.content.image->placeholder(oversized)",
        "messages.content.document->placeholder(oversized)",
    )
    first = cast(list[JsonObject], cast(list[JsonObject], degraded["messages"])[0]["content"])
    assert [block["type"] for block in first] == ["text", "text"]


def test_an_oversized_image_in_the_latest_user_turn_is_a_typed_400_stating_the_limit() -> None:
    """The caller can still resend the newest image, so it fails closed with the limit."""
    turns: list[JsonObject] = [
        {"role": "user", "content": [{"type": "text", "text": "what is this"}, _image("A" * 8)]},
        {"role": "assistant", "content": "a pixel"},
        {"role": "user", "content": [_image(_OVERSIZED_IMAGE)]},
    ]
    with pytest.raises(OpenAIProtocolError) as info:
        degrade_history_media(_body(turns))
    assert info.value.status_code == 400
    assert info.value.detail.param == "messages.2.content.0.source.data"
    assert f"at most {MAXIMUM_IMAGE_BASE64_BYTES:,} base64 characters" in info.value.detail.message
    assert "20 MiB decoded" in info.value.detail.message


def test_an_undecodable_image_in_the_latest_user_turn_keeps_the_decoder_rejection() -> None:
    """Only history degrades: the newest top-level image still fails closed."""
    payload = _body([{"role": "user", "content": [_image("@@not-base64@@")]}])
    assert degrade_history_media(payload) == (payload, ())
    with pytest.raises(OpenAIProtocolError) as info:
        decode_messages(payload)
    assert info.value.detail.param == "messages.0.content.0.source"


def test_decode_serves_a_history_whose_tool_result_image_was_oversized() -> None:
    """End to end: the replayed history decodes and the degrade is disclosed."""
    decoded = decode_messages(_body(_tool_turns(_image(_OVERSIZED_IMAGE))))
    request = decoded.request
    assert "messages.content.tool_result.image->placeholder(oversized)" in (
        request.ignored_parameters
    )
    tool_message = next(message for message in request.messages if message.role == "tool")
    assert not tool_message.images
    assert placeholder_text("image", "oversized") in (tool_message.content or "")


def test_a_messages_placeholder_carries_only_the_cache_control_key() -> None:
    """A Messages text block forbids ``prompt_cache_breakpoint``, so only the marker moves."""
    part = _image(
        _OVERSIZED_IMAGE,
        cache_control={"type": "ephemeral"},
        prompt_cache_breakpoint=True,
    )
    degraded, _ = degrade_history_media(_body(_tool_turns(part)))
    turn = cast(JsonObject, cast(list[object], degraded["messages"])[2])
    result = cast(JsonObject, cast(list[object], turn["content"])[0])
    placeholder = cast(JsonObject, cast(list[object], result["content"])[1])
    assert placeholder["cache_control"] == {"type": "ephemeral"}
    assert "prompt_cache_breakpoint" not in placeholder


def test_an_empty_final_prefill_does_not_turn_the_latest_user_turn_into_history() -> None:
    """The decoder drops an empty prefill, so the user turn before it stays strict."""
    payload = _body(
        [
            {"role": "user", "content": [_image(_OVERSIZED_IMAGE)]},
            {"role": "assistant", "content": ""},
        ]
    )
    with pytest.raises(OpenAIProtocolError) as info:
        degrade_history_media(payload)
    assert info.value.detail.param == "messages.0.content.0.source.data"


def test_a_marked_empty_final_turn_still_counts_as_the_models_turn() -> None:
    """The decoder keeps (and refuses) a marked empty turn, so earlier media is history."""
    payload = _body(
        [
            {"role": "user", "content": [_image(_OVERSIZED_IMAGE)]},
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "", "cache_control": {"type": "ephemeral"}}],
            },
        ]
    )
    degraded, disclosures = degrade_history_media(payload)
    assert degraded is not payload
    assert disclosures == ("messages.content.image->placeholder(oversized)",)
