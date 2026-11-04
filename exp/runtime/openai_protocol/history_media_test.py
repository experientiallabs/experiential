"""Tests for the disclosed degrade of unservable inline media in Chat and Responses histories."""

from __future__ import annotations

import base64
import binascii
import copy
import random
from typing import cast

import pytest

from exp.common.core.artifacts import JsonObject
from exp.common.models.content import MAXIMUM_DOCUMENT_BASE64_BYTES, MAXIMUM_IMAGE_BASE64_BYTES
from exp.runtime.openai_protocol.errors import OpenAIProtocolError
from exp.runtime.openai_protocol.history_media import (
    base64_decodable,
    data_url_reason,
    degrade_chat_media,
    degrade_responses_media,
    inline_reason,
    oversized_error,
    placeholder_text,
)
from exp.runtime.openai_protocol.requests import decode_chat, decode_responses

_PNG = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8"
    "z8DwHwAFAAH/q842iQAAAABJRU5ErkJggg=="
)
"""One valid single-pixel PNG, base64 encoded."""

_PNG_URL = f"data:image/png;base64,{_PNG}"
_OVERSIZED_URL = "data:image/png;base64," + "A" * (MAXIMUM_IMAGE_BASE64_BYTES + 4)
_OVERSIZED_PDF = "data:application/pdf;base64," + "A" * (MAXIMUM_DOCUMENT_BASE64_BYTES + 4)


def _chat_image(url: str) -> JsonObject:
    """Return one Chat ``image_url`` part."""
    return {"type": "image_url", "image_url": {"url": url}}


def _chat_body(messages: list[JsonObject]) -> JsonObject:
    """Return one minimal Chat Completions body around ``messages``."""
    return {"model": "coding", "messages": cast(list[object], messages)}


def _chat_tool_history(tool_image: JsonObject, latest: str = "go on") -> list[JsonObject]:
    """Return a Chat history whose tool message carries ``tool_image``."""
    return [
        {"role": "user", "content": "take a screenshot"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "screenshot", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": [{"type": "text", "text": "done"}, tool_image],
        },
        {"role": "user", "content": latest},
    ]


def test_inline_reason_and_data_url_reason_mirror_the_canonical_parsers() -> None:
    """Size, base64, and media type are judged; remote URLs pass through."""
    assert inline_reason("image", "image/png", _PNG) is None
    assert inline_reason("image", "IMAGE/PNG", _PNG) is None
    assert inline_reason("image", "image/bmp", _PNG) == "unsupported_media_type"
    assert inline_reason("image", None, "not base64!") == "undecodable"
    assert data_url_reason("image", "https://example.com/a.png", bare_base64=False) is None
    assert data_url_reason("image", _PNG_URL, bare_base64=False) is None
    assert data_url_reason("image", _OVERSIZED_URL, bare_base64=False) == "oversized"
    assert data_url_reason("image", "data:image/png,raw", bare_base64=False) == "undecodable"
    assert data_url_reason("image", f"data:image/bmp;base64,{_PNG}", bare_base64=False) == (
        "unsupported_media_type"
    )
    assert data_url_reason("document", "JVBERi0xLjcK", bare_base64=True) is None
    assert data_url_reason("document", "%%%", bare_base64=True) == "undecodable"
    assert data_url_reason("document", _OVERSIZED_PDF, bare_base64=True) == "oversized"


def test_oversized_error_states_the_limit() -> None:
    """The latest-turn rejection names the field and both forms of the ceiling."""
    error = oversized_error("document", "input.0.content.0.file_data", 50_000_000)
    assert error.status_code == 400
    assert error.detail.param == "input.0.content.0.file_data"
    assert f"at most {MAXIMUM_DOCUMENT_BASE64_BYTES:,} base64 characters" in error.detail.message
    assert "32 MiB decoded" in error.detail.message


def test_chat_tool_image_and_earlier_user_media_degrade_with_bounded_disclosures() -> None:
    """Tool screenshots and earlier user attachments become placeholders, once per reason."""
    messages = _chat_tool_history(_chat_image("data:image/png;base64,@@@"))
    messages.insert(
        0,
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "see these"},
                {**_chat_image(_OVERSIZED_URL), "cache_control": {"type": "ephemeral"}},
                _chat_image(_OVERSIZED_URL),
                {"type": "file", "file": {"file_data": _OVERSIZED_PDF, "filename": "a.pdf"}},
            ],
        },
    )
    messages.insert(1, {"role": "assistant", "content": "ok"})
    payload = _chat_body(messages)
    original = copy.deepcopy(payload)
    degraded, disclosures = degrade_chat_media(payload)
    assert payload == original
    assert disclosures == (
        "messages.content.image->placeholder(oversized)",
        "messages.content.document->placeholder(oversized)",
        "messages.tool.content.image->placeholder(undecodable)",
    )
    first = cast(list[JsonObject], cast(list[JsonObject], degraded["messages"])[0]["content"])
    assert first[1] == {
        "type": "text",
        "text": placeholder_text("image", "oversized"),
        "cache_control": {"type": "ephemeral"},
    }
    assert first[3]["type"] == "text"


def test_chat_latest_user_media_stays_strict() -> None:
    """An oversized latest attachment is a typed 400 that states the limit."""
    image = _chat_body([{"role": "user", "content": [_chat_image(_OVERSIZED_URL)]}])
    with pytest.raises(OpenAIProtocolError) as info:
        degrade_chat_media(image)
    assert info.value.detail.param == "messages.0.content.0.image_url.url"
    assert "20 MiB decoded" in info.value.detail.message
    file_part: JsonObject = {"type": "file", "file": {"file_data": _OVERSIZED_PDF}}
    with pytest.raises(OpenAIProtocolError) as file_info:
        degrade_chat_media(_chat_body([{"role": "user", "content": [file_part]}]))
    assert file_info.value.detail.param == "messages.0.content.0.file.file_data"
    undecodable = _chat_body([{"role": "user", "content": [_chat_image("data:image/png,raw")]}])
    assert degrade_chat_media(undecodable) == (undecodable, ())
    with pytest.raises(OpenAIProtocolError):
        decode_chat(undecodable)


def test_decode_chat_serves_a_history_whose_tool_image_was_oversized() -> None:
    """End to end: the replayed Chat history decodes and discloses the degrade."""
    decoded = decode_chat(_chat_body(_chat_tool_history(_chat_image(_OVERSIZED_URL))))
    assert "messages.tool.content.image->placeholder(oversized)" in (
        decoded.request.ignored_parameters
    )
    tool = next(message for message in decoded.request.messages if message.role == "tool")
    assert not tool.images
    assert placeholder_text("image", "oversized") in (tool.content or "")


def _responses_history(output_part: JsonObject) -> list[JsonObject]:
    """Return a Responses input whose function output carries ``output_part``."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "read it"},
                {"type": "input_file", "file_data": "%%%", "filename": "old.pdf"},
            ],
        },
        {"type": "function_call", "call_id": "call_1", "name": "read", "arguments": "{}"},
        {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": [{"type": "input_text", "text": "page:"}, output_part],
        },
        {"role": "user", "content": [{"type": "input_text", "text": "summarize"}]},
    ]


def test_responses_function_output_and_earlier_input_degrade() -> None:
    """Function outputs and earlier messages degrade with Responses text tags."""
    payload: JsonObject = {
        "model": "coding",
        "input": cast(
            list[object],
            _responses_history({"type": "input_image", "image_url": _OVERSIZED_URL}),
        ),
    }
    degraded, disclosures = degrade_responses_media(payload)
    assert disclosures == (
        "input.content.document->placeholder(undecodable)",
        "input.function_call_output.output.image->placeholder(oversized)",
    )
    items = cast(list[JsonObject], degraded["input"])
    output = cast(list[JsonObject], items[2]["output"])
    assert output[1] == {"type": "input_text", "text": placeholder_text("image", "oversized")}
    decoded = decode_responses(payload)
    assert set(disclosures) <= set(decoded.request.ignored_parameters)
    assert not decoded.request.documents
    assert not decoded.request.images


def test_responses_latest_user_media_stays_strict() -> None:
    """An oversized latest input image is a typed 400 that states the limit."""
    payload: JsonObject = {
        "model": "coding",
        "input": [
            {"role": "user", "content": [{"type": "input_image", "image_url": _OVERSIZED_URL}]}
        ],
    }
    with pytest.raises(OpenAIProtocolError) as info:
        decode_responses(payload)
    assert info.value.detail.param == "input.0.content.0.image_url"
    assert f"at most {MAXIMUM_IMAGE_BASE64_BYTES:,} base64 characters" in info.value.detail.message
    assert degrade_responses_media({"model": "coding", "input": "hi"}) == (
        {"model": "coding", "input": "hi"},
        (),
    )


@pytest.mark.parametrize("chunk", [4, 8, 12])
def test_base64_decodable_matches_strict_decoding_chunk_by_chunk(chunk: int) -> None:
    """The bounded validator gives ``b64decode(validate=True)``'s verdict at every boundary."""
    rng = random.Random(1291)
    alphabet = "AB+/=\n é-_"
    samples = ["", "QQ==", "QUI=", "QUJD", "QQ==QUJD", "QUJDRA==", "QUJD\n", "=QUJ", "QUJ"]
    samples += [
        "".join(rng.choice(alphabet) for _ in range(rng.randrange(0, 30))) for _ in range(4000)
    ]
    samples += [base64.b64encode(rng.randbytes(rng.randrange(0, 40))).decode() for _ in range(500)]
    for sample in samples:
        try:
            base64.b64decode(sample, validate=True)
            expected = True
        except (binascii.Error, ValueError):
            expected = False
        assert base64_decodable(sample, chunk=chunk) is expected, sample


def test_a_latest_turn_part_is_judged_by_size_alone() -> None:
    """The latest turn's media is never base64-validated here: the strict decoder owns it."""
    assert data_url_reason("image", "data:image/png;base64,@@@@", bare_base64=False) == (
        "undecodable"
    )
    assert (
        data_url_reason(
            "image", "data:image/png;base64,@@@@", bare_base64=False, check_decodable=False
        )
        is None
    )
    assert inline_reason("image", None, "A" * 8, check_decodable=False) is None


def test_a_malformed_data_url_in_history_is_undecodable_not_remote() -> None:
    """A ``data:`` value that does not parse degrades instead of wedging as a non-HTTP URL."""
    assert data_url_reason("image", "data:image/png;base64", bare_base64=False) == "undecodable"
    assert data_url_reason("image", "https://example.com/a.png", bare_base64=False) is None
    payload = _chat_body(_chat_tool_history(_chat_image("data:image/png;base64")))
    degraded, disclosures = degrade_chat_media(payload)
    assert disclosures == ("messages.tool.content.image->placeholder(undecodable)",)
    assert degraded is not payload


def test_every_user_message_of_the_latest_turn_stays_strict() -> None:
    """An image sent, then its question, are one turn: the image still gets the typed 400."""
    payload = _chat_body(
        [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": [_chat_image(_OVERSIZED_URL)]},
            {"role": "user", "content": "what is this?"},
        ]
    )
    with pytest.raises(OpenAIProtocolError) as info:
        degrade_chat_media(payload)
    assert info.value.detail.param == "messages.2.content.0.image_url.url"
    responses: JsonObject = {
        "model": "coding",
        "input": [
            {"role": "user", "content": [{"type": "input_image", "image_url": _OVERSIZED_URL}]},
            {"role": "user", "content": [{"type": "input_text", "text": "what is this?"}]},
        ],
    }
    with pytest.raises(OpenAIProtocolError):
        degrade_responses_media(responses)


def test_an_unchanged_history_is_returned_without_copies() -> None:
    """A servable history allocates no rebuilt lists: the same payload object comes back."""
    payload = _chat_body(_chat_tool_history(_chat_image(_PNG_URL)))
    degraded, disclosures = degrade_chat_media(payload)
    assert degraded is payload and disclosures == ()


@pytest.mark.parametrize(
    "item_type", ["compaction", "program", "program_output", "tool_search_output"]
)
def test_every_provider_authored_item_ends_the_latest_turn(item_type: str) -> None:
    """A hosted output item is the model speaking, so earlier user media is history."""
    payload: JsonObject = {
        "model": "coding",
        "input": [
            {"role": "user", "content": [{"type": "input_image", "image_url": _OVERSIZED_URL}]},
            {"type": item_type, "id": "item_1"},
            {"role": "user", "content": [{"type": "input_text", "text": "next"}]},
        ],
    }
    degraded, disclosures = degrade_responses_media(payload)
    assert disclosures == ("input.content.image->placeholder(oversized)",)
    assert degraded is not payload
