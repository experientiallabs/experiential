"""Tests for keeping a long conversation under the per-request document ceiling."""

from __future__ import annotations

from typing import cast

import pytest

from exp.common.core.artifacts import JsonObject
from exp.common.models.content import (
    MAXIMUM_DOCUMENTS_PER_REQUEST,
    DocumentContentPart,
    TextContentPart,
)
from exp.common.models.model import ToolCall
from exp.runtime.anthropic_protocol.requests import decode_messages
from exp.runtime.gateway.contracts import GatewayMessage
from exp.runtime.openai_protocol.errors import OpenAIProtocolError
from exp.runtime.openai_protocol.history_documents import (
    DOCUMENT_COUNT_DISCLOSURE,
    document_count_placeholder,
    trim_history_documents,
)
from exp.runtime.openai_protocol.requests import decode_chat

_PDF = "JVBERi0xLjcK"
"""A tiny base64 PDF header."""


def _user_with_document(name: str, *, marked: bool = False) -> GatewayMessage:
    """Return one user message carrying text and one named document."""
    return GatewayMessage(
        role="user",
        content="see",
        content_parts=(
            TextContentPart(text="see"),
            DocumentContentPart(
                data=_PDF,
                name=name,
                cache_control={"type": "ephemeral"} if marked else None,
            ),
        ),
    )


def _turns(count: int) -> list[GatewayMessage]:
    """Return ``count`` user turns each with one document, answered in between."""
    turns: list[GatewayMessage] = []
    for index in range(count):
        turns.append(_user_with_document(f"doc{index}.pdf", marked=index == 0))
        turns.append(GatewayMessage(role="assistant", content="ok"))
    return turns


def test_a_request_under_the_ceiling_is_unchanged() -> None:
    """Nothing is replaced while the documents fit."""
    messages = tuple(_turns(MAXIMUM_DOCUMENTS_PER_REQUEST))
    assert trim_history_documents(messages) == (messages, ())


def test_the_oldest_history_documents_become_placeholders() -> None:
    """Seven documents keep the five most recent and disclose once."""
    messages = (*_turns(6), _user_with_document("latest.pdf"))
    trimmed, disclosures = trim_history_documents(messages)
    assert disclosures == (DOCUMENT_COUNT_DISCLOSURE,)
    kept = [part.name for message in trimmed for part in message.documents]
    assert kept == ["doc2.pdf", "doc3.pdf", "doc4.pdf", "doc5.pdf", "latest.pdf"]
    first = trimmed[0]
    assert first.content_parts == ()
    assert first.content == "see" + document_count_placeholder("doc0.pdf")
    # The cache marker the oldest document carried stays at its position.
    assert first.provider_text_blocks[-1]["cache_control"] == {"type": "ephemeral"}


def test_tool_documents_are_history_even_in_the_latest_turn() -> None:
    """A document a tool returned is replaced before any latest-turn document."""
    tool = GatewayMessage(
        role="tool",
        tool_call_id="call_1",
        content="read",
        content_parts=(TextContentPart(text="read"), DocumentContentPart(data=_PDF, name="t.pdf")),
    )
    latest = [_user_with_document(f"new{index}.pdf") for index in range(5)]
    trimmed, disclosures = trim_history_documents((tool, *latest))
    assert disclosures == (DOCUMENT_COUNT_DISCLOSURE,)
    assert trimmed[0].content == "read" + document_count_placeholder("t.pdf")
    assert not trimmed[0].content_parts
    assert trimmed[1:] == tuple(latest)


def test_a_latest_turn_over_the_ceiling_stays_strict() -> None:
    """The latest turn's own documents are never replaced."""
    latest = tuple(_user_with_document(f"new{index}.pdf") for index in range(6))
    assert trim_history_documents(latest) == (latest, ())


def _claude_code_pdf_reads(count: int) -> list[JsonObject]:
    """Return a Messages history with ``count`` Read results, each a PDF."""
    turns: list[JsonObject] = [{"role": "user", "content": "read the papers"}]
    for index in range(count):
        turns.append(
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": f"toolu_{index}", "name": "Read", "input": {}}
                ],
            }
        )
        turns.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": f"toolu_{index}",
                        "content": [
                            {
                                "type": "document",
                                "source": {
                                    "type": "base64",
                                    "media_type": "application/pdf",
                                    "data": _PDF,
                                },
                            }
                        ],
                    }
                ],
            }
        )
    return turns


def test_a_long_messages_session_keeps_serving_past_the_document_ceiling() -> None:
    """Eight PDF reads in one session decode, keeping the five most recent."""
    decoded = decode_messages(
        {
            "model": "coding",
            "max_tokens": 64,
            "messages": cast(list[object], _claude_code_pdf_reads(8)),
        }
    )
    assert len(decoded.request.documents) == MAXIMUM_DOCUMENTS_PER_REQUEST
    assert DOCUMENT_COUNT_DISCLOSURE in decoded.request.ignored_parameters
    tools = [message for message in decoded.request.messages if message.role == "tool"]
    assert [bool(message.documents) for message in tools] == [False] * 3 + [True] * 5


def test_chat_latest_turn_over_the_ceiling_keeps_the_canonical_rejection() -> None:
    """Six files in the newest Chat message still fail with the count ceiling."""
    parts: list[object] = [
        {"type": "file", "file": {"file_data": f"data:application/pdf;base64,{_PDF}"}}
        for _ in range(MAXIMUM_DOCUMENTS_PER_REQUEST + 1)
    ]
    with pytest.raises(OpenAIProtocolError) as info:
        decode_chat({"model": "coding", "messages": [{"role": "user", "content": parts}]})
    assert f"at most {MAXIMUM_DOCUMENTS_PER_REQUEST} documents" in info.value.detail.message


def test_a_collapsed_tool_result_keeps_its_document_marker_on_the_message() -> None:
    """A tool result left text-only carries the replaced document's breakpoint on the message."""
    marked_tool = GatewayMessage(
        role="tool",
        tool_call_id="call_0",
        content="read",
        content_parts=(
            TextContentPart(text="read"),
            DocumentContentPart(data=_PDF, name="old.pdf", cache_control={"type": "ephemeral"}),
        ),
    )
    messages = (
        GatewayMessage(role="user", content="go"),
        GatewayMessage(
            role="assistant",
            tool_calls=(ToolCall(call_id="call_0", name="Read", arguments={}),),
        ),
        marked_tool,
        *_turns(MAXIMUM_DOCUMENTS_PER_REQUEST),
        GatewayMessage(role="user", content="next"),
    )
    trimmed, disclosures = trim_history_documents(messages)
    assert disclosures == (DOCUMENT_COUNT_DISCLOSURE,)
    tool = trimmed[2]
    assert tool.content_parts == ()
    assert tool.cache_control == {"type": "ephemeral"}
