"""Keep a long conversation under the per-request document ceiling.

Every surface admits at most :data:`MAXIMUM_DOCUMENTS_PER_REQUEST` documents
in one request (the canonical request's attachment ceiling, enforced on every
route). A client that resends its whole history each turn crosses that line
once its conversation has accumulated enough PDFs (each Read of a PDF by a
coding agent adds one), and from then on every turn fails identically.

So when a decoded request carries more documents than the ceiling, the
OLDEST history documents become text placeholders until the request fits,
keeping the most recent ones, and the substitution is disclosed in
``ignored_parameters``. The latest user turn (every user message after the
last assistant message) stays strict: its documents are never replaced, and
a turn that alone exceeds the ceiling keeps the canonical rejection. Tool
outputs are history the moment the tool returns them, so their documents
are candidates in every position.
"""

from __future__ import annotations

from exp.common.models.content import (
    MAXIMUM_DOCUMENTS_PER_REQUEST,
    MessageContentPart,
    TextContentPart,
)
from exp.runtime.gateway.contracts import GatewayMessage
from exp.runtime.models.providers.cache_policy import multimodal_text_cache_blocks

DOCUMENT_COUNT_DISCLOSURE = "messages.content.document->placeholder(document_count_limit)"
"""Disclosure recorded when history documents are replaced to fit the ceiling."""


def document_count_placeholder(name: str | None) -> str:
    """Return the text left where an older history document was.

    Args:
        name: The document's caller filename, if any.

    Returns:
        One bracketed line the model reads where the document was.
    """
    label = f" {name!r}" if name else ""
    return (
        f"[document{label} omitted from conversation history: a request carries at most "
        f"{MAXIMUM_DOCUMENTS_PER_REQUEST} documents, so only the most recent are kept]"
    )


def _latest_turn(messages: tuple[GatewayMessage, ...]) -> frozenset[int]:
    """Return the indexes of the user messages after the last assistant message."""
    last_assistant = max(
        (index for index, message in enumerate(messages) if message.role == "assistant"),
        default=-1,
    )
    return frozenset(
        index
        for index, message in enumerate(messages)
        if index > last_assistant and message.role == "user"
    )


def _replace_documents(message: GatewayMessage, positions: frozenset[int]) -> GatewayMessage:
    """Rebuild one message with the documents at ``positions`` as placeholders.

    Args:
        message: The canonical message.
        positions: Indexes into ``message.content_parts`` to replace.

    Returns:
        The rebuilt message, whose content still flattens its text parts.
    """
    parts: tuple[MessageContentPart, ...] = tuple(
        TextContentPart(
            text=document_count_placeholder(part.name), cache_control=part.cache_control
        )
        if index in positions and part.kind == "document"
        else part
        for index, part in enumerate(message.content_parts)
    )
    content = "".join(part.text for part in parts if part.kind == "text")
    blocks = () if message.role == "tool" else multimodal_text_cache_blocks(parts)
    cache_control = message.cache_control
    if all(part.kind == "text" for part in parts):
        if message.role == "tool" and cache_control is None:
            # A text-only tool result carries its breakpoint on the message,
            # so the last part marker (the replaced document's included) moves there.
            cache_control = next(
                (
                    part.cache_control
                    for part in reversed(parts)
                    if isinstance(part, TextContentPart) and part.cache_control is not None
                ),
                None,
            )
        parts = ()
    return message.model_copy(
        update={
            "content": content,
            "content_parts": parts,
            "provider_text_blocks": blocks,
            "cache_control": cache_control,
        }
    )


def trim_history_documents(
    messages: tuple[GatewayMessage, ...],
) -> tuple[tuple[GatewayMessage, ...], tuple[str, ...]]:
    """Replace the oldest history documents until the request fits the ceiling.

    Args:
        messages: The decoded canonical messages, in conversation order.

    Returns:
        The messages to serve and the disclosures, ``(messages, ())`` when
        the request already fits or its latest turn alone exceeds the ceiling.
    """
    located = [
        (message_index, part_index)
        for message_index, message in enumerate(messages)
        for part_index, part in enumerate(message.content_parts)
        if part.kind == "document"
    ]
    if len(located) <= MAXIMUM_DOCUMENTS_PER_REQUEST:
        return messages, ()
    latest = _latest_turn(messages)
    history = [entry for entry in located if entry[0] not in latest]
    if len(located) - len(history) > MAXIMUM_DOCUMENTS_PER_REQUEST:
        return messages, ()
    replaced: dict[int, set[int]] = {}
    for message_index, part_index in history[: len(located) - MAXIMUM_DOCUMENTS_PER_REQUEST]:
        replaced.setdefault(message_index, set()).add(part_index)
    rebuilt = tuple(
        _replace_documents(message, frozenset(replaced[index])) if index in replaced else message
        for index, message in enumerate(messages)
    )
    return rebuilt, (DOCUMENT_COUNT_DISCLOSURE,)
