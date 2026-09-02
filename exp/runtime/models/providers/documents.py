"""Per-dialect encoders for caller document (PDF) parts.

One canonical document part translates differently on every provider wire,
so each dialect's exact shape lives here and the message translators stay
about roles and ordering. Inline base64 rides every document-capable wire; a
remote URL is a provider-side fetch that only the OpenAI Responses and
Anthropic Messages wires perform, so every other wire rejects it as a
capability the rung cannot preserve, which lets a waterfall narrow past it
instead of dropping the caller's document. A provider handle rides only the
wire of the provider that minted it: an OpenAI ``file_id`` on the Chat
``file`` part and the Responses ``input_file`` part, an Anthropic ``file``
source, a Gemini Files or ``gs://`` ``file_data`` URI, or a Bedrock
``s3Location``.
"""

from __future__ import annotations

import re

from exp.common.core.artifacts import JsonObject
from exp.common.models.content import MAXIMUM_DOCUMENT_NAME_CHARACTERS, DocumentContentPart
from exp.runtime.models.providers.errors import ProviderCapabilityError
from exp.runtime.models.providers.media_handles import (
    ANTHROPIC_HANDLE_PROVIDERS,
    BEDROCK_HANDLE_PROVIDERS,
    GEMINI_HANDLE_PROVIDERS,
    OPENAI_HANDLE_PROVIDERS,
    bedrock_s3_location,
    require_handle_provider,
)

PDF_URL_DIALECTS = frozenset({"openai_responses", "anthropic_messages"})
"""Dialects whose provider fetches a caller document URL on the gateway's behalf.

Chat Completions ``file`` parts carry ``file_data`` or an uploaded
``file_id`` only, so the ``openai_compatible`` dialect is deliberately absent.
"""

PDF_CAPABILITY = "pdf_input"
"""Capability literal naming a wire's ability to carry a caller PDF."""

PDF_URL_CAPABILITY = "pdf_url_input"
"""Capability literal naming a provider-side fetch of a caller document URL."""

DEFAULT_DOCUMENT_FILENAME = "document.pdf"
"""Filename emitted on wires that require one when the caller sent none."""

_BEDROCK_NAME_DISALLOWED = re.compile(r"[^A-Za-z0-9 \-()\[\]]+")
_BEDROCK_REPEATED_SPACE = re.compile(r" {2,}")


def openai_chat_document_part(document: DocumentContentPart) -> JsonObject:
    """Encode one document as a Chat Completions ``file`` content part.

    Args:
        document: Canonical document part from the caller's message.

    Returns:
        The ``{"type": "file", "file": {...}}`` part with inline
        ``file_data`` and the caller's filename (or a neutral default), or
        with the OpenAI Files ``file_id`` for a handle.

    Raises:
        ProviderCapabilityError: The document is a remote URL, which the
            Chat Completions ``file`` part cannot carry, or a handle from
            another provider.
    """
    if document.handle is not None:
        require_handle_provider(document.handle, OPENAI_HANDLE_PROVIDERS)
        return {"type": "file", "file": {"file_id": document.handle.reference}}
    if document.data is None:
        raise ProviderCapabilityError(capability=PDF_URL_CAPABILITY)
    return {
        "type": "file",
        "file": {
            "filename": document.name or DEFAULT_DOCUMENT_FILENAME,
            "file_data": document.data_url(),
        },
    }


def responses_document_part(document: DocumentContentPart) -> JsonObject:
    """Encode one document as a Responses ``input_file`` content part.

    Inline bytes ride ``file_data`` beside a filename; a remote document
    rides ``file_url``, which the provider fetches itself; an OpenAI Files
    handle rides ``file_id``.

    Raises:
        ProviderCapabilityError: The document is a handle from another provider.
    """
    if document.handle is not None:
        require_handle_provider(document.handle, OPENAI_HANDLE_PROVIDERS)
        return {"type": "input_file", "file_id": document.handle.reference}
    if document.data is None:
        return {"type": "input_file", "file_url": document.url or ""}
    return {
        "type": "input_file",
        "filename": document.name or DEFAULT_DOCUMENT_FILENAME,
        "file_data": document.data_url(),
    }


def anthropic_document_block(document: DocumentContentPart) -> JsonObject:
    """Encode one document as an Anthropic ``document`` content block.

    A caller title re-emits as the block ``title`` and a caller cache
    breakpoint re-emits on the block it was placed on: this wire caches a
    marked document, and a lost marker silently returns the whole prefix to
    full input billing.

    Raises:
        ProviderCapabilityError: The document is a handle from another provider.
    """
    source: JsonObject
    if document.handle is not None:
        require_handle_provider(document.handle, ANTHROPIC_HANDLE_PROVIDERS)
        source = {"type": "file", "file_id": document.handle.reference}
    elif document.data is None:
        source = {"type": "url", "url": document.url or ""}
    else:
        source = {"type": "base64", "media_type": document.media_type, "data": document.data}
    block: JsonObject = {"type": "document", "source": source}
    if document.name is not None:
        block["title"] = document.name
    if document.cache_control is not None:
        block["cache_control"] = document.cache_control
    return block


def gemini_document_part(document: DocumentContentPart) -> JsonObject:
    """Encode one document as a Gemini ``inline_data`` or ``file_data`` part.

    Args:
        document: Canonical document part from the caller's message.

    Returns:
        The native Gemini part carrying the PDF bytes, or the ``file_data``
        reference for a Gemini Files or ``gs://`` handle.

    Raises:
        ProviderCapabilityError: The document is a remote URL, which this
            wire cannot fetch on the caller's behalf, or a handle from
            another provider.
    """
    if document.handle is not None:
        require_handle_provider(document.handle, GEMINI_HANDLE_PROVIDERS)
        return {
            "file_data": {
                "file_uri": document.handle.reference,
                "mime_type": document.media_type,
            }
        }
    if document.data is None:
        raise ProviderCapabilityError(capability=PDF_URL_CAPABILITY)
    return {"inline_data": {"mime_type": document.media_type, "data": document.data}}


def bedrock_document_name(document: DocumentContentPart, ordinal: int) -> str:
    """Return a Converse-valid ``name`` for one document.

    Converse restricts the name to alphanumerics, single spaces, hyphens,
    parentheses, and square brackets, at most 200 characters, and shows it
    to the model. Disallowed runs in the caller's name (a ``.pdf`` suffix,
    underscores) collapse to one hyphen; a missing or fully stripped name
    falls back to a neutral ordinal name.

    Args:
        document: Canonical document part from the caller's message.
        ordinal: One-based position of the document within its message.

    Returns:
        A name Converse accepts.
    """
    if document.name is None:
        return f"document-{ordinal}"
    cleaned = _BEDROCK_NAME_DISALLOWED.sub("-", document.name)
    cleaned = _BEDROCK_REPEATED_SPACE.sub(" ", cleaned).strip(" -")
    return cleaned[:MAXIMUM_DOCUMENT_NAME_CHARACTERS] or f"document-{ordinal}"


def bedrock_document_block(document: DocumentContentPart, ordinal: int) -> JsonObject:
    """Encode one document as a Bedrock Converse ``document`` block.

    The Converse REST body carries the bytes base64 encoded, which is
    exactly the canonical inline form; the block also requires a valid
    ``name`` and the bare ``format``.

    Args:
        document: Canonical document part from the caller's message.
        ordinal: One-based position of the document within its message.

    Returns:
        The native Converse content block.

    Raises:
        ProviderCapabilityError: The document is a remote URL, which this
            wire cannot fetch on the caller's behalf, or a handle from
            another provider.
    """
    source: JsonObject
    if document.handle is not None:
        require_handle_provider(document.handle, BEDROCK_HANDLE_PROVIDERS)
        source = {"s3Location": bedrock_s3_location(document.handle)}
    elif document.data is None:
        raise ProviderCapabilityError(capability=PDF_URL_CAPABILITY)
    else:
        source = {"bytes": document.data}
    return {
        "document": {
            "name": bedrock_document_name(document, ordinal),
            "format": "pdf",
            "source": source,
        }
    }
