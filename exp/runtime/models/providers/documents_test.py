"""Tests for per-dialect PDF document encoding."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from exp.common.core.artifacts import JsonObject
from exp.common.models.content import DocumentContentPart
from exp.runtime.models.providers.documents import (
    PDF_URL_DIALECTS,
    anthropic_document_block,
    bedrock_document_block,
    bedrock_document_name,
    gemini_document_part,
    openai_chat_document_part,
    responses_document_part,
)
from exp.runtime.models.providers.errors import ProviderCapabilityError

_PDF_BASE64 = "JVBERi0xLjQKJSBtaW5pbWFsIHBkZgo="
"""One short PDF header, base64 encoded."""

_INLINE = DocumentContentPart(data=_PDF_BASE64, name="brief.pdf")
"""One inline document with a caller filename."""

_UNNAMED = DocumentContentPart(data=_PDF_BASE64)
"""One inline document without a caller filename."""

_REMOTE = DocumentContentPart(url="https://example.com/brief.pdf")
"""One document the provider would have to fetch itself."""


def test_openai_chat_inlines_as_a_file_part() -> None:
    """Chat Completions carries the bytes as a nested ``file`` object."""
    assert openai_chat_document_part(_INLINE) == {
        "type": "file",
        "file": {
            "filename": "brief.pdf",
            "file_data": f"data:application/pdf;base64,{_PDF_BASE64}",
        },
    }


def test_openai_chat_names_an_unnamed_document() -> None:
    """The Chat wire always sends a filename, so one is supplied when absent."""
    encoded = openai_chat_document_part(_UNNAMED)
    file = encoded["file"]
    assert isinstance(file, dict)
    assert file["filename"] == "document.pdf"


def test_responses_uses_input_file_with_file_data_or_file_url() -> None:
    """The Responses wire splits inline bytes and remote URLs into two fields."""
    assert responses_document_part(_INLINE) == {
        "type": "input_file",
        "filename": "brief.pdf",
        "file_data": f"data:application/pdf;base64,{_PDF_BASE64}",
    }
    assert responses_document_part(_REMOTE) == {
        "type": "input_file",
        "file_url": "https://example.com/brief.pdf",
    }


def test_anthropic_splits_base64_and_url_sources() -> None:
    """Anthropic declares the carrier and media type on the source, title beside."""
    assert anthropic_document_block(_INLINE) == {
        "type": "document",
        "source": {"type": "base64", "media_type": "application/pdf", "data": _PDF_BASE64},
        "title": "brief.pdf",
    }
    assert anthropic_document_block(_REMOTE) == {
        "type": "document",
        "source": {"type": "url", "url": "https://example.com/brief.pdf"},
    }


def test_anthropic_re_emits_a_cache_marker_placed_on_the_document() -> None:
    """A breakpoint on the document survives to the wire that honors it."""
    marked = _UNNAMED.model_copy(update={"cache_control": {"type": "ephemeral"}})
    assert anthropic_document_block(marked) == {
        "type": "document",
        "source": {"type": "base64", "media_type": "application/pdf", "data": _PDF_BASE64},
        "cache_control": {"type": "ephemeral"},
    }


def test_gemini_uses_inline_data() -> None:
    """Gemini carries the bytes as an ``inline_data`` part with the PDF mime type."""
    assert gemini_document_part(_INLINE) == {
        "inline_data": {"mime_type": "application/pdf", "data": _PDF_BASE64}
    }


def test_bedrock_uses_a_named_pdf_document_block() -> None:
    """Bedrock Converse requires a name, the bare format, and encoded bytes."""
    assert bedrock_document_block(_INLINE, 1) == {
        "document": {"name": "brief-pdf", "format": "pdf", "source": {"bytes": _PDF_BASE64}}
    }
    assert bedrock_document_block(_UNNAMED, 2) == {
        "document": {"name": "document-2", "format": "pdf", "source": {"bytes": _PDF_BASE64}}
    }


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Quarterly Report (Q3) [final].pdf", "Quarterly Report (Q3) [final]-pdf"),
        ("a/b\\c:d*e?f", "a-b-c-d-e-f"),
        ("   spaced    out  ", "spaced out"),
        ("!!!", "document-7"),
    ],
)
def test_bedrock_document_names_are_sanitized(name: str, expected: str) -> None:
    """Names keep only the characters Converse accepts and never go empty."""
    assert bedrock_document_name(DocumentContentPart(data=_PDF_BASE64, name=name), 7) == expected


@pytest.mark.parametrize(
    "encode",
    [openai_chat_document_part, gemini_document_part, lambda part: bedrock_document_block(part, 1)],
)
def test_inline_only_wires_reject_a_remote_url_as_a_capability(
    encode: Callable[[DocumentContentPart], JsonObject],
) -> None:
    """A wire without a URL carrier fails as a capability, so a waterfall can narrow."""
    with pytest.raises(ProviderCapabilityError) as error:
        encode(_REMOTE)
    assert error.value.capability == "pdf_url_input"


def test_only_responses_and_anthropic_fetch_urls() -> None:
    """The URL dialect set matches the encoders that emit a URL carrier."""
    assert PDF_URL_DIALECTS == {"openai_responses", "anthropic_messages"}
