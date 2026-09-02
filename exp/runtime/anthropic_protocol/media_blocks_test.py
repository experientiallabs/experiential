"""Tests for the Anthropic image and document block decoders."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from exp.common.models.content import MediaHandle
from exp.runtime.anthropic_protocol.media_blocks import (
    DocumentBlock,
    ImageBlock,
    document_part_from_block,
    image_part_from_block,
)
from exp.runtime.openai_protocol.errors import OpenAIProtocolError

_PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8"
    "z8DwHwAFAAH/q842iQAAAABJRU5ErkJggg=="
)


def test_file_sources_decode_to_anthropic_handles_with_their_marker() -> None:
    """``source.type: file`` wraps the id as an Anthropic handle and keeps the marker."""
    image = image_part_from_block(
        ImageBlock.model_validate(
            {
                "type": "image",
                "source": {"type": "file", "file_id": "file_img"},
                "cache_control": {"type": "ephemeral"},
            }
        ),
        "messages.0.content.0",
    )
    assert image.handle == MediaHandle(provider="anthropic", reference="file_img")
    assert (image.data, image.url) == (None, None)
    assert image.cache_control == {"type": "ephemeral"}
    document = document_part_from_block(
        DocumentBlock.model_validate(
            {"type": "document", "source": {"type": "file", "file_id": "file_doc"}, "title": "T"}
        ),
        "messages.0.content.1",
    )
    assert document.handle == MediaHandle(provider="anthropic", reference="file_doc")
    assert document.name == "T"


def test_bucket_uris_in_url_sources_decode_to_bucket_handles() -> None:
    """An ``s3://`` or ``gs://`` URL source is a Bedrock or Vertex handle, not a fetch."""
    image = image_part_from_block(
        ImageBlock.model_validate(
            {"type": "image", "source": {"type": "url", "url": "s3://bkt/cat.webp"}}
        ),
        "messages.0.content.0",
    )
    assert image.handle == MediaHandle(provider="bedrock", reference="s3://bkt/cat.webp")
    assert image.media_type == "image/webp"
    document = document_part_from_block(
        DocumentBlock.model_validate(
            {"type": "document", "source": {"type": "url", "url": "gs://bkt/brief.pdf"}}
        ),
        "messages.0.content.0",
    )
    assert document.handle == MediaHandle(provider="vertex", reference="gs://bkt/brief.pdf")
    assert document.url is None


def test_plain_sources_still_decode_without_a_handle() -> None:
    """Base64 and http(s) sources never grow a handle."""
    inline = image_part_from_block(
        ImageBlock.model_validate(
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": _PNG_BASE64},
            }
        ),
        "messages.0.content.0",
    )
    remote = document_part_from_block(
        DocumentBlock.model_validate(
            {"type": "document", "source": {"type": "url", "url": "https://x.test/a.pdf"}}
        ),
        "messages.0.content.0",
    )
    assert inline.handle is None and inline.data == _PNG_BASE64
    assert remote.handle is None and remote.url == "https://x.test/a.pdf"


@pytest.mark.parametrize(
    "source",
    [
        {"type": "file"},
        {"type": "file", "file_id": "file-openai-shaped"},
        {"type": "url", "url": "s3://bkt/no-suffix"},
    ],
)
def test_malformed_handle_sources_fail_at_the_source_field(source: dict[str, str]) -> None:
    """A missing id, a foreign id shape, or a suffixless object names ``source``."""
    with pytest.raises(OpenAIProtocolError) as error:
        image_part_from_block(
            ImageBlock.model_validate({"type": "image", "source": source}), "messages.0.content.0"
        )
    assert error.value.detail.param == "messages.0.content.0.source"
    assert "Anthropic Files id" in error.value.detail.message


@pytest.mark.parametrize(
    ("block_type", "source"),
    [
        ("image", {"type": "file", "file_id": "file_abc", "url": "https://x.test/a.png"}),
        ("image", {"type": "url", "url": "https://x.test/a.png", "data": _PNG_BASE64}),
        ("document", {"type": "file", "file_id": "file_abc", "data": "JVBERi0x"}),
        ("document", {"type": "base64", "media_type": "application/pdf", "file_id": "file_abc"}),
    ],
)
def test_sources_carrying_two_carriers_fail_validation(
    block_type: str, source: dict[str, str]
) -> None:
    """A source may only carry the fields of the carrier its ``type`` selects."""
    model = ImageBlock if block_type == "image" else DocumentBlock
    with pytest.raises(ValidationError) as error:
        model.model_validate({"type": block_type, "source": source})
    assert "accepts only" in str(error.value)
