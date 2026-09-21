"""Tests for pydantic-to-protocol error rendering; the decoders' end-to-end
messages are exercised in requests_test.py."""

from __future__ import annotations

from exp.runtime.openai_protocol.validation_errors import cleaned_location


def test_cleaned_location_drops_branch_labels_but_keeps_a_same_named_final_field() -> None:
    """A union-branch label is never the last segment; a caller field may be."""
    assert cleaned_location(("input", 1, "reasoning", "id")) == ("input", "1", "id")
    assert cleaned_location(("messages", 0, "reasoning")) == ("messages", "0", "reasoning")
    assert cleaned_location(("body", "input", "tuple[...]", 0, "_ResponseMessage", "role")) == (
        "input",
        "0",
        "role",
    )
    assert cleaned_location(("messages", 0, "content", 1, "image_url", "image_url", "url")) == (
        "messages",
        "0",
        "content",
        "1",
        "image_url",
        "url",
    )
