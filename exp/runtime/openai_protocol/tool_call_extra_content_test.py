"""Tests for the echoed tool-call ``extra_content`` vendor block."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from exp.common.models.model import MAXIMUM_THOUGHT_SIGNATURE_CHARACTERS
from exp.runtime.openai_protocol.tool_call_extra_content import (
    ToolCallExtraContent,
    thought_signature,
)


def test_google_thought_signature_is_read_from_the_documented_shape() -> None:
    """The shape Google's OpenAI-compatible endpoint returns yields its signature."""
    extra = ToolCallExtraContent.model_validate(
        {"google": {"thought_signature": "c2lnbmF0dXJlLWE="}}
    )
    assert thought_signature(extra) == "c2lnbmF0dXJlLWE="


def test_a_direct_signature_is_read_when_extra_content_carries_none() -> None:
    """The top-level spelling is the signature unless the canonical block names one."""
    assert thought_signature(None, "c2lnbmF0dXJlLWE=") == "c2lnbmF0dXJlLWE="
    empty = ToolCallExtraContent.model_validate({})
    assert thought_signature(empty, "c2lnbmF0dXJlLWE=") == "c2lnbmF0dXJlLWE="


@pytest.mark.parametrize("payload", [None, {}, {"google": None}])
def test_absent_signatures_read_as_none(payload: dict[str, None] | None) -> None:
    """No object, an empty object, or a null Google block carries no signature."""
    extra = None if payload is None else ToolCallExtraContent.model_validate(payload)
    assert thought_signature(extra) is None


@pytest.mark.parametrize(
    "payload",
    [
        {"google": {"thought_signature": "not base64!"}},
        {"google": {"thought_signature": "c2lnbmF0dXJl"[:-1]}},
        {"google": {"thought_signature": ""}},
        {"google": {"thought_signature": "QQ==" * (MAXIMUM_THOUGHT_SIGNATURE_CHARACTERS // 4 + 1)}},
        {"google": {"thought_signature": "c2ln", "extra": 1}},
        {"openai": {}},
    ],
)
def test_malformed_or_unknown_content_is_refused(payload: dict[str, object]) -> None:
    """Non-base64, empty, oversized, or unknown keys fail instead of reaching Google."""
    with pytest.raises(ValidationError):
        ToolCallExtraContent.model_validate(payload)
