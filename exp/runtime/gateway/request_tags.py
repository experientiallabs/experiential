"""Bounded caller attribution tags, never provider request metadata."""

from __future__ import annotations

import unicodedata
from typing import Annotated

from pydantic import AfterValidator, Field, StringConstraints


def _public_key(value: str) -> str:
    """Keep platform-owned keys out of the caller attribution namespace."""
    if value.lower().startswith("explabs."):
        raise ValueError("the explabs. tag prefix is reserved")
    return value


def _safe_value(value: str) -> str:
    """Require UTF-8 scalar text without Unicode control characters."""
    if any(unicodedata.category(character) in {"Cc", "Cs"} for character in value):
        raise ValueError("tag values must be UTF-8 text without control characters")
    return value


RequestTagKey = Annotated[
    str,
    StringConstraints(strict=True, pattern=r"^[A-Za-z][A-Za-z0-9_.-]{0,63}$"),
    AfterValidator(_public_key),
]
RequestTagValue = Annotated[
    str,
    StringConstraints(strict=True, min_length=1, max_length=256),
    AfterValidator(_safe_value),
]
RequestTags = Annotated[dict[RequestTagKey, RequestTagValue], Field(max_length=16)]
"""Validated public tags; keys are case-sensitive except the reserved prefix."""
