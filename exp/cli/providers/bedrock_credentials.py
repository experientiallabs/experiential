"""Bedrock credential-mode inference for interactive provider setup."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

_BEDROCK_BEARER_ENV = "AWS_BEARER_TOKEN_BEDROCK"
_AWS_ACCESS_KEY_ID_ENV = "AWS_ACCESS_KEY_ID"
_AWS_SECRET_ACCESS_KEY_ENV = "AWS_SECRET_ACCESS_KEY"


def infer_bedrock_auth(
    environment: Mapping[str, str] | None,
) -> tuple[str | None, str | None, Literal["access_key_pair", "api_key"] | None]:
    """Return secret locator, access-ID locator, and mode from standard AWS variables."""
    if environment is None:
        return None, None, None
    if environment.get(_BEDROCK_BEARER_ENV, "").strip():
        return _BEDROCK_BEARER_ENV, None, "api_key"
    if (
        environment.get(_AWS_ACCESS_KEY_ID_ENV, "").strip()
        and environment.get(_AWS_SECRET_ACCESS_KEY_ENV, "").strip()
    ):
        return _AWS_SECRET_ACCESS_KEY_ENV, _AWS_ACCESS_KEY_ID_ENV, "access_key_pair"
    return None, None, None
