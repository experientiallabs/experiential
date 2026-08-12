"""Validation for opaque provider resource IDs persisted by offline SFT."""

from __future__ import annotations

from urllib.parse import urlsplit

from wmo.common.core.artifacts import SecretBoundaryError, assert_text_secret_free
from wmo.optimize.model.sft.training_contracts import TinkerSFTError


def validate_provider_resource_id(value: str, *, label: str) -> str:
    """Reject URLs, credentials, and secret-like material before artifact persistence.

    Args:
        value: Provider-returned opaque resource identifier.
        label: Human-readable resource kind for safe error messages.

    Returns:
        The unchanged validated opaque provider resource ID.

    Raises:
        TinkerSFTError: The value is a URL, malformed, or contains credential-like material.
    """
    if len(value) > 2048 or not value or any(character.isspace() for character in value):
        raise TinkerSFTError(f"Tinker {label} is not a safe opaque provider resource ID")
    parsed = urlsplit(value)
    if parsed.scheme in {"http", "https", "ftp"} or not parsed.scheme:
        raise TinkerSFTError(f"Tinker {label} must be an opaque provider resource ID, not a URL")
    if parsed.username is not None or parsed.password is not None:
        raise TinkerSFTError(f"Tinker {label} must not contain URI user information")
    if parsed.query or parsed.fragment or "@" in value:
        raise TinkerSFTError(f"Tinker {label} must not contain URI query, fragment, or user info")
    if not parsed.netloc and not parsed.path:
        raise TinkerSFTError(f"Tinker {label} has no provider resource component")
    try:
        assert_text_secret_free(value)
    except SecretBoundaryError as exc:
        raise TinkerSFTError(f"Tinker {label} contains credential-like material") from exc
    return value
