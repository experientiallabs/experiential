"""Tests for caller-facing model discovery objects."""

from __future__ import annotations

import pytest

from exp.runtime.gateway.discovery import public_model_object, require_granted_authority
from exp.runtime.openai_protocol.errors import OpenAIProtocolError

_AUTHORITY = ("coding", "revision-one", "a" * 64)


def test_public_model_object_keeps_the_openai_keys_and_adds_authority() -> None:
    """The enriched object stays a valid OpenAI model for official clients."""
    assert public_model_object(_AUTHORITY) == {
        "id": "coding",
        "object": "model",
        "created": 0,
        "owned_by": "wmo",
        "wmo": {"alias_revision_id": "revision-one", "catalog_sha256": "a" * 64},
    }


def test_require_granted_authority_returns_the_exact_granted_triple() -> None:
    """A granted alias resolves to its own frozen authority."""
    assert require_granted_authority((_AUTHORITY,), "coding") == _AUTHORITY


def test_unknown_and_ungranted_aliases_raise_the_identical_404() -> None:
    """The 404 never distinguishes an unknown alias from an ungranted one."""
    with pytest.raises(OpenAIProtocolError) as ungranted:
        require_granted_authority((_AUTHORITY,), "other-model")
    with pytest.raises(OpenAIProtocolError) as unknown:
        require_granted_authority((), "coding")

    assert ungranted.value.status_code == 404
    assert ungranted.value.detail == unknown.value.detail
    assert ungranted.value.detail.code == "model_not_found"
    assert "other-model" not in ungranted.value.detail.message
    assert (
        "GET /v1/models lists the model aliases available to this key."
        in ungranted.value.detail.message
    )
