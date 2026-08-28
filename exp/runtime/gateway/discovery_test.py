"""Tests for caller-facing OpenAI model discovery objects."""

from __future__ import annotations

import pytest

from exp.runtime.gateway.discovery import (
    public_model_list,
    public_model_object,
    require_granted_authority,
)
from exp.runtime.openai_protocol.errors import OpenAIProtocolError

_AUTHORITY = ("coding", "revision-one", "a" * 64)
_MODEL_OBJECT = {
    "id": "coding",
    "object": "model",
    "created": 0,
    "owned_by": "exp",
}


def test_public_model_object_has_only_openai_model_fields() -> None:
    """The detail endpoint never leaks gateway-specific extension fields."""
    assert public_model_object(_AUTHORITY) == _MODEL_OBJECT


def test_public_model_list_has_only_the_openai_list_shape() -> None:
    """List entries and their envelope contain no gateway-specific fields."""
    assert public_model_list((_AUTHORITY,)) == {
        "object": "list",
        "data": [_MODEL_OBJECT],
    }
    assert public_model_list(()) == {"object": "list", "data": []}


def test_require_granted_authority_returns_the_exact_granted_triple() -> None:
    """A granted alias resolves to its own frozen authority."""
    assert require_granted_authority((_AUTHORITY,), "coding") == _AUTHORITY


def test_unknown_and_ungranted_aliases_raise_the_identical_404() -> None:
    """The detail route never confirms whether an ungranted alias exists."""
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
