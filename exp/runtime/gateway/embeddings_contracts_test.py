"""Tests for the canonical embeddings request contract."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from exp.runtime.gateway.contracts import GatewayApiSurface
from exp.runtime.gateway.embeddings_contracts import EmbeddingsRequest


def test_embeddings_request_is_message_less_and_defaults_optionals() -> None:
    """The parallel embeddings contract needs only inputs and pins its surface."""
    request = EmbeddingsRequest(inputs=("hello",))

    assert request.surface == GatewayApiSurface.EMBEDDINGS
    assert request.inputs == ("hello",)
    assert request.dimensions is None
    assert request.encoding_format is None
    assert request.user is None


def test_embeddings_request_rejects_empty_input_sets() -> None:
    """An empty input list or an empty input string is invalid at the contract boundary."""
    with pytest.raises(ValidationError, match="at least 1 item"):
        EmbeddingsRequest(inputs=())
    with pytest.raises(ValidationError, match="must not be empty"):
        EmbeddingsRequest(inputs=("ok", ""))
    with pytest.raises(ValidationError, match="greater than 0"):
        EmbeddingsRequest(inputs=("ok",), dimensions=0)
