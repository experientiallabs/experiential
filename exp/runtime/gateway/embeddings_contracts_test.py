"""Tests for the canonical embeddings request contract."""

from __future__ import annotations

import pytest
from pydantic import JsonValue, ValidationError

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


def test_embeddings_request_preserves_token_sequences_through_json() -> None:
    """Token batches round-trip without losing integer IDs or logical input boundaries."""
    request = EmbeddingsRequest(inputs=((0, 100257), (42,)))
    assert request.inputs == ((0, 100257), (42,))
    assert EmbeddingsRequest.model_validate_json(request.model_dump_json()) == request
    assert request.model_dump(mode="json")["inputs"] == [[0, 100257], [42]]


@pytest.mark.parametrize(
    "inputs", [[], [[]], [[True]], [[1.0]], [[-1]], ["text", [1]], [1, 2], [[1], "text"]]
)
def test_embeddings_request_rejects_invalid_canonical_batches(inputs: JsonValue) -> None:
    """The canonical contract requires homogeneous nonempty batches and strict token IDs."""
    with pytest.raises(ValidationError):
        EmbeddingsRequest.model_validate({"inputs": inputs})


def test_embeddings_request_attributes_the_end_user_from_the_user_field() -> None:
    """``attribution_label`` mirrors the chat contract: the ``user`` field or nothing."""
    assert EmbeddingsRequest(inputs=("hi",)).attribution_label is None
    assert EmbeddingsRequest(inputs=("hi",), user="tenant-7").attribution_label == "tenant-7"
