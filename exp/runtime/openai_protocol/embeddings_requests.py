"""Decode the OpenAI Embeddings body into the canonical embeddings surface.

Lives beside :mod:`exp.runtime.openai_protocol.requests` (which owns the
line-budgeted chat and Responses decoders) and reuses its validation helpers,
so every public surface decodes through one manifest, one official-SDK shape
check, and one closed wire model.
"""

from __future__ import annotations

from typing import Literal, cast

from openai.types import EmbeddingCreateParams
from pydantic import Field, JsonValue, TypeAdapter, ValidationError, field_validator

from exp.common.core.artifacts import ContractModel, JsonObject
from exp.runtime.gateway.embeddings_contracts import (
    EmbeddingInputs,
    EmbeddingsRequest,
    EmbeddingTokenIds,
)
from exp.runtime.openai_protocol.errors import unsupported_field
from exp.runtime.openai_protocol.manifest import EMBEDDINGS_MANIFEST, validate_manifest
from exp.runtime.openai_protocol.requests import _validate_official, _validate_wire
from exp.runtime.openai_protocol.validation_errors import validation_protocol_error
from exp.runtime.openai_protocol.wire_models import _WireModel


class _EmbeddingsRequest(_WireModel):
    """Closed OpenAI embeddings profile with text and pre-tokenized inputs.

    Attributes:
        model: Public model alias requested by the caller.
        input: One text/token input or a homogeneous batch of either form.
        dimensions: Optional positive vector width requested from the provider.
        encoding_format: Optional float or base64 response encoding.
        user: Optional gateway-only attribution, at most 1,024 characters.
        stream: Literal false convenience, validated and omitted upstream.
    """

    model: str = Field(min_length=1, max_length=256)
    input: str | EmbeddingTokenIds | EmbeddingInputs
    dimensions: int | None = Field(default=None, gt=0)
    encoding_format: Literal["float", "base64"] | None = None
    user: str | None = Field(default=None, max_length=1024)
    stream: Literal[False] = False

    @field_validator("stream", mode="before")
    @classmethod
    def _require_nonstreaming(cls, value: JsonValue) -> Literal[False]:
        """Accept only literal false, without coercing zero or string values."""
        if value is not False:
            raise ValueError("embeddings do not stream; omit stream or set it to false")
        return False

    @field_validator("input")
    @classmethod
    def _require_nonempty_input(
        cls, value: str | EmbeddingTokenIds | EmbeddingInputs
    ) -> str | EmbeddingTokenIds | EmbeddingInputs:
        """Reject empty texts or batches without treating token ID zero as empty."""
        if isinstance(value, str):
            if not value:
                raise ValueError("input must not be an empty string")
            return value
        if not value:
            raise ValueError("input must not be an empty array")
        if any(not item for item in value if not isinstance(item, int)):
            raise ValueError("input array must not contain empty inputs")
        return value

    def batch_inputs(self) -> EmbeddingInputs:
        """Normalize one string or flat token sequence to one logical batch item."""
        if isinstance(self.input, str):
            return (self.input,)
        if isinstance(self.input[0], int):
            # The validated union is homogeneous; a flat sequence contains only token IDs.
            return (cast(EmbeddingTokenIds, self.input),)
        return cast(EmbeddingInputs, self.input)


# object-parametrized: EmbeddingCreateParams is one TypedDict, unlike the chat/responses unions.
_EMBEDDINGS_OFFICIAL: TypeAdapter[object] = TypeAdapter[object](EmbeddingCreateParams)


class DecodedEmbeddingsRequest(ContractModel):
    """Public alias plus its canonical embeddings request.

    Distinct from :class:`exp.runtime.openai_protocol.requests.DecodedGatewayRequest`
    because the embeddings surface carries its own message-less, non-streaming
    request contract.
    """

    alias: str = Field(min_length=1, max_length=256)
    request: EmbeddingsRequest


def decode_embeddings(payload: JsonObject) -> DecodedEmbeddingsRequest:
    """Decode one Embeddings body into the canonical embeddings surface.

    The embeddings surface has no idempotency protocol yet: keyed replay is a
    future add, so an inbound ``Idempotency-Key`` header is ignored upstream
    rather than keying this decode (which therefore takes no header arguments).

    Args:
        payload: Parsed JSON request body.

    Returns:
        Public alias and canonical embeddings request.

    Raises:
        OpenAIProtocolError: The body is invalid, unknown, or unsupported.
    """
    validate_manifest(payload, EMBEDDINGS_MANIFEST)
    if payload.get("stream") is True:
        raise unsupported_field("stream")
    _validate_official(_EMBEDDINGS_OFFICIAL, payload, extension_fields={"stream"})
    request = _validate_wire(_EmbeddingsRequest, payload)
    try:
        canonical = EmbeddingsRequest(
            inputs=request.batch_inputs(),
            dimensions=request.dimensions,
            encoding_format=request.encoding_format,
            user=request.user,
        )
    except ValidationError as exc:
        raise validation_protocol_error(exc) from exc
    return DecodedEmbeddingsRequest(alias=request.model, request=canonical)
