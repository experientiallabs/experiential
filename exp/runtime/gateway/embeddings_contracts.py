"""Canonical embeddings request contract, parallel to the chat ``GatewayRequest``."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator

from exp.common.core.artifacts import ContractModel
from exp.runtime.gateway.contracts import GatewayApiSurface


class EmbeddingsRequest(ContractModel):
    """Canonical, provider-neutral embeddings request.

    Deliberately parallel to :class:`~exp.runtime.gateway.contracts.GatewayRequest`
    rather than a mode of it: the embeddings surface is message-less and
    non-streaming, while ``GatewayRequest`` hard-requires ``messages`` and is
    admitted stream-only. Reusing that contract would have forced either a
    message-less exception or a stream-forced embeddings dispatch, so the two
    surfaces stay separate. It lives in its own module so the already
    line-budgeted ``contracts`` module carries only the shared enum member.
    """

    surface: Literal[GatewayApiSurface.EMBEDDINGS] = GatewayApiSurface.EMBEDDINGS
    inputs: tuple[str, ...] = Field(min_length=1)
    dimensions: int | None = Field(default=None, gt=0)
    encoding_format: Literal["float", "base64"] | None = None
    user: str | None = Field(default=None, max_length=1024)
    """End-user attribution from the OpenAI ``user`` field: content-free and never a credential."""

    @field_validator("inputs")
    @classmethod
    def _require_nonempty_inputs(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Reject empty input strings, mirroring the provider's own rejection.

        Args:
            value: Ordered visible text inputs to embed.

        Returns:
            The unchanged validated inputs.

        Raises:
            ValueError: An input string is empty.
        """
        if any(not text for text in value):
            raise ValueError("embedding inputs must not be empty strings")
        return value
