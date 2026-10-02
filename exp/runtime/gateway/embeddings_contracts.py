"""Canonical embeddings request contract, parallel to the chat ``GatewayRequest``."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, field_validator

from exp.common.core.artifacts import ContractModel
from exp.runtime.gateway.contracts import GatewayApiSurface, GatewayRequest
from exp.runtime.gateway.decisions_contracts import DecisionRequest
from exp.runtime.gateway.images_contracts import ImagesRequest
from exp.runtime.gateway.ledger_valuation import require_representable_nano_usd

type EmbeddingTokenIds = Annotated[
    tuple[Annotated[int, Field(strict=True, ge=0)], ...], Field(min_length=1)
]
"""One nonempty token sequence in the selected model's tokenizer vocabulary."""

type EmbeddingInputs = tuple[str, ...] | tuple[EmbeddingTokenIds, ...]
"""A homogeneous ordered batch of texts or pre-tokenized inputs, one vector per item."""


class EmbeddingsRequest(ContractModel):
    """Canonical, provider-neutral embeddings request.

    This message-less, non-streaming surface is separate from completion requests.
    Each batch item produces one vector. Token sequences retain their provider-specific
    IDs; the gateway never decodes or translates between tokenizer vocabularies.

    Attributes:
        surface: Fixed embeddings API surface.
        inputs: Nonempty homogeneous batch of text strings or token sequences.
        dimensions: Optional positive output dimensionality, enforced by the provider.
        encoding_format: Optional float or base64 vector encoding.
        user: Optional gateway-only end-user attribution, at most 1,024 characters.
    """

    surface: Literal[GatewayApiSurface.EMBEDDINGS] = GatewayApiSurface.EMBEDDINGS
    inputs: EmbeddingInputs = Field(min_length=1)
    dimensions: int | None = Field(default=None, gt=0)
    encoding_format: Literal["float", "base64"] | None = None
    user: str | None = Field(default=None, max_length=1024)

    @property
    def attribution_label(self) -> str | None:
        """The end-user attribution label, per the OpenAI spec.

        The embeddings body carries only the ``user`` field (no
        ``safety_identifier``), so the label is exactly that field; hosts read
        it off every serving request at accept, whichever surface it came in on.

        Returns:
            The attribution label, or ``None`` when the caller sent no ``user``.
        """
        return self.user

    @field_validator("inputs")
    @classmethod
    def _require_nonempty_inputs(cls, value: EmbeddingInputs) -> EmbeddingInputs:
        """Reject empty input strings; token sequences are nonempty by type.

        Args:
            value: Ordered text or token inputs to embed.

        Returns:
            The unchanged validated inputs.

        Raises:
            ValueError: An input string is empty.
        """
        if any(not item for item in value):
            raise ValueError("embedding inputs must not be empty strings")
        return value


ServingRequest = GatewayRequest | EmbeddingsRequest | ImagesRequest | DecisionRequest
"""One admitted serving request across every public surface.

The money, auth, and accounting seams widen from ``GatewayRequest`` to this
union so a chat-assuming reader cannot duck-type onto an embeddings request and
touch an absent leg (messages, output tokens): ``ty`` enumerates every reader
that must now handle the embeddings and images arms, and each branches
exhaustively.
"""


def embeddings_input_ceiling_nano_usd(
    *,
    input_tokens: int,
    input_rate: int | None,
) -> int | None:
    """Return the conservative input-only reservation ceiling for one embeddings call.

    ``input_tokens`` is the request's estimated input reservation (there is no
    output leg and no excluded provider carrier), so only the input rate
    applies. A missing rate unprices the route (``None``); a ceiling past the
    int8 ledger column raises ``NanoUsdOverflowError`` like the completion path.
    """
    if input_rate is None:
        return None
    return require_representable_nano_usd(
        (input_tokens * input_rate + 999_999) // 1_000_000,
        what="embeddings reservation ceiling",
    )
