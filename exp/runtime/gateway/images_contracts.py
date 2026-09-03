"""Canonical image-generation request contract, parallel to the chat ``GatewayRequest``."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from exp.common.core.artifacts import ContractModel, canonical_json_bytes
from exp.runtime.gateway.contracts import GatewayApiSurface

ImageSize = Literal[
    "auto",
    "256x256",
    "512x512",
    "1024x1024",
    "1536x1024",
    "1024x1536",
    "1792x1024",
    "1024x1792",
]
ImageQuality = Literal["standard", "hd", "low", "medium", "high", "auto"]

# The largest output-token count one generated image can bill on the OpenAI
# GPT image models (high quality, 1024x1536 or 1536x1024). The reservation
# ceiling multiplies it by ``n``; settlement charges the provider's count.
MAXIMUM_IMAGE_OUTPUT_TOKENS = 6_240


class ImagesRequest(ContractModel):
    """Canonical, provider-neutral image-generation request.

    Parallel to :class:`~exp.runtime.gateway.contracts.GatewayRequest` like the
    embeddings contract: prompt-in, images-out, never streamed. Only the
    OpenAI Images API's generation parameters are carried; ``user`` is the
    end-user attribution label and never a credential.
    """

    surface: Literal[GatewayApiSurface.IMAGES] = GatewayApiSurface.IMAGES
    prompt: str = Field(min_length=1, max_length=32_000)
    n: int = Field(default=1, ge=1, le=10)
    size: ImageSize | None = None
    quality: ImageQuality | None = None
    background: Literal["transparent", "opaque", "auto"] | None = None
    output_format: Literal["png", "jpeg", "webp"] | None = None
    output_compression: int | None = Field(default=None, ge=0, le=100)
    moderation: Literal["low", "auto"] | None = None
    response_format: Literal["url", "b64_json"] | None = None
    style: Literal["vivid", "natural"] | None = None
    user: str | None = Field(default=None, max_length=1024)

    @property
    def attribution_label(self) -> str | None:
        """The end-user attribution label (the ``user`` field), per the OpenAI spec."""
        return self.user


def images_ceiling_micro_usd(
    request: ImagesRequest,
    *,
    input_rate: int | None,
    output_rate: int | None,
    maximum: int,
) -> int | None:
    """Return the conservative reservation ceiling for one token-priced image call.

    GPT image models bill prompt tokens at the input rate and generated image
    tokens at the output rate; the prompt's canonical byte length upper-bounds
    the input tokens and ``n`` times the largest per-image token count bounds the
    output. A lane without both rates is unpriced for images (``None``): the
    per-image priced models (dall-e) wait for the typed billed-units ledger.
    """
    if input_rate is None or output_rate is None:
        return None
    input_ceiling = len(canonical_json_bytes(request)) * input_rate
    output_ceiling = request.n * MAXIMUM_IMAGE_OUTPUT_TOKENS * output_rate
    ceiling = (input_ceiling + output_ceiling + 999_999) // 1_000_000
    return ceiling if ceiling <= maximum else None
