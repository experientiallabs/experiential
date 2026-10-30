"""Canonical image-generation request contract, parallel to the chat ``GatewayRequest``."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from exp.common.core.artifacts import ContractModel
from exp.common.models.catalog_prices import BilledUnitKind, GatewayTokenPrices
from exp.runtime.gateway.contracts import GatewayApiSurface
from exp.runtime.gateway.ledger_valuation import require_representable_nano_usd

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


# The largest decoded input image an edit carries (base64 travels 4/3 larger).
# Stability's edit tools accept at most 9.4 megapixels; a lossless PNG of that
# size stays under this bound.
MAXIMUM_EDIT_IMAGE_BYTES = 20 * 1024 * 1024
_MAXIMUM_EDIT_IMAGE_BASE64 = (MAXIMUM_EDIT_IMAGE_BYTES + 2) // 3 * 4

EditOptionValue = str | int | float


class ImageEditInput(ContractModel):
    """The input images of one ``/v1/images/edits`` request, base64 encoded.

    Attributes:
        image: The image to edit (the multipart ``image`` part).
        mask: The optional mask (the multipart ``mask`` part); white marks the
            region to change.
        style_image: The optional style reference (the multipart
            ``style_image`` part), used by style transfer.
    """

    image: str = Field(min_length=1, max_length=_MAXIMUM_EDIT_IMAGE_BASE64)
    mask: str | None = Field(default=None, min_length=1, max_length=_MAXIMUM_EDIT_IMAGE_BASE64)
    style_image: str | None = Field(
        default=None, min_length=1, max_length=_MAXIMUM_EDIT_IMAGE_BASE64
    )


class ImagesRequest(ContractModel):
    """Canonical, provider-neutral image-generation or image-edit request.

    Parallel to :class:`~exp.runtime.gateway.contracts.GatewayRequest` like the
    embeddings contract: prompt-in (plus input images for an edit), images-out,
    never streamed. The OpenAI Images API's generation parameters are carried;
    ``user`` is the end-user attribution label and never a credential.

    Attributes:
        surface: Always the images surface (generations and edits share it).
        prompt: The text prompt; required for a generation, optional for an
            edit whose model needs none (background removal, fast upscale).
        n: Images requested.
        size: OpenAI output size.
        quality: OpenAI output quality.
        background: OpenAI background control.
        output_format: Encoded image format.
        output_compression: Lossy compression level.
        moderation: OpenAI moderation level.
        response_format: ``url`` or ``b64_json``.
        style: DALL-E 3 style.
        user: End-user attribution label.
        edit: The input images when the request edits an image
            (``/v1/images/edits``); None for a generation.
        edit_options: Provider image-tool controls an edit carries beyond the
            OpenAI fields (``seed``, ``negative_prompt``, ``creativity``...),
            validated by the decoder and forwarded only to a rung that
            accepts each one.
    """

    surface: Literal[GatewayApiSurface.IMAGES] = GatewayApiSurface.IMAGES
    prompt: str = Field(default="", max_length=32_000)
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
    edit: ImageEditInput | None = None
    edit_options: dict[str, EditOptionValue] | None = None

    @model_validator(mode="after")
    def _prompt_or_edit(self) -> ImagesRequest:
        """Require a prompt on a generation and keep edit controls on edits.

        Returns:
            The validated request.

        Raises:
            ValueError: A generation without a prompt, or edit options
                without an edit.
        """
        if self.edit is None and not self.prompt:
            raise ValueError("an image generation requires a prompt")
        if self.edit is None and self.edit_options:
            raise ValueError("edit options require an image edit")
        return self

    @property
    def attribution_label(self) -> str | None:
        """The end-user attribution label (the ``user`` field), per the OpenAI spec."""
        return self.user


def image_variant(request: ImagesRequest) -> str:
    """The priced variant of one request on a per-image unit card.

    ``quality/size`` with omitted parts dropped (``"hd/1024x1792"``,
    ``"1024x1024"``, or ``""``). A card without that exact key prices the
    request at its flat ``""`` rate when it has one.

    Args:
        request: The canonical request.

    Returns:
        The variant key.
    """
    return "/".join(part for part in (request.quality, request.size) if part)


def per_image_rate(prices: GatewayTokenPrices, request: ImagesRequest) -> int | None:
    """The lane's nano-USD per image for this request, or None when not per-image priced.

    Args:
        prices: The deployment's price card.
        request: The canonical request.

    Returns:
        The rate, or None when the card is absent, of another kind, or has no
        rate for the request's variant.
    """
    units = prices.units
    if units is None or units.kind != BilledUnitKind.IMAGE:
        return None
    return units.rate_for(image_variant(request))


def images_ceiling_nano_usd(
    request: ImagesRequest,
    *,
    input_tokens: int,
    input_rate: int | None,
    output_rate: int | None,
    image_rate: int | None = None,
) -> int | None:
    """Return the conservative reservation ceiling for one image call.

    A per-image priced lane (``image_rate``) reserves ``n`` images at its rate,
    which is also the most it can settle. GPT image models bill prompt tokens
    at the input rate and generated image tokens at the output rate;
    ``input_tokens`` is the prompt's estimated input reservation and ``n``
    times the largest per-image token count bounds the output. A token lane
    without both rates is unpriced (``None``). A ceiling past the int8 ledger
    column raises ``NanoUsdOverflowError``.
    """
    if image_rate is not None:
        return require_representable_nano_usd(
            request.n * image_rate, what="images reservation ceiling"
        )
    if input_rate is None or output_rate is None:
        return None
    input_ceiling = input_tokens * input_rate
    output_ceiling = request.n * MAXIMUM_IMAGE_OUTPUT_TOKENS * output_rate
    return require_representable_nano_usd(
        (input_ceiling + output_ceiling + 999_999) // 1_000_000,
        what="images reservation ceiling",
    )
