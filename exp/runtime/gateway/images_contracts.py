"""Canonical image-generation request contract, parallel to the chat ``GatewayRequest``."""

from __future__ import annotations

from typing import Annotated, Literal

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

# The most source images one edit carries: OpenAI's GPT image models accept up
# to 16; xAI takes up to 5 and Stability exactly one, so a rung refuses more.
MAXIMUM_EDIT_IMAGES = 16

# The largest image-input token count one source image (or mask) of a GPT
# image edit can bill. A 1536x1024 source measured 1,536 image tokens on
# gpt-image-2.5-sunburst; high input fidelity adds a few thousand more. The
# reservation ceiling multiplies it by the image parts; settlement charges the
# provider's count.
MAXIMUM_EDIT_INPUT_IMAGE_TOKENS = 8_192

# The unit-card variant that prices an image an edit returns, on a lane whose
# provider bills edited images apart from generated ones (xAI). Its rate must
# be authored by name: an edit never falls back to the generation rate.
EDIT_VARIANT = "edit"

EditOptionValue = str | int | float


class ImageEditInput(ContractModel):
    """The input images of one ``/v1/images/edits`` request, base64 encoded.

    Attributes:
        image: The first image to edit (the multipart ``image`` part, or the
            first ``image[]`` part).
        additional_images: Further source images in upload order (every
            ``image[]`` part after the first); a rung that takes one image
            refuses a request carrying any.
        mask: The optional mask (the multipart ``mask`` part); white marks the
            region to change.
        style_image: The optional style reference (the multipart
            ``style_image`` part), used by style transfer.
    """

    image: str = Field(min_length=1, max_length=_MAXIMUM_EDIT_IMAGE_BASE64)
    additional_images: tuple[
        Annotated[str, Field(min_length=1, max_length=_MAXIMUM_EDIT_IMAGE_BASE64)], ...
    ] = Field(default=(), max_length=MAXIMUM_EDIT_IMAGES - 1)
    mask: str | None = Field(default=None, min_length=1, max_length=_MAXIMUM_EDIT_IMAGE_BASE64)
    style_image: str | None = Field(
        default=None, min_length=1, max_length=_MAXIMUM_EDIT_IMAGE_BASE64
    )

    @property
    def images(self) -> tuple[str, ...]:
        """Every source image in upload order."""
        return (self.image, *self.additional_images)

    @property
    def input_image_parts(self) -> int:
        """Image parts a provider reads: the source images plus any mask."""
        return len(self.images) + (self.mask is not None)


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
        input_fidelity: OpenAI edit input fidelity (how closely a GPT image
            edit preserves the source images' details); edits only.
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
    input_fidelity: Literal["high", "low"] | None = None
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
            ValueError: A generation without a prompt, or edit options or
                input fidelity without an edit.
        """
        if self.edit is None and not self.prompt:
            raise ValueError("an image generation requires a prompt")
        if self.edit is None and self.edit_options:
            raise ValueError("edit options require an image edit")
        if self.edit is None and self.input_fidelity is not None:
            raise ValueError("input_fidelity requires an image edit")
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


def unit_variant(prices: GatewayTokenPrices, request: ImagesRequest) -> str:
    """The variant one lane's per-image card prices this request at.

    An edit on a card that authors :data:`EDIT_VARIANT` is priced at that
    variant (its produced images and, through ``input_rates``, its source
    images); every other request at :func:`image_variant`.

    Args:
        prices: The deployment's price card.
        request: The canonical request.

    Returns:
        The variant key.
    """
    units = prices.units
    if request.edit is not None and units is not None and EDIT_VARIANT in units.rates:
        return EDIT_VARIANT
    return image_variant(request)


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
    return units.rate_for(unit_variant(prices, request))


def per_input_image_rate(prices: GatewayTokenPrices, request: ImagesRequest) -> int | None:
    """The lane's nano-USD per consumed source image of an edit, or None.

    Args:
        prices: The deployment's price card.
        request: The canonical request.

    Returns:
        The card's input rate for the request's variant, or None when the card
        is absent, of another kind, or prices no consumed images.
    """
    units = prices.units
    if units is None or units.kind != BilledUnitKind.IMAGE:
        return None
    return units.input_rate_for(unit_variant(prices, request))


def edit_unit_card_complete(prices: GatewayTokenPrices, request: ImagesRequest) -> bool:
    """Whether a per-image card prices both legs of a provider-billed edit.

    A provider that bills an edit's source images beside its output (xAI)
    is admitted for an edit only on a card carrying an :data:`EDIT_VARIANT`
    rate and an input rate for it; never at the generation rate or with the
    source images unbilled.

    Args:
        prices: The deployment's price card.
        request: The canonical edit request.

    Returns:
        True when both legs have a rate.
    """
    units = prices.units
    return (
        request.edit is not None
        and units is not None
        and units.kind == BilledUnitKind.IMAGE
        and EDIT_VARIANT in units.rates
        and units.input_rate_for(EDIT_VARIANT) is not None
    )


def images_ceiling_nano_usd(
    request: ImagesRequest,
    *,
    input_tokens: int,
    input_rate: int | None,
    output_rate: int | None,
    image_rate: int | None = None,
    image_input_rate: int | None = None,
    image_output_rate: int | None = None,
    input_image_rate: int | None = None,
) -> int | None:
    """Return the conservative reservation ceiling for one image call.

    A per-image priced lane (``image_rate``) reserves ``n`` images at its rate,
    which is also the most it can settle, plus each source image of an edit at
    ``input_image_rate`` when the card prices consumed images. GPT image models
    bill prompt tokens at the input rate and generated image tokens at the
    output rate (the image-output rate when the card sets a higher one);
    ``input_tokens`` is the prompt's estimated input reservation and ``n``
    times the largest per-image token count bounds the output. An edit's
    source images and mask each add :data:`MAXIMUM_EDIT_INPUT_IMAGE_TOKENS` at
    the higher of the input and image-input rates. A token lane without both
    rates is unpriced (``None``). A ceiling past the int8 ledger column raises
    ``NanoUsdOverflowError``.
    """
    edit_parts = 0 if request.edit is None else request.edit.input_image_parts
    if image_rate is not None:
        consumed = 0 if input_image_rate is None else edit_parts * input_image_rate
        return require_representable_nano_usd(
            request.n * image_rate + consumed, what="images reservation ceiling"
        )
    if input_rate is None or output_rate is None:
        return None
    image_input = max(input_rate, image_input_rate or 0)
    image_output = max(output_rate, image_output_rate or 0)
    input_ceiling = input_tokens * input_rate
    input_ceiling += edit_parts * MAXIMUM_EDIT_INPUT_IMAGE_TOKENS * image_input
    output_ceiling = request.n * MAXIMUM_IMAGE_OUTPUT_TOKENS * image_output
    return require_representable_nano_usd(
        (input_ceiling + output_ceiling + 999_999) // 1_000_000,
        what="images reservation ceiling",
    )
