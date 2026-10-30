"""Tests for the canonical image-generation contract and its reservation ceiling."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from exp.common.models import BilledUnitKind, GatewayTokenPrices, GatewayUnitPrices
from exp.runtime.gateway.contracts import GatewayApiSurface
from exp.runtime.gateway.images_contracts import (
    MAXIMUM_IMAGE_OUTPUT_TOKENS,
    ImageEditInput,
    ImagesRequest,
    image_variant,
    images_ceiling_nano_usd,
    per_image_rate,
)
from exp.runtime.gateway.ledger_valuation import NanoUsdOverflowError


def test_images_request_defaults_to_one_image_and_records_the_surface() -> None:
    """A bare prompt is one image on the images surface with every control unset."""
    request = ImagesRequest(prompt="a cat")
    assert request.surface is GatewayApiSurface.IMAGES
    assert request.n == 1
    assert request.size is None
    assert request.attribution_label is None
    assert ImagesRequest(prompt="a cat", user="tenant-7").attribution_label == "tenant-7"


@pytest.mark.parametrize(
    "overrides",
    [{"n": 0}, {"n": 11}, {"prompt": ""}, {"output_compression": 101}, {"size": "9x9"}],
)
def test_images_request_rejects_out_of_contract_controls(overrides: dict[str, object]) -> None:
    """The contract keeps the OpenAI bounds: n in 1..10, known sizes, 0..100 compression."""
    values: dict[str, object] = {"prompt": "a cat", **overrides}
    with pytest.raises(ValidationError):
        ImagesRequest.model_validate(values)


def test_reservation_ceiling_prices_prompt_tokens_and_maximum_image_tokens() -> None:
    """The ceiling prices the prompt estimate at the input rate and n images at the output rate."""
    request = ImagesRequest(prompt="a cat", n=2)
    ceiling = images_ceiling_nano_usd(
        request, input_tokens=10, input_rate=5_000_000, output_rate=40_000_000
    )
    assert ceiling is not None
    output_only = (2 * MAXIMUM_IMAGE_OUTPUT_TOKENS * 40_000_000) // 1_000_000
    assert ceiling == output_only + 50
    # A lane without an output rate (per-image priced) is unpriced for images.
    assert (
        images_ceiling_nano_usd(request, input_tokens=10, input_rate=5_000_000, output_rate=None)
        is None
    )
    # A ceiling past the int8 ledger column is refused by name, never wrapped
    # or read as "unpriced".
    with pytest.raises(NanoUsdOverflowError):
        images_ceiling_nano_usd(request, input_tokens=10**16, input_rate=10**12, output_rate=1)


def test_the_image_variant_is_quality_then_size() -> None:
    """Omitted parts drop out, so a bare request prices at the flat rate."""
    assert image_variant(ImagesRequest(prompt="p", quality="hd", size="1024x1792")) == (
        "hd/1024x1792"
    )
    assert image_variant(ImagesRequest(prompt="p", size="1024x1024")) == "1024x1024"
    assert image_variant(ImagesRequest(prompt="p")) == ""


def test_a_per_image_card_prices_its_variant_or_flat_rate() -> None:
    """Variants price exactly; others fall back to the flat rate when one exists."""
    card = GatewayTokenPrices(
        units=GatewayUnitPrices(kind=BilledUnitKind.IMAGE, rates={"": 20, "hd/1024x1024": 80})
    )
    assert per_image_rate(card, ImagesRequest(prompt="p", quality="hd", size="1024x1024")) == 80
    assert per_image_rate(card, ImagesRequest(prompt="p", size="256x256")) == 20
    strict = GatewayTokenPrices(
        units=GatewayUnitPrices(kind=BilledUnitKind.IMAGE, rates={"hd/1024x1024": 80})
    )
    assert per_image_rate(strict, ImagesRequest(prompt="p")) is None
    seconds = GatewayTokenPrices(
        units=GatewayUnitPrices(kind=BilledUnitKind.VIDEO_SECOND, rates={"": 1})
    )
    assert per_image_rate(seconds, ImagesRequest(prompt="p")) is None


def test_a_per_image_ceiling_is_n_images() -> None:
    """The unit reservation is n times the rate, regardless of token rates."""
    request = ImagesRequest(prompt="p", n=3)
    assert (
        images_ceiling_nano_usd(
            request, input_tokens=10, input_rate=None, output_rate=None, image_rate=20_000_000
        )
        == 60_000_000
    )


def test_a_generation_requires_a_prompt_and_an_edit_may_omit_it() -> None:
    """Background removal or fast upscale carries no prompt; a generation must."""
    with pytest.raises(ValidationError, match="requires a prompt"):
        ImagesRequest(prompt="")
    edit = ImagesRequest(edit=ImageEditInput(image="aW1hZ2U="))
    assert edit.prompt == ""
    assert image_variant(edit) == ""


def test_edit_options_require_an_edit() -> None:
    """Image-tool controls never ride a plain generation."""
    with pytest.raises(ValidationError, match="edit options require"):
        ImagesRequest(prompt="a fox", edit_options={"seed": 1})
    request = ImagesRequest(
        edit=ImageEditInput(image="aW1hZ2U=", mask="bWFzaw=="), edit_options={"seed": 1}
    )
    assert request.edit_options == {"seed": 1}
