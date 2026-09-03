"""Tests for the canonical image-generation contract and its reservation ceiling."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from exp.runtime.gateway.contracts import GatewayApiSurface
from exp.runtime.gateway.images_contracts import (
    MAXIMUM_IMAGE_OUTPUT_TOKENS,
    ImagesRequest,
    images_ceiling_micro_usd,
)


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


def test_reservation_ceiling_prices_prompt_bytes_and_maximum_image_tokens() -> None:
    """The ceiling bounds prompt bytes at the input rate and n images at the output rate."""
    request = ImagesRequest(prompt="a cat", n=2)
    ceiling = images_ceiling_micro_usd(
        request, input_rate=5_000_000, output_rate=40_000_000, maximum=10**12
    )
    assert ceiling is not None
    output_only = (2 * MAXIMUM_IMAGE_OUTPUT_TOKENS * 40_000_000) // 1_000_000
    assert ceiling >= output_only
    assert ceiling < output_only + 5_000_000
    # A lane without an output rate (per-image priced) is unpriced for images.
    assert (
        images_ceiling_micro_usd(request, input_rate=5_000_000, output_rate=None, maximum=10**12)
        is None
    )
    assert images_ceiling_micro_usd(request, input_rate=1, output_rate=1, maximum=0) is None
