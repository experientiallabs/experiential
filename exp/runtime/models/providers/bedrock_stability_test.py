"""Tests for the Stability AI Bedrock image wire."""

from __future__ import annotations

import pytest

from exp.runtime.models.providers.bedrock_stability import (
    STABILITY_MODELS,
    StabilityParameterError,
    stability_aspect_ratio,
    stability_body,
    stability_model,
)

_IMAGE = "aW1hZ2U="


@pytest.mark.parametrize(
    ("model_id", "key"),
    [
        ("stability.sd3-5-large-v1:0", "sd3-5-large"),
        ("stability.stable-image-core-v1:1", "stable-image-core"),
        ("stability.stable-image-ultra-v1:1", "stable-image-ultra"),
        ("us.stability.stable-image-inpaint-v1:0", "stable-image-inpaint"),
        ("us.stability.stable-outpaint-v1:0", "stable-outpaint"),
        ("stability.stable-fast-upscale-v1:0", "stable-fast-upscale"),
        ("eu.stability.stable-style-transfer-v1:0", "stable-style-transfer"),
    ],
)
def test_bedrock_ids_resolve_to_their_model(model_id: str, key: str) -> None:
    """Region prefixes and version suffixes resolve to the model's contract."""
    model = stability_model(model_id)
    assert model is not None
    assert model.key == key


@pytest.mark.parametrize(
    "model_id",
    [
        "anthropic.claude-sonnet-4-5",
        "stability.stable-diffusion-xl-v1",
        "stability.unknown-tool-v1:0",
        "xx.stability.stable-outpaint-v1:0",
    ],
)
def test_other_ids_are_not_stability_models(model_id: str) -> None:
    """Every non-Stability or unknown id answers None."""
    assert stability_model(model_id) is None


def test_every_known_tool_is_listed_once() -> None:
    """The table covers the 3 generation models and 13 Image Services tools."""
    assert len(STABILITY_MODELS) == 16
    assert sorted(key for key, model in STABILITY_MODELS.items() if model.generates) == [
        "sd3-5-large",
        "stable-image-core",
        "stable-image-ultra",
    ]


@pytest.mark.parametrize(
    ("size", "ratio"),
    [
        (None, None),
        ("auto", None),
        ("1024x1024", "1:1"),
        ("512x512", "1:1"),
        ("1536x1024", "3:2"),
        ("1024x1536", "2:3"),
        ("1792x1024", "16:9"),
        ("1024x1792", "9:16"),
    ],
)
def test_sizes_map_to_the_nearest_aspect_ratio(size: str | None, ratio: str | None) -> None:
    """OpenAI sizes become Stability aspect ratios; no size keeps the default."""
    assert stability_aspect_ratio(size) == ratio


def test_a_generation_carries_prompt_ratio_and_format() -> None:
    """A text-to-image body carries only the fields the model takes."""
    model = STABILITY_MODELS["stable-image-ultra"]
    body = stability_body(
        model, operation="generate", prompt="a red fox", size="1792x1024", output_format="jpeg"
    )
    assert body == {"prompt": "a red fox", "aspect_ratio": "16:9", "output_format": "jpeg"}


def test_a_default_generation_is_prompt_only() -> None:
    """Without size or format the provider defaults (1:1, PNG) apply."""
    body = stability_body(
        STABILITY_MODELS["sd3-5-large"],
        operation="generate",
        prompt="a fox",
        size=None,
        output_format=None,
    )
    assert body == {"prompt": "a fox"}


def test_an_edit_tool_refuses_text_to_image() -> None:
    """An edit-only tool on the generations surface names the right surface."""
    with pytest.raises(StabilityParameterError) as raised:
        stability_body(
            STABILITY_MODELS["stable-image-inpaint"],
            operation="generate",
            prompt="x",
            size=None,
            output_format=None,
        )
    assert raised.value.param == "model"
    assert "/v1/images/edits" in str(raised.value)


def test_core_refuses_an_edit() -> None:
    """Stable Image Core has no image-to-image mode."""
    with pytest.raises(StabilityParameterError) as raised:
        stability_body(
            STABILITY_MODELS["stable-image-core"],
            operation="edit",
            prompt="x",
            size=None,
            output_format=None,
            image=_IMAGE,
        )
    assert raised.value.param == "model"


def test_sd35_image_to_image_carries_mode_and_strength() -> None:
    """SD 3.5 Large edits run image-to-image with a default strength."""
    body = stability_body(
        STABILITY_MODELS["sd3-5-large"],
        operation="edit",
        prompt="make it night",
        size=None,
        output_format="png",
        image=_IMAGE,
        options={"seed": 7},
    )
    assert body == {
        "prompt": "make it night",
        "image": _IMAGE,
        "mode": "image-to-image",
        "strength": 0.7,
        "seed": 7,
        "output_format": "png",
    }


def test_ultra_image_to_image_takes_the_callers_strength() -> None:
    """Ultra has no mode field; the caller's strength replaces the default."""
    body = stability_body(
        STABILITY_MODELS["stable-image-ultra"],
        operation="edit",
        prompt="watercolor",
        size=None,
        output_format=None,
        image=_IMAGE,
        options={"strength": 0.3},
    )
    assert body == {"prompt": "watercolor", "image": _IMAGE, "strength": 0.3}


def test_inpaint_carries_mask_and_its_options() -> None:
    """Inpaint forwards the mask and every option it accepts."""
    body = stability_body(
        STABILITY_MODELS["stable-image-inpaint"],
        operation="edit",
        prompt="a cat",
        size=None,
        output_format="webp",
        image=_IMAGE,
        mask="bWFzaw==",
        options={"grow_mask": 5, "style_preset": "anime", "negative_prompt": "dogs"},
    )
    assert body == {
        "prompt": "a cat",
        "image": _IMAGE,
        "mask": "bWFzaw==",
        "grow_mask": 5,
        "style_preset": "anime",
        "negative_prompt": "dogs",
        "output_format": "webp",
    }


def test_remove_background_needs_no_prompt() -> None:
    """Background removal is image-only."""
    body = stability_body(
        STABILITY_MODELS["stable-image-remove-background"],
        operation="edit",
        prompt="",
        size=None,
        output_format=None,
        image=_IMAGE,
    )
    assert body == {"image": _IMAGE}


@pytest.mark.parametrize(
    ("key", "options", "kwargs", "param"),
    [
        ("stable-image-inpaint", {}, {}, "prompt"),
        ("stable-image-control-sketch", {}, {}, "prompt"),
        ("stable-image-search-recolor", {}, {"prompt": "red"}, "select_prompt"),
        ("stable-image-search-replace", {}, {"prompt": "a dog"}, "search_prompt"),
        ("stable-outpaint", {}, {}, "left"),
        ("stable-outpaint", {"left": 0, "up": 0}, {}, "left"),
        ("stable-style-transfer", {}, {}, "style_image"),
        ("stable-fast-upscale", {"seed": 3}, {}, "seed"),
        ("stable-image-remove-background", {}, {"mask": "bWFzaw=="}, "mask"),
        ("stable-outpaint", {"left": 10}, {"style_image": _IMAGE}, "style_image"),
    ],
)
def test_missing_or_foreign_inputs_are_refused_by_name(
    key: str, options: dict[str, str | int | float], kwargs: dict[str, str], param: str
) -> None:
    """A required input absent, or a control the tool lacks, names the parameter."""
    with pytest.raises(StabilityParameterError) as raised:
        stability_body(
            STABILITY_MODELS[key],
            operation="edit",
            prompt=kwargs.get("prompt", ""),
            size=None,
            output_format=None,
            image=_IMAGE,
            mask=kwargs.get("mask"),
            style_image=kwargs.get("style_image"),
            options=options,
        )
    assert raised.value.param == param


def test_an_edit_without_an_image_is_refused() -> None:
    """The image part is the one input every edit needs."""
    with pytest.raises(StabilityParameterError) as raised:
        stability_body(
            STABILITY_MODELS["stable-fast-upscale"],
            operation="edit",
            prompt="",
            size=None,
            output_format=None,
        )
    assert raised.value.param == "image"


def test_style_transfer_carries_init_and_style_images() -> None:
    """Style transfer's input travels as init_image beside style_image."""
    body = stability_body(
        STABILITY_MODELS["stable-style-transfer"],
        operation="edit",
        prompt="",
        size=None,
        output_format=None,
        image=_IMAGE,
        style_image="c3R5bGU=",
        options={"style_strength": 0.5},
    )
    assert body == {"init_image": _IMAGE, "style_image": "c3R5bGU=", "style_strength": 0.5}


def test_outpaint_carries_its_margins() -> None:
    """Outpaint forwards the margins it extends."""
    body = stability_body(
        STABILITY_MODELS["stable-outpaint"],
        operation="edit",
        prompt="",
        size=None,
        output_format=None,
        image=_IMAGE,
        options={"left": 128, "down": 64, "creativity": 0.5},
    )
    assert body == {"image": _IMAGE, "left": 128, "down": 64, "creativity": 0.5}
