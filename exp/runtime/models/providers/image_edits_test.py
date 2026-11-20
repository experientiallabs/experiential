"""Tests for the OpenAI multipart and xAI JSON image-edit wires."""

from __future__ import annotations

import base64

import pytest

from exp.runtime.gateway.images_contracts import ImageEditInput, ImagesRequest
from exp.runtime.models.providers.image_edits import (
    ImageEditParameterError,
    edits_xai_images,
    image_media_type,
    images_edits_url,
    openai_image_edit_fields,
    xai_image_edit_request,
)

_PNG = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16).decode("ascii")
_JPEG = base64.b64encode(b"\xff\xd8\xff\xe0" + b"\x00" * 16).decode("ascii")
_WEBP = base64.b64encode(b"RIFF\x00\x00\x00\x00WEBPVP8 " + b"\x00" * 8).decode("ascii")


def _edit(**fields: object) -> ImagesRequest:
    """Build one canonical edit with a PNG source unless the fields say otherwise."""
    edit = fields.pop("edit", ImageEditInput(image=_PNG))
    return ImagesRequest.model_validate({"prompt": "watercolor", "edit": edit, **fields})


def test_the_edits_endpoint_sits_beside_the_generations_endpoint() -> None:
    """The edit URL swaps only the last path segment and keeps Azure's query."""
    assert images_edits_url("https://api.openai.com/v1/images/generations") == (
        "https://api.openai.com/v1/images/edits"
    )
    assert images_edits_url(
        "https://r.openai.azure.com/openai/deployments/d/images/generations"
        "?api-version=2025-04-01-preview"
    ) == (
        "https://r.openai.azure.com/openai/deployments/d/images/edits"
        "?api-version=2025-04-01-preview"
    )
    assert images_edits_url("https://bedrock.example/model/x/invoke") is None


def test_openai_edit_fields_carry_every_gpt_image_control_but_response_format() -> None:
    """Every GPT image edit control rides as a text field; ``user`` and ``url`` never do."""
    request = _edit(
        n=2,
        size="1024x1536",
        quality="low",
        background="transparent",
        input_fidelity="high",
        output_format="webp",
        output_compression=80,
        moderation="low",
        response_format="b64_json",
        user="tenant-7",
    )
    assert openai_image_edit_fields("gpt-image-2.5-sunburst", request) == {
        "model": "gpt-image-2.5-sunburst",
        "n": 2,
        "prompt": "watercolor",
        "size": "1024x1536",
        "quality": "low",
        "background": "transparent",
        "input_fidelity": "high",
        "output_format": "webp",
        "output_compression": 80,
        "moderation": "low",
    }


@pytest.mark.parametrize(
    ("request_fields", "param"),
    [
        ({"edit": ImageEditInput(image=_PNG, style_image=_PNG)}, "style_image"),
        ({"edit_options": {"seed": 7}}, "seed"),
        ({"prompt": ""}, "prompt"),
    ],
)
def test_openai_edits_refuse_stability_controls_by_name(
    request_fields: dict[str, object], param: str
) -> None:
    """A promptless edit or a Stability-only control excludes the rung, never dropped."""
    with pytest.raises(ImageEditParameterError) as raised:
        openai_image_edit_fields("gpt-image-1.5", _edit(**request_fields))
    assert raised.value.param == param


def test_one_xai_source_rides_as_image_and_several_as_images() -> None:
    """xAI takes data URIs: ``image`` for one source, ``images`` for two to five."""
    one = xai_image_edit_request("grok-imagine-image-2.0", _edit())
    assert one == {
        "model": "grok-imagine-image-2.0",
        "prompt": "watercolor",
        "n": 1,
        "image": {"url": f"data:image/png;base64,{_PNG}", "type": "image_url"},
        "response_format": "b64_json",
    }
    several = xai_image_edit_request(
        "grok-imagine-image-2.0",
        _edit(n=2, edit=ImageEditInput(image=_PNG, additional_images=(_JPEG, _WEBP))),
    )
    assert "image" not in several
    assert several["n"] == 2
    assert several["images"] == [
        {"url": f"data:image/png;base64,{_PNG}", "type": "image_url"},
        {"url": f"data:image/jpeg;base64,{_JPEG}", "type": "image_url"},
        {"url": f"data:image/webp;base64,{_WEBP}", "type": "image_url"},
    ]


@pytest.mark.parametrize(
    ("request_fields", "param"),
    [
        ({"edit": ImageEditInput(image=_PNG, mask=_PNG)}, "mask"),
        ({"edit": ImageEditInput(image=_PNG, additional_images=(_PNG,) * 5)}, "image"),
        ({"edit": ImageEditInput(image=base64.b64encode(b"GIF89a....").decode())}, "image"),
        ({"size": "1024x1024"}, "size"),
        ({"quality": "high"}, "quality"),
        ({"input_fidelity": "high"}, "input_fidelity"),
        ({"output_format": "png"}, "output_format"),
        ({"moderation": "low"}, "moderation"),
        ({"edit_options": {"seed": 7}}, "seed"),
        ({"prompt": ""}, "prompt"),
    ],
)
def test_xai_edits_refuse_what_the_wire_cannot_carry(
    request_fields: dict[str, object], param: str
) -> None:
    """A mask, a sixth image, a foreign format, or an OpenAI-only control is refused."""
    with pytest.raises(ImageEditParameterError) as raised:
        xai_image_edit_request("grok-imagine-image-2.0", _edit(**request_fields))
    assert raised.value.param == param


def test_only_grok_imagine_image_models_edit_on_xai() -> None:
    """The Grok Imagine image family edits; other xAI image ids do not."""
    assert edits_xai_images("grok-imagine-image-2.0")
    assert edits_xai_images("x-ai/grok-imagine-image")
    assert not edits_xai_images("grok-2-image")


def test_image_media_types_are_read_from_leading_bytes() -> None:
    """PNG, JPEG, and WebP are recognized; anything else is None."""
    assert image_media_type(_PNG) == "image/png"
    assert image_media_type(_JPEG) == "image/jpeg"
    assert image_media_type(_WEBP) == "image/webp"
    assert image_media_type("not base64!") is None
