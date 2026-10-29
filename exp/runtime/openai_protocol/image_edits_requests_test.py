"""Tests for the OpenAI Images edit upload decoder."""

from __future__ import annotations

import pytest

from exp.common.core.artifacts import JsonObject
from exp.runtime.openai_protocol.errors import OpenAIProtocolError
from exp.runtime.openai_protocol.image_edits_requests import decode_image_edits

_IMAGE = "aW1hZ2U="


def _upload(fields: dict[str, str], files: dict[str, str] | None = None) -> JsonObject:
    """Build one parsed upload as the data plane sends it."""
    return {"fields": dict(fields), "files": dict({"image": _IMAGE} if files is None else files)}


def test_an_edit_decodes_its_fields_images_and_options() -> None:
    """OpenAI fields, image parts, and typed tool controls all land canonically."""
    decoded = decode_image_edits(
        _upload(
            {
                "model": "inpaint",
                "prompt": "a cat",
                "n": "1",
                "output_format": "jpeg",
                "response_format": "b64_json",
                "user": "end-user-7",
                "seed": "42",
                "grow_mask": "6",
                "creativity": "0.35",
                "style_preset": "anime",
                "negative_prompt": "dogs",
            },
            {"image": _IMAGE, "mask": "bWFzaw=="},
        )
    )
    assert decoded.alias == "inpaint"
    request = decoded.request
    assert request.prompt == "a cat"
    assert request.output_format == "jpeg"
    assert request.user == "end-user-7"
    assert request.edit is not None
    assert request.edit.image == _IMAGE
    assert request.edit.mask == "bWFzaw=="
    assert request.edit.style_image is None
    assert request.edit_options == {
        "seed": 42,
        "grow_mask": 6,
        "creativity": 0.35,
        "style_preset": "anime",
        "negative_prompt": "dogs",
    }


def test_the_sdk_list_spelling_of_image_is_accepted() -> None:
    """``image[]`` with one entry is the same image part."""
    decoded = decode_image_edits(_upload({"model": "m"}, {"image[]": _IMAGE}))
    assert decoded.request.edit is not None
    assert decoded.request.edit.image == _IMAGE
    assert decoded.request.prompt == ""
    assert decoded.request.edit_options is None


@pytest.mark.parametrize(
    ("upload", "param", "code"),
    [
        (_upload({"model": "m"}, {}), "image", "invalid_request"),
        (_upload({"model": "m"}, {"image": _IMAGE, "image[]": _IMAGE}), "image", "invalid_request"),
        (_upload({"model": "m"}, {"image": _IMAGE, "thumbnail": _IMAGE}), "thumbnail", None),
        (_upload({}), "model", "invalid_request"),
        (_upload({"model": "m", "stream": "true"}), "stream", "unsupported_parameter"),
        (_upload({"model": "m", "partial_images": "2"}), "partial_images", None),
        (_upload({"model": "m", "input_fidelity": "high"}), "input_fidelity", None),
        (_upload({"model": "m", "frobnicate": "1"}), "frobnicate", "invalid_request"),
        (_upload({"model": "m", "seed": "-1"}), "seed", "invalid_request"),
        (_upload({"model": "m", "seed": "seven"}), "seed", "invalid_request"),
        (_upload({"model": "m", "creativity": "0.05"}), "creativity", "invalid_request"),
        (_upload({"model": "m", "left": "2001"}), "left", "invalid_request"),
        (_upload({"model": "m", "style_preset": "baroque"}), "style_preset", "invalid_request"),
        (_upload({"model": "m", "aspect_ratio": "7:3"}), "aspect_ratio", "invalid_request"),
        (_upload({"model": "m", "select_prompt": ""}), "select_prompt", "invalid_request"),
        (_upload({"model": "m", "n": "two"}), "n", "invalid_request"),
        (_upload({"model": "m", "output_compression": "x"}), "output_compression", None),
    ],
)
def test_invalid_uploads_are_refused_by_field(
    upload: JsonObject, param: str, code: str | None
) -> None:
    """Every malformed, unknown, or unsupported input names its field."""
    with pytest.raises(OpenAIProtocolError) as raised:
        decode_image_edits(upload)
    assert raised.value.status_code == 400
    assert raised.value.detail.param == param
    if code is not None:
        assert raised.value.detail.code == code


def test_openai_enumerations_are_still_validated() -> None:
    """A bad size or format fails canonical validation, naming the field."""
    with pytest.raises(OpenAIProtocolError) as raised:
        decode_image_edits(_upload({"model": "m", "size": "huge"}))
    assert raised.value.status_code == 400


def test_a_malformed_upload_object_is_refused() -> None:
    """Non-string members mean the data plane's upload is malformed."""
    with pytest.raises(OpenAIProtocolError):
        decode_image_edits({"fields": {"model": 7}, "files": {"image": _IMAGE}})
    with pytest.raises(OpenAIProtocolError):
        decode_image_edits({"fields": [], "files": {"image": _IMAGE}})
