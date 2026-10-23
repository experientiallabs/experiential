"""Tests for decoding OpenAI ``POST /v1/videos`` bodies."""

from __future__ import annotations

import base64

import pytest

from exp.common.core.artifacts import JsonObject
from exp.runtime.openai_protocol.errors import OpenAIProtocolError
from exp.runtime.openai_protocol.video_requests import decode_video_create

_PNG = b"\x89PNG\r\n\x1a\nframe"


def test_the_sdk_multipart_shape_decodes_with_string_seconds() -> None:
    """Form fields are strings: seconds arrives as ``"8"`` and becomes an integer."""
    decoded = decode_video_create(
        {"model": "veo", "prompt": "a boat", "seconds": "8", "size": "1280x720"},
        uploaded_reference=("image/png", _PNG),
    )
    assert decoded.alias == "veo"
    assert decoded.request.seconds == 8
    assert decoded.reference_image == _PNG
    assert decoded.request.input_reference is not None
    assert decoded.request.input_reference.bytes == len(_PNG)


def test_a_json_data_url_reference_decodes() -> None:
    """A JSON body may carry the first frame as a base64 data URL."""
    url = "data:image/png;base64," + base64.b64encode(_PNG).decode()
    decoded = decode_video_create(
        {"model": "veo", "prompt": "p", "seconds": 4, "input_reference": {"image_url": url}}
    )
    assert decoded.reference_image == _PNG


def test_defaults_apply_when_omitted() -> None:
    """Seconds and size default to OpenAI's documented 4 and 720x1280."""
    decoded = decode_video_create({"model": "veo", "prompt": "p"})
    assert (decoded.request.seconds, decoded.request.size) == (4, "720x1280")


@pytest.mark.parametrize(
    ("fields", "param", "code"),
    [
        ({"prompt": "p"}, "model", "missing_required_parameter"),
        ({"model": "veo"}, "prompt", "missing_required_parameter"),
        ({"model": "veo", "prompt": "p", "seconds": "eight"}, "seconds", "invalid_value"),
        ({"model": "veo", "prompt": "p", "seconds": True}, "seconds", "invalid_value"),
        ({"model": "veo", "prompt": "p", "seconds": "21"}, "seconds", "invalid_value"),
        ({"model": "veo", "prompt": "p", "size": "1792x1024"}, "size", "unsupported_value"),
        ({"model": "veo", "prompt": "p", "characters": []}, "characters", None),
        (
            {"model": "veo", "prompt": "p", "input_reference": {"image_url": "https://x/a.png"}},
            "input_reference.image_url",
            "unsupported_value",
        ),
        (
            {"model": "veo", "prompt": "p", "input_reference": {"file_id": "file_1"}},
            "input_reference.file_id",
            "unsupported_value",
        ),
        (
            {"model": "veo", "prompt": "p", "input_reference": {"image_url": "data:image/png,raw"}},
            "input_reference.image_url",
            "invalid_value",
        ),
        (
            {
                "model": "veo",
                "prompt": "p",
                "input_reference": {"image_url": "data:text/plain;base64,aGk="},
            },
            "input_reference",
            "unsupported_value",
        ),
    ],
)
def test_invalid_fields_are_refused_by_name(
    fields: JsonObject, param: str, code: str | None
) -> None:
    """Every refusal is a 400 naming the responsible field."""
    with pytest.raises(OpenAIProtocolError) as refused:
        decode_video_create(fields)
    assert refused.value.status_code == 400
    assert refused.value.detail.param == param
    if code is not None:
        assert refused.value.detail.code == code


def test_a_reference_sent_twice_is_refused() -> None:
    """A file part and a JSON reference cannot both name the first frame."""
    with pytest.raises(OpenAIProtocolError) as refused:
        decode_video_create(
            {"model": "veo", "prompt": "p", "input_reference": {"image_url": "data:x"}},
            uploaded_reference=("image/png", _PNG),
        )
    assert refused.value.detail.param == "input_reference"


def test_an_oversized_reference_is_refused() -> None:
    """Images past 10 MiB are refused before any provider call."""
    with pytest.raises(OpenAIProtocolError) as refused:
        decode_video_create(
            {"model": "veo", "prompt": "p"},
            uploaded_reference=("image/png", b"x" * (10 * 1024 * 1024 + 1)),
        )
    assert refused.value.detail.param == "input_reference"
