"""Decode OpenAI ``POST /v1/videos`` bodies into the canonical video request.

The official SDK always sends multipart form data (``client.videos.create``
encodes every field as a form part and the optional ``input_reference`` as a
file part); curl and other HTTP clients may send JSON. The native data plane
parses either into the same shape: the text fields as one object, plus the
uploaded reference image's bytes when a file part was present. A JSON
``input_reference`` object names its image as ``image_url``, which must be a
base64 data URL: a remote URL would make the gateway fetch arbitrary hosts,
so it is refused with a pointer to the multipart upload instead.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
from dataclasses import dataclass, field

from pydantic import ValidationError

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.video.contracts import (
    DEFAULT_VIDEO_SECONDS,
    DEFAULT_VIDEO_SIZE,
    MAXIMUM_REFERENCE_IMAGE_BYTES,
    MAXIMUM_VIDEO_SECONDS,
    REFERENCE_IMAGE_TYPES,
    VIDEO_SIZES,
    VideoReferenceImage,
    VideoRequest,
)
from exp.runtime.openai_protocol.errors import OpenAIProtocolError
from exp.runtime.openai_protocol.manifest import VIDEO_MANIFEST, validate_manifest
from exp.runtime.openai_protocol.validation_errors import validation_protocol_error


@dataclass(frozen=True)
class DecodedVideoRequest:
    """One decoded create request.

    Attributes:
        alias: The public model alias.
        request: The canonical, content-bearing request.
        reference_image: The first-frame image bytes, when one was supplied.
    """

    alias: str
    request: VideoRequest
    reference_image: bytes | None = field(default=None, repr=False)


def _invalid(param: str, message: str, code: str = "invalid_value") -> OpenAIProtocolError:
    """Build one field-specific 400."""
    return OpenAIProtocolError(status_code=400, code=code, message=message, param=param)


def _seconds(value: object) -> int:
    """Read ``seconds`` as the SDK's string or a JSON integer.

    Raises:
        OpenAIProtocolError: The value is not a whole number of seconds in range.
    """
    if value is None:
        return DEFAULT_VIDEO_SECONDS
    if isinstance(value, bool):
        raise _invalid("seconds", 'seconds must be a whole number of seconds, such as "8".')
    if isinstance(value, str) and value.strip().isdecimal():
        value = int(value.strip())
    if not isinstance(value, int) or not 1 <= value <= MAXIMUM_VIDEO_SECONDS:
        raise _invalid(
            "seconds",
            f'seconds must be a whole number from 1 to {MAXIMUM_VIDEO_SECONDS}, such as "8".',
        )
    return value


def _size(value: object) -> str:
    """Read ``size`` and require a size some provider renders.

    Raises:
        OpenAIProtocolError: The size is not a string or not a known size.
    """
    if value is None:
        return DEFAULT_VIDEO_SIZE
    if not isinstance(value, str) or value not in VIDEO_SIZES:
        sizes = ", ".join(VIDEO_SIZES)
        raise _invalid("size", f"size must be one of {sizes}.", code="unsupported_value")
    return value


def _reference_from_json(value: object) -> tuple[str, bytes]:
    """Read a JSON ``input_reference`` object holding a base64 data URL.

    Returns:
        The image's declared media type and decoded bytes.

    Raises:
        OpenAIProtocolError: Any other shape, a remote URL, or a file id.
    """
    if not isinstance(value, dict):
        raise _invalid(
            "input_reference",
            "input_reference must be an uploaded file or an object with an image_url data URL.",
        )
    if "file_id" in value:
        raise _invalid(
            "input_reference.file_id",
            "file_id references are not supported; upload the image as multipart form data.",
            code="unsupported_value",
        )
    url = value.get("image_url")
    if set(value) != {"image_url"} or not isinstance(url, str):
        raise _invalid("input_reference", "input_reference must carry exactly one image_url.")
    if not url.startswith("data:"):
        raise _invalid(
            "input_reference.image_url",
            "image_url must be a base64 data URL; upload remote images as multipart form data.",
            code="unsupported_value",
        )
    header, separator, data = url[5:].partition(",")
    media_type, _semicolon, encoding = header.partition(";")
    if not separator or encoding != "base64":
        raise _invalid("input_reference.image_url", "image_url must be a base64 data URL.")
    try:
        return media_type, base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise _invalid("input_reference.image_url", "image_url holds invalid base64.") from exc


def _reference_facts(content_type: str, image: bytes) -> VideoReferenceImage:
    """Validate one reference image's type and size into its facts.

    Raises:
        OpenAIProtocolError: The type is not JPEG, PNG, or WebP, or the image is
            empty or too large.
    """
    if content_type not in REFERENCE_IMAGE_TYPES:
        raise _invalid(
            "input_reference",
            "input_reference must be a JPEG, PNG, or WebP image.",
            code="unsupported_value",
        )
    if not image or len(image) > MAXIMUM_REFERENCE_IMAGE_BYTES:
        raise _invalid(
            "input_reference",
            f"input_reference must be 1 byte to {MAXIMUM_REFERENCE_IMAGE_BYTES // (1024 * 1024)} "
            "MiB.",
        )
    return VideoReferenceImage.model_validate(
        {
            "content_type": content_type,
            "bytes": len(image),
            "sha256": hashlib.sha256(image).hexdigest(),
        }
    )


def decode_video_create(
    fields: JsonObject,
    *,
    uploaded_reference: tuple[str, bytes] | None = None,
) -> DecodedVideoRequest:
    """Decode one create request's fields into the canonical surface.

    Args:
        fields: The JSON body, or the multipart text fields (each a string).
        uploaded_reference: The multipart ``input_reference`` file part's
            content type and bytes, when one was uploaded.

    Returns:
        The alias, canonical request, and reference bytes.

    Raises:
        OpenAIProtocolError: Any field is missing, invalid, or unsupported.
    """
    payload = dict(fields)
    if uploaded_reference is not None:
        if "input_reference" in payload:
            raise _invalid(
                "input_reference", "Send input_reference once, as a file or as an object."
            )
        payload["input_reference"] = "<uploaded file>"
    validate_manifest(payload, VIDEO_MANIFEST)
    model = payload.get("model")
    if not isinstance(model, str) or not model.strip():
        raise _invalid("model", "model is required.", code="missing_required_parameter")
    prompt = payload.get("prompt")
    if not isinstance(prompt, str):
        raise _invalid("prompt", "prompt is required.", code="missing_required_parameter")
    reference: tuple[str, bytes] | None = uploaded_reference
    if uploaded_reference is None and payload.get("input_reference") is not None:
        reference = _reference_from_json(payload["input_reference"])
    try:
        request = VideoRequest(
            prompt=prompt,
            seconds=_seconds(payload.get("seconds")),
            size=_size(payload.get("size")),
            input_reference=None if reference is None else _reference_facts(*reference),
        )
    except ValidationError as exc:
        raise validation_protocol_error(exc) from exc
    return DecodedVideoRequest(
        alias=model.strip(),
        request=request,
        reference_image=None if reference is None else reference[1],
    )
