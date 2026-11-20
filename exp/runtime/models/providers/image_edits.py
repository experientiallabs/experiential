# Copyright (c) 2026 Experiential Labs. All rights reserved.
"""Image-edit wires for OpenAI GPT image models and xAI Grok Imagine.

Two OpenAI-wire providers edit images, each on its own request shape:

- **OpenAI Images edits** (``gpt-image-*`` on OpenAI, and on an Azure
  connection's OpenAI Image API wire): ``POST {root}/images/edits`` as
  ``multipart/form-data``. The source images ride as ``image`` (one) or
  repeated ``image[]`` parts (up to 16), the optional ``mask`` as its own
  part, and the controls as text fields. The control plane builds the text
  fields (:func:`openai_image_edit_fields`); the data plane keeps the uploaded
  image bytes and encodes the multipart body at dispatch. GPT image models
  never receive ``response_format`` (they always answer base64). The answer
  bills tokens: ``usage.input_tokens`` with an ``input_tokens_details`` split
  into text and image tokens, and image ``output_tokens``.
- **xAI image edits** (``grok-imagine-image*``): ``POST {root}/images/edits``
  as JSON, the source images as data URIs in ``image`` (one) or ``images``
  (two to five), always asking for ``b64_json`` (xAI keeps no hosted images
  for Zero Data Retention teams). xAI bills each edited image returned and
  each source image read, so such a rung bills per image on both legs.

A control a rung's wire cannot carry is refused by name
(:class:`ImageEditParameterError`), never silently dropped: the admission
excludes that rung and answers the refusal when no other rung serves.
"""

from __future__ import annotations

import base64
import binascii
from urllib.parse import urlsplit, urlunsplit

from exp.common.core.artifacts import JsonObject, JsonValue
from exp.runtime.gateway.images_contracts import ImagesRequest

MAXIMUM_XAI_EDIT_IMAGES = 5
"""Source images one xAI edit carries ("up to 5 source images in a single request")."""

_GENERATIONS_SUFFIX = "/images/generations"

# OpenAI edit controls forwarded as text fields when present. ``user`` is
# attribution only and ``response_format`` is never sent to a GPT image model.
_OPENAI_EDIT_FIELDS = (
    "size",
    "quality",
    "background",
    "input_fidelity",
    "output_format",
    "output_compression",
    "moderation",
)

# OpenAI controls xAI's edit endpoint has no field for.
_XAI_UNSUPPORTED_CONTROLS = (
    "size",
    "quality",
    "background",
    "input_fidelity",
    "output_format",
    "output_compression",
    "moderation",
    "style",
)

# Leading bytes of the image formats the edit wires accept.
_IMAGE_SIGNATURES = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
)


class ImageEditParameterError(ValueError):
    """A request control one edit rung's wire cannot carry.

    Attributes:
        param: The public request parameter at fault.
    """

    def __init__(self, param: str, message: str) -> None:
        """Name the offending parameter and the reason.

        Args:
            param: Public request parameter.
            message: Caller-facing reason.
        """
        super().__init__(message)
        self.param = param


def images_edits_url(images_url: str) -> str | None:
    """Return the ``/images/edits`` endpoint beside an ``/images/generations`` one.

    The query (Azure's ``api-version``) is kept.

    Args:
        images_url: A rung's resolved OpenAI-wire image-generation endpoint.

    Returns:
        The edits endpoint, or None when the URL is not a generations endpoint.
    """
    parsed = urlsplit(images_url)
    path = parsed.path.rstrip("/")
    if not path.endswith(_GENERATIONS_SUFFIX):
        return None
    edits = path.removesuffix(_GENERATIONS_SUFFIX) + "/images/edits"
    return urlunsplit((parsed.scheme, parsed.netloc, edits, parsed.query, parsed.fragment))


def edits_xai_images(model_id: str) -> bool:
    """Whether one xAI image model edits images (the Grok Imagine image family).

    Args:
        model_id: The served xAI model id.

    Returns:
        True for ``grok-imagine-image`` and its versions.
    """
    return model_id.rsplit("/", 1)[-1].lower().startswith("grok-imagine-image")


def image_media_type(encoded: str) -> str | None:
    """Identify one base64 source image's format from its leading bytes.

    Args:
        encoded: Standard base64 image bytes.

    Returns:
        ``image/png``, ``image/jpeg`` or ``image/webp``, or None for anything else.
    """
    try:
        head = base64.b64decode(encoded[:24], validate=True)
    except (binascii.Error, ValueError):
        return None
    for signature, media_type in _IMAGE_SIGNATURES:
        if head.startswith(signature):
            return media_type
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "image/webp"
    return None


def _refuse_foreign_edit_controls(request: ImagesRequest) -> None:
    """Refuse a promptless edit and the Stability-only controls on an OpenAI-wire rung.

    Both OpenAI and xAI require an edit prompt, while a Stability tool (background
    removal, fast upscale) needs none, so a promptless edit excludes these rungs
    instead of reaching a provider that refuses it without failover.
    """
    if not request.prompt:
        raise ImageEditParameterError("prompt", "This model edits only with a prompt; send one.")
    edit = request.edit
    if edit is not None and edit.style_image is not None:
        raise ImageEditParameterError("style_image", "This model takes no style_image.")
    for name in request.edit_options or {}:
        raise ImageEditParameterError(name, f"This model takes no {name}.")


def openai_image_edit_fields(model_id: str, request: ImagesRequest) -> JsonObject:
    """Build the text fields of one OpenAI Images edit (``multipart/form-data``).

    The data plane appends the source images (``image``, or repeated
    ``image[]`` when there are several) and any ``mask`` as file parts.

    Args:
        model_id: The rung's wire model id (an Azure deployment name on Azure).
        request: The canonical edit request.

    Returns:
        The text fields, by name.

    Raises:
        ImageEditParameterError: The request has no prompt, or carries a
            control this wire cannot carry (a style image or a Stability edit
            option).
    """
    _refuse_foreign_edit_controls(request)
    if request.style is not None:
        raise ImageEditParameterError("style", "GPT image models take no style.")
    fields: JsonObject = {"model": model_id, "n": request.n, "prompt": request.prompt}
    for name in _OPENAI_EDIT_FIELDS:
        value = getattr(request, name)
        if value is not None:
            fields[name] = value
    return fields


def xai_image_edit_request(model_id: str, request: ImagesRequest) -> JsonObject:
    """Build one xAI ``/images/edits`` JSON body from a canonical edit.

    Args:
        model_id: The rung's xAI model id.
        request: The canonical edit request.

    Returns:
        The JSON body: the prompt, the source images as data URIs (``image``
        for one, ``images`` for several), ``n``, and ``b64_json`` output.

    Raises:
        ImageEditParameterError: The request carries something xAI's edit
            cannot carry: no prompt, a mask, more than five images, an image that is not
            PNG, JPEG or WebP, or an OpenAI-only control.
    """
    edit = request.edit
    if edit is None:
        raise ImageEditParameterError("image", "An image edit must carry an image part.")
    _refuse_foreign_edit_controls(request)
    if edit.mask is not None:
        raise ImageEditParameterError("mask", "This model edits without a mask; omit mask.")
    for control in _XAI_UNSUPPORTED_CONTROLS:
        if getattr(request, control) is not None:
            raise ImageEditParameterError(control, f"This model's edits take no {control}.")
    if len(edit.images) > MAXIMUM_XAI_EDIT_IMAGES:
        raise ImageEditParameterError(
            "image", f"This model edits at most {MAXIMUM_XAI_EDIT_IMAGES} images per request."
        )
    sources: list[JsonValue] = []
    for encoded in edit.images:
        media_type = image_media_type(encoded)
        if media_type is None:
            raise ImageEditParameterError("image", "Send each image as PNG, JPEG or WebP.")
        sources.append({"url": f"data:{media_type};base64,{encoded}", "type": "image_url"})
    body: JsonObject = {"model": model_id, "prompt": request.prompt, "n": request.n}
    if len(sources) == 1:
        body["image"] = sources[0]
    else:
        body["images"] = sources
    # xAI refuses hosted output for Zero Data Retention teams; the gateway
    # hosts nothing either, so the answer is always inline base64.
    body["response_format"] = "b64_json"
    return body
