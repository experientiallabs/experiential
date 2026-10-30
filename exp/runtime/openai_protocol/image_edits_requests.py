"""Decode one OpenAI Images edit upload into the canonical images surface.

The data plane parses the ``/v1/images/edits`` multipart body itself: its text
fields arrive as strings and its image parts as base64 strings, in one JSON
object ``{"fields": {name: text}, "files": {name: base64}}``. The OpenAI edit
fields map onto :class:`~exp.runtime.gateway.images_contracts.ImagesRequest`
with an :class:`~exp.runtime.gateway.images_contracts.ImageEditInput`; the
image-tool controls Stability AI's Bedrock models take (``seed``,
``negative_prompt``, ``creativity``, outpaint margins...) are typed and range
checked here and travel as ``edit_options``, forwarded only to a rung that
accepts each one. Unknown fields are refused by name.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Final

from pydantic import ValidationError

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.images_contracts import EditOptionValue, ImagesRequest
from exp.runtime.openai_protocol.errors import OpenAIProtocolError
from exp.runtime.openai_protocol.images_requests import DecodedImagesRequest
from exp.runtime.openai_protocol.validation_errors import validation_protocol_error

# The image parts an edit may carry; ``image[]`` is the SDK's list spelling of
# ``image`` and must name exactly one image (every served edit model takes one).
_FILE_PARTS: Final[frozenset[str]] = frozenset({"image", "image[]", "mask", "style_image"})

# OpenAI edit fields carried on the canonical request.
_OPENAI_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "model",
        "prompt",
        "n",
        "size",
        "quality",
        "background",
        "output_format",
        "output_compression",
        "response_format",
        "user",
    }
)

# OpenAI edit fields the buffered surface does not serve.
_UNSUPPORTED_FIELDS: Final[dict[str, str]] = {
    "stream": "Streaming image edits are not supported; omit stream.",
    "partial_images": "Streaming partial images are not supported; omit partial_images.",
    "input_fidelity": "input_fidelity is not supported by any served edit model; omit it.",
}

_STYLE_PRESETS: Final[frozenset[str]] = frozenset(
    {
        "3d-model",
        "analog-film",
        "anime",
        "cinematic",
        "comic-book",
        "digital-art",
        "enhance",
        "fantasy-art",
        "isometric",
        "line-art",
        "low-poly",
        "modeling-compound",
        "neon-punk",
        "origami",
        "photographic",
        "pixel-art",
        "tile-texture",
    }
)

_ASPECT_RATIOS: Final[frozenset[str]] = frozenset(
    {"16:9", "1:1", "21:9", "2:3", "3:2", "4:5", "5:4", "9:16", "9:21"}
)


def _text(limit: int) -> Callable[[str, str], EditOptionValue]:
    """Build a parser for one bounded free-text control.

    Args:
        limit: Maximum characters.

    Returns:
        The parser.
    """

    def parse(name: str, raw: str) -> EditOptionValue:
        """Accept non-empty text up to the limit.

        Args:
            name: Field name.
            raw: Field text.

        Returns:
            The text.

        Raises:
            ValueError: The text is empty or too long.
        """
        if not raw or len(raw) > limit:
            raise ValueError(f"{name} must be 1 to {limit} characters.")
        return raw

    return parse


def _integer(minimum: int, maximum: int) -> Callable[[str, str], EditOptionValue]:
    """Build a parser for one bounded integer control.

    Args:
        minimum: Smallest accepted value.
        maximum: Largest accepted value.

    Returns:
        The parser.
    """

    def parse(name: str, raw: str) -> EditOptionValue:
        """Parse a decimal integer inside the bounds.

        Args:
            name: Field name.
            raw: Field text.

        Returns:
            The integer.

        Raises:
            ValueError: Not an integer, or out of range.
        """
        try:
            value = int(raw.strip())
        except ValueError:
            raise ValueError(f"{name} must be an integer.") from None
        if not minimum <= value <= maximum:
            raise ValueError(f"{name} must be between {minimum} and {maximum}.")
        return value

    return parse


def _number(minimum: float, maximum: float) -> Callable[[str, str], EditOptionValue]:
    """Build a parser for one bounded decimal control.

    Args:
        minimum: Smallest accepted value.
        maximum: Largest accepted value.

    Returns:
        The parser.
    """

    def parse(name: str, raw: str) -> EditOptionValue:
        """Parse a finite decimal inside the bounds.

        Args:
            name: Field name.
            raw: Field text.

        Returns:
            The number.

        Raises:
            ValueError: Not a number, or out of range.
        """
        try:
            value = float(raw.strip())
        except ValueError:
            raise ValueError(f"{name} must be a number.") from None
        if not minimum <= value <= maximum:
            raise ValueError(f"{name} must be between {minimum} and {maximum}.")
        return value

    return parse


def _choice(allowed: frozenset[str]) -> Callable[[str, str], EditOptionValue]:
    """Build a parser for one enumerated control.

    Args:
        allowed: The accepted values.

    Returns:
        The parser.
    """

    def parse(name: str, raw: str) -> EditOptionValue:
        """Accept one listed value.

        Args:
            name: Field name.
            raw: Field text.

        Returns:
            The value.

        Raises:
            ValueError: The value is not listed.
        """
        if raw not in allowed:
            raise ValueError(f"{name} must be one of {', '.join(sorted(allowed))}.")
        return raw

    return parse


# Image-tool controls (Stability AI on Bedrock), with the providers' ranges.
EDIT_OPTION_PARSERS: Final[dict[str, Callable[[str, str], EditOptionValue]]] = {
    "negative_prompt": _text(10_000),
    "select_prompt": _text(10_000),
    "search_prompt": _text(10_000),
    "seed": _integer(0, 4_294_967_294),
    "style_preset": _choice(_STYLE_PRESETS),
    "aspect_ratio": _choice(_ASPECT_RATIOS),
    "grow_mask": _integer(0, 20),
    "left": _integer(0, 2_000),
    "right": _integer(0, 2_000),
    "up": _integer(0, 2_000),
    "down": _integer(0, 2_000),
    "creativity": _number(0.1, 1.0),
    "control_strength": _number(0.0, 1.0),
    "fidelity": _number(0.0, 1.0),
    "composition_fidelity": _number(0.0, 1.0),
    "style_strength": _number(0.0, 1.0),
    "change_strength": _number(0.1, 1.0),
    "strength": _number(0.0, 1.0),
}


def _invalid(param: str, message: str) -> OpenAIProtocolError:
    """Build one field-named 400.

    Args:
        param: The public field at fault.
        message: What is wrong and how to fix it.

    Returns:
        The protocol error.
    """
    return OpenAIProtocolError(
        status_code=400, code="invalid_request", message=message, param=param
    )


def _string_map(payload: JsonObject, key: str) -> dict[str, str]:
    """Read one ``{name: text}`` object from the parsed upload.

    Args:
        payload: The parsed upload.
        key: ``fields`` or ``files``.

    Returns:
        The mapping.

    Raises:
        OpenAIProtocolError: The member is not an object of strings.
    """
    value = payload.get(key, {})
    if not isinstance(value, dict) or not all(isinstance(item, str) for item in value.values()):
        raise _invalid(key, "The image edit upload is malformed.")
    return {str(name): str(item) for name, item in value.items()}


def decode_image_edits(payload: JsonObject) -> DecodedImagesRequest:
    """Decode one parsed ``/v1/images/edits`` upload.

    Args:
        payload: ``{"fields": {...}, "files": {...}}`` from the data plane.

    Returns:
        Public alias and canonical image-edit request.

    Raises:
        OpenAIProtocolError: A field is unknown, unsupported, or invalid, or
            the image part is missing.
    """
    fields = _string_map(payload, "fields")
    files = _string_map(payload, "files")
    for name in files:
        if name not in _FILE_PARTS:
            raise _invalid(name, f"Unknown image part {name!r}; send image, mask or style_image.")
    if "image" in files and "image[]" in files:
        raise _invalid("image", "Send the image to edit once, as image.")
    image = files.get("image") or files.get("image[]")
    if image is None:
        raise _invalid("image", "An image edit must carry an image part.")
    options: dict[str, EditOptionValue] = {}
    for name, raw in fields.items():
        if name in _UNSUPPORTED_FIELDS:
            raise OpenAIProtocolError(
                status_code=400,
                code="unsupported_parameter",
                message=_UNSUPPORTED_FIELDS[name],
                param=name,
            )
        if name in _OPENAI_FIELDS:
            continue
        parser = EDIT_OPTION_PARSERS.get(name)
        if parser is None:
            raise _invalid(name, f"Unknown image edit field {name!r}.")
        try:
            options[name] = parser(name, raw)
        except ValueError as exc:
            raise _invalid(name, str(exc)) from None
    model = fields.get("model", "")
    if not model or len(model) > 256:
        raise _invalid("model", "An image edit must name a model (1 to 256 characters).")
    request: JsonObject = {
        name: fields[name]
        for name in ("prompt", "size", "quality", "background", "output_format", "response_format")
        if name in fields
    }
    for name in ("n", "output_compression"):
        if name in fields:
            try:
                request[name] = int(fields[name])
            except ValueError:
                raise _invalid(name, f"{name} must be an integer.") from None
    if "user" in fields:
        request["user"] = fields["user"]
    edit: JsonObject = {"image": image}
    for part in ("mask", "style_image"):
        if part in files:
            edit[part] = files[part]
    request["edit"] = edit
    if options:
        request["edit_options"] = dict(options)
    try:
        canonical = ImagesRequest.model_validate(request)
    except ValidationError as exc:
        raise validation_protocol_error(exc) from exc
    return DecodedImagesRequest(alias=model, request=canonical)
