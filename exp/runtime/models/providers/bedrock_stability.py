"""Stability AI image models on Amazon Bedrock: the ``InvokeModel`` image wire.

Bedrock serves Stability's three generation models on demand
(``stability.sd3-5-large-v1:0``, ``stability.stable-image-core-v1:1``,
``stability.stable-image-ultra-v1:1``) and its thirteen Image Services tools
(upscale, inpaint, outpaint, erase, remove background, search-and-recolor,
search-and-replace, control sketch and structure, style guide, style transfer)
through the cross-region ``us.stability.*-v1:0`` inference profiles. Every
model takes one JSON body on ``POST /model/{modelId}/invoke`` and answers
``{"images": [base64], "seeds": [...], "finish_reasons": [null | reason]}``
with exactly one image; a non-null finish reason is a content-filter refusal
(``"Filter reason: prompt"``) or an ``"Inference error"``.

This module maps the gateway's OpenAI-shaped image requests onto those bodies:
generation models on ``/v1/images/generations`` (``size`` becomes the nearest
``aspect_ratio``), and every model that takes an input image on
``/v1/images/edits`` (the multipart ``image`` part, plus ``mask`` and, for
style transfer, ``style_image``). A control the model's wire cannot carry is
refused with the parameter named, never dropped. Request and response fields
follow the AWS Bedrock user guide (Stable Diffusion 3.5 Large and Stability AI
Image Services pages, read 2026-10-23).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final, Literal

from exp.common.core.artifacts import JsonObject

StabilityOperation = Literal["generate", "edit"]


@dataclass(frozen=True)
class StabilityModel:
    """One Stability model's wire contract on Bedrock.

    Attributes:
        key: Version- and region-free model key (``stable-image-inpaint``).
        generates: Whether the model answers a text-only generation request.
        edits: Whether the model takes an input image (``/v1/images/edits``).
        prompt_required: Whether an edit request must carry a prompt.
        image_field: The body field the edit's input image travels in.
        options: Optional edit controls the model's wire accepts.
        aspect_ratio: Whether a generation carries ``aspect_ratio``.
    """

    key: str
    generates: bool
    edits: bool
    prompt_required: bool
    image_field: str
    options: frozenset[str]
    aspect_ratio: bool = False


_STYLE = frozenset({"style_preset"})
_GENERATIVE = frozenset({"negative_prompt", "seed"})
_MASKED = frozenset({"grow_mask"})

STABILITY_MODELS: Final[dict[str, StabilityModel]] = {
    model.key: model
    for model in (
        # Generation models. SD 3.5 Large and Ultra also run image-to-image
        # on the edits surface (the body's ``image`` plus ``strength``).
        StabilityModel(
            key="sd3-5-large",
            generates=True,
            edits=True,
            prompt_required=True,
            image_field="image",
            options=_GENERATIVE | frozenset({"strength"}),
            aspect_ratio=True,
        ),
        StabilityModel(
            key="stable-image-ultra",
            generates=True,
            edits=True,
            prompt_required=True,
            image_field="image",
            options=_GENERATIVE | frozenset({"strength"}),
            aspect_ratio=True,
        ),
        StabilityModel(
            key="stable-image-core",
            generates=True,
            edits=False,
            prompt_required=True,
            image_field="image",
            options=frozenset(),
            aspect_ratio=True,
        ),
        # Upscale.
        StabilityModel(
            key="stable-conservative-upscale",
            generates=False,
            edits=True,
            prompt_required=True,
            image_field="image",
            options=_GENERATIVE | frozenset({"creativity"}),
        ),
        StabilityModel(
            key="stable-creative-upscale",
            generates=False,
            edits=True,
            prompt_required=True,
            image_field="image",
            options=_GENERATIVE | _STYLE | frozenset({"creativity"}),
        ),
        StabilityModel(
            key="stable-fast-upscale",
            generates=False,
            edits=True,
            prompt_required=False,
            image_field="image",
            options=frozenset(),
        ),
        # Edit.
        StabilityModel(
            key="stable-image-inpaint",
            generates=False,
            edits=True,
            prompt_required=True,
            image_field="image",
            options=_GENERATIVE | _STYLE | _MASKED,
        ),
        StabilityModel(
            key="stable-outpaint",
            generates=False,
            edits=True,
            prompt_required=False,
            image_field="image",
            options=frozenset({"seed", "creativity", "left", "right", "up", "down"}) | _STYLE,
        ),
        StabilityModel(
            key="stable-image-search-recolor",
            generates=False,
            edits=True,
            prompt_required=True,
            image_field="image",
            options=_GENERATIVE | _STYLE | _MASKED | frozenset({"select_prompt"}),
        ),
        StabilityModel(
            key="stable-image-search-replace",
            generates=False,
            edits=True,
            prompt_required=True,
            image_field="image",
            options=_GENERATIVE | _STYLE | _MASKED | frozenset({"search_prompt"}),
        ),
        StabilityModel(
            key="stable-image-erase-object",
            generates=False,
            edits=True,
            prompt_required=False,
            image_field="image",
            options=frozenset({"seed"}) | _MASKED,
        ),
        StabilityModel(
            key="stable-image-remove-background",
            generates=False,
            edits=True,
            prompt_required=False,
            image_field="image",
            options=frozenset(),
        ),
        # Control.
        StabilityModel(
            key="stable-image-control-sketch",
            generates=False,
            edits=True,
            prompt_required=True,
            image_field="image",
            options=_GENERATIVE | _STYLE | frozenset({"control_strength"}),
        ),
        StabilityModel(
            key="stable-image-control-structure",
            generates=False,
            edits=True,
            prompt_required=True,
            image_field="image",
            options=_GENERATIVE | _STYLE | frozenset({"control_strength"}),
        ),
        StabilityModel(
            key="stable-image-style-guide",
            generates=False,
            edits=True,
            prompt_required=True,
            image_field="image",
            options=_GENERATIVE | _STYLE | frozenset({"fidelity", "aspect_ratio"}),
        ),
        StabilityModel(
            key="stable-style-transfer",
            generates=False,
            edits=True,
            prompt_required=False,
            image_field="init_image",
            options=_GENERATIVE
            | frozenset({"composition_fidelity", "style_strength", "change_strength"}),
        ),
    )
}

# Edit models that accept a ``mask`` part (else the mask is refused by name).
MASK_MODELS: Final[frozenset[str]] = frozenset(
    {"stable-image-inpaint", "stable-image-erase-object"}
)

# The models whose edit requires a second image (``style_image``).
STYLE_IMAGE_MODELS: Final[frozenset[str]] = frozenset({"stable-style-transfer"})

# Prompt-bearing edit options that, like the prompt, are free text.
REQUIRED_EDIT_OPTIONS: Final[dict[str, str]] = {
    "stable-image-search-recolor": "select_prompt",
    "stable-image-search-replace": "search_prompt",
}

# OpenAI ``size`` values onto Stability's aspect-ratio enum: the nearest ratio.
_SIZE_ASPECT_RATIOS: Final[dict[str, str]] = {
    "256x256": "1:1",
    "512x512": "1:1",
    "1024x1024": "1:1",
    "1536x1024": "3:2",
    "1024x1536": "2:3",
    "1792x1024": "16:9",
    "1024x1792": "9:16",
}

_MODEL_ID = re.compile(r"^(?:(?:us|eu|apac|global)\.)?stability\.(?P<key>[a-z0-9-]+?)-v\d+:\d+$")


def stability_model(model_id: str) -> StabilityModel | None:
    """The Stability wire contract for one Bedrock model id, or None.

    Args:
        model_id: Bedrock model or inference-profile id
            (``stability.sd3-5-large-v1:0``, ``us.stability.stable-outpaint-v1:0``).

    Returns:
        The model's contract, or None when the id is not a known Stability model.
    """
    match = _MODEL_ID.match(model_id)
    if match is None:
        return None
    return STABILITY_MODELS.get(match.group("key"))


class StabilityParameterError(ValueError):
    """A request control the Stability rung's wire cannot carry.

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


def stability_aspect_ratio(size: str | None) -> str | None:
    """Map an OpenAI ``size`` onto Stability's nearest ``aspect_ratio``.

    Args:
        size: The request's ``size`` (``None`` and ``"auto"`` keep the default).

    Returns:
        The aspect ratio, or None for the provider default (1:1).
    """
    if size is None or size == "auto":
        return None
    return _SIZE_ASPECT_RATIOS[size]


def stability_body(
    model: StabilityModel,
    *,
    operation: StabilityOperation,
    prompt: str,
    size: str | None,
    output_format: str | None,
    image: str | None = None,
    mask: str | None = None,
    style_image: str | None = None,
    options: dict[str, str | int | float] | None = None,
) -> JsonObject:
    """Build one Bedrock ``InvokeModel`` body for a Stability model.

    Args:
        model: The rung's Stability contract.
        operation: ``generate`` (text-to-image) or ``edit`` (input image).
        prompt: The caller's prompt (empty when an edit carried none).
        size: The request's OpenAI ``size`` (generation only).
        output_format: ``png`` / ``jpeg`` / ``webp``, or None for PNG.
        image: Base64 input image (edit only).
        mask: Base64 mask (inpaint and erase only).
        style_image: Base64 style image (style transfer only).
        options: Validated Stability edit controls (``seed``, ``creativity``...).

    Returns:
        The exact JSON body.

    Raises:
        StabilityParameterError: The request carries a control or input this
            model's wire cannot carry, or omits one it requires.
    """
    options = dict(options or {})
    body: JsonObject = {}
    if operation == "generate":
        if not model.generates:
            raise StabilityParameterError(
                "model", "This model edits an input image; send it to /v1/images/edits."
            )
        body["prompt"] = prompt
        aspect_ratio = stability_aspect_ratio(size)
        if aspect_ratio is not None:
            body["aspect_ratio"] = aspect_ratio
    else:
        if not model.edits:
            raise StabilityParameterError(
                "model", "This model only generates images; send it to /v1/images/generations."
            )
        if image is None:
            raise StabilityParameterError("image", "An image edit must carry an image part.")
        if model.prompt_required and not prompt:
            raise StabilityParameterError("prompt", "This model requires a prompt.")
        if prompt:
            body["prompt"] = prompt
        body[model.image_field] = image
        if model.key in ("sd3-5-large", "stable-image-ultra"):
            # Image-to-image needs a strength; 0.7 keeps the input's layout.
            if model.key == "sd3-5-large":
                body["mode"] = "image-to-image"
            body["strength"] = options.pop("strength", 0.7)
        if mask is not None:
            if model.key not in MASK_MODELS:
                raise StabilityParameterError("mask", "This model takes no mask.")
            body["mask"] = mask
        if model.key in STYLE_IMAGE_MODELS:
            if style_image is None:
                raise StabilityParameterError(
                    "style_image", "Style transfer requires a style_image part."
                )
            body["style_image"] = style_image
        elif style_image is not None:
            raise StabilityParameterError("style_image", "This model takes no style_image.")
        required = REQUIRED_EDIT_OPTIONS.get(model.key)
        if required is not None and not options.get(required):
            raise StabilityParameterError(required, f"This model requires {required}.")
        if model.key == "stable-outpaint" and not any(
            options.get(side) for side in ("left", "right", "up", "down")
        ):
            raise StabilityParameterError(
                "left", "Outpainting needs at least one of left, right, up or down above 0."
            )
        for name, value in options.items():
            if name not in model.options:
                raise StabilityParameterError(name, f"This model does not accept {name}.")
            body[name] = value
    if output_format is not None:
        body["output_format"] = output_format
    return body
