"""Azure image-generation wires: the OpenAI Image API, MAI images, and BFL FLUX.2.

An Azure connection (Azure OpenAI or a Foundry resource) serves image models on
three different wires, chosen by the model family rather than by the
connection (verified against Microsoft Learn 2026-10-23):

- **OpenAI Image API** (``gpt-image-*``, ``FLUX-1.1-pro``, ``FLUX.1-Kontext-pro``):
  ``{resource}/openai/deployments/{deployment}/images/generations?api-version=...``
  (or the ``/openai/v1`` root with the deployment in ``model``), ``api-key``
  header, the OpenAI body unchanged.
- **MAI image API** (``MAI-Image-2.5``, ``-2.5-Flash``, ``-2.5-Pro``, ``-2.6``,
  ``-2.6-Flash``): ``{resource}/mai/v1/images/generations``, ``api-key``
  header, body ``{model, prompt, width, height}``; one PNG per call returned as
  ``data[0].b64_json`` with no token usage. Width and height are each at least
  768 px; the pixel product is at most 1,048,576 on the 2.5 models and
  2,359,296 on the 2.6 models.
- **BFL provider API** (``FLUX.2-pro``, ``FLUX.2-flex``):
  ``{resource}/providers/blackforestlabs/v1/{flux-2-pro|flux-2-flex}?api-version=preview``,
  ``Authorization: Bearer <api key>``, body ``{model, prompt, width, height,
  n, output_format}``; one image returned as ``data[0].b64_json`` with no
  token usage.

MAI and BFL answers carry no usage, so their rungs must bill per image (an
``image`` unit card); the admission excludes a rung without one. Both return
exactly one image, so a request for ``n > 1`` is excluded from those rungs.
Neither wire has OpenAI's ``quality``, ``style``, ``background``,
``moderation`` or ``output_compression``; a request carrying one is excluded
rather than silently changed, and a ``url`` response is answered with
``b64_json`` (these wires host nothing to link to), as on xAI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
from urllib.parse import urlsplit, urlunsplit

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.images_contracts import ImagesRequest

AzureImagesWire = Literal["openai_images", "mai", "bfl"]

OPENAI_IMAGES_API_VERSION = "2025-04-01-preview"
BFL_API_VERSION = "preview"
MAI_MINIMUM_SIDE = 768
MAI_25_MAXIMUM_PIXELS = 1_048_576
MAI_26_MAXIMUM_PIXELS = 2_359_296

# OpenAI controls neither the MAI nor the BFL wire can honour.
_UNREPRESENTED_FIELDS = ("quality", "style", "background", "moderation", "output_compression")


class AzureImagesUnsupportedError(ValueError):
    """The request carries something this Azure image wire cannot preserve."""

    def __init__(self, param: str, message: str) -> None:
        """Record the offending public parameter and a caller-facing message."""
        super().__init__(message)
        self.param = param
        self.message = message


@dataclass(frozen=True)
class AzureImagesDispatch:
    """One Azure rung's resolved image endpoint, headers, and body."""

    url: str
    headers: dict[str, str]
    body: JsonObject
    wire: AzureImagesWire


def _normalized(model_id: str) -> str:
    return model_id.strip().lower().replace("_", "-")


def azure_images_wire(*model_ids: str) -> AzureImagesWire:
    """Classify an Azure image model by its catalog id or deployment name.

    Args:
        model_ids: The catalog exact model id and the Azure deployment name; the
            first that names a MAI or FLUX.2 model decides.

    Returns:
        ``"mai"``, ``"bfl"``, or ``"openai_images"`` (every other image model).
    """
    for model_id in model_ids:
        name = _normalized(model_id)
        if name.startswith("mai-image"):
            return "mai"
        if name.startswith(("flux.2", "flux-2")):
            return "bfl"
    return "openai_images"


def _bfl_model_path(*model_ids: str) -> str:
    """The BFL provider path segment (``flux-2-pro`` or ``flux-2-flex``)."""
    for model_id in model_ids:
        name = _normalized(model_id)
        if name.startswith(("flux.2", "flux-2")):
            return "flux-2-flex" if "flex" in name else "flux-2-pro"
    return "flux-2-pro"  # pragma: no cover - only reached for a bfl classification.


def _mai_maximum_pixels(*model_ids: str) -> int:
    """The pixel-product ceiling for one MAI model (2.6 models allow more)."""
    for model_id in model_ids:
        if _normalized(model_id).startswith("mai-image-2.6"):
            return MAI_26_MAXIMUM_PIXELS
    return MAI_25_MAXIMUM_PIXELS


def _resource_root(url: str) -> str:
    """The ``scheme://host`` root of one Azure endpoint (the wires hang off it)."""
    parsed = urlsplit(url)
    return urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))


def _dimensions(request: ImagesRequest) -> tuple[int, int] | None:
    """The requested width and height, or None for the provider default."""
    if request.size is None or request.size == "auto":
        return None
    width, height = request.size.split("x")
    return int(width), int(height)


def _require_single_image(request: ImagesRequest) -> None:
    if request.n != 1:
        raise AzureImagesUnsupportedError(
            "n", "This model returns one image per request; send n=1 and repeat the request."
        )


def _require_representable(request: ImagesRequest) -> None:
    for field in _UNREPRESENTED_FIELDS:
        if getattr(request, field) is not None:
            raise AzureImagesUnsupportedError(
                field, f"This model does not support {field!r}; remove it and resend the request."
            )


def azure_images_dispatch(
    *,
    images_url: str,
    headers: dict[str, str],
    deployment: str,
    exact_model_id: str,
    request: ImagesRequest,
    openai_body: JsonObject,
) -> AzureImagesDispatch:
    """Resolve the endpoint, headers, and body one Azure image rung dispatches.

    Args:
        images_url: The connection's OpenAI-wire images endpoint (its root
            identifies the Azure resource).
        headers: The connection's static headers (``api-key``).
        deployment: The Azure deployment name the request is served by.
        exact_model_id: The catalog's exact model id for the lane.
        request: The canonical image request.
        openai_body: The OpenAI Image API body already built for this rung.

    Returns:
        The dispatch for this rung.

    Raises:
        AzureImagesUnsupportedError: The request cannot be served faithfully on
            this rung's wire (``n > 1``, a size the model cannot render, or an
            OpenAI-only control).
    """
    wire = azure_images_wire(exact_model_id, deployment)
    root = _resource_root(images_url)
    if wire == "openai_images":
        # The Foundry model-inference surface (``{resource}/models``) has no
        # images route; the Image API lives on the deployments path of the same
        # resource. The deployments and ``/openai/v1`` spellings already do.
        url = images_url
        if "/openai/" not in urlsplit(images_url).path:
            url = (
                f"{root}/openai/deployments/{deployment}/images/generations"
                f"?api-version={OPENAI_IMAGES_API_VERSION}"
            )
        return AzureImagesDispatch(url=url, headers=dict(headers), body=openai_body, wire=wire)
    _require_single_image(request)
    _require_representable(request)
    dimensions = _dimensions(request)
    if wire == "mai":
        if request.output_format not in (None, "png"):
            raise AzureImagesUnsupportedError(
                "output_format", "This model returns PNG only; remove output_format or send png."
            )
        body: JsonObject = {"model": deployment, "prompt": request.prompt}
        if dimensions is not None:
            width, height = dimensions
            maximum = _mai_maximum_pixels(exact_model_id, deployment)
            if min(width, height) < MAI_MINIMUM_SIDE or width * height > maximum:
                raise AzureImagesUnsupportedError(
                    "size",
                    f"This model renders sizes with each side at least {MAI_MINIMUM_SIDE} px "
                    f"and at most {maximum:,} pixels; choose another size.",
                )
            body["width"] = width
            body["height"] = height
        return AzureImagesDispatch(
            url=f"{root}/mai/v1/images/generations", headers=dict(headers), body=body, wire=wire
        )
    if request.output_format == "webp":
        raise AzureImagesUnsupportedError(
            "output_format", "This model returns PNG or JPEG; send png or jpeg."
        )
    body = {"model": deployment, "prompt": request.prompt, "n": 1}
    if dimensions is not None:
        body["width"], body["height"] = dimensions
    if request.output_format is not None:
        body["output_format"] = request.output_format
    api_key = headers.get("api-key", "")
    bfl_headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    path = _bfl_model_path(exact_model_id, deployment)
    return AzureImagesDispatch(
        url=f"{root}/providers/blackforestlabs/v1/{path}?api-version={BFL_API_VERSION}",
        headers=bfl_headers,
        body=body,
        wire=wire,
    )
