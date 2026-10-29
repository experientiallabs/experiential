"""Tests for the Azure image wires (OpenAI Image API, MAI images, BFL FLUX.2)."""

from __future__ import annotations

import pytest

from exp.runtime.gateway.images_contracts import ImagesRequest
from exp.runtime.models.providers.azure_images import (
    AzureImagesDispatch,
    AzureImagesUnsupportedError,
    azure_images_dispatch,
    azure_images_wire,
)

_DEPLOYMENTS_URL = (
    "https://res.openai.azure.com/openai/deployments/my-deploy/images/generations"
    "?api-version=2025-04-01-preview"
)
_HEADERS = {"api-key": "secret-key", "Content-Type": "application/json"}


def _dispatch(
    request: ImagesRequest,
    *,
    exact: str,
    deployment: str = "my-deploy",
    url: str = _DEPLOYMENTS_URL,
) -> AzureImagesDispatch:
    return azure_images_dispatch(
        images_url=url,
        headers=_HEADERS,
        deployment=deployment,
        exact_model_id=exact,
        request=request,
        openai_body={"model": deployment, "prompt": request.prompt, "n": request.n},
    )


@pytest.mark.parametrize(
    ("ids", "wire"),
    [
        (("mai-image-2.6", "anything"), "mai"),
        (("catalog-x", "MAI-Image-2.5-Flash"), "mai"),
        (("flux.2-pro", "d"), "bfl"),
        (("x", "FLUX.2-flex"), "bfl"),
        (("flux-1.1-pro", "d"), "openai_images"),
        (("flux.1-kontext-pro", "d"), "openai_images"),
        (("gpt-image-1.5", "d"), "openai_images"),
    ],
)
def test_wire_follows_the_model_family(ids: tuple[str, str], wire: str) -> None:
    assert azure_images_wire(*ids) == wire


def test_openai_image_api_keeps_the_deployments_url_and_body() -> None:
    dispatch = _dispatch(ImagesRequest(prompt="a fox", n=2, quality="high"), exact="gpt-image-1.5")
    assert dispatch.url == _DEPLOYMENTS_URL
    assert dispatch.headers == _HEADERS
    assert dispatch.body == {"model": "my-deploy", "prompt": "a fox", "n": 2}


def test_model_inference_surface_is_moved_to_the_deployments_images_path() -> None:
    dispatch = _dispatch(
        ImagesRequest(prompt="a fox"),
        exact="flux-1.1-pro",
        url="https://res.services.ai.azure.com/models/images/generations?api-version=2024-05-01-preview",
    )
    assert dispatch.url == (
        "https://res.services.ai.azure.com/openai/deployments/my-deploy/images/generations"
        "?api-version=2025-04-01-preview"
    )


def test_mai_maps_size_to_width_and_height_on_the_mai_route() -> None:
    dispatch = _dispatch(
        ImagesRequest(prompt="a fox", size="1536x1024"), exact="mai-image-2.6", deployment="mai-dep"
    )
    assert dispatch.wire == "mai"
    assert dispatch.url == "https://res.openai.azure.com/mai/v1/images/generations"
    assert dispatch.headers == _HEADERS
    assert dispatch.body == {"model": "mai-dep", "prompt": "a fox", "width": 1536, "height": 1024}


def test_mai_omits_dimensions_for_the_default_size() -> None:
    dispatch = _dispatch(ImagesRequest(prompt="a fox", size="auto"), exact="mai-image-2.5")
    assert dispatch.body == {"model": "my-deploy", "prompt": "a fox"}


@pytest.mark.parametrize(
    ("request_fields", "exact", "param"),
    [
        ({"size": "512x512"}, "mai-image-2.6", "size"),
        ({"size": "1536x1024"}, "mai-image-2.5", "size"),  # 1.57 MP > the 2.5 ceiling.
        ({"n": 2}, "mai-image-2.6", "n"),
        ({"quality": "high"}, "mai-image-2.6", "quality"),
        ({"style": "vivid"}, "flux.2-pro", "style"),
        ({"output_format": "jpeg"}, "mai-image-2.6", "output_format"),
        ({"output_format": "webp"}, "flux.2-pro", "output_format"),
        ({"n": 3}, "flux.2-flex", "n"),
    ],
)
def test_unrepresentable_requests_are_refused_not_changed(
    request_fields: dict[str, object], exact: str, param: str
) -> None:
    with pytest.raises(AzureImagesUnsupportedError) as raised:
        _dispatch(ImagesRequest.model_validate({"prompt": "a fox", **request_fields}), exact=exact)
    assert raised.value.param == param


def test_bfl_uses_the_provider_route_and_bearer_key() -> None:
    dispatch = _dispatch(
        ImagesRequest(prompt="a fox", size="1024x1024", output_format="jpeg"),
        exact="flux.2-flex",
        deployment="flex-dep",
    )
    assert dispatch.wire == "bfl"
    assert dispatch.url == (
        "https://res.openai.azure.com/providers/blackforestlabs/v1/flux-2-flex?api-version=preview"
    )
    assert dispatch.headers == {
        "Authorization": "Bearer secret-key",
        "Content-Type": "application/json",
    }
    assert dispatch.body == {
        "model": "flex-dep",
        "prompt": "a fox",
        "n": 1,
        "width": 1024,
        "height": 1024,
        "output_format": "jpeg",
    }


def test_bfl_pro_path_and_default_size() -> None:
    dispatch = _dispatch(ImagesRequest(prompt="a fox"), exact="flux.2-pro")
    assert dispatch.url.endswith("/providers/blackforestlabs/v1/flux-2-pro?api-version=preview")
    assert dispatch.body == {"model": "my-deploy", "prompt": "a fox", "n": 1}
