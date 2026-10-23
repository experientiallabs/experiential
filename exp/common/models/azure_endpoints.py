"""Azure resource-endpoint surface inference shared by catalog setup and the Azure client."""

from __future__ import annotations

from typing import Literal
from urllib.parse import urlsplit

AzureApiSurface = Literal["openai_deployments", "model_inference"]
"""Azure wire surface a connection speaks: classic deployments or Foundry model inference."""

_FOUNDRY_HOST_SUFFIXES = (".services.ai.azure.com", ".inference.ai.azure.com")
_AZURE_OPENAI_HOST_SUFFIX = ".openai.azure.com"
_MODEL_INFERENCE_ROOT_SUFFIXES = ("/models", "/openai/v1")


def infer_azure_api_surface(endpoint: str) -> AzureApiSurface | None:
    """Infer the Azure wire surface one resource endpoint serves.

    Azure AI Foundry resources (``*.services.ai.azure.com``) serve the model-inference surface,
    which carries provider-specific sampling fields such as ``top_k``. Azure OpenAI resources
    (``*.openai.azure.com``) serve only the deployment surface.

    Args:
        endpoint: Azure resource endpoint from a connection.

    Returns:
        The surface the host is known to serve, or ``None`` for an unrecognized host such as a
        private endpoint or a local recording proxy.
    """
    host = urlsplit(endpoint).hostname
    if host is None:
        return None
    host = host.lower()
    if host.endswith(_AZURE_OPENAI_HOST_SUFFIX):
        return "openai_deployments"
    if any(host.endswith(suffix) for suffix in _FOUNDRY_HOST_SUFFIXES):
        return "model_inference"
    return None


def strip_model_inference_root(value: str) -> str:
    """Remove the route suffix one Azure model-inference endpoint spelling carries.

    The model-inference surface serves ``/models`` directly off the resource, so the bare resource,
    its terminal ``/models`` form, and the Azure OpenAI ``/openai/v1`` root all name one resource.

    Args:
        value: Endpoint or endpoint path, with or without a trailing slash.

    Returns:
        The value reduced to the resource itself.
    """
    trimmed = value.rstrip("/")
    for suffix in _MODEL_INFERENCE_ROOT_SUFFIXES:
        if trimmed.lower().endswith(suffix):
            return trimmed[: -len(suffix)].rstrip("/")
    return trimmed
