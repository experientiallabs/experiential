"""Tests for Azure resource-endpoint surface inference."""

from __future__ import annotations

from exp.common.models.azure_endpoints import infer_azure_api_surface, strip_model_inference_root


def test_azure_surface_inference_follows_the_resource_host() -> None:
    """Foundry hosts serve model inference, Azure OpenAI hosts serve deployments."""
    assert infer_azure_api_surface("https://resource.openai.azure.com") == "openai_deployments"
    assert (
        infer_azure_api_surface("https://Resource.Services.AI.Azure.com/models")
        == "model_inference"
    )
    assert infer_azure_api_surface("https://resource.inference.ai.azure.com") == "model_inference"
    assert infer_azure_api_surface("https://gateway.example.test/tenant-a") is None


def test_model_inference_root_spellings_name_one_resource() -> None:
    """The bare resource, its /models form, and the /openai/v1 root reduce to one resource."""
    resource = "https://resource.services.ai.azure.com"
    assert strip_model_inference_root(resource) == resource
    assert strip_model_inference_root(f"{resource}/models/") == resource
    assert strip_model_inference_root(f"{resource}/OpenAI/v1") == resource
