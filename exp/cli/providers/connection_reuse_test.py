"""Interactive connection reuse keeps distinct Azure wire surfaces separate."""

import pytest

from exp.cli.providers.connection_reuse import reused_connection
from exp.common.models import ProviderConnection
from exp.common.models.azure_endpoints import AzureApiSurface


@pytest.mark.parametrize("configured", ["openai_deployments", "responses"])
@pytest.mark.parametrize("requested", ["openai_deployments", "responses"])
def test_azure_reuses_only_the_same_wire_surface(
    configured: AzureApiSurface, requested: AzureApiSurface
) -> None:
    """Choosing Responses cannot silently reuse a Chat connection or vice versa."""
    existing = ProviderConnection(
        name="azure-existing",
        provider="azure",
        api_key_env="AZURE_FIXTURE_KEY",
        base_url="https://resource.openai.azure.com",
        api_version="v1",
        azure_api_surface=configured,
    )
    result = reused_connection(
        (existing,),
        provider="azure",
        api_key_env=existing.api_key_env,
        base_url=existing.base_url,
        api_version=existing.api_version,
        azure_api_surface=requested,
        region=None,
    )
    assert result == (existing if configured == requested else None)
