"""Tests for the first-party Platform login command."""

from __future__ import annotations

import io
from pathlib import Path

import pytest
import typer
from rich.console import Console

from exp.cli import auth
from exp.cli.providers.experiential_cloud import hosted_credential_binding
from exp.common.auth import ProviderAuthStore
from exp.common.models import (
    BillingSource,
    DiscoveredModel,
    GatewayDeploymentMetadata,
    ModelCapabilities,
    ModelCatalog,
    ModelRecord,
    ProviderConnection,
    load_model_catalog,
    write_model_catalog,
)
from exp.runtime.gateway.management import GatewayManagement
from exp.runtime.models.providers import ProviderEndpoint


class _AccountModelLister:
    """Return the model identities visible to the logged-in hosted account."""

    def __init__(self, api_key: str = "xpl_browser_key") -> None:
        """Expect one test-only credential while preserving the endpoint assertion."""
        self.api_key = api_key

    def list_models(self, endpoint: ProviderEndpoint) -> tuple[DiscoveredModel, ...]:
        """Assert login uses the hosted connection and return all account models."""
        assert endpoint == ProviderEndpoint(
            provider="openai-compatible",
            api_key=self.api_key,
            base_url="https://api.preview.experientiallabs.ai/v1",
        )
        return (
            DiscoveredModel(provider="openai-compatible", model="exp-chat"),
            DiscoveredModel(provider="openai-compatible", model="exp-reasoning"),
        )


def test_login_persists_platform_key_in_the_shared_cloud_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The top-level login stores the browser key where provider setup reads it."""
    transcript = io.StringIO()
    console = Console(file=transcript, force_terminal=True, no_color=True)
    store = ProviderAuthStore(tmp_path / "auth.json")
    environment = {"EXP_GATEWAY_URL": "https://api.preview.experientiallabs.ai/v1"}

    monkeypatch.setattr(
        auth,
        "hosted_platform_login",
        lambda _connection, **_kwargs: "xpl_browser_key",
    )

    root = tmp_path / ".exp"
    auth.run_login(
        console=console,
        environment=environment,
        store=store,
        root=root,
        lister=_AccountModelLister(),
    )

    assert (
        store.get(
            "experiential-cloud",
            binding=hosted_credential_binding(environment),
        )
        == "xpl_browser_key"
    )
    catalog = load_model_catalog(root / "models.toml")
    assert catalog.connections["experiential-cloud"].provider == "openai-compatible"
    assert {record.model for record in catalog.models.values()} == {
        "exp-chat",
        "exp-reasoning",
    }
    assert {record.billing_source.value for record in catalog.models.values()} == {"host_managed"}
    assert "xpl_browser_key" not in transcript.getvalue()
    assert "Synced Experiential Cloud:" in transcript.getvalue()
    assert "models." in transcript.getvalue()
    assert "Logged in to Experiential Cloud." in transcript.getvalue()


def test_login_rejects_endpoint_replacement_for_an_active_gateway(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A preview-origin login cannot rebind a connection used by an active deployment."""
    root = tmp_path / ".exp"
    old_environment = {"EXP_GATEWAY_URL": "https://api.experientiallabs.ai/v1"}
    old_connection = ProviderConnection(
        name="experiential-cloud",
        provider="openai-compatible",
        api_key_env="EXPLABS_API_KEY",
        base_url=old_environment["EXP_GATEWAY_URL"],
    )
    write_model_catalog(
        root / "models.toml",
        ModelCatalog(
            connections={"experiential-cloud": old_connection.catalog_config()},
            models={
                "exp-chat": ModelRecord(
                    connection="experiential-cloud",
                    model="exp-chat",
                    billing_source=BillingSource.HOST_MANAGED,
                    capabilities=ModelCapabilities(),
                    gateway=GatewayDeploymentMetadata(exact_model_id="exp-chat"),
                )
            },
        ),
    )
    store = ProviderAuthStore(tmp_path / "auth.json")
    store.put(
        "experiential-cloud",
        "xpl_old_key",
        binding=hosted_credential_binding(old_environment),
    )
    new_environment = {"EXP_GATEWAY_URL": "https://api.preview.experientiallabs.ai/v1"}
    monkeypatch.setattr(auth, "hosted_platform_login", lambda _connection, **_kwargs: "xpl_new_key")

    with pytest.raises(typer.BadParameter, match="active gateway deployments"):
        auth.run_login(
            console=Console(file=io.StringIO(), no_color=True),
            environment=new_environment,
            store=store,
            root=root,
            lister=_AccountModelLister("xpl_new_key"),
        )

    assert (
        store.get(
            "experiential-cloud",
            binding=hosted_credential_binding(old_environment),
        )
        == "xpl_old_key"
    )


def test_login_rejects_endpoint_replacement_for_a_sqlite_owned_gateway_connection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A gateway-owned provider remains protected before any catalog model is deployed."""
    root = tmp_path / ".exp"
    old_environment = {"EXP_GATEWAY_URL": "https://api.experientiallabs.ai/v1"}
    old_connection = ProviderConnection(
        name="experiential-cloud",
        provider="openai-compatible",
        api_key_env="EXPLABS_API_KEY",
        base_url=old_environment["EXP_GATEWAY_URL"],
    )
    manager = GatewayManagement(root)
    manager.initialize()
    manager.upsert_provider_connection(
        connection_id=old_connection.name,
        config=old_connection.catalog_config(),
    )
    store = ProviderAuthStore(tmp_path / "auth.json")
    store.put(
        "experiential-cloud",
        "xpl_old_key",
        binding=hosted_credential_binding(old_environment),
    )
    new_environment = {"EXP_GATEWAY_URL": "https://api.preview.experientiallabs.ai/v1"}
    monkeypatch.setattr(auth, "hosted_platform_login", lambda _connection, **_kwargs: "xpl_new_key")

    with pytest.raises(typer.BadParameter, match="active gateway authority"):
        auth.run_login(
            console=Console(file=io.StringIO(), no_color=True),
            environment=new_environment,
            store=store,
            root=root,
            lister=_AccountModelLister("xpl_new_key"),
        )

    assert (
        store.get(
            "experiential-cloud",
            binding=hosted_credential_binding(old_environment),
        )
        == "xpl_old_key"
    )
