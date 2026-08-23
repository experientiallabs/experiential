"""Tests for the first-party Platform login command."""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from rich.console import Console

from exp.cli import auth
from exp.cli.providers.experiential_cloud import hosted_credential_binding
from exp.common.auth import ProviderAuthStore
from exp.common.models import DiscoveredModel, load_model_catalog
from exp.runtime.models.providers import ProviderEndpoint


class _AccountModelLister:
    """Return the model identities visible to the logged-in hosted account."""

    def list_models(self, endpoint: ProviderEndpoint) -> tuple[DiscoveredModel, ...]:
        """Assert login uses the hosted connection and return all account models."""
        assert endpoint == ProviderEndpoint(
            provider="openai-compatible",
            api_key="xpl_browser_key",
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
