"""Behavior tests for local gateway composition and loopback-only lifecycle routes."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from wmo.cli.gateway.catalog import upsert_connection, upsert_singleton_deployment
from wmo.common.models import (
    ConnectionConfig,
    GatewayDeploymentCapabilities,
    GatewayTokenPrices,
    ModelCapabilities,
)
from wmo.runtime.gateway.lifecycle import (
    GatewayLifecycleError,
    gateway_instance_lock,
    load_local_gateway,
)
from wmo.runtime.gateway.management import GatewayManagement


def test_local_gateway_preflights_real_state_and_serves_health_and_usage(
    tmp_path: Path,
) -> None:
    """Real SQLite state reaches readiness and content-free loopback surfaces."""
    manager, raw_key = _configured_gateway(tmp_path)

    runtime = load_local_gateway(
        tmp_path,
        graceful_timeout_seconds=1,
        environment={"TEST_PROVIDER_KEY": "provider-secret-canary"},
    )

    with TestClient(runtime.app) as client:
        assert client.get("/health/live").json() == {"status": "live"}
        assert client.get("/health/ready").json() == {"status": "ready"}
        models = client.get(
            "/v1/models",
            headers={"Authorization": f"Bearer {raw_key}"},
        )
        assert models.status_code == 200
        assert [item["id"] for item in models.json()["data"]] == ["coding"]
        usage = client.get("/usage.json")
        page = client.get("/usage")

    assert runtime.state.ready is False
    assert usage.status_code == 200
    assert usage.json()["schema_version"] == 1
    assert usage.json()["identities"][0]["identity_id"] == "default"
    assert page.status_code == 200
    assert "provider-secret-canary" not in page.text
    assert raw_key not in page.text
    assert manager.status().active_aliases == 1


def test_instance_lock_rejects_a_second_owner_for_the_same_root(tmp_path: Path) -> None:
    """A second local process cannot concurrently own one gateway database."""
    with gateway_instance_lock(tmp_path, port=8000):
        with pytest.raises(GatewayLifecycleError, match="already owns"):
            with gateway_instance_lock(tmp_path, port=9000):
                raise AssertionError("second lock unexpectedly acquired")


def test_readiness_requires_an_explicit_grant(tmp_path: Path) -> None:
    """Configured aliases remain unavailable until an identity is granted access."""
    manager = GatewayManagement(tmp_path)
    manager.initialize()
    manager.create_identity(identity_id="default", display_name="Default")

    with pytest.raises(GatewayLifecycleError, match="no granted active alias"):
        load_local_gateway(tmp_path, graceful_timeout_seconds=1, environment={})


def test_missing_secret_marks_only_its_direct_alias_unavailable(tmp_path: Path) -> None:
    """One absent provider secret does not block another complete granted alias."""
    manager, _raw_key = _configured_gateway(tmp_path)
    upsert_connection(
        tmp_path,
        name="missing-provider",
        connection=ConnectionConfig(
            provider="openai-compatible",
            base_url="http://127.0.0.1:9/v1",
            api_key_env="MISSING_PROVIDER_KEY",
        ),
        replace=False,
    )
    normalized, snapshot, _changed = upsert_singleton_deployment(
        tmp_path,
        deployment_alias="broken",
        connection_name="missing-provider",
        provider_model="missing-model",
        exact_model_id="missing-exact-model",
        revision=None,
        capabilities=ModelCapabilities(),
        gateway_capabilities=GatewayDeploymentCapabilities(supports_streaming=True),
        prices=GatewayTokenPrices(),
        pricing_source=None,
        replace=False,
    )
    manager.activate_direct_alias(
        alias_id="broken",
        alias_name="broken",
        revision_id="revision-broken",
        pool_id="broken",
        snapshot_ref=f"catalog-snapshots/{snapshot.name}",
        catalog_sha256=normalized.identity_sha256(),
    )
    manager.add_grant(identity_id="default", alias_id="broken")

    runtime = load_local_gateway(
        tmp_path,
        graceful_timeout_seconds=1,
        environment={"TEST_PROVIDER_KEY": "available"},
    )

    assert runtime.reconciled_expired_requests == 0


def _configured_gateway(root: Path) -> tuple[GatewayManagement, str]:
    """Create one explicit direct alias, identity, grant, and key in real SQLite."""
    manager = GatewayManagement(root)
    manager.initialize()
    upsert_connection(
        root,
        name="provider-main",
        connection=ConnectionConfig(
            provider="openai-compatible",
            base_url="http://127.0.0.1:9/v1",
            api_key_env="TEST_PROVIDER_KEY",
        ),
        replace=False,
    )
    normalized, snapshot, _changed = upsert_singleton_deployment(
        root,
        deployment_alias="coding",
        connection_name="provider-main",
        provider_model="provider-model-exact",
        exact_model_id="model-revision-exact",
        revision=None,
        capabilities=ModelCapabilities(),
        gateway_capabilities=GatewayDeploymentCapabilities(supports_streaming=True),
        prices=GatewayTokenPrices(),
        pricing_source=None,
        replace=False,
    )
    manager.activate_direct_alias(
        alias_id="coding",
        alias_name="coding",
        revision_id="revision-one",
        pool_id="coding",
        snapshot_ref=f"catalog-snapshots/{snapshot.name}",
        catalog_sha256=normalized.identity_sha256(),
    )
    manager.create_identity(identity_id="default", display_name="Default")
    manager.add_grant(identity_id="default", alias_id="coding")
    issued = manager.issue_key(identity_id="default", key_id="key-one")
    return manager, issued.raw_key
