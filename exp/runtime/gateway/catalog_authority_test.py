"""Behavior tests for authored gateway catalog mutation atomicity."""

from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import pytest
from pydantic import ValidationError

from exp.common.models import (
    BillingSource,
    ConnectionConfig,
    GatewayDeploymentCapabilities,
    GatewayDeploymentMetadata,
    GatewayEquivalenceCertification,
    GatewayPoolRecord,
    GatewayTokenPrices,
    ModelCapabilities,
    ModelCatalog,
    ModelRecord,
    ModelRoles,
    normalize_gateway_catalog,
    write_model_catalog,
)
from exp.runtime.gateway.catalog_authority import (
    _write_catalog_snapshot,
    authored_snapshot_path,
    upsert_singleton_deployment,
)
from exp.runtime.gateway.management import GatewayManagement


def _pooled_catalog(root: Path) -> Path:
    """Author a catalog whose two deployments form one declared pool named 'wf'."""
    connection = ConnectionConfig(
        provider="openai-compatible",
        base_url="http://127.0.0.1:9/v1",
        api_key_env="TEST_PROVIDER_KEY",
    )
    record = ModelRecord(
        connection="test",
        model="test-model",
        billing_source=BillingSource.CUSTOMER_MANAGED,
        capabilities=ModelCapabilities(),
        gateway=GatewayDeploymentMetadata(exact_model_id="exact-one"),
    )
    catalog = ModelCatalog(
        connections={"test": connection},
        models={"primary": record, "secondary": record.model_copy()},
        gateway_pools={
            "wf": GatewayPoolRecord(
                exact_model_id="exact-one",
                deployment_aliases=("primary", "secondary"),
                equivalence=GatewayEquivalenceCertification(
                    certification_id="cert-one",
                    provenance="operator-verified equivalence for tests",
                    evidence_sha256=sha256(b"evidence").hexdigest(),
                    certified_at=datetime(2026, 1, 1, tzinfo=UTC),
                ),
            )
        },
        roles=ModelRoles(),
    )
    path = root / "models.toml"
    write_model_catalog(path, catalog)
    return path


def test_rejected_singleton_deployment_leaves_the_authored_catalog_unchanged(
    tmp_path: Path,
) -> None:
    """A deployment alias colliding with a declared pool fails without persisting."""
    path = _pooled_catalog(tmp_path)
    before = path.read_bytes()

    with pytest.raises(ValidationError, match="pool IDs must be unique"):
        upsert_singleton_deployment(
            tmp_path,
            deployment_alias="wf",
            connection_name="test",
            provider_model="other-model",
            exact_model_id="exact-two",
            revision=None,
            capabilities=ModelCapabilities(),
            gateway_capabilities=GatewayDeploymentCapabilities(supports_streaming=True),
            prices=GatewayTokenPrices(),
            pricing_source=None,
            replace=False,
        )

    assert path.read_bytes() == before


def test_write_catalog_snapshot_repairs_a_stale_content_addressed_file(tmp_path: Path) -> None:
    """C1 root-cause fix: a normalized snapshot whose on-disk bytes no longer hash
    to their own content-addressed name (a stale or partially written file) is
    rewritten instead of blindly trusted, so a self-inconsistent snapshot can
    never be pinned by this write path."""
    catalog = ModelCatalog(
        connections={"openai": ConnectionConfig(provider="openai")},
        models={
            "m": ModelRecord(
                connection="openai",
                model="gpt-test",
                billing_source=BillingSource.CUSTOMER_MANAGED,
            )
        },
    )
    normalized = normalize_gateway_catalog(catalog)

    snapshot = _write_catalog_snapshot(tmp_path, catalog, normalized)
    assert sha256(snapshot.read_bytes()).hexdigest() == normalized.identity_sha256()

    # A stale/corrupt file already sitting at the content-addressed path.
    snapshot.write_bytes(b'{"stale": true}')
    repaired = _write_catalog_snapshot(tmp_path, catalog, normalized)

    assert repaired == snapshot
    # The bytes on disk once again hash to their own content-addressed name.
    assert sha256(snapshot.read_bytes()).hexdigest() == normalized.identity_sha256()


def test_valid_singleton_deployment_still_persists_after_validation(tmp_path: Path) -> None:
    """A non-colliding deployment is validated first and then written durably."""
    path = _pooled_catalog(tmp_path)

    normalized, snapshot, changed = upsert_singleton_deployment(
        tmp_path,
        deployment_alias="extra",
        connection_name="test",
        provider_model="other-model",
        exact_model_id="exact-two",
        revision=None,
        capabilities=ModelCapabilities(),
        gateway_capabilities=GatewayDeploymentCapabilities(supports_streaming=True),
        prices=GatewayTokenPrices(),
        pricing_source=None,
        replace=False,
    )

    assert changed
    assert snapshot.exists()
    assert "extra" in {pool.pool_id for pool in normalized.pools}
    assert b"extra" in path.read_bytes()


def test_same_mode_bedrock_credential_rotation_refreshes_authored_snapshot(
    tmp_path: Path,
) -> None:
    """A locator-only rotation rebinds without changing normalized route identity."""
    manager = GatewayManagement(tmp_path)
    manager.initialize()

    def connection(secret_env: str, access_env: str) -> ConnectionConfig:
        return ConnectionConfig(
            provider="bedrock",
            region="us-west-2",
            api_key_env=secret_env,
            aws_access_key_id_env=access_env,
            bedrock_auth_mode="access_key_pair",
        )

    old = connection("OLD_SECRET", "OLD_ACCESS")
    manager.upsert_provider_connection(connection_id="bedrock-main", config=old)

    def upsert(config: ConnectionConfig, *, replace: bool) -> Path:
        _normalized, snapshot, _changed = upsert_singleton_deployment(
            tmp_path,
            deployment_alias="bedrock",
            connection_name="bedrock-main",
            provider_model="amazon.nova-lite-v1:0",
            exact_model_id="nova-lite",
            revision=None,
            capabilities=ModelCapabilities(supports_completions=True),
            gateway_capabilities=GatewayDeploymentCapabilities(supports_streaming=True),
            prices=GatewayTokenPrices(),
            pricing_source=None,
            replace=replace,
            serving_connections={"bedrock-main": config},
        )
        return snapshot

    first_snapshot = upsert(old, replace=False)

    new = connection("NEW_SECRET", "NEW_ACCESS")
    manager.upsert_provider_connection(
        connection_id="bedrock-main",
        config=new,
        replace=True,
    )
    second_snapshot = upsert(new, replace=True)

    assert second_snapshot == first_snapshot
    authored = ModelCatalog.model_validate_json(
        authored_snapshot_path(second_snapshot).read_bytes()
    )
    assert authored.connections["bedrock-main"] == new
    assert manager.provider_bindings(authored)
