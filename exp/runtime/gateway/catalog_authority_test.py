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
    write_model_catalog,
)
from exp.runtime.gateway.catalog_authority import upsert_singleton_deployment


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
