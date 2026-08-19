"""Tests for project-backed gateway alias compatibility migrations."""

from pathlib import Path

from wmo.common.models import (
    BillingSource,
    ConnectionConfig,
    GatewayDeploymentCapabilities,
    GatewayDeploymentMetadata,
    ModelCatalog,
    ModelRecord,
    load_model_catalog,
    write_model_catalog,
)
from wmo.runtime.gateway.project_alias import _migrate_legacy_project_gateway_metadata


def test_legacy_project_candidates_gain_only_shared_streaming_transport(tmp_path: Path) -> None:
    """Upgrade absent gateway metadata without overriding explicit declarations."""
    catalog = ModelCatalog(
        schema_version=2,
        connections={"provider": ConnectionConfig(provider="openai")},
        models={
            "legacy": ModelRecord(
                connection="provider",
                model="legacy-model",
                billing_source=BillingSource.CUSTOMER_MANAGED,
            ),
            "explicit": ModelRecord(
                connection="provider",
                model="explicit-model",
                billing_source=BillingSource.CUSTOMER_MANAGED,
                gateway=GatewayDeploymentMetadata(
                    capabilities=GatewayDeploymentCapabilities(supports_strict_tools=True)
                ),
            ),
        },
    )
    write_model_catalog(tmp_path / "models.toml", catalog)

    changed = _migrate_legacy_project_gateway_metadata(
        tmp_path,
        aliases=("legacy", "explicit"),
    )
    migrated = load_model_catalog(tmp_path / "models.toml")

    assert changed is True
    assert migrated.models["legacy"].gateway == GatewayDeploymentMetadata(
        capabilities=GatewayDeploymentCapabilities(supports_streaming=True)
    )
    assert migrated.models["explicit"].gateway == catalog.models["explicit"].gateway
    assert (
        _migrate_legacy_project_gateway_metadata(
            tmp_path,
            aliases=("legacy", "explicit"),
        )
        is False
    )
