"""Tests for project-backed gateway alias compatibility migrations."""

from pathlib import Path
from typing import cast

import pytest

from wmo.common.models import (
    BillingSource,
    CandidateTokenPrice,
    ConnectionConfig,
    GatewayDeploymentCapabilities,
    GatewayDeploymentMetadata,
    ModelCatalog,
    ModelRecord,
    PricingSnapshot,
    load_model_catalog,
    write_model_catalog,
)
from wmo.runtime.gateway.project_activation import ProjectActivation, ProjectActivationError
from wmo.runtime.gateway.project_alias import (
    _migrate_legacy_project_gateway_metadata,
    prepare_project_gateway_alias,
)
from wmo.runtime.models import RuntimeModelCatalog
from wmo.runtime.router.runtime_test import _fixture


class _MismatchedRepository:
    """Return one activation without enforcing the requested lookup authority."""

    def __init__(self, activation: ProjectActivation) -> None:
        """Store one deliberately mismatched activation."""
        self.activation = activation

    def load(
        self,
        project_ref: str,
        activation_ref: str | None,
        *,
        runtime_catalog: RuntimeModelCatalog,
    ) -> ProjectActivation:
        """Return the stored activation while ignoring requested references."""
        del project_ref, activation_ref, runtime_catalog
        return self.activation


def _activation(*, project_ref: str, activation_ref: str) -> ProjectActivation:
    """Build one internally consistent activation with selected external references."""
    policy, manifest, bank, _snapshots, _client = _fixture()
    policy = policy.model_copy(update={"policy_id": activation_ref})
    pricing = PricingSnapshot(
        schema_version=1,
        created_at=policy.created_at,
        code_revision="test",
        pricing_snapshot_id=policy.pricing_snapshot_id,
        candidate_prices=tuple(
            CandidateTokenPrice(
                candidate_alias=alias,
                input_usd_per_million_tokens=1,
                output_usd_per_million_tokens=2,
            )
            for alias in bank.candidate_aliases
        ),
    )
    return ProjectActivation(
        project_ref=project_ref,
        activation_ref=activation_ref,
        policy=policy,
        bank_manifest=manifest,
        bank=bank,
        pricing=pricing,
        pricing_sha256=policy.pricing_snapshot_sha256,
    )


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


@pytest.mark.parametrize("mismatch", ["project", "activation"])
def test_alias_preparation_rejects_wrong_authority_before_persisting(
    tmp_path: Path,
    mismatch: str,
) -> None:
    """Wrong repository authority cannot initialize or mutate gateway state."""
    project_ref = "other-project" if mismatch == "project" else "project-one"
    activation_ref = "other-activation" if mismatch == "activation" else "activation-one"
    repository = _MismatchedRepository(
        _activation(project_ref=project_ref, activation_ref=activation_ref)
    )

    with pytest.raises(ProjectActivationError, match=f"returned {mismatch} reference"):
        prepare_project_gateway_alias(
            "project-one",
            tmp_path,
            policy_id="activation-one",
            project_repository=repository,
            runtime_catalog=cast(RuntimeModelCatalog, object()),
        )

    assert not (tmp_path / "gateway").exists()
    assert not (tmp_path / "models.toml").exists()
