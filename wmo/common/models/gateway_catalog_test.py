"""Tests for conservative gateway deployment normalization."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from wmo.common.core.artifacts import ArtifactInput, sha256_json
from wmo.common.models.catalog import (
    BillingSource,
    ConnectionConfig,
    GatewayDeploymentMetadata,
    GatewayEquivalenceCertification,
    GatewayPoolRecord,
    ModelCatalog,
    ModelRecord,
    SFTModelProvenance,
)
from wmo.common.models.gateway_catalog import (
    ExactModelDeployment,
    ExactModelPool,
    normalize_gateway_catalog,
)
from wmo.common.models.model import ModelCapabilities, ModelSnapshot

_DIGEST = "a" * 64


def test_singleton_identity_includes_connection_model_revision_and_full_capabilities() -> None:
    """Endpoint and capability variants cannot collide during legacy singleton migration."""
    catalog = ModelCatalog(
        connections={
            "first": ConnectionConfig(
                provider="openai-compatible",
                base_url="https://first.example.test/v1",
            ),
            "second": ConnectionConfig(
                provider="openai-compatible",
                base_url="https://second.example.test/v1",
            ),
        },
        models={
            "first-low": ModelRecord(
                connection="first",
                model="same-name",
                billing_source=BillingSource.CUSTOMER_MANAGED,
                revision="2026-08-01",
                capabilities=ModelCapabilities(reasoning_effort="low"),
            ),
            "first-high": ModelRecord(
                connection="first",
                model="same-name",
                billing_source=BillingSource.CUSTOMER_MANAGED,
                revision="2026-08-01",
                capabilities=ModelCapabilities(reasoning_effort="high"),
            ),
            "second-low": ModelRecord(
                connection="second",
                model="same-name",
                billing_source=BillingSource.CUSTOMER_MANAGED,
                revision="2026-08-01",
                capabilities=ModelCapabilities(reasoning_effort="low"),
            ),
        },
    )

    normalized = normalize_gateway_catalog(catalog)
    by_alias = {item.source_alias: item for item in normalized.deployments}

    assert len({item.exact_model_id for item in normalized.deployments}) == 3
    assert by_alias["first-low"].connection_sha256 != by_alias["second-low"].connection_sha256
    assert by_alias["first-low"].capabilities_sha256 != by_alias["first-high"].capabilities_sha256
    assert tuple(pool.pool_id for pool in normalized.pools) == (
        "first-high",
        "first-low",
        "second-low",
    )


def test_identical_alias_records_remain_separate_singleton_pools() -> None:
    """Duplicate metadata may share exact identity without being silently grouped for failover."""
    catalog = ModelCatalog(
        connections={"openai": ConnectionConfig(provider="openai")},
        models={
            "coding-a": ModelRecord(
                connection="openai",
                model="gpt-coding",
                billing_source=BillingSource.CUSTOMER_MANAGED,
            ),
            "coding-b": ModelRecord(
                connection="openai",
                model="gpt-coding",
                billing_source=BillingSource.CUSTOMER_MANAGED,
            ),
        },
    )

    normalized = normalize_gateway_catalog(catalog)

    assert normalized.deployments[0].exact_model_id == normalized.deployments[1].exact_model_id
    assert normalized.pools[0].pool_id != normalized.pools[1].pool_id
    assert all(len(pool.deployment_ids) == 1 for pool in normalized.pools)


def test_billing_source_changes_catalog_identity_without_changing_exact_model() -> None:
    """Billing ownership is frozen in deployment identity but not model equivalence."""
    connection = ConnectionConfig(provider="openai")
    customer = ModelCatalog(
        connections={"openai": connection},
        models={
            "coding": ModelRecord(
                connection="openai",
                model="gpt-coding",
                billing_source=BillingSource.CUSTOMER_MANAGED,
            )
        },
    )
    host = customer.model_copy(
        update={
            "models": {
                "coding": customer.models["coding"].model_copy(
                    update={"billing_source": BillingSource.HOST_MANAGED}
                )
            }
        }
    )

    customer_normalized = normalize_gateway_catalog(customer)
    host_normalized = normalize_gateway_catalog(host)

    assert customer_normalized.deployments[0].exact_model_id == (
        host_normalized.deployments[0].exact_model_id
    )
    assert customer_normalized.deployments[0].billing_source == BillingSource.CUSTOMER_MANAGED
    assert host_normalized.deployments[0].billing_source == BillingSource.HOST_MANAGED
    assert customer_normalized.identity_sha256() != host_normalized.identity_sha256()


def test_legacy_normalized_deployment_defaults_to_customer_managed_billing() -> None:
    """A normalized payload written before billing attribution decodes conservatively."""
    deployment = ExactModelDeployment.model_validate(
        {
            "deployment_id": "coding",
            "source_alias": "coding",
            "exact_model_id": "exact-coding",
            "connection": "openai",
            "provider": "openai",
            "provider_model": "gpt-coding",
            "connection_sha256": "a" * 64,
            "capabilities_sha256": "b" * 64,
        }
    )

    assert deployment.billing_source == BillingSource.CUSTOMER_MANAGED


def test_tinker_and_sft_records_are_not_gateway_deployments() -> None:
    """Training handles retain provenance without being treated as operational model routes."""
    sampling_handle = "tinker://sampling/run-1"
    provenance = SFTModelProvenance(
        source_dataset=ArtifactInput(artifact_id="dataset", sha256=_DIGEST),
        optimization_config=ArtifactInput(artifact_id="config", sha256="b" * 64),
        training_spec_sha256="c" * 64,
        run_id="run-1",
        model_id="model-1",
        model_sha256="d" * 64,
        result_id="result-1",
        result_sha256="e" * 64,
        base_model=ModelSnapshot(
            provider="tinker",
            model_id="base",
            billing_source=BillingSource.CUSTOMER_MANAGED,
            capabilities_sha256="f" * 64,
            connection_sha256="0" * 64,
        ),
        connection_config_sha256="1" * 64,
        sampling_handle_sha256=sha256_json({"sampling_handle": sampling_handle}),
    )
    catalog = ModelCatalog(
        connections={
            "openai": ConnectionConfig(provider="openai"),
            "tinker": ConnectionConfig(provider="tinker"),
        },
        models={
            "regular": ModelRecord(
                connection="openai",
                model="gpt-coding",
                billing_source=BillingSource.CUSTOMER_MANAGED,
            ),
            "training": ModelRecord(
                connection="openai",
                model=sampling_handle,
                billing_source=BillingSource.CUSTOMER_MANAGED,
                sft_provenance=provenance,
            ),
            "tinker-handle": ModelRecord(
                connection="tinker",
                model="base",
                billing_source=BillingSource.CUSTOMER_MANAGED,
            ),
        },
    )

    normalized = normalize_gateway_catalog(catalog)

    assert tuple(item.source_alias for item in normalized.deployments) == ("regular",)
    assert normalized.identity_sha256() == normalize_gateway_catalog(catalog).identity_sha256()


def test_operator_certified_pool_preserves_explicit_deployment_order_and_provenance() -> None:
    """Only one authored certification groups deployments and fixes waterfall priority."""
    certification = GatewayEquivalenceCertification(
        certification_id="certification-one",
        provenance="operator comparison run 2026-08-18",
        evidence_sha256=_DIGEST,
        certified_at=datetime(2026, 8, 18, tzinfo=UTC),
    )
    catalog = ModelCatalog(
        connections={
            "anthropic": ConnectionConfig(provider="anthropic"),
            "openai": ConnectionConfig(provider="openai"),
        },
        models={
            "route-a": ModelRecord(
                connection="openai",
                model="provider-a",
                billing_source=BillingSource.CUSTOMER_MANAGED,
                gateway=GatewayDeploymentMetadata(exact_model_id="exact-certified"),
            ),
            "route-b": ModelRecord(
                connection="anthropic",
                model="provider-b",
                billing_source=BillingSource.CUSTOMER_MANAGED,
                gateway=GatewayDeploymentMetadata(exact_model_id="exact-certified"),
            ),
        },
        gateway_pools={
            "certified-pool": GatewayPoolRecord(
                exact_model_id="exact-certified",
                deployment_aliases=("route-b", "route-a"),
                equivalence=certification,
            )
        },
    )

    normalized = normalize_gateway_catalog(catalog)

    assert normalized.pools == (
        ExactModelPool(
            pool_id="certified-pool",
            exact_model_id="exact-certified",
            deployment_ids=("route-b", "route-a"),
            equivalence=certification,
        ),
    )
    assert normalized.identity_sha256() == normalize_gateway_catalog(catalog).identity_sha256()


def test_equivalence_catalog_rejects_implicit_false_or_ambiguous_grouping() -> None:
    """Missing exact declarations, training handles, and repeated membership fail closed."""
    certification = GatewayEquivalenceCertification(
        certification_id="certification-one",
        provenance="operator comparison run",
        evidence_sha256=_DIGEST,
        certified_at=datetime(2026, 8, 18, tzinfo=UTC),
    )
    with pytest.raises(ValidationError, match="declare exact model identity"):
        ModelCatalog(
            connections={"openai": ConnectionConfig(provider="openai")},
            models={
                "route-a": ModelRecord(
                    connection="openai",
                    model="provider-a",
                    billing_source=BillingSource.CUSTOMER_MANAGED,
                ),
                "route-b": ModelRecord(
                    connection="openai",
                    model="provider-b",
                    billing_source=BillingSource.CUSTOMER_MANAGED,
                ),
            },
            gateway_pools={
                "pool": GatewayPoolRecord(
                    exact_model_id="exact-certified",
                    deployment_aliases=("route-a", "route-b"),
                    equivalence=certification,
                )
            },
        )
    with pytest.raises(ValidationError, match="more than one pool"):
        ModelCatalog(
            connections={"openai": ConnectionConfig(provider="openai")},
            models={
                alias: ModelRecord(
                    connection="openai",
                    model=alias,
                    billing_source=BillingSource.CUSTOMER_MANAGED,
                    gateway=GatewayDeploymentMetadata(exact_model_id="exact-certified"),
                )
                for alias in ("route-a", "route-b", "route-c")
            },
            gateway_pools={
                "pool-a": GatewayPoolRecord(
                    exact_model_id="exact-certified",
                    deployment_aliases=("route-a", "route-b"),
                    equivalence=certification,
                ),
                "pool-b": GatewayPoolRecord(
                    exact_model_id="exact-certified",
                    deployment_aliases=("route-b", "route-c"),
                    equivalence=certification,
                ),
            },
        )


def test_normalized_multi_deployment_pool_requires_operator_certification() -> None:
    """The runtime snapshot cannot construct implicit multi-route equivalence."""
    with pytest.raises(ValidationError, match="operator equivalence certification"):
        ExactModelPool(
            pool_id="unsafe",
            exact_model_id="exact-one",
            deployment_ids=("route-a", "route-b"),
        )
