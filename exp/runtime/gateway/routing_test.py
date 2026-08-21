"""Tests for catalog-backed public alias metadata lookup."""

from __future__ import annotations

from datetime import UTC, datetime

from exp.common.models.catalog import (
    GatewayDeploymentMetadata,
    GatewayEquivalenceCertification,
    GatewayTokenPrices,
)
from exp.common.models.gateway_catalog import (
    ExactModelDeployment,
    ExactModelPool,
    NormalizedGatewayCatalog,
)
from exp.common.models.model import ModelCapabilities
from exp.runtime.gateway.routing import CatalogRouteResolver

_REVISION = "revision-one"


def _deployment(
    *,
    deployment_id: str,
    source_alias: str,
    exact_model_id: str = "exact-one",
) -> ExactModelDeployment:
    """Build one priced completion deployment for lookup tests.

    Args:
        deployment_id: Catalog deployment identifier.
        source_alias: Source alias recorded on the deployment.
        exact_model_id: Exact logical model identity shared with its pool.

    Returns:
        A secret-free deployment with declared tools, output limit, and prices.
    """
    return ExactModelDeployment(
        deployment_id=deployment_id,
        source_alias=source_alias,
        exact_model_id=exact_model_id,
        connection="hosted",
        provider="openai-compatible",
        provider_model="hosted-model",
        connection_sha256="b" * 64,
        capabilities_sha256="c" * 64,
        capabilities=ModelCapabilities(
            supports_tools=True,
            supports_structured_output=True,
            maximum_output_tokens=8_192,
        ),
        gateway=GatewayDeploymentMetadata(
            prices=GatewayTokenPrices(
                input_micro_usd_per_million_tokens=900_000,
                output_micro_usd_per_million_tokens=900_000,
            )
        ),
    )


def _resolver(
    deployments: tuple[ExactModelDeployment, ...],
    pools: tuple[ExactModelPool, ...],
) -> tuple[CatalogRouteResolver, str]:
    """Index one catalog snapshot and return the resolver plus its digest.

    Args:
        deployments: Deployments stored in the snapshot.
        pools: Exact-model pools stored in the snapshot.

    Returns:
        The resolver and the catalog digest used as the revision key.
    """
    catalog = NormalizedGatewayCatalog(deployments=deployments, pools=pools)
    digest = catalog.identity_sha256()
    return CatalogRouteResolver({(_REVISION, digest): catalog}), digest


def test_published_metadata_uses_the_deployment_named_by_the_public_alias() -> None:
    """A direct alias copies fields from the deployment that shares its name."""
    deployment = _deployment(deployment_id="coding", source_alias="coding")
    resolver, digest = _resolver(
        (deployment,),
        (
            ExactModelPool(
                pool_id="coding",
                exact_model_id="exact-one",
                deployment_ids=("coding",),
            ),
        ),
    )

    metadata = resolver.published_metadata(
        alias="coding",
        revision_id=_REVISION,
        catalog_sha256=digest,
    )

    assert metadata is not None
    assert metadata.supports_completions is True
    assert metadata.supports_tools is True
    assert metadata.maximum_output_tokens == 8_192
    assert metadata.input_micro_usd_per_million_tokens == 900_000
    assert metadata.context_window_tokens is None
    assert metadata.cached_input_micro_usd_per_million_tokens is None


def test_published_metadata_accepts_a_singleton_pool_named_by_the_public_alias() -> None:
    """A public alias may name the singleton pool rather than the deployment ID."""
    deployment = _deployment(deployment_id="deployment-one", source_alias="source-one")
    resolver, digest = _resolver(
        (deployment,),
        (
            ExactModelPool(
                pool_id="coding",
                exact_model_id="exact-one",
                deployment_ids=("deployment-one",),
            ),
        ),
    )

    metadata = resolver.published_metadata(
        alias="coding",
        revision_id=_REVISION,
        catalog_sha256=digest,
    )

    assert metadata is not None
    assert metadata.output_micro_usd_per_million_tokens == 900_000


def test_published_metadata_stays_closed_for_multi_deployment_pools() -> None:
    """A pool with more than one route does not pick a deployment to advertise."""
    first = _deployment(deployment_id="one", source_alias="one")
    second = _deployment(deployment_id="two", source_alias="two")
    resolver, digest = _resolver(
        (first, second),
        (
            ExactModelPool(
                pool_id="coding",
                exact_model_id="exact-one",
                deployment_ids=("one", "two"),
                equivalence=GatewayEquivalenceCertification(
                    certification_id="certification-one",
                    provenance="operator comparison for listing fail-closed lookup",
                    evidence_sha256="d" * 64,
                    certified_at=datetime(2026, 8, 18, tzinfo=UTC),
                ),
            ),
        ),
    )

    assert (
        resolver.published_metadata(
            alias="coding",
            revision_id=_REVISION,
            catalog_sha256=digest,
        )
        is None
    )


def test_published_metadata_stays_closed_when_the_alias_is_not_in_the_catalog() -> None:
    """Project aliases and other unmatched names publish no extra fields."""
    deployment = _deployment(deployment_id="source-one", source_alias="source-one")
    resolver, digest = _resolver(
        (deployment,),
        (
            ExactModelPool(
                pool_id="pool-one",
                exact_model_id="exact-one",
                deployment_ids=("source-one",),
            ),
        ),
    )

    assert (
        resolver.published_metadata(
            alias="public-model",
            revision_id=_REVISION,
            catalog_sha256=digest,
        )
        is None
    )
