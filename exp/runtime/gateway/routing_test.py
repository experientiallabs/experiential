"""Tests for catalog-backed public alias metadata lookup."""

from __future__ import annotations

import unittest.mock
from datetime import UTC, datetime

import pytest

from exp.common.models.catalog import (
    GatewayDeploymentMetadata,
    GatewayEquivalenceCertification,
    GatewayTokenPrices,
)
from exp.common.models.gateway_catalog import (
    SNAPSHOT_SCHEMA_VERSION,
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
    input_price: int = 900_000,
) -> ExactModelDeployment:
    """Build one priced completion deployment for lookup tests.

    Args:
        deployment_id: Catalog deployment identifier.
        source_alias: Source alias recorded on the deployment.
        exact_model_id: Exact logical model identity shared with its pool.
        input_price: Configured input micro-USD per million tokens.

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
                input_micro_usd_per_million_tokens=input_price,
                output_micro_usd_per_million_tokens=900_000,
            )
        ),
    )


def _catalog(
    deployments: tuple[ExactModelDeployment, ...],
    pools: tuple[ExactModelPool, ...],
) -> tuple[NormalizedGatewayCatalog, str]:
    """Build one catalog snapshot and its digest.

    Args:
        deployments: Deployments stored in the snapshot.
        pools: Exact-model pools stored in the snapshot.

    Returns:
        The catalog and the digest used as the revision key.
    """
    catalog = NormalizedGatewayCatalog(deployments=deployments, pools=pools)
    return catalog, catalog.identity_sha256()


def _resolver(
    catalog: NormalizedGatewayCatalog,
    digest: str,
    listing_pools: dict[tuple[str, str, str], str] | None = None,
) -> CatalogRouteResolver:
    """Index one catalog with optional authoritative listing targets.

    Args:
        catalog: Snapshot served by the resolver.
        digest: Frozen catalog digest.
        listing_pools: Direct-target pool IDs keyed by granted alias authority.

    Returns:
        A resolver ready for metadata lookup.
    """
    return CatalogRouteResolver(
        {(_REVISION, digest): catalog},
        listing_pools=listing_pools,
    )


def test_published_metadata_uses_the_revision_direct_pool_not_the_public_name() -> None:
    """A differently named public alias still publishes its frozen direct pool."""
    deployment = _deployment(deployment_id="deployment-one", source_alias="source-one")
    catalog, digest = _catalog(
        (deployment,),
        (
            ExactModelPool(
                pool_id="pool-one",
                exact_model_id="exact-one",
                deployment_ids=("deployment-one",),
            ),
        ),
    )

    metadata = _resolver(
        catalog,
        digest,
        {("public-model", _REVISION, digest): "pool-one"},
    ).published_metadata(
        alias="public-model",
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


def test_published_metadata_ignores_a_deployment_that_only_shares_the_public_name() -> None:
    """A name collision with another deployment does not override the frozen target."""
    decoy = _deployment(
        deployment_id="public-model",
        source_alias="public-model",
        exact_model_id="exact-decoy",
        input_price=1,
    )
    target = _deployment(deployment_id="deployment-one", source_alias="source-one")
    catalog, digest = _catalog(
        (decoy, target),
        (
            ExactModelPool(
                pool_id="decoy-pool",
                exact_model_id="exact-decoy",
                deployment_ids=("public-model",),
            ),
            ExactModelPool(
                pool_id="pool-one",
                exact_model_id="exact-one",
                deployment_ids=("deployment-one",),
            ),
        ),
    )

    metadata = _resolver(
        catalog,
        digest,
        {("public-model", _REVISION, digest): "pool-one"},
    ).published_metadata(
        alias="public-model",
        revision_id=_REVISION,
        catalog_sha256=digest,
    )

    assert metadata is not None
    assert metadata.input_micro_usd_per_million_tokens == 900_000


def test_published_metadata_stays_closed_for_multi_deployment_pools() -> None:
    """A pool with more than one route does not pick a deployment to advertise."""
    first = _deployment(deployment_id="one", source_alias="one")
    second = _deployment(deployment_id="two", source_alias="two")
    catalog, digest = _catalog(
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
        _resolver(
            catalog,
            digest,
            {("coding", _REVISION, digest): "coding"},
        ).published_metadata(
            alias="coding",
            revision_id=_REVISION,
            catalog_sha256=digest,
        )
        is None
    )


def test_published_metadata_stays_closed_when_the_revision_has_no_direct_pool() -> None:
    """Project aliases and other unmapped names stay identity-only, even on name hits."""
    deployment = _deployment(deployment_id="public-model", source_alias="public-model")
    catalog, digest = _catalog(
        (deployment,),
        (
            ExactModelPool(
                pool_id="pool-one",
                exact_model_id="exact-one",
                deployment_ids=("public-model",),
            ),
        ),
    )

    assert (
        _resolver(catalog, digest).published_metadata(
            alias="public-model",
            revision_id=_REVISION,
            catalog_sha256=digest,
        )
        is None
    )


def _single_pool_catalog() -> NormalizedGatewayCatalog:
    """One valid single-deployment catalog for the roll-safety index guards."""
    deployment = _deployment(deployment_id="deployment-one", source_alias="source-one")
    catalog, _digest = _catalog(
        (deployment,),
        (
            ExactModelPool(
                pool_id="pool-one",
                exact_model_id="exact-one",
                deployment_ids=("deployment-one",),
            ),
        ),
    )
    return catalog


def test_index_rejects_a_same_version_catalog_under_the_wrong_digest() -> None:
    """A same-version catalog that does not reproduce its key digest is
    corruption and still fails closed when the resolver indexes it."""
    with pytest.raises(ValueError, match="wrong digest"):
        CatalogRouteResolver({(_REVISION, "b" * 64): _single_pool_catalog()})


def test_index_hashes_a_shared_catalog_object_once_and_shares_its_view() -> None:
    """Keys sharing one immutable catalog verify with ONE identity computation.

    A repoint mints hundreds of alias keys against the same snapshot object, and
    per-key hashing re-hashed the same multi-megabyte production document 732
    times (~35 s of a ~51 s catalog state build). The digest check must stay
    per KEY (a shared object pinned under a second, wrong digest still fails
    closed), while the expensive hash and the indexed view are per OBJECT.
    """
    catalog = _single_pool_catalog()
    digest = catalog.identity_sha256()
    calls = 0
    original = type(catalog).identity_sha256

    def counting(self: NormalizedGatewayCatalog) -> str:
        nonlocal calls
        calls += 1
        return original(self)

    keys = [(f"alias-revision-{index:03d}", digest) for index in range(50)]
    with unittest.mock.patch.object(type(catalog), "identity_sha256", counting):
        resolver = CatalogRouteResolver(dict.fromkeys(keys, catalog))
    assert calls == 1
    views = resolver._catalogs  # noqa: SLF001 -- pinning the shared-view invariant
    assert all(views[key] is views[keys[0]] for key in keys)

    with pytest.raises(ValueError, match="wrong digest"):
        CatalogRouteResolver({keys[0]: catalog, ("alias-revision-bad", "b" * 64): catalog})


def test_index_serves_a_cross_version_catalog_under_its_pinned_digest() -> None:
    """Roll-safety guard: a snapshot authored by a newer engine build (a higher
    schema_version, a digest this build cannot recompute) is indexed under its
    pinned digest and resolves rather than hard-failing route admission, so a
    rolling deploy never turns a route lookup into a fleet-wide error."""
    foreign = _single_pool_catalog().model_copy(
        update={"schema_version": SNAPSHOT_SCHEMA_VERSION + 1}
    )
    pinned = "b" * 64
    resolver = CatalogRouteResolver(
        {(_REVISION, pinned): foreign},
        listing_pools={("public-model", _REVISION, pinned): "pool-one"},
    )

    metadata = resolver.published_metadata(
        alias="public-model",
        revision_id=_REVISION,
        catalog_sha256=pinned,
    )

    assert metadata is not None
