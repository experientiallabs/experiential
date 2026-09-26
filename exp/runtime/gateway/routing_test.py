"""Tests for catalog-backed public alias metadata lookup."""

from __future__ import annotations

import asyncio
import json
import sqlite3
import unittest.mock
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Literal, cast

import pytest

from exp.common.models import load_model_catalog, normalize_gateway_catalog
from exp.common.models.catalog import GatewayDeploymentMetadata, GatewayTokenPrices
from exp.common.models.gateway_catalog import (
    SNAPSHOT_SCHEMA_VERSION,
    ExactModelDeployment,
    ExactModelPool,
    NormalizedGatewayCatalog,
)
from exp.common.models.gateway_chains_test import chain
from exp.common.models.gateway_pools import GatewayEquivalenceCertification
from exp.common.models.model import ModelCapabilities
from exp.runtime.gateway import model_plan
from exp.runtime.gateway.contracts import (
    AuthorizationSnapshot,
    DirectTarget,
    GatewayApiSurface,
    GatewayMessage,
    GatewayRequest,
    ProjectSelection,
    ProjectTarget,
)
from exp.runtime.gateway.ledger import SQLiteAttemptLedger
from exp.runtime.gateway.native_bridge import NativeBridgeError, NativeControlPlane
from exp.runtime.gateway.native_bridge_test import _chat_body, _configured_pool_gateway
from exp.runtime.gateway.native_components import NativeGatewayComponents
from exp.runtime.gateway.native_execution import select_route_deployments
from exp.runtime.gateway.routing import (
    CatalogRouteResolver,
    GatewayRoute,
    GatewayRoutingError,
    RouteResolver,
)
from exp.runtime.models import RuntimeModelCatalog

_REVISION = "revision-one"


def test_catalog_route_resolver_satisfies_the_route_resolver_protocol() -> None:
    """CatalogRouteResolver structurally satisfies the exported Protocol.

    The annotated binding makes ``ty`` verify the public resolution seam at
    type-check time, so a wrapper annotated ``RouteResolver`` catches a missing
    or drifted resolve_* method statically.
    """
    resolver: RouteResolver = CatalogRouteResolver({})
    assert resolver is not None
    assert "requires_model_chain_authority" in RouteResolver.__dict__


class _ForwardingRouteResolver:
    """Expose the required public resolver seam without forwarding arbitrary private methods."""

    def __init__(self, delegate: RouteResolver) -> None:
        """Keep one revision-aware delegate and an observable classification count."""
        self.delegate = delegate
        self.classifications = 0

    def requires_model_chain_authority(self, authorization: AuthorizationSnapshot) -> bool:
        """Delegate exact selected-root classification without guessing a permissive result."""
        self.classifications += 1
        return self.delegate.requires_model_chain_authority(authorization)

    def resolve_direct(self, authorization: AuthorizationSnapshot) -> GatewayRoute:
        """Delegate direct resolution after the serving authority gate."""
        return self.delegate.resolve_direct(authorization)

    def resolve_deployment_hint(
        self, authorization: AuthorizationSnapshot, deployment_id: str
    ) -> GatewayRoute:
        """Delegate authenticated deployment-hint resolution."""
        return self.delegate.resolve_deployment_hint(authorization, deployment_id)

    def resolve_project_blocking(
        self,
        *,
        authorization: AuthorizationSnapshot,
        request: GatewayRequest,
        episode_namespace: tuple[str, str, str, str],
    ) -> GatewayRoute:
        """Delegate project selection inside its exact frozen catalog."""
        return self.delegate.resolve_project_blocking(
            authorization=authorization, request=request, episode_namespace=episode_namespace
        )


@pytest.mark.parametrize("target_kind", ["direct", "project", "chain"])
@pytest.mark.parametrize("operation", ["claim_scope", "admit"])
def test_forwarding_resolver_preserves_actual_serving_authority(
    tmp_path: Path, target_kind: Literal["direct", "project", "chain"], operation: str
) -> None:
    """A conforming wrapper serves plain targets but cannot grant unproved chain execution."""
    manager, key = _configured_pool_gateway(tmp_path)
    store = manager.require_initialized()
    authored = load_model_catalog(tmp_path / "models.toml")
    normalized = normalize_gateway_catalog(authored)
    pool = normalized.pools[0]
    if target_kind == "chain":
        normalized = normalized.model_copy(
            update={
                "model_chains": (
                    chain(pool.exact_model_id, *pool.deployment_ids).model_copy(
                        update={"pool_id": pool.pool_id}
                    ),
                )
            }
        )
    digest = normalized.identity_sha256()
    target = (
        ProjectTarget(project_ref="project", activation_ref="activation", catalog_sha256=digest)
        if target_kind == "project"
        else DirectTarget(pool_id=pool.pool_id)
    )
    reference = manager.aliases()[0].snapshot_ref
    assert reference is not None
    if target_kind == "chain":
        reference = "remote/wrapped.json"
        store.register_catalog_snapshot(
            organization_id=manager.organization_id,
            snapshot_ref=reference,
            catalog_sha256=digest,
        )
    store.activate_alias_revision(
        organization_id=manager.organization_id,
        alias_id="coding",
        alias_name="coding",
        revision_id="wrapped-revision",
        target=target,
        snapshot_ref=reference,
        catalog_sha256=digest,
    )

    class Selection(_ExactAProjectResolver):
        """Select the already-authorized fixture pool without provider I/O."""

        def select_blocking(
            self,
            *,
            target: ProjectTarget,
            request: GatewayRequest,
            episode_namespace: tuple[str, str, str, str],
            deadline_monotonic: float,
        ) -> ProjectSelection:
            """Return one exact configured deployment without model-reference expansion."""
            return ProjectSelection(
                exact_model_id=pool.exact_model_id,
                selected_alias=normalized.deployments[0].source_alias,
                activation_ref=target.activation_ref,
            )

    wrapper = _ForwardingRouteResolver(
        CatalogRouteResolver(
            {("wrapped-revision", digest): normalized}, project_resolver=Selection()
        )
    )
    resolver: RouteResolver = wrapper
    control = NativeControlPlane(
        cast(
            NativeGatewayComponents,
            SimpleNamespace(
                store=store,
                ledger=SQLiteAttemptLedger(manager.database_path),
                routes=resolver,
                runtime_catalogs={
                    ("wrapped-revision", digest): RuntimeModelCatalog(
                        authored, environment={"TEST_PROVIDER_KEY": "test-only"}
                    )
                },
            ),
        )
    )
    argument = json.dumps({"raw_key": key, "body": _chat_body(), "idempotency_key": "wrapper"})
    if target_kind == "chain":
        with pytest.raises(NativeBridgeError) as error:
            getattr(control, operation)(argument)
        assert (
            json.loads(error.value.public_error_json)["code"] == "model_chain_authority_unavailable"
        )
    else:
        result = json.loads(getattr(control, operation)(argument))
        assert result["alias_revision_id"] == "wrapped-revision"
        if operation == "admit":
            assert result["route"]
    assert wrapper.classifications == 1
    with sqlite3.connect(manager.database_path) as connection:
        requests = connection.execute("SELECT count(*) FROM gateway_requests").fetchone()[0]
        attempts = connection.execute("SELECT count(*) FROM gateway_attempts").fetchone()[0]
    assert requests == int(target_kind != "chain" and operation == "admit")
    assert attempts == 0


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
        input_price: Configured input nano-USD per million tokens.

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
                input_nano_usd_per_million_tokens=input_price,
                output_nano_usd_per_million_tokens=900_000,
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
    assert metadata.input_nano_usd_per_million_tokens == 900_000
    assert metadata.context_window_tokens is None
    assert metadata.cached_input_nano_usd_per_million_tokens is None


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
    assert metadata.input_nano_usd_per_million_tokens == 900_000


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


def test_direct_route_carries_the_pools_throttle_cache_threshold() -> None:
    """The authored cache-stakes threshold rides the execution snapshot like failover_mode."""
    deployments = (
        _deployment(deployment_id="route-a", source_alias="route-a"),
        _deployment(deployment_id="route-b", source_alias="route-b"),
    )
    certification = GatewayEquivalenceCertification(
        certification_id="certification-threshold",
        provenance="operator comparison run 2026-09-10",
        evidence_sha256="e" * 64,
        certified_at=datetime(2026, 9, 10, tzinfo=UTC),
    )
    pool = ExactModelPool(
        pool_id="pool-threshold",
        exact_model_id="exact-one",
        deployment_ids=("route-a", "route-b"),
        equivalence=certification,
        failover_mode="maximize_cache",
        throttle_cache_threshold=0.5,
    )
    catalog, digest = _catalog(deployments, (pool,))
    resolver = _resolver(catalog, digest)
    authorization = AuthorizationSnapshot(
        request_id="request-one",
        organization_id="organization-one",
        identity_id="identity-one",
        virtual_key_id="key-one",
        alias="public-model",
        alias_revision_id=_REVISION,
        target=DirectTarget(pool_id="pool-threshold"),
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        catalog_sha256=digest,
        canonical_request_sha256="d" * 64,
        deadline_monotonic=1.0,
    )

    route = resolver.resolve_direct(authorization)

    assert route.snapshot.failover_mode == "maximize_cache"
    assert route.snapshot.throttle_cache_threshold == 0.5
    # An unauthored pool leaves the snapshot's threshold unset.
    unauthored, unauthored_digest = _catalog(
        deployments, (pool.model_copy(update={"throttle_cache_threshold": None}),)
    )
    plain = _resolver(unauthored, unauthored_digest).resolve_direct(
        authorization.model_copy(update={"catalog_sha256": unauthored_digest})
    )
    assert plain.snapshot.throttle_cache_threshold is None


def _hint_authorization(digest: str, pool_id: str) -> AuthorizationSnapshot:
    """Build one direct-target authorization over the given frozen catalog."""
    return AuthorizationSnapshot(
        request_id="request-one",
        organization_id="organization-one",
        identity_id="identity-one",
        virtual_key_id="key-one",
        alias="public-model",
        alias_revision_id=_REVISION,
        target=DirectTarget(pool_id=pool_id),
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        catalog_sha256=digest,
        canonical_request_sha256="d" * 64,
        deadline_monotonic=1.0,
    )


def test_deployment_hint_pins_the_issuing_rung_first_with_the_pool_as_fallbacks() -> None:
    """A reasoning continuation walks the pool's ladder after its issuing rung.

    The hinted deployment leads (it alone can unseal the request's reasoning)
    and the pool's other certified rungs follow in pool order, each flagged for
    the reasoning strip and recorded as ``reasoning_continuation_failover``,
    so a throttle on the issuing rung no longer ends the request after one
    attempt. The pool's failover policy rides the snapshot unchanged.
    """
    deployments = tuple(
        _deployment(deployment_id=name, source_alias=name) for name in ("route-a", "route-b")
    )
    pool = ExactModelPool(
        pool_id="pool-two",
        exact_model_id="exact-one",
        deployment_ids=("route-a", "route-b"),
        equivalence=GatewayEquivalenceCertification(
            certification_id="certification-pinned",
            provenance="operator comparison run 2026-09-13",
            evidence_sha256="e" * 64,
            certified_at=datetime(2026, 9, 13, tzinfo=UTC),
        ),
        failover_mode="maximize_availability",
        throttle_cache_threshold=0.25,
    )
    catalog, digest = _catalog(deployments, (pool,))
    resolver = _resolver(catalog, digest)

    route = resolver.resolve_deployment_hint(_hint_authorization(digest, "pool-two"), "route-b")

    assert route.route_reason == "reasoning_continuation"
    assert route.reasoning_pinned_deployment_id == "route-b"
    assert [item.deployment_id for item in route.deployments] == ["route-b", "route-a"]
    assert route.snapshot.deployment_ids == ("route-b", "route-a")
    assert route.snapshot.failover_mode == "maximize_availability"
    assert route.snapshot.throttle_cache_threshold == 0.25
    issuing, fallback = route.deployments
    assert route.requires_reasoning_strip(issuing) is False
    assert route.requires_reasoning_strip(fallback) is True
    assert route.attempt_route_reason(issuing) == "reasoning_continuation"
    assert route.attempt_route_reason(fallback) == "reasoning_continuation_failover"
    # A plain direct route never flags a rung for the strip.
    direct = resolver.resolve_direct(_hint_authorization(digest, "pool-two"))
    assert direct.reasoning_pinned_deployment_id is None
    assert all(not direct.requires_reasoning_strip(item) for item in direct.deployments)
    assert direct.attempt_route_reason(direct.deployments[1]) == "direct"


def _chained_catalog(*, unrelated_models: int = 0) -> NormalizedGatewayCatalog:
    """Build authored A-to-B order with a same-exact A suffix and implicit B chain."""
    deployments = tuple(
        _deployment(deployment_id=name, source_alias=name, exact_model_id=model)
        for name, model in (("a1", "a"), ("a2", "a"), ("b1", "b"))
    )
    unrelated = tuple(
        _deployment(deployment_id=f"c{i}", source_alias=f"c{i}", exact_model_id=f"c{i}")
        for i in range(unrelated_models)
    )
    return NormalizedGatewayCatalog(
        deployments=(*deployments, *unrelated),
        pools=(
            ExactModelPool(
                pool_id="pool-a",
                exact_model_id="a",
                deployment_ids=("a1", "a2"),
                equivalence=GatewayEquivalenceCertification(
                    certification_id="certification-a",
                    provenance="verified fixture",
                    evidence_sha256="e" * 64,
                    certified_at=datetime(2026, 9, 16, tzinfo=UTC),
                ),
            ),
            ExactModelPool(pool_id="pool-b", exact_model_id="b", deployment_ids=("b1",)),
            *(
                ExactModelPool(
                    pool_id=f"pool-c{i}", exact_model_id=f"c{i}", deployment_ids=(f"c{i}",)
                )
                for i in range(unrelated_models)
            ),
        ),
        model_chains=(chain("a", "a1", ">b", "a2"),),
    )


class _ExactAProjectResolver:
    """Select only A and authenticate only A's exact-pool continuation deployments."""

    async def select(
        self,
        *,
        target: ProjectTarget,
        request: GatewayRequest,
        episode_namespace: tuple[str, str, str, str],
        deadline_monotonic: float,
    ) -> ProjectSelection:
        """Return the same frozen A selection on the async entrypoint."""
        return self.select_blocking(
            target=target,
            request=request,
            episode_namespace=episode_namespace,
            deadline_monotonic=deadline_monotonic,
        )

    def select_blocking(
        self,
        *,
        target: ProjectTarget,
        request: GatewayRequest,
        episode_namespace: tuple[str, str, str, str],
        deadline_monotonic: float,
    ) -> ProjectSelection:
        """Return A without embeddings, runtime activation, or provider work."""
        return ProjectSelection(
            exact_model_id="a", selected_alias="a1", activation_ref=target.activation_ref
        )

    def authorize_deployment_hint(
        self, *, target: ProjectTarget, deployment: ExactModelDeployment
    ) -> None:
        """Reject any hinted exact model not selected by this frozen project."""
        if deployment.exact_model_id != "a":
            raise GatewayRoutingError(
                "reasoning continuation deployment is not a project candidate"
            )


@pytest.mark.parametrize(
    "surface",
    [GatewayApiSurface.CHAT_COMPLETIONS, GatewayApiSurface.RESPONSES, GatewayApiSurface.MESSAGES],
)
def test_project_routes_and_hints_stay_in_selected_exact_pool(
    surface: GatewayApiSurface,
) -> None:
    """Both project entrypoints and hints keep A exact while direct A may traverse B."""
    catalog = _chained_catalog()
    digest = catalog.identity_sha256()
    resolver = CatalogRouteResolver(
        {(_REVISION, digest): catalog}, project_resolver=_ExactAProjectResolver()
    )
    direct_auth = _hint_authorization(digest, "pool-a").model_copy(update={"surface": surface})
    auth = direct_auth.model_copy(
        update={
            "target": ProjectTarget(
                project_ref="project-one", activation_ref="activation-one", catalog_sha256=digest
            )
        }
    )
    request = GatewayRequest(surface=surface, messages=(GatewayMessage(role="user", content="hi"),))
    namespace = ("org", "identity", _REVISION, "episode")
    with unittest.mock.patch.object(
        model_plan, "expand_model_chain", wraps=model_plan.expand_model_chain
    ) as expansion:
        routes = (
            asyncio.run(
                resolver.resolve(authorization=auth, request=request, episode_namespace=namespace)
            ),
            resolver.resolve_project_blocking(
                authorization=auth, request=request, episode_namespace=namespace
            ),
            resolver.resolve_deployment_hint(auth, "a1"),
            resolver.resolve_deployment_hint(auth, "a2"),
        )
    expansion.assert_not_called()
    for route in routes:
        assert route.snapshot.authorization == auth
        assert route.snapshot.exact_model_id == "a"
        assert route.snapshot.pool_id == "pool-a"
        assert set(route.snapshot.deployment_ids) == {"a1", "a2"}
        assert tuple(d.deployment_id for d in route.deployments) == route.snapshot.deployment_ids
        assert all(d.exact_model_id == "a" for d in route.deployments)
        assert route.snapshot.model_stages == ()
        assert route.snapshot.traversal_events == ()
    assert routes[-1].snapshot.deployment_ids == ("a2", "a1")
    with pytest.raises(GatewayRoutingError, match="not a project candidate"):
        resolver.resolve_deployment_hint(auth, "b1")
    direct = resolver.resolve_direct(direct_auth)
    assert direct.snapshot.deployment_ids == ("a1", "b1", "a2")
    assert [s.exact_model_id for s in direct.snapshot.model_stages] == ["a", "b", "a"]
    hint = resolver.resolve_deployment_hint(
        direct_auth.model_copy(update={"descendant_start_authorized": True}), "b1"
    )
    assert hint.snapshot.deployment_ids == ("b1", "a2")
    assert hint.snapshot.exact_model_id == "a"
    assert hint.snapshot.pool_id == "pool-a"


@pytest.mark.parametrize(
    "surface",
    [
        GatewayApiSurface.CHAT_COMPLETIONS,
        GatewayApiSurface.RESPONSES,
        GatewayApiSurface.MESSAGES,
    ],
)
def test_child_hint_requires_explicit_start_authority_without_blocking_root_suffix(
    surface: GatewayApiSurface,
) -> None:
    """A child grant cannot substitute for root preflight, while root suffix pins remain valid."""
    catalog = _chained_catalog()
    digest = catalog.identity_sha256()
    resolver = _resolver(catalog, digest)
    auth = _hint_authorization(digest, "pool-a").model_copy(update={"surface": surface})
    assert auth.descendant_start_authorized is False
    with pytest.raises(GatewayRoutingError, match="requires explicit authorization"):
        resolver.resolve_deployment_hint(auth, "b1")
    for issuer, expected in (("a1", ("a1", "b1", "a2")), ("a2", ("a2",))):
        root = resolver.resolve_deployment_hint(auth, issuer)
        assert root.snapshot.deployment_ids == expected
        assert root.reasoning_pinned_deployment_id == issuer
    authorized = auth.model_copy(update={"descendant_start_authorized": True})
    child = resolver.resolve_deployment_hint(authorized, "b1")
    assert child.snapshot.deployment_ids == ("b1", "a2")
    assert child.snapshot.stage_for_depth(0).ancestry == ("a", "b")
    assert child.snapshot.authorization == authorized
    # A real root pin narrowed after a forward failure keeps its original issuer,
    # not a newly authorized child start.
    root = resolver.resolve_deployment_hint(auth, "a1")
    fallback = select_route_deployments(root, (1, 2))
    assert fallback.snapshot.deployment_ids == ("b1", "a2")
    assert fallback.reasoning_pinned_deployment_id == "a1"
    assert fallback.snapshot.authorization.descendant_start_authorized is False


def test_child_start_authorization_never_adds_unreachable_or_revoked_hints() -> None:
    """Explicit child-start permission preserves the pinned graph's deployment boundaries."""
    catalog = _chained_catalog(unrelated_models=1)
    digest = catalog.identity_sha256()
    resolver = _resolver(catalog, digest)
    auth = _hint_authorization(digest, "pool-a").model_copy(
        update={"descendant_start_authorized": True}
    )
    for hint in ("missing", "c0"):
        with pytest.raises(GatewayRoutingError, match="not reachable"):
            resolver.resolve_deployment_hint(auth, hint)
    with pytest.raises(GatewayRoutingError, match="snapshot is not active"):
        resolver.resolve_deployment_hint(
            auth.model_copy(update={"alias_revision_id": "revoked"}), "b1"
        )
    resolver.swap_catalogs({}, project_resolver=None, listing_pools={})
    with pytest.raises(GatewayRoutingError, match="snapshot is not active"):
        resolver.resolve_deployment_hint(auth, "b1")


def test_index_builds_immutable_chain_maps_once_per_shared_catalog() -> None:
    """Index unrelated singleton defaults once, not during direct or hint routing."""
    catalog = _chained_catalog(unrelated_models=100)
    digest = catalog.identity_sha256()
    keys = [(f"revision-{i}", digest) for i in range(50)]
    with unittest.mock.patch.object(
        NormalizedGatewayCatalog,
        "chains_by_model",
        autospec=True,
        side_effect=NormalizedGatewayCatalog.chains_by_model,
    ) as chains:
        resolver = CatalogRouteResolver(dict.fromkeys(keys, catalog))
        assert chains.call_count == 1
        views = resolver._catalogs  # noqa: SLF001 - verify revision-scoped immutable indexes.
        view = views[keys[0]]
        assert all(views[key] is view for key in keys)
        assert isinstance(view.chains, MappingProxyType)
        assert isinstance(view.pools, MappingProxyType)
        assert isinstance(view.deployments, MappingProxyType)
        assert set(view.chains) == {"a", "b", *(f"c{i}" for i in range(100))}
        for revision, _ in keys[:3]:
            auth = _hint_authorization(digest, "pool-a").model_copy(
                update={"alias_revision_id": revision, "descendant_start_authorized": True}
            )
            assert resolver.resolve_direct(auth).snapshot.deployment_ids == ("a1", "b1", "a2")
            assert resolver.resolve_deployment_hint(auth, "b1").snapshot.deployment_ids == (
                "b1",
                "a2",
            )
        assert chains.call_count == 1
        unrelated_auth = auth.model_copy(update={"target": DirectTarget(pool_id="pool-c0")})
        with unittest.mock.patch.object(model_plan, "expand_model_chain") as expansion:
            assert resolver.resolve_direct(unrelated_auth).snapshot.model_stages == ()
        expansion.assert_not_called()


def test_direct_hint_expands_only_its_authorized_root_once() -> None:
    """A child pin slices the already-authorized root plan rather than expanding either twice."""
    catalog = _chained_catalog()
    digest = catalog.identity_sha256()
    resolver = _resolver(catalog, digest)
    with unittest.mock.patch.object(
        model_plan, "expand_model_chain", wraps=model_plan.expand_model_chain
    ) as expansion:
        route = resolver.resolve_deployment_hint(
            _hint_authorization(digest, "pool-a").model_copy(
                update={"descendant_start_authorized": True}
            ),
            "b1",
        )
    assert expansion.call_count == 1
    assert expansion.call_args.args[0] == "a"
    assert route.snapshot.deployment_ids == ("b1", "a2")
    assert route.snapshot.stage_for_depth(0).exact_model_id == "b"
    assert route.snapshot.pool_id == "pool-a"


def test_chain_indexes_keep_same_revision_catalog_generations_separate_after_swap() -> None:
    """New default child policy and authored order never leak into old-digest authorizations."""
    old = _chained_catalog()
    old_digest = old.identity_sha256()
    new = NormalizedGatewayCatalog(
        deployments=old.deployments,
        pools=(old.pools[0], old.pools[1].model_copy(update={"failover_mode": "maximize_cache"})),
        model_chains=(chain("a", "a1", "a2", ">b"),),
    )
    new_digest = new.identity_sha256()
    old_auth = _hint_authorization(old_digest, "pool-a").model_copy(
        update={"descendant_start_authorized": True}
    )
    new_auth = _hint_authorization(new_digest, "pool-a").model_copy(
        update={"descendant_start_authorized": True}
    )
    resolver = _resolver(old, old_digest)
    frozen = resolver.resolve_direct(old_auth)
    resolver.swap_catalogs(
        {(_REVISION, old_digest): old, (_REVISION, new_digest): new},
        project_resolver=None,
        listing_pools={},
    )
    assert resolver.resolve_direct(old_auth) == frozen
    assert resolver.resolve_direct(new_auth).snapshot.deployment_ids == ("a1", "a2", "b1")
    old_hint = resolver.resolve_deployment_hint(old_auth, "b1")
    new_hint = resolver.resolve_deployment_hint(new_auth, "b1")
    assert old_hint.snapshot.deployment_ids == ("b1", "a2")
    assert old_hint.snapshot.stage_for_depth(0).failover_mode == "maximize_availability"
    assert old_hint.snapshot.authorization.catalog_sha256 == old_digest
    assert new_hint.snapshot.deployment_ids == ("b1",)
    assert new_hint.snapshot.stage_for_depth(0).failover_mode == "maximize_cache"
    assert new_hint.snapshot.authorization.catalog_sha256 == new_digest


def test_catalog_without_authored_chains_skips_default_chain_indexing() -> None:
    """Direct-only catalogs never pay to synthesize unused chain defaults."""
    catalog = _single_pool_catalog()
    digest = catalog.identity_sha256()
    with unittest.mock.patch.object(NormalizedGatewayCatalog, "chains_by_model") as chains:
        resolver = _resolver(catalog, digest)
        auth = _hint_authorization(digest, "pool-one")
        assert resolver.resolve_direct(auth).snapshot.model_stages == ()
        assert resolver.resolve_deployment_hint(auth, "deployment-one").snapshot.model_stages == ()
    chains.assert_not_called()


def test_deployment_hint_on_a_single_rung_pool_has_no_fallbacks() -> None:
    """A one-deployment pool pins its only rung and has nothing to fail over to."""
    catalog = _single_pool_catalog()
    digest = catalog.identity_sha256()
    resolver = _resolver(catalog, digest)

    route = resolver.resolve_deployment_hint(
        _hint_authorization(digest, "pool-one"), "deployment-one"
    )

    assert route.deployment.deployment_id == "deployment-one"
    assert route.fallback_deployments == ()
    assert route.snapshot.deployment_ids == ("deployment-one",)
    assert route.reasoning_pinned_deployment_id == "deployment-one"
    assert route.route_reason == "reasoning_continuation"
