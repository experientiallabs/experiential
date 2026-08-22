"""Resolve authorized direct and project targets without executing provider work."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass

from exp.common.core.artifacts import ArtifactId, ContractModel, stable_id
from exp.common.models.gateway_catalog import (
    ExactModelDeployment,
    ExactModelPool,
    NormalizedGatewayCatalog,
)
from exp.common.routing.policy import RoutingDecision
from exp.runtime.gateway.contracts import (
    AuthorizationSnapshot,
    DirectTarget,
    ExecutionSnapshot,
    GatewayRequest,
    ProjectSelection,
    ProjectTarget,
)
from exp.runtime.gateway.discovery import PublishedAliasMetadata, published_alias_metadata
from exp.runtime.gateway.interfaces import ProjectTargetResolver
from exp.runtime.models.providers.async_transport import ProviderDeadlineExceeded, RequestDeadline
from exp.runtime.openai_protocol.model_adapter import model_request as gateway_model_request
from exp.runtime.router.runtime import RouterRuntime


class GatewayRoutingError(ValueError):
    """An authorized target cannot resolve inside its frozen catalog snapshot."""


class GatewayRoute(ContractModel):
    """One immutable ordered exact-model route ready for provider execution."""

    snapshot: ExecutionSnapshot
    deployment: ExactModelDeployment
    fallback_deployments: tuple[ExactModelDeployment, ...] = ()
    route_reason: str
    fallback_reason: str | None = None

    @property
    def deployments(self) -> tuple[ExactModelDeployment, ...]:
        """Return every certified deployment in deterministic operational order."""
        return (self.deployment, *self.fallback_deployments)


@dataclass(frozen=True)
class _CatalogView:
    """One revision-scoped catalog plus its indexed pools and deployments."""

    catalog: NormalizedGatewayCatalog
    pools: Mapping[str, ExactModelPool]
    deployments: Mapping[str, ExactModelDeployment]


class CatalogRouteResolver:
    """Resolve direct pools or injected project selections against one exact catalog."""

    def __init__(
        self,
        catalogs: Mapping[tuple[str, str], NormalizedGatewayCatalog],
        *,
        project_resolver: ProjectTargetResolver | None = None,
        listing_pools: Mapping[tuple[str, str, str], str] | None = None,
    ) -> None:
        """Index one immutable catalog and optional learned-selection seam.

        Args:
            catalogs: Alias-revision and digest pairs mapped to normalized snapshots.
            project_resolver: Optional resolver for project-backed targets.
            listing_pools: Direct-target pool IDs keyed by granted alias, revision,
                and catalog digest. Project aliases are omitted and stay identity-only.
        """
        self._project_resolver = project_resolver
        self._listing_pools = dict(listing_pools or {})
        self._catalogs = _index_catalogs(catalogs)

    def swap_catalogs(
        self,
        catalogs: Mapping[tuple[str, str], NormalizedGatewayCatalog],
        *,
        project_resolver: ProjectTargetResolver | None,
        listing_pools: Mapping[tuple[str, str, str], str],
    ) -> None:
        """Atomically replace the served catalog index with one validated superset.

        Callers must include every revision that an in-flight authorization may
        still reference so requests never observe a partially loaded catalog.

        Args:
            catalogs: Alias-revision and digest pairs mapped to normalized snapshots.
            project_resolver: Replacement resolver covering all retained activations.
            listing_pools: Direct-target pool IDs keyed by granted alias, revision,
                and catalog digest, covering the replacement generation.

        Raises:
            ValueError: One catalog does not match its declared digest.
        """
        indexed = _index_catalogs(catalogs)
        self._project_resolver = project_resolver
        self._listing_pools = dict(listing_pools)
        self._catalogs = indexed

    def published_metadata(
        self,
        *,
        alias: str,
        revision_id: str,
        catalog_sha256: str,
    ) -> PublishedAliasMetadata | None:
        """Return catalog-backed listing fields for one granted public alias.

        Lookup uses the alias revision's authoritative direct pool, never a public
        name that happens to match a deployment or source alias. Multi-deployment
        pools and project aliases publish nothing extra.

        Args:
            alias: Granted public alias name.
            revision_id: Active alias revision loaded in this process.
            catalog_sha256: Frozen catalog digest bound to that revision.

        Returns:
            Declared capability, limit, and price fields, or ``None`` when the alias
            has no unique catalog deployment on its frozen direct target.
        """
        pool_id = self._listing_pools.get((alias, revision_id, catalog_sha256))
        if pool_id is None:
            return None
        view = self._catalogs.get((revision_id, catalog_sha256))
        if view is None:
            return None
        pool = view.pools.get(pool_id)
        if pool is None or len(pool.deployment_ids) != 1:
            return None
        return published_alias_metadata(view.deployments.get(pool.deployment_ids[0]))

    async def resolve(
        self,
        *,
        authorization: AuthorizationSnapshot,
        request: GatewayRequest,
        episode_namespace: tuple[str, str, str, str],
    ) -> GatewayRoute:
        """Resolve authorized authority into one singleton provider deployment.

        Args:
            authorization: Frozen authenticated alias revision and target.
            request: Canonical request visible to learned selection.
            episode_namespace: Tenant-isolated sticky selection identity.

        Returns:
            Frozen exact model, pool, and one launch deployment.

        Raises:
            GatewayRoutingError: Catalog identity or target resolution is invalid.
        """
        target = authorization.target
        if isinstance(target, DirectTarget):
            return self.resolve_direct(authorization)
        view = self._catalogs.get((authorization.alias_revision_id, authorization.catalog_sha256))
        if view is None:
            raise GatewayRoutingError("authorized catalog snapshot is not active for this revision")
        if target.catalog_sha256 != authorization.catalog_sha256:
            raise GatewayRoutingError("project target catalog differs from authorized authority")
        selection = await self._select_project(
            target=target,
            request=request,
            episode_namespace=episode_namespace,
            deadline_monotonic=authorization.deadline_monotonic,
        )
        return self._project_route(view=view, authorization=authorization, selection=selection)

    def resolve_project_blocking(
        self,
        *,
        authorization: AuthorizationSnapshot,
        request: GatewayRequest,
        episode_namespace: tuple[str, str, str, str],
    ) -> GatewayRoute:
        """Resolve one project target from a caller thread without an event loop.

        The native engine's control-plane bridge runs on Rust worker threads, so
        it uses this synchronous path. It applies the same frozen-catalog checks,
        the same selection seam, and the same route construction as the async
        resolver, so the two engines cannot drift on project routing.

        Args:
            authorization: Frozen authenticated alias revision and target.
            request: Canonical request visible to learned selection.
            episode_namespace: Tenant-isolated sticky selection identity.

        Returns:
            Frozen exact model, pool, and one launch deployment.

        Raises:
            GatewayRoutingError: Catalog identity or target resolution is invalid.
            ProviderDeadlineExceeded: No request-wide time remains for selection.
        """
        target = authorization.target
        if not isinstance(target, ProjectTarget):
            raise GatewayRoutingError("blocking project resolution requires a project target")
        view = self._catalogs.get((authorization.alias_revision_id, authorization.catalog_sha256))
        if view is None:
            raise GatewayRoutingError("authorized catalog snapshot is not active for this revision")
        if target.catalog_sha256 != authorization.catalog_sha256:
            raise GatewayRoutingError("project target catalog differs from authorized authority")
        if self._project_resolver is None:
            raise GatewayRoutingError("project target is not activated in this process")
        try:
            selection = self._project_resolver.select_blocking(
                target=target,
                request=request,
                episode_namespace=episode_namespace,
                deadline_monotonic=authorization.deadline_monotonic,
            )
        except (GatewayRoutingError, ProviderDeadlineExceeded):
            raise
        except Exception as exc:
            raise GatewayRoutingError("project selection failed") from exc
        return self._project_route(view=view, authorization=authorization, selection=selection)

    def _project_route(
        self,
        *,
        view: _CatalogView,
        authorization: AuthorizationSnapshot,
        selection: ProjectSelection,
    ) -> GatewayRoute:
        """Map one learned selection to its unambiguous frozen pool and route."""
        selected_deployments = tuple(
            item
            for item in view.catalog.deployments
            if item.source_alias == selection.selected_alias
            and item.exact_model_id == selection.exact_model_id
        )
        if len(selected_deployments) != 1:
            raise GatewayRoutingError(
                "project selection requires one unambiguous frozen deployment"
            )
        deployment = selected_deployments[0]
        pools = tuple(
            item
            for item in view.catalog.pools
            if item.exact_model_id == selection.exact_model_id
            and deployment.deployment_id in item.deployment_ids
        )
        if len(pools) != 1:
            raise GatewayRoutingError(
                "project selection requires one unambiguous certified exact-model pool"
            )
        pool = pools[0]
        return self._route(
            view=view,
            authorization=authorization,
            pool=pool,
            route_reason="learned_router",
            fallback_reason=selection.fallback_reason,
        )

    def resolve_direct(self, authorization: AuthorizationSnapshot) -> GatewayRoute:
        """Resolve one direct-target authorization without event-loop work.

        Direct pools resolve entirely inside frozen in-memory catalogs, so
        callers without a running event loop (the Rust engine's control-plane
        bridge) share this path with the async resolver.

        Args:
            authorization: Frozen authenticated alias revision and target.

        Returns:
            Frozen exact model, pool, and one launch deployment.

        Raises:
            GatewayRoutingError: The target is project-backed or the catalog
                identity cannot resolve inside its frozen snapshot.
        """
        view = self._catalogs.get((authorization.alias_revision_id, authorization.catalog_sha256))
        if view is None:
            raise GatewayRoutingError("authorized catalog snapshot is not active for this revision")
        target = authorization.target
        if not isinstance(target, DirectTarget):
            raise GatewayRoutingError("project targets require asynchronous learned selection")
        pool = self._pool(view, target.pool_id)
        return self._route(
            view=view,
            authorization=authorization,
            pool=pool,
            route_reason="direct",
            fallback_reason=None,
        )

    async def _select_project(
        self,
        *,
        target: ProjectTarget,
        request: GatewayRequest,
        episode_namespace: tuple[str, str, str, str],
        deadline_monotonic: float,
    ) -> ProjectSelection:
        """Call the injected selection-only project bridge."""
        if self._project_resolver is None:
            raise GatewayRoutingError("project target is not activated in this process")
        try:
            return await self._project_resolver.select(
                target=target,
                request=request,
                episode_namespace=episode_namespace,
                deadline_monotonic=deadline_monotonic,
            )
        except (GatewayRoutingError, ProviderDeadlineExceeded):
            raise
        except Exception as exc:
            raise GatewayRoutingError("project selection failed") from exc

    def _pool(self, view: _CatalogView, pool_id: str) -> ExactModelPool:
        """Return one named frozen pool or fail closed."""
        pool = view.pools.get(pool_id)
        if pool is None:
            raise GatewayRoutingError("authorized direct pool is absent from the frozen catalog")
        return pool

    def _route(
        self,
        *,
        view: _CatalogView,
        authorization: AuthorizationSnapshot,
        pool: ExactModelPool,
        route_reason: str,
        fallback_reason: str | None,
    ) -> GatewayRoute:
        """Build one ordered execution route from a certified exact-model pool."""
        deployments: list[ExactModelDeployment] = []
        for deployment_id in pool.deployment_ids:
            deployment = view.deployments.get(deployment_id)
            if deployment is None or deployment.exact_model_id != pool.exact_model_id:
                raise GatewayRoutingError("frozen pool deployment identity is invalid")
            deployments.append(deployment)
        return GatewayRoute(
            snapshot=ExecutionSnapshot(
                authorization=authorization,
                exact_model_id=pool.exact_model_id,
                pool_id=pool.pool_id,
                deployment_ids=pool.deployment_ids,
            ),
            deployment=deployments[0],
            fallback_deployments=tuple(deployments[1:]),
            route_reason=route_reason,
            fallback_reason=fallback_reason,
        )


def _index_catalogs(
    catalogs: Mapping[tuple[str, str], NormalizedGatewayCatalog],
) -> dict[tuple[str, str], _CatalogView]:
    """Index digest-verified catalogs by alias revision and catalog digest.

    Args:
        catalogs: Alias-revision and digest pairs mapped to normalized snapshots.

    Returns:
        Fully built revision-scoped catalog views.

    Raises:
        ValueError: One catalog does not match its declared digest.
    """
    indexed: dict[tuple[str, str], _CatalogView] = {}
    for key, catalog in catalogs.items():
        revision_id, catalog_sha256 = key
        if catalog.identity_sha256() != catalog_sha256:
            raise ValueError(f"catalog for alias revision {revision_id!r} has the wrong digest")
        indexed[key] = _CatalogView(
            catalog=catalog,
            pools={pool.pool_id: pool for pool in catalog.pools},
            deployments={
                deployment.deployment_id: deployment for deployment in catalog.deployments
            },
        )
    return indexed


class RouterProjectTargetResolver:
    """Run synchronous ``RouterRuntime.select`` in a bounded selection worker lane."""

    def __init__(
        self,
        activations: Mapping[tuple[str, str, str], RouterRuntime],
        exact_models_by_alias: Mapping[tuple[str, str, str, str], str],
        *,
        maximum_outstanding_selections: int = 4,
    ) -> None:
        """Bind frozen activations and an exact-model projection.

        Args:
            activations: Project, activation, and catalog digest mapped to verified
                runtimes, so each retained revision keeps its own selection policy.
            exact_models_by_alias: Project, activation, catalog, and candidate alias mappings.
            maximum_outstanding_selections: Running plus detached selection calls allowed.
        """
        if maximum_outstanding_selections < 1:
            raise ValueError("maximum_outstanding_selections must be at least one")
        self._activations = dict(activations)
        self._exact_models_by_alias = dict(exact_models_by_alias)
        # One worker pool is the single aggregate bound on concurrent
        # selections, shared by the event-loop path and the blocking path.
        # Timed-out submissions are cancelled so abandoned work never
        # occupies a worker ahead of live requests.
        self._selection_workers = ThreadPoolExecutor(
            max_workers=maximum_outstanding_selections,
            thread_name_prefix="exp-router-selection",
        )

    async def select(
        self,
        *,
        target: ProjectTarget,
        request: GatewayRequest,
        episode_namespace: tuple[str, str, str, str],
        deadline_monotonic: float,
    ) -> ProjectSelection:
        """Select one logical model without invoking ``RouterRuntime.complete``.

        Args:
            target: Frozen project activation target.
            request: Canonical gateway request converted only for learned selection.
            episode_namespace: Tenant-isolated sticky episode identity.
            deadline_monotonic: Absolute request-wide deadline.

        Returns:
            Exact logical model and content-free learned selection details.
        """
        runtime = self._runtime(target)
        deadline = RequestDeadline(deadline_monotonic)
        deadline.attempt_timeout()
        model_request = gateway_model_request(request)
        episode_id = project_episode_identity(episode_namespace)
        decision = runtime._reuse_sticky_selection(  # noqa: SLF001 - selection-only bridge.
            model_request,
            episode_id=episode_id,
        )
        if decision is None:
            submitted = self._selection_workers.submit(
                runtime._select_unretained,  # noqa: SLF001 - selection-only bridge.
                model_request,
                episode_id=episode_id,
            )
            wrapped = asyncio.wrap_future(submitted)
            wrapped.add_done_callback(_consume_abandoned_selection)
            try:
                async with asyncio.timeout(deadline.attempt_timeout()):
                    prepared = await asyncio.shield(wrapped)
            except TimeoutError as exc:
                submitted.cancel()
                raise ProviderDeadlineExceeded("router selection deadline exceeded") from exc
            deadline.attempt_timeout()
            decision = runtime._retain_prepared_selection(
                model_request,
                episode_id=episode_id,
                prepared=prepared,
            )
        return self._selection(target, decision)

    def select_blocking(
        self,
        *,
        target: ProjectTarget,
        request: GatewayRequest,
        episode_namespace: tuple[str, str, str, str],
        deadline_monotonic: float,
    ) -> ProjectSelection:
        """Select one logical model synchronously on the caller's thread.

        The frozen ``RouterRuntime`` selection primitives are thread-safe, so
        this runs the same sticky reuse, unretained selection, and retention
        sequence as :meth:`select` without an event loop. The selection runs
        on the shared bounded worker pool, so a request whose deadline
        expires mid-selection fails immediately: still-queued work is
        cancelled and a running detached worker finishes without publishing
        sticky state.

        Args:
            target: Frozen project activation target.
            request: Canonical gateway request converted only for learned selection.
            episode_namespace: Tenant-isolated sticky episode identity.
            deadline_monotonic: Absolute request-wide deadline.

        Returns:
            Exact logical model and content-free learned selection details.
        """
        runtime = self._runtime(target)
        deadline = RequestDeadline(deadline_monotonic)
        deadline.attempt_timeout()
        model_request = gateway_model_request(request)
        episode_id = project_episode_identity(episode_namespace)
        decision = runtime._reuse_sticky_selection(  # noqa: SLF001 - selection-only bridge.
            model_request,
            episode_id=episode_id,
        )
        if decision is None:
            future = self._selection_workers.submit(
                runtime._select_unretained,  # noqa: SLF001 - selection-only bridge.
                model_request,
                episode_id=episode_id,
            )
            try:
                prepared = future.result(timeout=deadline.attempt_timeout())
            except FutureTimeoutError as exc:
                future.cancel()
                raise ProviderDeadlineExceeded("router selection deadline exceeded") from exc
            deadline.attempt_timeout()
            decision = runtime._retain_prepared_selection(  # noqa: SLF001 - selection-only.
                model_request,
                episode_id=episode_id,
                prepared=prepared,
            )
        return self._selection(target, decision)

    def _runtime(self, target: ProjectTarget) -> RouterRuntime:
        """Return the loaded frozen runtime for one activation or fail closed."""
        runtime = self._activations.get(
            (target.project_ref, target.activation_ref, target.catalog_sha256)
        )
        if runtime is None:
            raise GatewayRoutingError("project activation is not loaded")
        return runtime

    def _selection(self, target: ProjectTarget, decision: RoutingDecision) -> ProjectSelection:
        """Project one routing decision onto its frozen exact-model identity."""
        exact_model_id = self._exact_models_by_alias.get(
            (
                target.project_ref,
                target.activation_ref,
                target.catalog_sha256,
                decision.selected_alias,
            )
        )
        if exact_model_id is None:
            raise GatewayRoutingError("router selected alias has no frozen exact-model identity")
        return ProjectSelection(
            exact_model_id=exact_model_id,
            selected_alias=decision.selected_alias,
            activation_ref=target.activation_ref,
            fallback_reason=decision.fallback_reason,
        )


def _consume_abandoned_selection[SelectionT](wrapped: asyncio.Future[SelectionT]) -> None:
    """Retrieve a detached selection outcome so late failures are not logged as leaks."""
    if not wrapped.cancelled():
        wrapped.exception()


def project_episode_identity(
    namespace: tuple[ArtifactId, ArtifactId, ArtifactId, str],
) -> str:
    """Encode tenant-scoped episode components without delimiter collisions.

    Args:
        namespace: Organization, identity, alias revision, and caller episode key.

    Returns:
        Stable content-addressed identity with explicit component boundaries.
    """
    organization_id, identity_id, alias_revision_id, episode_key = namespace
    return stable_id(
        "gateway-project-episode",
        {
            "organization_id": organization_id,
            "identity_id": identity_id,
            "alias_revision_id": alias_revision_id,
            "episode_key": episode_key,
        },
    )
