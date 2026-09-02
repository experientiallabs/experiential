"""Resolve authorized direct and project targets without executing provider work."""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
from collections.abc import Mapping
from concurrent.futures import Future, wait
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass

from exp.common.core.artifacts import ArtifactId, ContractModel, stable_id
from exp.common.models import ModelRequest
from exp.common.models.gateway_catalog import (
    ExactModelDeployment,
    ExactModelPool,
    NormalizedGatewayCatalog,
    is_foreign_snapshot,
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
from exp.runtime.router.runtime import _PreparedSelection as PreparedSelection  # noqa: PLC2701

_SELECTION_DRAIN_SECONDS = 5.0

_logger = logging.getLogger(__name__)


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

    def resolve_deployment_hint(
        self,
        authorization: AuthorizationSnapshot,
        deployment_id: str,
    ) -> GatewayRoute:
        """Resolve one untrusted carrier hint only inside current alias authority."""
        view = self._catalogs.get((authorization.alias_revision_id, authorization.catalog_sha256))
        if view is None:
            raise GatewayRoutingError("authorized catalog snapshot is not active for this revision")
        target = authorization.target
        if isinstance(target, DirectTarget):
            pools = (self._pool(view, target.pool_id),)
        else:
            if target.catalog_sha256 != authorization.catalog_sha256:
                raise GatewayRoutingError(
                    "project target catalog differs from authorized authority"
                )
            deployment = view.deployments.get(deployment_id)
            if deployment is None:
                raise GatewayRoutingError("reasoning carrier deployment identity is invalid")
            self._authorize_project_deployment_hint(target, deployment)
            pools = tuple(
                pool for pool in view.catalog.pools if deployment_id in pool.deployment_ids
            )
        matching = tuple(pool for pool in pools if deployment_id in pool.deployment_ids)
        if len(matching) != 1:
            raise GatewayRoutingError("reasoning carrier deployment is not unambiguous")
        pool = matching[0]
        deployment = view.deployments.get(deployment_id)
        if deployment is None or deployment.exact_model_id != pool.exact_model_id:
            raise GatewayRoutingError("reasoning carrier deployment identity is invalid")
        return GatewayRoute(
            snapshot=ExecutionSnapshot(
                authorization=authorization,
                exact_model_id=pool.exact_model_id,
                pool_id=pool.pool_id,
                deployment_ids=(deployment_id,),
                failover_mode=pool.failover_mode,
            ),
            deployment=deployment,
            route_reason="reasoning_continuation",
            fallback_reason=None,
        )

    def _authorize_project_deployment_hint(
        self,
        target: ProjectTarget,
        deployment: ExactModelDeployment,
    ) -> None:
        """Require the loaded project resolver to authenticate candidate membership."""
        resolver = self._project_resolver
        authorize = (
            None if resolver is None else getattr(resolver, "authorize_deployment_hint", None)
        )
        if not callable(authorize):
            raise GatewayRoutingError(
                "project resolver cannot authenticate reasoning continuation deployments"
            )
        try:
            authorize(target=target, deployment=deployment)
        except GatewayRoutingError:
            raise
        except Exception as exc:
            raise GatewayRoutingError(
                "project resolver rejected reasoning continuation deployment"
            ) from exc

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
                failover_mode=pool.failover_mode,
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

    The pinned ``catalog_sha256`` stays the identity/attribution key for every
    revision. A same-version catalog must reproduce it exactly, so a mismatch is
    corruption and still raises. A cross-version snapshot (served through the
    hydration reader's tolerant path during a rolling deploy) is expected not to
    reproduce it; that catalog is indexed under its pinned digest without the
    byte-exact check, so a roll never hard-fails route resolution.

    Args:
        catalogs: Alias-revision and digest pairs mapped to normalized snapshots.

    Returns:
        Fully built revision-scoped catalog views.

    Raises:
        ValueError: A same-version catalog does not match its declared digest.
    """
    indexed: dict[tuple[str, str], _CatalogView] = {}
    for key, catalog in catalogs.items():
        revision_id, catalog_sha256 = key
        if not is_foreign_snapshot(catalog) and catalog.identity_sha256() != catalog_sha256:
            raise ValueError(f"catalog for alias revision {revision_id!r} has the wrong digest")
        indexed[key] = _CatalogView(
            catalog=catalog,
            pools={pool.pool_id: pool for pool in catalog.pools},
            deployments={
                deployment.deployment_id: deployment for deployment in catalog.deployments
            },
        )
    return indexed


@dataclass(frozen=True)
class _QueuedSelection:
    """One submitted selection waiting for a daemon worker to pick it up."""

    future: Future[PreparedSelection]
    runtime: RouterRuntime
    model_request: ModelRequest
    episode_id: str
    deadline: RequestDeadline


class SelectionWorkerPool:
    """Bounded selection lane owned for the process rather than one generation.

    One pool is the single aggregate bound on concurrent frozen selections,
    shared by the event-loop path, the blocking native path, and every
    resolver generation an alias-authority reload installs. Timed-out
    submissions are cancelled while queued, and a worker re-checks the
    request deadline before embedding, so abandoned work never runs ahead
    of live requests. Workers are daemon threads: a selection blocked inside
    a synchronous embedding call past the shutdown drain bound can only
    discard its own result (selection touches no ledger and no policy), and
    it never pins interpreter exit.
    """

    def __init__(self, *, maximum_outstanding_selections: int = 4) -> None:
        """Open one bounded daemon worker lane.

        Args:
            maximum_outstanding_selections: Running plus detached selection calls allowed.

        Raises:
            ValueError: The requested bound admits no selection at all.
        """
        if maximum_outstanding_selections < 1:
            raise ValueError("maximum_outstanding_selections must be at least one")
        self._lock = threading.Lock()
        self._closed = False
        self._queue: queue.SimpleQueue[_QueuedSelection | None] = queue.SimpleQueue()
        self._outstanding: set[Future[PreparedSelection]] = set()
        self._threads = tuple(
            threading.Thread(
                target=self._work,
                name=f"exp-router-selection-{index}",
                daemon=True,
            )
            for index in range(maximum_outstanding_selections)
        )
        for thread in self._threads:
            thread.start()

    def submit(
        self,
        runtime: RouterRuntime,
        model_request: ModelRequest,
        *,
        episode_id: str,
        deadline: RequestDeadline,
    ) -> Future[PreparedSelection]:
        """Queue one deadline-guarded unretained selection in the shared lane.

        Raises:
            RuntimeError: The lane is shut down and accepts no new selection.
        """
        submitted: Future[PreparedSelection] = Future()
        with self._lock:
            if self._closed:
                raise RuntimeError("selection worker pool is shut down")
            self._outstanding.add(submitted)
            self._queue.put(
                _QueuedSelection(
                    future=submitted,
                    runtime=runtime,
                    model_request=model_request,
                    episode_id=episode_id,
                    deadline=deadline,
                )
            )
        submitted.add_done_callback(self._forget)
        return submitted

    def shutdown(self, *, drain_timeout_seconds: float = _SELECTION_DRAIN_SECONDS) -> None:
        """Stop accepting selections, drop queued work, and drain running work.

        A selection already inside a synchronous provider embedding call cannot
        be preempted, so shutdown waits a bounded time for it and reports the
        remainder instead of blocking teardown. A reported straggler runs on a
        daemon thread, so it cannot keep the process alive after shutdown.

        Args:
            drain_timeout_seconds: Bound on waiting for running selections.
        """
        abandoned: list[_QueuedSelection] = []
        with self._lock:
            self._closed = True
            while True:
                try:
                    item = self._queue.get_nowait()
                except queue.Empty:
                    break
                if item is not None:
                    abandoned.append(item)
            for _ in self._threads:
                self._queue.put(None)
            outstanding = frozenset(self._outstanding)
        for item in abandoned:
            item.future.cancel()
        _, running = wait(outstanding, timeout=drain_timeout_seconds)
        if running:
            _logger.warning(
                "gateway shutdown left %d router selection call(s) running", len(running)
            )

    def _work(self) -> None:
        """Run queued selections on one daemon worker until the stop sentinel."""
        while True:
            item = self._queue.get()
            if item is None:
                return
            if not item.future.set_running_or_notify_cancel():
                continue
            try:
                prepared = _select_within_deadline(
                    item.runtime,
                    item.model_request,
                    episode_id=item.episode_id,
                    deadline=item.deadline,
                )
            except BaseException as failure:  # noqa: BLE001 - relayed to the waiting caller.
                item.future.set_exception(failure)
            else:
                item.future.set_result(prepared)

    def _forget(self, completed: Future[PreparedSelection]) -> None:
        """Drop one settled selection from the outstanding drain set."""
        with self._lock:
            self._outstanding.discard(completed)


class RouterProjectTargetResolver:
    """Run synchronous ``RouterRuntime.select`` in a bounded selection worker lane."""

    def __init__(
        self,
        activations: Mapping[tuple[str, str, str], RouterRuntime],
        exact_models_by_alias: Mapping[tuple[str, str, str, str], str],
        *,
        maximum_outstanding_selections: int = 4,
        selection_workers: SelectionWorkerPool | None = None,
    ) -> None:
        """Bind frozen activations and an exact-model projection.

        Args:
            activations: Project, activation, and catalog digest mapped to verified
                runtimes, so each retained revision keeps its own selection policy.
            exact_models_by_alias: Project, activation, catalog, and candidate alias mappings.
            maximum_outstanding_selections: Running plus detached selection calls allowed
                when this resolver opens its own lane.
            selection_workers: Pool shared with every other resolver generation, so an
                alias-authority reload cannot split the aggregate selection bound.
        """
        self._activations = dict(activations)
        self._exact_models_by_alias = dict(exact_models_by_alias)
        self._selection_workers = selection_workers or SelectionWorkerPool(
            maximum_outstanding_selections=maximum_outstanding_selections
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
                runtime,
                model_request,
                episode_id=episode_id,
                deadline=deadline,
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
                runtime,
                model_request,
                episode_id=episode_id,
                deadline=deadline,
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

    def authorize_deployment_hint(
        self,
        *,
        target: ProjectTarget,
        deployment: ExactModelDeployment,
    ) -> None:
        """Require one hinted deployment to be a candidate of this activation."""
        self._runtime(target)
        exact_model_id = self._exact_models_by_alias.get(
            (
                target.project_ref,
                target.activation_ref,
                target.catalog_sha256,
                deployment.source_alias,
            )
        )
        if exact_model_id != deployment.exact_model_id:
            raise GatewayRoutingError(
                "reasoning continuation deployment is not a project activation candidate"
            )

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


def _select_within_deadline(
    runtime: RouterRuntime,
    model_request: ModelRequest,
    *,
    episode_id: str,
    deadline: RequestDeadline,
) -> PreparedSelection:
    """Run one unretained selection only while its request deadline is live.

    The deadline check runs on the worker thread immediately before any
    embedding work, so a submission that expired while queued (racing its
    caller's cancellation) fails fast instead of occupying the bounded
    worker with a discarded selection.
    """
    deadline.attempt_timeout()
    return runtime._select_unretained(  # noqa: SLF001 - selection-only bridge.
        model_request,
        episode_id=episode_id,
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
