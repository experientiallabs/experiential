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
from typing import Protocol

from exp.common.core.artifacts import ContractModel
from exp.common.models import ModelRequest
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
from exp.runtime.gateway.model_plan import (
    _CatalogView,
    _index_catalogs,
    model_execution_snapshot,
    project_stage_selection,
    stage_start_authorized,
)
from exp.runtime.gateway.project_episode_identity import (
    project_episode_identity as project_episode_identity,
)
from exp.runtime.gateway.request_policy import RequestedRouteId
from exp.runtime.models.providers.async_transport import ProviderDeadlineExceeded, RequestDeadline
from exp.runtime.openai_protocol.model_adapter import model_request as gateway_model_request
from exp.runtime.router.runtime import RouterRuntime
from exp.runtime.router.runtime import _PreparedSelection as PreparedSelection  # noqa: PLC2701

_SELECTION_DRAIN_SECONDS = 5.0

_logger = logging.getLogger(__name__)


class GatewayRoutingError(ValueError):
    """An authorized target cannot resolve inside its frozen catalog snapshot."""


REASONING_CONTINUATION_ROUTE_REASON = "reasoning_continuation"
"""Route reason of a request whose active sealed reasoning pins its issuing rung first."""

REASONING_CONTINUATION_FAILOVER_ROUTE_REASON = "reasoning_continuation_failover"
"""Attempt route reason of a pinned continuation served by a non-issuing rung.

Recorded on every attempt dispatched past the issuing rung: the sealed reasoning
only that rung's credential could unseal was stripped from the attempt's payload,
so the ledger shows the request ran without its thinking continuity.
"""


class GatewayRoute(ContractModel):
    """One immutable ordered exact-model route ready for provider execution.

    Attributes:
        resolved_route_id: Optional host attestation that the requested public handle leads.
    """

    snapshot: ExecutionSnapshot
    deployment: ExactModelDeployment
    fallback_deployments: tuple[ExactModelDeployment, ...] = ()
    route_reason: str
    fallback_reason: str | None = None
    resolved_route_id: RequestedRouteId | None = None
    reasoning_pinned_deployment_id: str | None = None
    """The deployment whose credential sealed the request's active reasoning.

    ``None`` on every route without gateway-sealed reasoning. When set, that
    rung alone can replay the unsealed reasoning; every other rung is a failover
    fallback that ``requires_reasoning_strip`` and is recorded under
    ``REASONING_CONTINUATION_FAILOVER_ROUTE_REASON``. Sealed blocks never reach
    another provider's payload: the strip removes them before the fallback
    payload is built, and the payload builders still reject a foreign block.
    """

    @property
    def deployments(self) -> tuple[ExactModelDeployment, ...]:
        """Return every certified deployment in deterministic operational order."""
        return (self.deployment, *self.fallback_deployments)

    def requires_reasoning_strip(self, deployment: ExactModelDeployment) -> bool:
        """Return whether ``deployment`` must dispatch without the pinned sealed reasoning.

        True only on a reasoning-pinned route for a rung other than the issuing
        one: that rung cannot unseal the reasoning, so the post-user-boundary
        sealed blocks leave its payload while messages, tool calls, tool
        results, and visible text all stay. The stated loss is the model's
        thinking continuity across that tool call and the issuing provider's
        prompt cache for the turn.
        """
        pinned = self.reasoning_pinned_deployment_id
        return pinned is not None and deployment.deployment_id != pinned

    def attempt_route_reason(self, deployment: ExactModelDeployment) -> str:
        """Return the route reason the ledger records for an attempt on ``deployment``.

        The route's own reason, except a pinned continuation served by a
        non-issuing rung, recorded as ``REASONING_CONTINUATION_FAILOVER_ROUTE_REASON``
        so the ledger shows which attempts ran without thinking continuity.
        """
        if self.requires_reasoning_strip(deployment):
            return REASONING_CONTINUATION_FAILOVER_ROUTE_REASON
        return self.route_reason


class RouteResolver(Protocol):
    """Structural contract for the gateway's authorized route resolver.

    Platform wrappers can delegate to ``CatalogRouteResolver`` and annotate
    themselves against this protocol to catch missing or drifted methods.
    The seam includes selected-root classification before serving authorization
    and the three ways authority becomes a frozen :class:`GatewayRoute`.
    Catalog-swap, listing metadata, and lifecycle methods remain private.
    """

    def requires_model_chain_authority(self, authorization: AuthorizationSnapshot) -> bool:
        """Classify the selected root in its exact revision before acceptance or replay.

        Wrappers must delegate classification to their catalog authority and
        propagate invalid or missing catalog errors. This result identifies
        required host enforcement; it does not itself grant serving permission.
        """
        ...

    def resolve_direct(self, authorization: AuthorizationSnapshot) -> GatewayRoute:
        """Resolve one direct-target authorization without event-loop work."""
        ...

    def resolve_deployment_hint(
        self,
        authorization: AuthorizationSnapshot,
        deployment_id: str,
    ) -> GatewayRoute:
        """Resolve one canonical carrier-hint deployment inside current authority."""
        ...

    def resolve_project_blocking(
        self,
        *,
        authorization: AuthorizationSnapshot,
        request: GatewayRequest,
        episode_namespace: tuple[str, str, str, str],
    ) -> GatewayRoute:
        """Resolve one project target from a caller thread without an event loop."""
        ...


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

    def requires_model_chain_authority(self, authorization: AuthorizationSnapshot) -> bool:
        """Classify the exact selected root, not unrelated models in the shared catalog."""
        view = self._catalogs.get((authorization.alias_revision_id, authorization.catalog_sha256))
        if view is None:
            raise GatewayRoutingError("authorized catalog snapshot is not active for this revision")
        if isinstance(authorization.target, ProjectTarget):
            if authorization.target.catalog_sha256 != authorization.catalog_sha256:
                raise GatewayRoutingError(
                    "project target catalog differs from authorized authority"
                )
            return False
        return view.catalog.requires_model_chain_authority(pool_id=authorization.target.pool_id)

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
        """Resolve one untrusted carrier hint only inside current alias authority.

        Hints must name a canonical deployment in the authorized revision's
        reachable pools. BYOK and organization variant IDs do not resolve here:
        the caller must first map them to canonical IDs. Catalog membership
        never substitutes for explicit authority to start on a child model.

        Args:
            authorization: Frozen authenticated alias revision and target.
            deployment_id: Canonical deployment id carried on a reasoning
                continuation, resolved only within the authorized revision.

        Returns:
            The issuing rung first, followed by its permitted forward suffix.
            Other rungs require reasoning stripping: only the issuer can unseal
            the active reasoning. Eligible operational failures may continue
            without it; a singleton has no fallback. Child starts require the
            host's explicit root funding and policy authorization.

        Raises:
            GatewayRoutingError: Snapshot, membership, identity, or child-start
                authority is invalid.
        """
        view = self._catalogs.get((authorization.alias_revision_id, authorization.catalog_sha256))
        if view is None:
            raise GatewayRoutingError("authorized catalog snapshot is not active for this revision")
        target = authorization.target
        if isinstance(target, DirectTarget):
            root_pool = self._pool(view, target.pool_id)
            plan = model_execution_snapshot(
                view.catalog, authorization, root_pool, chains=view.chains, pools=view.pools
            )
            if deployment_id not in plan.deployment_ids:
                raise GatewayRoutingError(
                    "reasoning carrier deployment is not reachable in current authority"
                )
            stage = plan.stage_for_depth(plan.deployment_ids.index(deployment_id))
            if not stage_start_authorized(plan, stage):
                raise GatewayRoutingError(
                    "descendant reasoning start requires explicit authorization"
                )
            pools = (self._pool(view, stage.pool_id),)
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
        if not isinstance(target, DirectTarget):
            plan = model_execution_snapshot(
                view.catalog, authorization, pool, chains=view.chains, pools=view.pools
            )
        pinned_depth = plan.deployment_ids.index(deployment_id)
        if plan.model_stages:
            # A pin may lead its own segment, never resurrect preceding ancestors.
            pinned_stage = plan.stage_for_depth(pinned_depth).stage_index
            eligible = tuple(
                i
                for i in range(len(plan.deployment_ids))
                if plan.stage_for_depth(i).stage_index >= pinned_stage and i != pinned_depth
            )
        else:
            eligible = tuple(i for i in range(len(plan.deployment_ids)) if i != pinned_depth)
        snapshot = project_stage_selection(plan, (pinned_depth, *eligible))
        fallbacks = [view.deployments[d] for d in snapshot.deployment_ids[1:]]
        return GatewayRoute(
            snapshot=snapshot,
            deployment=deployment,
            fallback_deployments=tuple(fallbacks),
            route_reason=REASONING_CONTINUATION_ROUTE_REASON,
            fallback_reason=None,
            reasoning_pinned_deployment_id=deployment_id,
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
        try:
            snapshot = model_execution_snapshot(
                view.catalog, authorization, pool, chains=view.chains, pools=view.pools
            )
        except ValueError as exc:
            raise GatewayRoutingError(str(exc)) from exc
        deployments: list[ExactModelDeployment] = []
        for depth, deployment_id in enumerate(snapshot.deployment_ids):
            deployment = view.deployments.get(deployment_id)
            if (
                deployment is None
                or deployment.exact_model_id != snapshot.stage_for_depth(depth).exact_model_id
            ):
                raise GatewayRoutingError("frozen pool deployment identity is invalid")
            deployments.append(deployment)
        return GatewayRoute(
            snapshot=snapshot,
            deployment=deployments[0],
            fallback_deployments=tuple(deployments[1:]),
            route_reason=route_reason,
            fallback_reason=fallback_reason,
        )


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
