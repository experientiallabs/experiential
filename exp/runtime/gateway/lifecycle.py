"""Local gateway composition, readiness, usage routes, and process ownership."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import cast

from fastapi import FastAPI
from filelock import FileLock, Timeout

from exp.common.config import ARTIFACT_DIR
from exp.common.core.artifacts import sha256_json
from exp.common.models import (
    ModelCatalog,
    NormalizedGatewayCatalog,
    normalize_gateway_catalog,
)
from exp.runtime.gateway.composition import (
    GatewayLifecycleState,
    GatewayRuntime,
    GatewayRuntimeConfig,
    create_gateway_runtime,
)
from exp.runtime.gateway.contracts import (
    AuthorizationSnapshot,
    DirectTarget,
    ExecutionSnapshot,
    GatewayApiSurface,
    GatewayRequest,
    ProjectTarget,
)
from exp.runtime.gateway.execution import GatewayExecutor
from exp.runtime.gateway.interfaces import GatewayControlStore, ProjectTargetResolver
from exp.runtime.gateway.ledger import SQLiteAttemptLedger
from exp.runtime.gateway.management import GatewayAliasView, GatewayManagement
from exp.runtime.gateway.project_activation import (
    ProjectActivation,
    ProjectActivationError,
    ProjectActivationRepository,
    require_project_activation_authority,
)
from exp.runtime.gateway.routing import (
    CatalogRouteResolver,
    GatewayRoutingError,
    RouterProjectTargetResolver,
)
from exp.runtime.gateway.service import GatewayService
from exp.runtime.gateway.sqlite.store import SQLiteGatewayStore, SystemGatewayClock
from exp.runtime.gateway.usage import read_usage_report
from exp.runtime.models import ModelConnectionError, RuntimeModelCatalog
from exp.runtime.models.credentials import ModelCredentialError
from exp.runtime.openai_protocol.state import ResponseContinuationStore, ResponseReplayStore
from exp.runtime.router.errors import RouterApplicationError
from exp.runtime.router.runtime import DecisionSink, RouterRuntime, RouterRuntimeIntegrityError

_DEFAULT_GRACEFUL_TIMEOUT_SECONDS = 10.0
_RETIRED_REVISION_RETENTION_SECONDS = 600.0

_logger = logging.getLogger(__name__)


class GatewayLifecycleError(ValueError):
    """Local gateway configuration cannot form one ready execution snapshot."""


@dataclass(frozen=True)
class LocalGatewayRuntime:
    """Fully composed local service and its loopback application."""

    runtime: GatewayRuntime
    reconciled_expired_requests: int
    reconciled_unknown_attempts: int

    @property
    def app(self) -> FastAPI:
        """Return the shared managed FastAPI application."""
        return self.runtime.app

    @property
    def service(self) -> GatewayService:
        """Return the shared injected gateway service."""
        return self.runtime.service

    @property
    def state(self) -> GatewayLifecycleState:
        """Return the shared process-local readiness state."""
        return self.runtime.state

    async def preflight(self) -> ExecutionSnapshot:
        """Preflight the shared gateway composition seam."""
        return await self.runtime.preflight()

    async def readiness(self) -> bool:
        """Return current shared runtime readiness."""
        return await self.runtime.readiness()

    async def drain(self, *, timeout_seconds: float | None = None) -> bool:
        """Drain the shared runtime within the selected bound."""
        return await self.runtime.drain(timeout_seconds=timeout_seconds)

    async def shutdown(self) -> bool:
        """Shut down the shared runtime within its local bound."""
        return await self.runtime.shutdown()


@dataclass(frozen=True)
class _AliasAuthorityState:
    """One fully validated generation of granted alias serving authority."""

    authorities: frozenset[tuple[str, str, str]]
    normalized_catalogs: Mapping[tuple[str, str], NormalizedGatewayCatalog]
    runtime_catalogs: Mapping[tuple[str, str], RuntimeModelCatalog]
    activations: Mapping[tuple[str, str, str], RouterRuntime]
    exact_models: Mapping[tuple[str, str, str, str], str]
    listing_pools: Mapping[tuple[str, str, str], str]
    proof: ExecutionSnapshot


class _AliasAuthorityReloader:
    """Swap fully validated authority generations while retaining in-flight revisions.

    A candidate generation is loaded and digest-verified completely before it is
    published, so requests never observe a half-loaded catalog. Revisions retired
    by a swap stay resolvable for a bounded retention window so requests that
    authorized against them finish on the revision they started with.
    """

    def __init__(
        self,
        *,
        loader: Callable[[], _AliasAuthorityState],
        state: _AliasAuthorityState,
        routes: CatalogRouteResolver,
        executor: GatewayExecutor,
        retention_seconds: float = _RETIRED_REVISION_RETENTION_SECONDS,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        """Bind the validated startup generation and its reload seam.

        Args:
            loader: Builds one complete candidate generation from current authority.
            state: Fully validated startup generation.
            routes: Shared route resolver whose catalog index this reloader swaps.
            executor: Shared executor whose runtime catalogs this reloader swaps.
            retention_seconds: How long retired revisions stay resolvable.
            monotonic: Monotonic clock used for retirement bookkeeping.
        """
        self._loader = loader
        self._routes = routes
        self._executor = executor
        self._retention_seconds = retention_seconds
        self._monotonic = monotonic
        self._lock = threading.Lock()
        self._retired: dict[tuple[str, str], float] = {}
        self._state = state

    @property
    def state(self) -> _AliasAuthorityState:
        """Return the current immutable authority generation."""
        return self._state

    def refresh_if_drifted(self, authority: tuple[str, str, str]) -> _AliasAuthorityState:
        """Reload authority once when one authorized revision is not currently served.

        Args:
            authority: Alias, revision, and catalog digest triple from SQLite authority.

        Returns:
            The current generation after at most one reload attempt.

        Raises:
            GatewayRoutingError: The changed authority cannot form a valid generation;
                the previous generation keeps serving unchanged.
        """
        with self._lock:
            state = self._state
            if authority in state.authorities:
                return state
            try:
                loaded = self._loader()
            except GatewayLifecycleError as exc:
                _logger.warning("gateway alias authority reload failed: %s", exc)
                raise GatewayRoutingError(
                    "alias authority changed but the new revision failed to load; "
                    "previously ready revisions keep serving; "
                    f"fix the alias configuration and retry: {exc}"
                ) from exc
            self._swap(state, loaded)
            return self._state

    def _swap(self, previous: _AliasAuthorityState, loaded: _AliasAuthorityState) -> None:
        """Publish one validated generation while retaining recent retired revisions."""
        now = self._monotonic()
        for key in previous.normalized_catalogs:
            if key not in loaded.normalized_catalogs:
                self._retired.setdefault(key, now)
        for key in tuple(self._retired):
            if key in loaded.normalized_catalogs:
                del self._retired[key]
        self._retired = {
            key: retired_at
            for key, retired_at in self._retired.items()
            if now - retired_at <= self._retention_seconds
        }
        retained = tuple(key for key in self._retired if key in previous.normalized_catalogs)
        normalized = {
            **{key: previous.normalized_catalogs[key] for key in retained},
            **dict(loaded.normalized_catalogs),
        }
        runtime = {
            **{key: previous.runtime_catalogs[key] for key in retained},
            **dict(loaded.runtime_catalogs),
        }
        digests = {digest for _revision, digest in normalized}
        exact_models = {
            key: value
            for key, value in {**dict(previous.exact_models), **dict(loaded.exact_models)}.items()
            if key[2] in digests
        }
        active_projects = {(key[0], key[1], key[2]) for key in exact_models}
        activations = {
            key: value
            for key, value in {**dict(previous.activations), **dict(loaded.activations)}.items()
            if key in active_projects
        }
        listing_pools = {
            key: value
            for key, value in {
                **dict(previous.listing_pools),
                **dict(loaded.listing_pools),
            }.items()
            if (key[1], key[2]) in normalized
        }
        merged = _AliasAuthorityState(
            authorities=loaded.authorities,
            normalized_catalogs=normalized,
            runtime_catalogs=runtime,
            activations=activations,
            exact_models=exact_models,
            listing_pools=listing_pools,
            proof=loaded.proof,
        )
        self._executor.swap_catalogs(runtime)
        self._routes.swap_catalogs(
            normalized,
            project_resolver=_project_resolver(activations, exact_models),
            listing_pools=listing_pools,
        )
        self._state = merged


@dataclass(frozen=True)
class _ReadyControlStore:
    """Filter public authority through the current hot-reloadable ready generation."""

    store: SQLiteGatewayStore
    reloader: _AliasAuthorityReloader

    def authenticate_key(self, *, raw_key: str) -> None:
        """Delegate authentication without consulting alias readiness."""
        self.store.authenticate_key(raw_key=raw_key)

    def granted_aliases(self, *, raw_key: str) -> tuple[str, ...]:
        """Return granted aliases whose active revision is currently served."""
        return tuple(
            alias for alias, _revision, _digest in self.granted_alias_authorities(raw_key=raw_key)
        )

    def granted_alias_authorities(self, *, raw_key: str) -> tuple[tuple[str, str, str], ...]:
        """Return granted authority triples whose active revision is currently served."""
        granted = tuple(self.store.granted_alias_authorities(raw_key=raw_key))
        state = self.reloader.state
        drifted = next((item for item in granted if item not in state.authorities), None)
        if drifted is not None:
            try:
                state = self.reloader.refresh_if_drifted(drifted)
            except GatewayRoutingError:
                state = self.reloader.state
        return tuple(item for item in granted if item in state.authorities)

    def authorize_request(
        self,
        *,
        raw_key: str,
        alias: str,
        request: GatewayRequest,
        deadline_monotonic: float,
    ) -> AuthorizationSnapshot:
        """Authorize only an alias revision proven ready, reloading once on drift."""
        authorization = self.store.authorize_request(
            raw_key=raw_key,
            alias=alias,
            request=request,
            deadline_monotonic=deadline_monotonic,
        )
        authority = (
            authorization.alias,
            authorization.alias_revision_id,
            authorization.catalog_sha256,
        )
        state = self.reloader.state
        if authority not in state.authorities:
            state = self.reloader.refresh_if_drifted(authority)
        if authority not in state.authorities:
            raise GatewayRoutingError("authorized alias revision is unavailable in this process")
        return authorization


@contextmanager
def gateway_instance_lock(root: Path, *, port: int) -> Iterator[None]:
    """Hold the single live gateway owner lock for one EXP root.

    Args:
        root: EXP root whose gateway database and snapshots are served.
        port: Requested loopback port, included only in actionable diagnostics.

    Yields:
        None while this process exclusively owns the local gateway.

    Raises:
        GatewayLifecycleError: Another process currently owns the root.
    """
    state_dir = root / "gateway"
    state_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    lock = FileLock(state_dir / "run.lock", timeout=0, mode=0o600)
    try:
        lock.acquire()
    except Timeout:
        raise GatewayLifecycleError(
            f"another gateway process already owns {root} (requested port {port})"
        ) from None
    try:
        yield
    finally:
        lock.release()


@dataclass(frozen=True)
class LocalGatewayComponents:
    """Loaded authority, accounting, and routing shared by both gateway engines.

    The Python engine composes these into a ``GatewayRuntime``; the Rust
    engine's control-plane bridge uses them directly for admission and
    settlement over the same SQLite state and the same hot-reloadable
    authority generations.
    """

    manager: GatewayManagement
    store: GatewayControlStore
    ledger: SQLiteAttemptLedger
    routes: CatalogRouteResolver
    executor: GatewayExecutor
    reloader: _AliasAuthorityReloader
    reconciled_expired_requests: int
    reconciled_unknown_attempts: int

    @property
    def runtime_catalogs(self) -> Mapping[tuple[str, str], RuntimeModelCatalog]:
        """Return the current generation's runtime catalogs."""
        return self.reloader.state.runtime_catalogs

    @property
    def readiness(self) -> tuple[ExecutionSnapshot, ...]:
        """Return the current generation's credential-free route proof."""
        return (self.reloader.state.proof,)

    @property
    def organization_id(self) -> str:
        """Return the single local organization identity."""
        return self.manager.organization_id


def load_gateway_components(
    root: Path = Path(ARTIFACT_DIR),
    *,
    environment: Mapping[str, str] | None = None,
    project_repository: ProjectActivationRepository | None = None,
    decision_sink: DecisionSink | None = None,
    only_aliases: frozenset[str] | None = None,
) -> LocalGatewayComponents:
    """Load granted active aliases into engine-neutral gateway components.

    Args:
        root: Initialized EXP root. Defaults to the local ``.exp`` root.
        environment: Optional provider credential mapping used by tests.
        project_repository: Repository for verified immutable project activations.
        decision_sink: Optional aggregate-safe recorder for served project selections.
        only_aliases: Optional exact public aliases to expose.

    Returns:
        Hot-reloadable authority, ledger, routes, executor, and startup proof.

    Raises:
        GatewayLifecycleError: No granted alias can form a complete local route.
    """
    manager = GatewayManagement(root)
    store = manager.require_initialized()
    manager.migrate_legacy_provider_connections()
    ledger = SQLiteAttemptLedger(manager.database_path)
    expired, unknown = ledger.reconcile_crashed_requests(cleanup_grace=timedelta(seconds=5))

    def loader() -> _AliasAuthorityState:
        """Build one complete validated generation from current granted authority."""
        return _load_alias_state(
            manager,
            environment=environment,
            project_repository=project_repository,
            decision_sink=decision_sink,
            only_aliases=only_aliases,
        )

    state = loader()
    routes = CatalogRouteResolver(
        state.normalized_catalogs,
        project_resolver=_project_resolver(state.activations, state.exact_models),
        listing_pools=state.listing_pools,
    )
    executor = GatewayExecutor(state.runtime_catalogs, ledger)
    reloader = _AliasAuthorityReloader(
        loader=loader,
        state=state,
        routes=routes,
        executor=executor,
    )
    return LocalGatewayComponents(
        manager=manager,
        store=_ReadyControlStore(store=store, reloader=reloader),
        ledger=ledger,
        routes=routes,
        executor=executor,
        reloader=reloader,
        reconciled_expired_requests=expired,
        reconciled_unknown_attempts=unknown,
    )


def compose_local_gateway(
    components: LocalGatewayComponents,
    *,
    graceful_timeout_seconds: float = _DEFAULT_GRACEFUL_TIMEOUT_SECONDS,
    replay: ResponseReplayStore | None = None,
    continuations: ResponseContinuationStore | None = None,
) -> LocalGatewayRuntime:
    """Compose the loopback application over already loaded components.

    Args:
        components: Loaded authority, ledger, routes, executor, and reloader.
        graceful_timeout_seconds: Shutdown drain bound. Defaults to ten seconds.
        replay: Optional shared Chat and Responses replay state.
        continuations: Optional shared Responses continuation state.

    Returns:
        Composed service, application, health state, and recovery counts.

    Raises:
        GatewayLifecycleError: The drain bound is not positive.
    """
    if graceful_timeout_seconds <= 0:
        raise GatewayLifecycleError("graceful timeout must be positive")

    async def readiness_probe() -> ExecutionSnapshot:
        """Return the current generation's credential and route proof."""
        return components.reloader.state.proof

    runtime = create_gateway_runtime(
        config=GatewayRuntimeConfig(
            graceful_timeout_seconds=graceful_timeout_seconds,
            title="EXP local gateway",
        ),
        authority=components.store,
        ledger=components.ledger,
        routes=components.routes,
        executor=components.executor,
        clock=SystemGatewayClock(),
        readiness=readiness_probe,
        usage=lambda: read_usage_report(
            components.ledger,
            organization_id=components.organization_id,
        ),
        replay=replay,
        continuations=continuations,
    )
    return LocalGatewayRuntime(
        runtime=runtime,
        reconciled_expired_requests=components.reconciled_expired_requests,
        reconciled_unknown_attempts=components.reconciled_unknown_attempts,
    )


def load_local_gateway(
    root: Path = Path(ARTIFACT_DIR),
    *,
    graceful_timeout_seconds: float = _DEFAULT_GRACEFUL_TIMEOUT_SECONDS,
    environment: Mapping[str, str] | None = None,
    project_repository: ProjectActivationRepository | None = None,
    decision_sink: DecisionSink | None = None,
    replay: ResponseReplayStore | None = None,
    continuations: ResponseContinuationStore | None = None,
    only_aliases: frozenset[str] | None = None,
) -> LocalGatewayRuntime:
    """Load all granted active aliases and compose the loopback application.

    Args:
        root: Initialized EXP root. Defaults to the local ``.exp`` root.
        graceful_timeout_seconds: Shutdown drain bound. Defaults to ten seconds.
        environment: Optional provider credential mapping used by tests.
        project_repository: Repository for verified immutable project activations.
        decision_sink: Optional aggregate-safe recorder for served project selections.
        replay: Optional shared Chat and Responses replay state.
        continuations: Optional shared Responses continuation state.
        only_aliases: Optional exact public aliases to expose from the shared application.

    Returns:
        Composed service, application, health state, and recovery counts.

    Raises:
        GatewayLifecycleError: No granted alias can form a complete local route.
    """
    if graceful_timeout_seconds <= 0:
        raise GatewayLifecycleError("graceful timeout must be positive")
    components = load_gateway_components(
        root,
        environment=environment,
        project_repository=project_repository,
        decision_sink=decision_sink,
        only_aliases=only_aliases,
    )
    return compose_local_gateway(
        components,
        graceful_timeout_seconds=graceful_timeout_seconds,
        replay=replay,
        continuations=continuations,
    )


def _project_resolver(
    activations: Mapping[tuple[str, str, str], RouterRuntime],
    exact_models: Mapping[tuple[str, str, str, str], str],
) -> ProjectTargetResolver | None:
    """Build one selection-only project bridge when any activation is loaded."""
    if not activations:
        return None
    return cast(
        ProjectTargetResolver,
        RouterProjectTargetResolver(activations, exact_models),
    )


def _load_alias_state(
    manager: GatewayManagement,
    *,
    environment: Mapping[str, str] | None,
    project_repository: ProjectActivationRepository | None,
    decision_sink: DecisionSink | None,
    only_aliases: frozenset[str] | None,
) -> _AliasAuthorityState:
    """Load and validate every granted active alias into one complete generation.

    Args:
        manager: Initialized gateway management over SQLite authority.
        environment: Optional provider credential mapping used by tests.
        project_repository: Repository for verified immutable project activations.
        decision_sink: Optional aggregate-safe recorder for served project selections.
        only_aliases: Optional exact public aliases to expose.

    Returns:
        Fully validated authority generation ready for atomic publication.

    Raises:
        GatewayLifecycleError: No granted alias can form a complete local route.
    """
    aliases = _granted_active_aliases(manager)
    if only_aliases is not None:
        aliases = tuple(item for item in aliases if item.alias_name in only_aliases)
    if not aliases:
        raise GatewayLifecycleError(
            "gateway has no granted active alias; create an identity, alias, and grant first"
        )

    normalized_catalogs: dict[tuple[str, str], NormalizedGatewayCatalog] = {}
    runtime_catalogs: dict[tuple[str, str], RuntimeModelCatalog] = {}
    activations: dict[tuple[str, str, str], RouterRuntime] = {}
    exact_models: dict[tuple[str, str, str, str], str] = {}
    readiness: list[ExecutionSnapshot] = []
    unavailable_aliases: list[tuple[str, str]] = []

    for alias in aliases:
        try:
            revision_id, catalog_sha256 = _required_revision(alias)
            catalog, normalized = _load_snapshot(manager, alias)
        except GatewayLifecycleError as exc:
            unavailable_aliases.append((alias.alias_name, str(exc)))
            continue
        key = (revision_id, catalog_sha256)
        runtime_catalog = RuntimeModelCatalog(catalog, environment=environment)
        if alias.target_kind == "direct":
            try:
                proof = _direct_readiness(manager, alias, normalized, runtime_catalog)
            except (GatewayLifecycleError, ModelConnectionError, ModelCredentialError) as exc:
                unavailable_aliases.append((alias.alias_name, str(exc)))
                continue
            normalized_catalogs[key] = normalized
            runtime_catalogs[key] = runtime_catalog
            readiness.append(proof)
            continue
        if alias.target_kind != "project":
            unavailable_aliases.append((alias.alias_name, "unknown target kind"))
            continue
        if project_repository is None:
            unavailable_aliases.append(
                (alias.alias_name, "project alias requires a project activation repository")
            )
            continue
        try:
            project_ref = _required(alias.project_ref, "project reference", alias)
            activation_ref = _required(alias.activation_ref, "activation reference", alias)
            activation = project_repository.load(
                project_ref,
                activation_ref,
                runtime_catalog=runtime_catalog,
            )
            _require_activation_authority(
                activation,
                project_ref=project_ref,
                activation_ref=activation_ref,
            )
            runtime = RouterRuntime.from_activation(
                activation,
                runtime_catalog,
                decision_sink=decision_sink,
            )
            proof = _project_readiness(
                manager,
                alias,
                normalized,
                runtime,
                runtime_catalog,
                exact_models=exact_models,
            )
        except RouterApplicationError as exc:
            if not _caused_by_connection_error(exc):
                raise
            unavailable_aliases.append((alias.alias_name, str(exc)))
            continue
        except (
            GatewayLifecycleError,
            ModelConnectionError,
            ModelCredentialError,
            ProjectActivationError,
            RouterRuntimeIntegrityError,
        ) as exc:
            unavailable_aliases.append((alias.alias_name, str(exc)))
            continue
        activations[(project_ref, activation_ref, catalog_sha256)] = runtime
        normalized_catalogs[key] = normalized
        runtime_catalogs[key] = runtime_catalog
        readiness.append(proof)

    if not readiness:
        unavailable = "; ".join(
            f"{alias_name} ({reason})" for alias_name, reason in sorted(unavailable_aliases)
        )
        detail = f": {unavailable}" if unavailable else ""
        raise GatewayLifecycleError(
            "no granted active alias is locally available"
            f"{detail}; fix the listed provider configuration and rerun 'exp run'"
        )

    return _AliasAuthorityState(
        authorities=frozenset(
            (
                item.authorization.alias,
                item.authorization.alias_revision_id,
                item.authorization.catalog_sha256,
            )
            for item in readiness
        ),
        normalized_catalogs=normalized_catalogs,
        runtime_catalogs=runtime_catalogs,
        activations=activations,
        exact_models=exact_models,
        listing_pools={
            (
                item.authorization.alias,
                item.authorization.alias_revision_id,
                item.authorization.catalog_sha256,
            ): item.authorization.target.pool_id
            for item in readiness
            if isinstance(item.authorization.target, DirectTarget)
        },
        proof=readiness[0],
    )


def _granted_active_aliases(manager: GatewayManagement) -> tuple[GatewayAliasView, ...]:
    """Return active aliases that have at least one current identity grant."""
    active_identities = {
        identity.identity_id for identity in manager.identities() if identity.active
    }
    granted = {
        grant.alias_id for grant in manager.grants() if grant.identity_id in active_identities
    }
    return tuple(
        alias
        for alias in manager.aliases()
        if alias.active and alias.revision_id is not None and alias.alias_id in granted
    )


def _load_snapshot(
    manager: GatewayManagement,
    alias: GatewayAliasView,
) -> tuple[ModelCatalog, NormalizedGatewayCatalog]:
    """Load and cross-check one pinned normalized and authored catalog pair."""
    snapshot_ref = _required(alias.snapshot_ref, "snapshot reference", alias)
    snapshot = (manager.state_dir / snapshot_ref).resolve()
    state_dir = manager.state_dir.resolve()
    if not snapshot.is_relative_to(state_dir):
        raise GatewayLifecycleError("catalog snapshot reference escapes gateway state")
    authored = snapshot.with_suffix(".models.json")
    try:
        normalized = NormalizedGatewayCatalog.model_validate_json(snapshot.read_bytes())
        authored_catalog = ModelCatalog.model_validate_json(authored.read_bytes())
    except (OSError, ValueError) as exc:
        raise GatewayLifecycleError(
            f"alias {alias.alias_name!r} has an unreadable catalog snapshot"
        ) from exc
    catalog_sha256 = _required(alias.catalog_sha256, "catalog digest", alias)
    if normalized.identity_sha256() != catalog_sha256:
        raise GatewayLifecycleError(f"alias {alias.alias_name!r} catalog digest does not match")
    revision_id, _digest = _required_revision(alias)
    authorities = manager.ensure_alias_provider_bindings(
        alias_id=alias.alias_id,
        alias_revision_id=revision_id,
        catalog=authored_catalog,
    )
    connections = {item.connection_id: item.config for item in authorities}
    if set(connections) != set(authored_catalog.connections):
        raise GatewayLifecycleError(
            f"alias {alias.alias_name!r} provider bindings differ from its snapshot"
        )
    catalog = authored_catalog.model_copy(update={"connections": connections})
    if normalize_gateway_catalog(catalog) != normalized:
        raise GatewayLifecycleError(
            f"alias {alias.alias_name!r} authored catalog differs from normalized authority"
        )
    return catalog, normalized


def _direct_readiness(
    manager: GatewayManagement,
    alias: GatewayAliasView,
    catalog: NormalizedGatewayCatalog,
    runtime_catalog: RuntimeModelCatalog,
) -> ExecutionSnapshot:
    """Validate one ordered direct pool and return provider-idle readiness proof."""
    pool_id = _required(alias.pool_id, "pool ID", alias)
    pools = tuple(pool for pool in catalog.pools if pool.pool_id == pool_id)
    if len(pools) != 1:
        raise GatewayLifecycleError(f"alias {alias.alias_name!r} pool is unavailable")
    deployments_by_id = {item.deployment_id: item for item in catalog.deployments}
    for deployment_id in pools[0].deployment_ids:
        deployment = deployments_by_id.get(deployment_id)
        if deployment is None:
            raise GatewayLifecycleError(f"alias {alias.alias_name!r} deployment is unavailable")
        runtime_catalog.resolve(deployment.source_alias)
    authorization = _readiness_authorization(
        manager,
        alias,
        DirectTarget(pool_id=pool_id),
    )
    return ExecutionSnapshot(
        authorization=authorization,
        exact_model_id=pools[0].exact_model_id,
        pool_id=pool_id,
        deployment_ids=pools[0].deployment_ids,
    )


def _project_readiness(
    manager: GatewayManagement,
    alias: GatewayAliasView,
    catalog: NormalizedGatewayCatalog,
    runtime: RouterRuntime,
    runtime_catalog: RuntimeModelCatalog,
    *,
    exact_models: dict[tuple[str, str, str, str], str],
) -> ExecutionSnapshot:
    """Validate project candidate pools and return provider-idle readiness proof."""
    project_ref = _required(alias.project_ref, "project reference", alias)
    activation_ref = _required(alias.activation_ref, "activation reference", alias)
    catalog_sha256 = _required(alias.catalog_sha256, "catalog digest", alias)
    first_pool = None
    for candidate in runtime.policy.candidates:
        deployments = tuple(
            item for item in catalog.deployments if item.source_alias == candidate.alias
        )
        if len(deployments) != 1:
            raise GatewayLifecycleError(
                f"project alias {alias.alias_name!r} candidate {candidate.alias!r} "
                "does not name one deployment"
            )
        pools = tuple(
            item
            for item in catalog.pools
            if item.exact_model_id == deployments[0].exact_model_id
            and deployments[0].deployment_id in item.deployment_ids
        )
        if len(pools) != 1:
            raise GatewayLifecycleError(
                f"project alias {alias.alias_name!r} candidate {candidate.alias!r} "
                "does not name one unambiguous certified pool"
            )
        deployments_by_id = {item.deployment_id: item for item in catalog.deployments}
        for deployment_id in pools[0].deployment_ids:
            sibling = deployments_by_id.get(deployment_id)
            if sibling is None:
                raise GatewayLifecycleError(
                    f"project alias {alias.alias_name!r} pool deployment is unavailable"
                )
            runtime_catalog.resolve(sibling.source_alias)
        exact_models[(project_ref, activation_ref, catalog_sha256, candidate.alias)] = deployments[
            0
        ].exact_model_id
        if first_pool is None:
            first_pool = pools[0]
    if first_pool is None:
        raise GatewayLifecycleError(f"project alias {alias.alias_name!r} has no candidates")
    authorization = _readiness_authorization(
        manager,
        alias,
        ProjectTarget(
            project_ref=project_ref,
            activation_ref=activation_ref,
            catalog_sha256=catalog_sha256,
        ),
    )
    return ExecutionSnapshot(
        authorization=authorization,
        exact_model_id=first_pool.exact_model_id,
        pool_id=first_pool.pool_id,
        deployment_ids=first_pool.deployment_ids,
    )


def _caused_by_connection_error(exception: BaseException) -> bool:
    """Return whether a project activation failed only at local client construction."""
    current: BaseException | None = exception
    while current is not None:
        if isinstance(current, (ModelConnectionError, ModelCredentialError)):
            return True
        current = current.__cause__
    return False


def _readiness_authorization(
    manager: GatewayManagement,
    alias: GatewayAliasView,
    target: DirectTarget | ProjectTarget,
) -> AuthorizationSnapshot:
    """Build a non-dispatchable content-free proof for service preflight."""
    revision_id, catalog_sha256 = _required_revision(alias)
    return AuthorizationSnapshot(
        request_id="readiness-probe",
        organization_id=manager.organization_id,
        identity_id="readiness-probe",
        virtual_key_id="readiness-probe",
        alias=alias.alias_name,
        alias_revision_id=revision_id,
        target=target,
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        catalog_sha256=catalog_sha256,
        canonical_request_sha256=sha256_json(
            {"kind": "gateway-readiness-v1", "alias_revision_id": revision_id}
        ),
        refusal_failover=alias.refusal_failover,
        deadline_monotonic=time.monotonic() + 30,
    )


def _required_revision(alias: GatewayAliasView) -> tuple[str, str]:
    """Return required alias revision and catalog digest values."""
    return (
        _required(alias.revision_id, "revision ID", alias),
        _required(alias.catalog_sha256, "catalog digest", alias),
    )


def _required(value: str | None, name: str, alias: GatewayAliasView) -> str:
    """Return one required active-alias field or fail with safe context."""
    if value is None:
        raise GatewayLifecycleError(f"alias {alias.alias_name!r} is missing {name}")
    return value


def _require_activation_authority(
    activation: ProjectActivation,
    *,
    project_ref: str,
    activation_ref: str,
) -> None:
    """Require repository output to match the exact authorized project target."""
    try:
        require_project_activation_authority(
            activation,
            project_ref=project_ref,
            activation_ref=activation_ref,
        )
    except ProjectActivationError as exc:
        raise GatewayLifecycleError(str(exc)) from exc
