"""Local gateway component loading, hot-reloadable authority, and process ownership."""

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

from filelock import FileLock, Timeout

from exp.common.config import ARTIFACT_DIR
from exp.common.core.artifacts import sha256_json
from exp.common.models import (
    SNAPSHOT_SCHEMA_VERSION,
    ModelCatalog,
    NormalizedGatewayCatalog,
    is_foreign_snapshot,
    load_forward_compatible,
    normalize_gateway_catalog,
)
from exp.runtime.gateway.contracts import (
    AuthorizationSnapshot,
    DirectTarget,
    ExecutionSnapshot,
    GatewayApiSurface,
    GatewayRequest,
    ProjectTarget,
)
from exp.runtime.gateway.group_commit import GroupCommitAttemptLedger
from exp.runtime.gateway.interfaces import GatewayControlStore, ProjectTargetResolver
from exp.runtime.gateway.ledger import SQLiteAttemptLedger
from exp.runtime.gateway.management import GatewayAliasView, GatewayManagement
from exp.runtime.gateway.project_activation import (
    ProjectActivationError,
    ProjectActivationRepository,
    require_project_activation_authority,
)
from exp.runtime.gateway.routing import (
    CatalogRouteResolver,
    GatewayRoutingError,
    RouterProjectTargetResolver,
    SelectionWorkerPool,
)
from exp.runtime.gateway.sqlite.store import SQLiteGatewayStore
from exp.runtime.models import ModelConnectionError, RuntimeModelCatalog
from exp.runtime.models.credentials import MissingModelCredentialError, ModelCredentialError
from exp.runtime.router.errors import RouterApplicationError
from exp.runtime.router.runtime import DecisionSink, RouterRuntime, RouterRuntimeIntegrityError

_DEFAULT_GRACEFUL_TIMEOUT_SECONDS = 10.0
_RETIRED_REVISION_RETENTION_SECONDS = 600.0

_logger = logging.getLogger(__name__)


class GatewayLifecycleError(ValueError):
    """Local gateway configuration cannot form one ready execution snapshot."""


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
    unavailable_aliases: tuple[tuple[str, str], ...] = ()


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
        selection_workers: SelectionWorkerPool,
        retention_seconds: float = _RETIRED_REVISION_RETENTION_SECONDS,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        """Bind the validated startup generation and its reload seam.

        Args:
            loader: Builds one complete candidate generation from current authority.
            state: Fully validated startup generation.
            routes: Shared route resolver whose catalog index this reloader swaps.
            selection_workers: Process-wide selection lane reused by every generation.
            retention_seconds: How long retired revisions stay resolvable.
            monotonic: Monotonic clock used for retirement bookkeeping.
        """
        self._loader = loader
        self._routes = routes
        self._selection_workers = selection_workers
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
            except Exception as exc:  # noqa: BLE001 - any load failure keeps the previous generation.
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
            unavailable_aliases=loaded.unavailable_aliases,
        )
        self._routes.swap_catalogs(
            normalized,
            project_resolver=_project_resolver(
                activations, exact_models, selection_workers=self._selection_workers
            ),
            listing_pools=listing_pools,
        )
        self._state = merged


def _revision_served(
    state: _AliasAuthorityState,
    authorization: AuthorizationSnapshot,
) -> bool:
    """Return whether the authorized revision's catalogs are loaded in this generation.

    A freshly minted SQLite authorization can name a revision retired by a
    concurrent activation; its retained normalized and runtime catalogs keep it
    servable for the retention window, exactly like admitted in-flight work.

    Args:
        state: Current immutable authority generation.
        authorization: Frozen authority snapshot minted by SQLite.

    Returns:
        True when both catalogs for the authorized revision are resolvable.
    """
    key = (authorization.alias_revision_id, authorization.catalog_sha256)
    return key in state.normalized_catalogs and key in state.runtime_catalogs


@dataclass(frozen=True)
class _ReadyControlStore:
    """Filter public authority through the current hot-reloadable ready generation."""

    store: SQLiteGatewayStore
    reloader: _AliasAuthorityReloader

    def authenticate_key(self, *, raw_key: str) -> None:
        """Delegate authentication without consulting alias readiness."""
        self.store.authenticate_key(raw_key=raw_key)

    def authenticated_identity(self, *, raw_key: str) -> tuple[str, str]:
        """Delegate key-owner resolution without consulting alias readiness."""
        return self.store.authenticated_identity(raw_key=raw_key)

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
        app_referer: str | None = None,
        app_title: str | None = None,
    ) -> AuthorizationSnapshot:
        """Authorize only an alias revision this process can serve, reloading once on drift.

        A revision retired by a concurrent activation stays authorized while its retained
        catalogs can still serve it, so a request whose SQLite authority was minted an instant
        before the swap is pinned to its revision instead of being rejected at the swap boundary.
        """
        authorization = self.store.authorize_request(
            raw_key=raw_key,
            alias=alias,
            request=request,
            deadline_monotonic=deadline_monotonic,
            app_referer=app_referer,
            app_title=app_title,
        )
        authority = (
            authorization.alias,
            authorization.alias_revision_id,
            authorization.catalog_sha256,
        )
        state = self.reloader.state
        if authority not in state.authorities and not _revision_served(state, authorization):
            state = self.reloader.refresh_if_drifted(authority)
        if authority not in state.authorities and not _revision_served(state, authorization):
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
    """Loaded authority, accounting, and routing for the native gateway engine.

    The native engine's control-plane bridge uses these directly for
    admission and settlement over shared SQLite state and hot-reloadable
    authority generations.
    """

    manager: GatewayManagement
    store: GatewayControlStore
    ledger: SQLiteAttemptLedger
    write_ledger: GroupCommitAttemptLedger
    routes: CatalogRouteResolver
    reloader: _AliasAuthorityReloader
    selection_workers: SelectionWorkerPool
    reconciled_expired_requests: int
    reconciled_unknown_attempts: int
    # The local launch serves no asynchronous batch lane; hosted compositions
    # supply a BatchControlPlane here to enable /v1/batches.
    batches: object | None = None

    @property
    def runtime_catalogs(self) -> Mapping[tuple[str, str], RuntimeModelCatalog]:
        """Return the current generation's runtime catalogs."""
        return self.reloader.state.runtime_catalogs

    @property
    def accounting_healthy(self) -> bool:
        """Return whether the shared group-commit writer can still land writes.

        The native bridge's readiness callback reads this composition health
        surface; per-settlement losses latch in the bridge's own registry.
        """
        return not self.write_ledger.closed

    @property
    def readiness(self) -> tuple[ExecutionSnapshot, ...]:
        """Return the current generation's credential-free route proof."""
        return (self.reloader.state.proof,)

    @property
    def unavailable_aliases(self) -> tuple[tuple[str, str], ...]:
        """Return the current generation's failed aliases with their exact reasons."""
        return self.reloader.state.unavailable_aliases

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
        Hot-reloadable authority, ledger, routes, and startup proof.

    Raises:
        GatewayLifecycleError: No granted alias can form a complete local route.
    """
    manager = GatewayManagement(root)
    store = manager.require_initialized()
    manager.migrate_legacy_provider_connections()
    ledger = SQLiteAttemptLedger(manager.database_path)
    write_ledger = GroupCommitAttemptLedger(ledger)
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
    selection_workers = SelectionWorkerPool()
    routes = CatalogRouteResolver(
        state.normalized_catalogs,
        project_resolver=_project_resolver(
            state.activations, state.exact_models, selection_workers=selection_workers
        ),
        listing_pools=state.listing_pools,
    )
    reloader = _AliasAuthorityReloader(
        loader=loader,
        state=state,
        routes=routes,
        selection_workers=selection_workers,
    )
    return LocalGatewayComponents(
        manager=manager,
        store=_ReadyControlStore(store=store, reloader=reloader),
        ledger=ledger,
        write_ledger=write_ledger,
        routes=routes,
        reloader=reloader,
        selection_workers=selection_workers,
        reconciled_expired_requests=expired,
        reconciled_unknown_attempts=unknown,
    )


def _project_resolver(
    activations: Mapping[tuple[str, str, str], RouterRuntime],
    exact_models: Mapping[tuple[str, str, str, str], str],
    *,
    selection_workers: SelectionWorkerPool,
) -> ProjectTargetResolver | None:
    """Build one selection-only project bridge on the shared worker lane."""
    if not activations:
        return None
    return cast(
        ProjectTargetResolver,
        RouterProjectTargetResolver(activations, exact_models, selection_workers=selection_workers),
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
    missing_credential_variables: set[str] = set()

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
                if isinstance(exc, MissingModelCredentialError):
                    unavailable_aliases.append((alias.alias_name, exc.detail))
                    missing_credential_variables.add(exc.environment_variable)
                else:
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
            try:
                require_project_activation_authority(
                    activation,
                    project_ref=project_ref,
                    activation_ref=activation_ref,
                )
            except ProjectActivationError as exc:
                raise GatewayLifecycleError(str(exc)) from exc
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
            if isinstance(exc, MissingModelCredentialError):
                unavailable_aliases.append((alias.alias_name, exc.detail))
                missing_credential_variables.add(exc.environment_variable)
            else:
                unavailable_aliases.append((alias.alias_name, str(exc)))
            continue
        activations[(project_ref, activation_ref, catalog_sha256)] = runtime
        normalized_catalogs[key] = normalized
        runtime_catalogs[key] = runtime_catalog
        readiness.append(proof)

    if not readiness:
        unavailable = "; ".join(f"{name} ({why})" for name, why in sorted(unavailable_aliases))
        detail = f": {unavailable}" if unavailable else ""
        if missing_credential_variables:
            keys = " ".join(f"{name}=YOUR_API_KEY" for name in sorted(missing_credential_variables))
            remediation = f"; run '{keys} exp'"
        else:
            remediation = "; fix the listed provider configuration and rerun 'exp'"
        message = f"no granted active alias is locally available{detail}{remediation}"
        raise GatewayLifecycleError(message)

    return _AliasAuthorityState(
        authorities=frozenset(_authority_key(item) for item in readiness),
        normalized_catalogs=normalized_catalogs,
        runtime_catalogs=runtime_catalogs,
        activations=activations,
        exact_models=exact_models,
        listing_pools={
            _authority_key(item): item.authorization.target.pool_id
            for item in readiness
            if isinstance(item.authorization.target, DirectTarget)
        },
        proof=readiness[0],
        unavailable_aliases=tuple(sorted(unavailable_aliases)),
    )


def _authority_key(item: ExecutionSnapshot) -> tuple[str, str, str]:
    """Return the alias, revision, and catalog digest key of one readiness proof."""
    proof = item.authorization
    return (proof.alias, proof.alias_revision_id, proof.catalog_sha256)


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
        # Read forward-compatibly: a snapshot authored by a NEWER engine build
        # during a rolling deploy may carry fields this build does not know, and
        # rejecting it would hard-fail every request until the roll finished.
        # Unknown fields are dropped; every other validation stays strict.
        normalized, normalized_dropped = load_forward_compatible(
            NormalizedGatewayCatalog, snapshot.read_bytes()
        )
        authored_catalog, authored_dropped = load_forward_compatible(
            ModelCatalog, authored.read_bytes()
        )
    except (OSError, ValueError) as exc:
        raise GatewayLifecycleError(
            f"alias {alias.alias_name!r} has an unreadable catalog snapshot"
        ) from exc
    catalog_sha256 = _required(alias.catalog_sha256, "catalog digest", alias)
    # A cross-version skew (this build's schema differs from the snapshot's) is
    # served through this build's own tolerant view, keyed by the pinned digest,
    # instead of the byte-exact digest checks below: the local normalizer is not
    # expected to reproduce another build's bytes. When the versions agree the
    # checks stay strict, so same-version corruption is still caught.
    foreign = is_foreign_snapshot(normalized)
    if normalized_dropped or authored_dropped or foreign:
        _logger.warning(
            "gateway alias %r catalog snapshot is cross-version (schema_version=%d, this build=%d, "
            "dropped_fields=%s); serving this build's tolerant view keyed by the pinned digest",
            alias.alias_name,
            normalized.schema_version,
            SNAPSHOT_SCHEMA_VERSION,
            bool(normalized_dropped or authored_dropped),
        )
    if not foreign and normalized.identity_sha256() != catalog_sha256:
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
    if not foreign and normalize_gateway_catalog(catalog) != normalized:
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
