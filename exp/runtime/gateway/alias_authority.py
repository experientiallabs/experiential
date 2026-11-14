"""Hot-reloadable granted alias serving authority for the local gateway.

One validated generation of alias authority is published at a time. A reload
replaces it only after the candidate generation is fully loaded, while
recently retired revisions stay resolvable for in-flight requests.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import cast

from exp.common.models import NormalizedGatewayCatalog
from exp.runtime.gateway.contracts import GatewayTarget
from exp.runtime.gateway.execution_snapshot import ExecutionSnapshot
from exp.runtime.gateway.interfaces import ProjectTargetResolver
from exp.runtime.gateway.routing import (
    CatalogRouteResolver,
    GatewayRoutingError,
    ProjectSelector,
    RouterProjectTargetResolver,
    SelectionWorkerPool,
)
from exp.runtime.models import RuntimeModelCatalog

_RETIRED_REVISION_RETENTION_SECONDS = 600.0
# How long a reload that still left one authority unserved suppresses another
# reload for that same authority. Without it a granted alias that loads
# UNAVAILABLE with no last-good revision forces a full load under the reload
# lock on every request and model listing that names it.
UNSERVED_AUTHORITY_RELOAD_BACKOFF_SECONDS = 5.0

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _ServedFallback:
    """A last-good prior revision served in place of a dead active revision.

    When an alias's active revision pins an unservable snapshot, admission
    re-keys the request to this prior revision so routing, attribution, and
    prices all follow the revision actually served.
    """

    alias_revision_id: str
    catalog_sha256: str
    target: GatewayTarget


@dataclass(frozen=True)
class _AliasAuthorityState:
    """One fully validated generation of granted alias serving authority."""

    authorities: frozenset[tuple[str, str, str]]
    normalized_catalogs: Mapping[tuple[str, str], NormalizedGatewayCatalog]
    runtime_catalogs: Mapping[tuple[str, str], RuntimeModelCatalog]
    activations: Mapping[tuple[str, str, str], ProjectSelector]
    exact_models: Mapping[tuple[str, str, str, str], str]
    listing_pools: Mapping[tuple[str, str, str], str]
    proof: ExecutionSnapshot
    unavailable_aliases: tuple[tuple[str, str], ...] = ()
    # (alias, dead active revision_id) -> the last-good prior revision serving it. Keyed by
    # alias too, so a shared revision id never routes one alias's request to another alias's
    # fallback target. Empty unless an alias's active pinned snapshot was unservable at load.
    fallback_revisions: Mapping[tuple[str, str], _ServedFallback] = field(default_factory=dict)


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
        # Authorities a reload left unserved, keyed to the monotonic time of that reload.
        self._unserved: dict[tuple[str, str, str], float] = {}
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
            The current generation after at most one reload attempt. An
            authority the last reload already left unserved is not reloaded
            again until ``UNSERVED_AUTHORITY_RELOAD_BACKOFF_SECONDS`` pass or
            a reload changes the served authorities.

        Raises:
            GatewayRoutingError: The changed authority cannot form a valid generation;
                the previous generation keeps serving unchanged.
        """
        with self._lock:
            state = self._state
            if authority in state.authorities:
                return state
            now = self._monotonic()
            unserved_at = self._unserved.get(authority)
            if (
                unserved_at is not None
                and now - unserved_at < UNSERVED_AUTHORITY_RELOAD_BACKOFF_SECONDS
            ):
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
            if not _authority_served(self._state, authority):
                self._unserved = {
                    key: at
                    for key, at in self._unserved.items()
                    if now - at < UNSERVED_AUTHORITY_RELOAD_BACKOFF_SECONDS
                }
                self._unserved[authority] = now
            return self._state

    def _swap(self, previous: _AliasAuthorityState, loaded: _AliasAuthorityState) -> None:
        """Publish one validated generation while retaining recent retired revisions."""
        now = self._monotonic()
        if loaded.authorities != previous.authorities:
            # Changed authority may now serve what an earlier reload could not.
            self._unserved.clear()
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
            fallback_revisions=loaded.fallback_revisions,
        )
        self._routes.swap_catalogs(
            normalized,
            project_resolver=_project_resolver(
                activations, exact_models, selection_workers=self._selection_workers
            ),
            listing_pools=listing_pools,
        )
        self._state = merged


def _authority_served(state: _AliasAuthorityState, authority: tuple[str, str, str]) -> bool:
    """Return whether ``state`` serves ``authority`` directly, retained, or by fallback.

    Args:
        state: Published authority generation.
        authority: Alias, revision, and catalog digest triple.

    Returns:
        True when a request naming ``authority`` would be served without a reload.
    """
    alias, revision_id, catalog_sha256 = authority
    key = (revision_id, catalog_sha256)
    return (
        authority in state.authorities
        or (key in state.normalized_catalogs and key in state.runtime_catalogs)
        or (alias, revision_id) in state.fallback_revisions
    )


def _project_resolver(
    activations: Mapping[tuple[str, str, str], ProjectSelector],
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
