"""Last-good fallback loading for aliases whose active revision is unservable.

Serving-path support for the persistent-hydration fix: when an alias's active
revision pins an unservable snapshot (a missing file, or a same-version
self-inconsistent digest), the pod loads the most recent PRIOR revision that is
present and valid and serves it in place of a hard 503. The shared per-revision
load and readiness helpers live in ``lifecycle``; this module reuses them so a
prior revision is validated exactly like an active one.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from exp.common.models import NormalizedGatewayCatalog
from exp.runtime.gateway.contracts import ExecutionSnapshot
from exp.runtime.gateway.management import GatewayAliasView, GatewayManagement
from exp.runtime.gateway.project_activation import (
    ProjectActivationError,
    ProjectActivationRepository,
    require_project_activation_authority,
)
from exp.runtime.models import ModelConnectionError, RuntimeModelCatalog
from exp.runtime.models.credentials import ModelCredentialError
from exp.runtime.router.errors import RouterApplicationError
from exp.runtime.router.runtime import DecisionSink, RouterRuntime, RouterRuntimeIntegrityError

# Bound on how many prior revisions the last-good fallback walks when an alias's
# active revision pins an unservable snapshot. A dead pin's immediate
# predecessor is almost always the good one; the cap bounds cold-start cost.
FALLBACK_REVISION_WALK_LIMIT = 5


@dataclass(frozen=True)
class LoadedRevision:
    """One fully loaded and readiness-proven alias revision awaiting registration."""

    key: tuple[str, str]
    normalized: NormalizedGatewayCatalog
    runtime_catalog: RuntimeModelCatalog
    proof: ExecutionSnapshot
    activation: tuple[tuple[str, str, str], RouterRuntime] | None


def load_alias_revision(
    manager: GatewayManagement,
    alias: GatewayAliasView,
    *,
    environment: Mapping[str, str] | None,
    project_repository: ProjectActivationRepository | None,
    decision_sink: DecisionSink | None,
    exact_models: dict[tuple[str, str, str, str], str],
) -> LoadedRevision:
    """Load and prove readiness for one exact (alias, revision) view.

    Shared by the last-good fallback so a prior revision is validated through the
    same snapshot load and provider-idle readiness as an active one. Raises the
    same failures the active-revision loop already handles, so a caller can walk
    candidates and keep the first that loads.

    Raises:
        GatewayLifecycleError: The snapshot is unservable or the route is invalid.
        ModelConnectionError, ModelCredentialError, ProjectActivationError,
        RouterApplicationError, RouterRuntimeIntegrityError: Provider or project
            readiness could not be proven.
    """
    # Imported lazily to avoid a module import cycle: ``lifecycle`` imports this
    # module's fallback entry point, and these per-revision helpers live there.
    from exp.runtime.gateway.lifecycle import (
        GatewayLifecycleError,
        _direct_readiness,
        _load_snapshot,
        _project_readiness,
        _required,
        _required_revision,
    )

    revision_id, catalog_sha256 = _required_revision(alias)
    catalog, normalized = _load_snapshot(manager, alias)
    key = (revision_id, catalog_sha256)
    runtime_catalog = RuntimeModelCatalog(catalog, environment=environment)
    if alias.target_kind == "direct":
        proof = _direct_readiness(manager, alias, normalized, runtime_catalog)
        return LoadedRevision(key, normalized, runtime_catalog, proof, None)
    if alias.target_kind != "project":
        raise GatewayLifecycleError(f"alias {alias.alias_name!r} has an unknown target kind")
    if project_repository is None:
        raise GatewayLifecycleError(
            f"alias {alias.alias_name!r} project alias requires a project activation repository"
        )
    project_ref = _required(alias.project_ref, "project reference", alias)
    activation_ref = _required(alias.activation_ref, "activation reference", alias)
    activation = project_repository.load(
        project_ref, activation_ref, runtime_catalog=runtime_catalog
    )
    try:
        require_project_activation_authority(
            activation, project_ref=project_ref, activation_ref=activation_ref
        )
    except ProjectActivationError as exc:
        raise GatewayLifecycleError(str(exc)) from exc
    runtime = RouterRuntime.from_activation(
        activation, runtime_catalog, decision_sink=decision_sink
    )
    proof = _project_readiness(
        manager, alias, normalized, runtime, runtime_catalog, exact_models=exact_models
    )
    return LoadedRevision(
        key,
        normalized,
        runtime_catalog,
        proof,
        ((project_ref, activation_ref, catalog_sha256), runtime),
    )


def load_last_good_fallback(
    manager: GatewayManagement,
    alias: GatewayAliasView,
    *,
    environment: Mapping[str, str] | None,
    project_repository: ProjectActivationRepository | None,
    decision_sink: DecisionSink | None,
    exact_models: dict[tuple[str, str, str, str], str],
) -> LoadedRevision | None:
    """Return the newest prior revision that loads and proves ready, or ``None``.

    Walks the alias's prior revisions newest-first (bounded by
    ``FALLBACK_REVISION_WALK_LIMIT``) and returns the first one whose snapshot is
    present, matches its own pinned digest, and passes readiness. Topology-
    agnostic: a prior revision whose snapshot file is not on this node simply
    fails to load and is skipped; when none load the caller degrades the alias to
    a retryable-unavailable, never a permanent hard-fail.
    """
    from exp.runtime.gateway.lifecycle import GatewayLifecycleError

    for prior in manager.prior_alias_revisions(alias, limit=FALLBACK_REVISION_WALK_LIMIT):
        try:
            return load_alias_revision(
                manager,
                prior,
                environment=environment,
                project_repository=project_repository,
                decision_sink=decision_sink,
                exact_models=exact_models,
            )
        except (
            GatewayLifecycleError,
            ModelConnectionError,
            ModelCredentialError,
            ProjectActivationError,
            RouterApplicationError,
            RouterRuntimeIntegrityError,
        ):
            continue
    return None
