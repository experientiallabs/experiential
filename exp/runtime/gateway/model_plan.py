"""Freeze ordered cross-model plans while preserving exact-pool identity."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from exp.common.models.gateway_catalog import (
    ExactModelDeployment,
    ExactModelPool,
    NormalizedGatewayCatalog,
    is_foreign_snapshot,
)
from exp.common.models.gateway_chains import (
    GatewayModelChain,
    ModelExecutionStage,
    expand_model_chain,
)
from exp.runtime.gateway.contracts import AuthorizationSnapshot, DirectTarget, ExecutionSnapshot


def model_execution_snapshot(
    catalog: NormalizedGatewayCatalog,
    authorization: AuthorizationSnapshot,
    root_pool: ExactModelPool,
    *,
    chains: Mapping[str, GatewayModelChain] | None = None,
    pools: Mapping[str, ExactModelPool] | None = None,
) -> ExecutionSnapshot:
    """Resolve one authorized catalog graph to a bounded immutable leaf cursor.

    Only conversational direct targets enter explicit model references. Project
    selections and other surfaces stay within their selected exact-model pool.
    Optional chain and pool indexes must belong to this same frozen catalog view.
    """
    direct = ExecutionSnapshot(
        authorization=authorization,
        exact_model_id=root_pool.exact_model_id,
        pool_id=root_pool.pool_id,
        deployment_ids=root_pool.deployment_ids,
        failover_mode=root_pool.failover_mode,
        throttle_cache_threshold=root_pool.throttle_cache_threshold,
        throttle_redial=root_pool.throttle_redial,
    )
    if not isinstance(authorization.target, DirectTarget) or authorization.surface not in (
        "chat_completions",
        "responses",
        "messages",
    ):
        return direct
    chain = next((c for c in catalog.model_chains if c.model_id == root_pool.exact_model_id), None)
    if chain is None:
        return direct
    if chain.pool_id != root_pool.pool_id:
        raise ValueError("authorized chain root does not match its exact pool")
    if chains is None:
        chains = catalog.chains_by_model()
    expanded = expand_model_chain(root_pool.exact_model_id, chains)
    if pools is None:
        pools = {pool.pool_id: pool for pool in catalog.pools}
    stages = tuple(
        ModelExecutionStage(
            stage_index=index,
            exact_model_id=segment.model_id,
            pool_id=segment.pool_id,
            deployment_ids=segment.deployment_ids,
            failover_mode=pools[segment.pool_id].failover_mode
            if policy is None
            else policy.failover_mode,
            throttle_cache_threshold=pools[segment.pool_id].throttle_cache_threshold
            if policy is None
            else policy.throttle_cache_threshold,
            throttle_redial=pools[segment.pool_id].throttle_redial
            if policy is None
            else policy.throttle_redial,
            chain_revision=segment.chain_revision,
            ancestry=segment.ancestry,
            rung_positions=segment.rung_positions,
        )
        for index, segment in enumerate(expanded.segments)
        for policy in (chains[segment.model_id].policy,)
    )
    if not stages:
        raise ValueError("authorized model chain is unavailable; repair or reset the chain")
    return ExecutionSnapshot(
        authorization=authorization,
        exact_model_id=direct.exact_model_id,
        pool_id=direct.pool_id,
        deployment_ids=tuple(d for stage in stages for d in stage.deployment_ids),
        failover_mode=direct.failover_mode,
        throttle_cache_threshold=direct.throttle_cache_threshold,
        throttle_redial=direct.throttle_redial,
        model_stages=stages,
        traversal_events=expanded.events,
    )


def stage_start_authorized(snapshot: ExecutionSnapshot, stage: ModelExecutionStage) -> bool:
    """Permit an initial hint only for the canonical root or an explicitly authorized child.

    This gate applies before projecting an untrusted hint, not to forward
    execution after the request has already entered its root waterfall.
    """
    return snapshot.authorization.descendant_start_authorized or (
        stage.exact_model_id == snapshot.exact_model_id and stage.pool_id == snapshot.pool_id
    )


def project_stage_selection(
    snapshot: ExecutionSnapshot,
    indexes: tuple[int, ...],
) -> ExecutionSnapshot:
    """Narrow or permute leaves without merging across authored reference boundaries."""
    if not snapshot.model_stages:
        if any(not 0 <= index < len(snapshot.deployment_ids) for index in indexes):
            raise ValueError("execution route depth is outside the authorized plan")
        return snapshot.model_copy(
            update={
                "deployment_ids": tuple(snapshot.deployment_ids[i] for i in indexes),
                "model_stages": (),
            }
        )
    stages: list[ModelExecutionStage] = []
    for index in indexes:
        source = snapshot.stage_for_depth(index)
        deployment_id = snapshot.deployment_ids[index]
        position = source.deployment_ids.index(deployment_id)
        positions = () if not source.rung_positions else (source.rung_positions[position],)
        if stages and stages[-1].stage_index == source.stage_index:
            previous = stages[-1]
            stages[-1] = previous.model_copy(
                update={
                    "deployment_ids": (*previous.deployment_ids, deployment_id),
                    "rung_positions": (*previous.rung_positions, *positions),
                }
            )
        else:
            stages.append(
                source.model_copy(
                    update={
                        "deployment_ids": (deployment_id,),
                        "rung_positions": positions,
                    }
                )
            )
    return snapshot.model_copy(
        update={
            "deployment_ids": tuple(snapshot.deployment_ids[i] for i in indexes),
            "model_stages": tuple(stages),
        }
    )


@dataclass(frozen=True)
class _CatalogView:
    """One revision-scoped catalog with immutable chain, pool, and deployment indexes.

    Attributes:
        catalog: Normalized catalog bound to the selected revision and digest.
        chains: Immutable authored-chain index keyed by canonical model identity.
        pools: Immutable exact-model pool index keyed by pool identity.
        deployments: Immutable deployment index used to resolve stage leaves.
    """

    catalog: NormalizedGatewayCatalog
    chains: Mapping[str, GatewayModelChain]
    pools: Mapping[str, ExactModelPool]
    deployments: Mapping[str, ExactModelDeployment]


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
    # Hundreds of alias revisions can share one immutable catalog. Hash and index
    # each distinct object once, but verify every key's digest independently.
    # ``catalogs`` keeps object IDs stable by retaining them for the whole loop.
    identity_by_object: dict[int, str] = {}
    view_by_object: dict[int, _CatalogView] = {}
    for key, catalog in catalogs.items():
        revision_id, catalog_sha256 = key
        if not is_foreign_snapshot(catalog):
            identity = identity_by_object.get(id(catalog))
            if identity is None:
                identity = catalog.identity_sha256()
                identity_by_object[id(catalog)] = identity
            if identity != catalog_sha256:
                raise ValueError(f"catalog for alias revision {revision_id!r} has the wrong digest")
        view = view_by_object.get(id(catalog))
        if view is None:
            view = _CatalogView(
                catalog=catalog,
                chains=MappingProxyType(catalog.chains_by_model() if catalog.model_chains else {}),
                pools=MappingProxyType({pool.pool_id: pool for pool in catalog.pools}),
                deployments=MappingProxyType(
                    {deployment.deployment_id: deployment for deployment in catalog.deployments}
                ),
            )
            view_by_object[id(catalog)] = view
        indexed[key] = view
    return indexed
