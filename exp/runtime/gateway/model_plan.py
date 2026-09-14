"""Freeze ordered cross-model plans while preserving exact-pool identity."""

from __future__ import annotations

from exp.common.models.gateway_catalog import ExactModelPool, NormalizedGatewayCatalog
from exp.common.models.gateway_chains import ModelExecutionStage, expand_model_chain
from exp.runtime.gateway.contracts import AuthorizationSnapshot, ExecutionSnapshot


def model_execution_snapshot(
    catalog: NormalizedGatewayCatalog,
    authorization: AuthorizationSnapshot,
    root_pool: ExactModelPool,
) -> ExecutionSnapshot:
    """Resolve one authorized catalog graph to a bounded immutable leaf cursor.

    Only conversational surfaces enter explicit model references. Other surfaces
    keep direct-model behavior and never inherit child pricing or capabilities.
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
    if authorization.surface not in ("chat_completions", "responses", "messages"):
        return direct
    chain = next((c for c in catalog.model_chains if c.model_id == root_pool.exact_model_id), None)
    if chain is None:
        return direct
    if chain.pool_id != root_pool.pool_id:
        raise ValueError("authorized chain root does not match its exact pool")
    chains = catalog.chains_by_model()
    expanded = expand_model_chain(root_pool.exact_model_id, chains)
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


def project_stage_selection(
    snapshot: ExecutionSnapshot,
    indexes: tuple[int, ...],
) -> ExecutionSnapshot:
    """Narrow or permute leaves without merging across authored reference boundaries."""
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
            "model_stages": tuple(stages) if snapshot.model_stages else (),
        }
    )


def segment_order(snapshot: ExecutionSnapshot, order: tuple[int, ...]) -> tuple[int, ...]:
    """Apply a rank only inside its original provider segment, never across a reference."""
    if not snapshot.model_stages:
        return order
    result: list[int] = []
    start = 0
    for stage in snapshot.model_stages:
        end = start + len(stage.deployment_ids)
        result.extend(index for index in order if start <= index < end)
        start = end
    return tuple(result)
