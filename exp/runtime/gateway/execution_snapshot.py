"""Route-bound execution plan frozen after exact-model selection."""

from __future__ import annotations

from pydantic import Field, model_validator

from exp.common.core.artifacts import ContractModel
from exp.common.models.dispatch_policy import GatewayThrottleRedialPolicy
from exp.common.models.gateway_catalog import (
    DeploymentId,
    ExactModelId,
    ExactModelPoolId,
    FailoverMode,
)
from exp.common.models.gateway_chains import ModelExecutionStage, ModelTraversalEvent
from exp.runtime.gateway.contracts import AuthorizationSnapshot


class ExecutionSnapshot(ContractModel):
    """Route-bound request plan created only after exact-model selection.

    Attributes:
        model_stages: Frozen ordered stage segments; empty for an unstaged exact-model route.
        traversal_events: Immutable expansion provenance, not mutable runtime visitation state.
    """

    authorization: AuthorizationSnapshot
    exact_model_id: ExactModelId
    pool_id: ExactModelPoolId
    deployment_ids: tuple[DeploymentId, ...] = Field(min_length=1)
    # The pool's per-model failover policy, carried onto the route so the
    # per-attempt retry/failover decision can honor it.
    failover_mode: FailoverMode = "maximize_availability"
    # Optional cache-stakes threshold; otherwise the failover mode decides.
    throttle_cache_threshold: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    # The pool's backoff-and-redial schedule for throttled rungs, carried so
    # the admission can hand the data plane its frozen retry facts and the
    # per-attempt decision can honor a post-backoff redial.
    throttle_redial: GatewayThrottleRedialPolicy | None = None
    # Rungs the host flagged for OpenRouter's per-request ZDR constraint; the
    # dispatch builder tightens each and fails closed on a wire that cannot.
    zdr_constrained_deployment_ids: tuple[DeploymentId, ...] = ()
    model_stages: tuple[ModelExecutionStage, ...] = ()
    traversal_events: tuple[ModelTraversalEvent, ...] = ()

    @model_validator(mode="after")
    def _require_stage_projection(self) -> ExecutionSnapshot:
        """Require the ordered stage leaves to exactly cover the dispatch cursor."""
        if (
            self.model_stages
            and tuple(d for s in self.model_stages for d in s.deployment_ids) != self.deployment_ids
        ):
            raise ValueError("execution stages must exactly cover ordered deployment_ids")
        return self

    def stage_for_depth(self, depth: int) -> ModelExecutionStage:
        """Return the exact destination authority, never substitute the root model."""
        if not 0 <= depth < len(self.deployment_ids):
            raise ValueError("execution route depth is outside the authorized plan")
        if not self.model_stages:
            return ModelExecutionStage(
                stage_index=0,
                exact_model_id=self.exact_model_id,
                pool_id=self.pool_id,
                deployment_ids=self.deployment_ids,
                failover_mode=self.failover_mode,
                throttle_cache_threshold=self.throttle_cache_threshold,
                throttle_redial=self.throttle_redial,
            )
        cursor = 0
        for stage in self.model_stages:
            cursor += len(stage.deployment_ids)
            if depth < cursor:
                return stage
        raise ValueError("execution stage projection is incomplete")
