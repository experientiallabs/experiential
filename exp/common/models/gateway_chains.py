"""Typed ordered model references above certified same-exact provider pools."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Literal

from pydantic import Field, model_validator

from exp.common.core.artifacts import ArtifactId, ContractModel
from exp.common.models.dispatch_policy import FailoverMode, GatewayThrottleRedialPolicy

MAXIMUM_MODEL_STAGES = 16
MAXIMUM_EXPANDED_RUNGS = 256


class GatewayDeploymentRung(ContractModel):
    """One direct deployment at its authored position in a model chain."""

    kind: Literal["deployment"] = "deployment"
    deployment_id: ArtifactId


class GatewayModelReferenceRung(ContractModel):
    """One explicit substitution, identified by canonical model identity, not alias."""

    kind: Literal["model"] = "model"
    model_id: ArtifactId


GatewayChainRung = Annotated[
    GatewayDeploymentRung | GatewayModelReferenceRung, Field(discriminator="kind")
]


class ModelStagePolicy(ContractModel):
    """Complete model failure policy, including an explicitly unauthored threshold."""

    failover_mode: FailoverMode
    throttle_cache_threshold: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    throttle_redial: GatewayThrottleRedialPolicy | None = None


class GatewayModelChain(ContractModel):
    """One revision-pinned model waterfall, including an explicit unavailable override.

    Direct rungs remain members of ``pool_id`` and certify only ``model_id``.
    An absent chain inherits that pool's direct order; an unavailable chain does not.
    The host must choose and authorize one overlay/default view before expansion.
    """

    model_id: ArtifactId
    pool_id: ArtifactId
    revision: str = Field(min_length=1, max_length=256)
    rungs: tuple[GatewayChainRung, ...] = ()
    available: bool = True
    policy: ModelStagePolicy | None = None

    @model_validator(mode="after")
    def _require_authored_order(self) -> GatewayModelChain:
        """Require one optional reference after an actual direct rung."""
        references = [rung for rung in self.rungs if rung.kind == "model"]
        if len(references) > 1:
            raise ValueError(
                "a model chain permits one model reference; remove the extra reference"
            )
        if self.rungs and self.rungs[0].kind != "deployment":
            raise ValueError("a model reference cannot be first; add a direct deployment before it")
        direct = [rung.deployment_id for rung in self.rungs if rung.kind == "deployment"]
        if len(direct) != len(set(direct)):
            raise ValueError("model chain direct deployments must not repeat")
        if self.available and not direct:
            raise ValueError(
                "an available model chain needs a direct deployment; repair or reset it"
            )
        return self


class ModelTraversalEvent(ContractModel):
    """Content-free provenance for one bounded, canonical traversal decision."""

    reason: Literal[
        "model_entered",
        "model_reference_already_visited",
        "model_unavailable",
        "traversal_exhausted",
    ]
    model_id: ArtifactId
    ancestry: tuple[ArtifactId, ...] = ()
    rung_position: int | None = Field(default=None, ge=0)
    # Number of direct leaves before this event, so a cursor never replays it.
    cursor: int = Field(ge=0)


class ExpandedModelSegment(ContractModel):
    """A contiguous provider segment which scheduling must never cross."""

    model_id: ArtifactId
    pool_id: ArtifactId
    chain_revision: str
    ancestry: tuple[ArtifactId, ...]
    deployment_ids: tuple[ArtifactId, ...] = Field(min_length=1)
    rung_positions: tuple[int, ...] = Field(min_length=1)


class ModelExecutionStage(ContractModel):
    """Authorized same-exact segment facts carried to each attempt's ledger callback."""

    stage_index: int = Field(ge=0)
    exact_model_id: ArtifactId
    pool_id: ArtifactId
    deployment_ids: tuple[ArtifactId, ...] = Field(min_length=1)
    failover_mode: FailoverMode = "maximize_availability"
    throttle_cache_threshold: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    throttle_redial: GatewayThrottleRedialPolicy | None = None
    chain_revision: str = "direct"
    ancestry: tuple[ArtifactId, ...] = ()
    rung_positions: tuple[int, ...] = ()


class ExpandedModelChain(ContractModel):
    """One finite forward traversal, frozen before any upstream dispatch."""

    segments: tuple[ExpandedModelSegment, ...]
    events: tuple[ModelTraversalEvent, ...]
    visited_model_ids: tuple[ArtifactId, ...]
    examined_rungs: int


class ModelChainConfigurationError(ValueError):
    """A model graph cannot be resolved safely; authoring must repair it."""


def expand_model_chain(
    root_model_id: str,
    chains: Mapping[str, GatewayModelChain],
) -> ExpandedModelChain:
    """Expand one pinned authorized graph in authored depth-first suffix order.

    Canonical model IDs enter at most once, including self and reciprocal edges.
    Distinct model and examined-rung caps are independent of the attempt budget.
    The graph contains only host-authorized chains; missing targets fail closed.
    A sticky descendant starts by slicing this result, not by expanding a new root,
    so its ancestors and their back-edges stay behind the request cursor.
    """
    segments: list[ExpandedModelSegment] = []
    events: list[ModelTraversalEvent] = []
    visited: list[str] = []
    examined = 0
    cursor = 0

    def visit(model_id: str, ancestry: tuple[str, ...], position: int | None) -> None:
        """Visit one canonical model once and preserve its forward parent suffix."""
        nonlocal examined, cursor
        if model_id in visited:
            events.append(
                ModelTraversalEvent(
                    reason="model_reference_already_visited",
                    model_id=model_id,
                    ancestry=ancestry,
                    rung_position=position,
                    cursor=cursor,
                )
            )
            return
        if len(visited) >= MAXIMUM_MODEL_STAGES:
            raise ModelChainConfigurationError(
                "model chain exceeds 16 canonical models; shorten the chain"
            )
        chain = chains.get(model_id)
        if chain is None or chain.model_id != model_id:
            raise ModelChainConfigurationError(
                "model reference is unavailable; repair or reset the chain"
            )
        visited.append(model_id)
        path = (*ancestry, model_id)
        events.append(
            ModelTraversalEvent(
                reason="model_entered",
                model_id=model_id,
                ancestry=ancestry,
                rung_position=position,
                cursor=cursor,
            )
        )
        if not chain.available:
            events.append(
                ModelTraversalEvent(
                    reason="model_unavailable",
                    model_id=model_id,
                    ancestry=ancestry,
                    cursor=cursor,
                )
            )
            return
        direct: list[str] = []
        positions: list[int] = []

        def flush() -> None:
            """Freeze the segment preceding a boundary, without merging its suffix."""
            if direct:
                segments.append(
                    ExpandedModelSegment(
                        model_id=model_id,
                        pool_id=chain.pool_id,
                        chain_revision=chain.revision,
                        ancestry=path,
                        deployment_ids=tuple(direct),
                        rung_positions=tuple(positions),
                    )
                )
                direct.clear()
                positions.clear()

        for rung_position, rung in enumerate(chain.rungs):
            examined += 1
            if examined > MAXIMUM_EXPANDED_RUNGS:
                raise ModelChainConfigurationError(
                    "model chain exceeds 256 examined rungs; shorten the chain"
                )
            if rung.kind == "deployment":
                direct.append(rung.deployment_id)
                positions.append(rung_position)
                cursor += 1
            else:
                flush()
                visit(rung.model_id, path, rung_position)
        flush()

    visit(root_model_id, (), None)
    events.append(
        ModelTraversalEvent(
            reason="traversal_exhausted",
            model_id=root_model_id,
            cursor=cursor,
        )
    )
    return ExpandedModelChain(
        segments=tuple(segments),
        events=tuple(events),
        visited_model_ids=tuple(visited),
        examined_rungs=examined,
    )
