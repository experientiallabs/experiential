"""Frozen, provenance-filtered memory inputs for simulator-only actor councils."""

from collections.abc import Sequence

from pydantic import Field

from exp.common.core.artifacts import ArtifactInput, ContractModel, JsonObject
from exp.common.project import ArtifactStore, artifact_input
from exp.simulation.world_model.experience_memory import (
    Eligibility,
    load_source,
    materialize_log,
    memory_version_id,
    memory_view,
    read_memory_version,
    require_memory_scope,
    reserved_task_lineages,
    summary_generations,
)
from exp.simulation.world_model.memory_tree import overview, overview_sources
from exp.simulation.world_model.overview_pin import pinned_summaries
from exp.simulation.world_model.runtime import load_grounded_world_model_artifact


class CouncilMemory(ContractModel):
    """An exact read-only memory checkpoint, never a moving head.

    Attributes:
        lineage: Existing memory bank owning the learning observations.
        checkpoint: Exact published version read by every actor in this council.
        overview_lines: Maximum overview rows supplied alongside selected full records.
    """

    lineage: str = Field(min_length=1)
    checkpoint: int = Field(ge=1, strict=True)
    overview_lines: int = Field(default=4, ge=1, strict=True)


class CouncilMemoryContext(ContractModel):
    """Eligible context handed to the existing session-research owner without feedback.

    Attributes:
        snapshot: Manifest-bound memory version fixed before actor inference.
        identity: Checkpoint, exclusions and withholding information retained in the session.
        records: Eligible original observations and provisional research for JEV selection.
        overview: Eligible budgeted lines and compressed summaries.
        sources: Typed source ancestry for each selectable record.
        overview_sources: Typed source ancestry of exactly the supplied overview.
    """

    snapshot: ArtifactInput
    identity: JsonObject
    records: dict[str, JsonObject]
    overview: tuple[JsonObject, ...]
    sources: dict[str, tuple[ArtifactInput, ...]]
    overview_sources: tuple[ArtifactInput, ...]


def read_council_memory(
    store: ArtifactStore,
    config: CouncilMemory,
    world_model: ArtifactInput,
    excluded_lineages: Sequence[str],
) -> CouncilMemoryContext:
    """Resolve frozen memory through the same eligibility and overview owners as fidelity.

    Args:
        store: Existing project artifact store.
        config: Exact immutable checkpoint and overview allowance.
        world_model: Predictor knowledge whose compatibility is verified independently of model.
        excluded_lineages: Every reserved actor task in this council, not guessed from actions.

    Returns:
        Typed context without native feedback, state initialization, or publication authority.
    """
    version = read_memory_version(store, config.lineage, config.checkpoint)
    artifact = load_grounded_world_model_artifact(store, world_model)
    require_memory_scope(
        version,
        world_model=world_model,
        artifact=artifact,
        selection_policy_sha256=version.selection_policy_sha256,
    )
    exclusions = tuple(
        sorted(set(excluded_lineages) | set(reserved_task_lineages(store, config.lineage)))
    )
    rule = Eligibility(reserved_task_lineages=exclusions, strict=True)
    view = memory_view(store, version, rule)
    materialize_log(store, version)

    def admits(reference: ArtifactInput) -> bool:
        """Apply the same authority decision to each line and each summary dependency."""
        return rule.admits(load_source(store, reference))

    rows = overview(
        store,
        config.lineage,
        version.log_length or 0,
        config.overview_lines,
        admits=admits,
        admits_line=lambda line: line.origin not in view.retired,
        generations=summary_generations(version),
        pinned=pinned_summaries(
            store, version, created_at=version.created_at, code_revision=version.code_revision
        ),
    )
    reference = artifact_input(
        store.read(memory_version_id(config.lineage, config.checkpoint)).manifest
    )
    return CouncilMemoryContext(
        snapshot=reference,
        identity={
            "lineage": config.lineage,
            "version": config.checkpoint,
            "mode": "frozen",
            "reserved": True,
            "snapshot": reference.model_dump(mode="json"),
            "reserved_task_lineages": list(exclusions),
            "withheld_records": view.withheld,
            "consumed_trajectories": len(version.consumed),
        },
        records=view.records,
        overview=tuple(rows),
        sources=view.sources,
        overview_sources=overview_sources(
            store, config.lineage, rows, summary_generations(version)
        ),
    )
