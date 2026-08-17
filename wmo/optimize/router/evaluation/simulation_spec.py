"""Phase-scoped immutable simulation specifications for router composition."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from wmo.common.core.artifacts import ArtifactInput, stable_id
from wmo.common.evaluations import EvaluationCell, EvaluationPlan
from wmo.common.rollouts import SimulationMode
from wmo.simulation.specs import SimulationSpec

if TYPE_CHECKING:
    from wmo.optimize.router.composition import RouterEvaluationSetup


def build_router_simulation_spec(
    plan: EvaluationPlan,
    plan_input: ArtifactInput,
    task_input: ArtifactInput,
    setup: RouterEvaluationSetup,
    maximum_cost_usd: float,
    code_revision: str,
    cells: tuple[EvaluationCell, ...],
    *,
    phase: Literal["fit", "heldout"],
) -> SimulationSpec:
    """Create one phase-scoped simulation spec over the exact fit-only RAG.

    The spec timestamp is the frozen plan's creation time, never the current invocation
    time, so a resumed run reproduces the exact spec digest stamped on completed rollouts.

    Args:
        plan: Frozen evaluation plan containing the selected cells.
        plan_input: Exact persisted evaluation-plan pointer.
        task_input: Exact persisted task-set pointer.
        setup: Reviewed candidates, protocols, retrieval pointer, and world-model settings.
        maximum_cost_usd: Finite provider-spend ceiling for this phase.
        code_revision: Exact source revision bound to generated artifacts.
        cells: Phase-specific plan cells eligible for simulation.
        phase: Fit or held-out phase label used in stable identity.

    Returns:
        Sparse immutable specification that binds the same fit RAG in either phase.
    """
    cell_ids = tuple(
        sorted(cell.cell_id for cell in cells if getattr(cell, "execution", None) == "simulate")
    )
    binding = {
        "phase": phase,
        "plan": plan_input.model_dump(mode="json"),
        "task_set": task_input.model_dump(mode="json"),
        "cells": cell_ids,
        "settings": setup.world_model_settings.model_dump(mode="json"),
        "agent_id": setup.agent_id,
        "seed": setup.seed,
        "maximum_steps": setup.maximum_steps,
        "maximum_concurrency": setup.maximum_concurrency,
        "maximum_cost_usd": maximum_cost_usd,
        "code_revision": code_revision,
    }
    if setup.simulation_completion_input is not None:
        binding["simulation_completion"] = setup.simulation_completion_input.model_dump(mode="json")
    spec_inputs = [
        plan_input,
        task_input,
        setup.fit_rag_input,
        setup.world_model_settings.grounded_world_model_input,
    ]
    if setup.simulation_completion_input is not None:
        spec_inputs.append(setup.simulation_completion_input)
    return SimulationSpec(
        schema_version=1,
        created_at=plan.created_at,
        inputs=tuple(sorted(spec_inputs, key=lambda item: item.artifact_id)),
        code_revision=code_revision,
        simulation_id=stable_id("simulation", binding),
        evaluation_plan_id=plan.plan_id,
        cell_ids=cell_ids,
        agent_id=setup.agent_id,
        mode=SimulationMode.WORLD_MODEL,
        world_model=setup.world_model_settings,
        seed=setup.seed,
        maximum_steps=setup.maximum_steps,
        maximum_concurrency=setup.maximum_concurrency,
        maximum_cost_usd=maximum_cost_usd,
    )
