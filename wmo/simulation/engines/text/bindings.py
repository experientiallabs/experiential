"""Frozen text-simulation resolution bindings and their immutable artifact envelope."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import cast

from pydantic import Field, field_validator

from wmo.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactId,
    ArtifactInput,
    JsonValue,
    Sha256,
    sha256_json,
    stable_id,
)
from wmo.common.evaluations import EvaluationCell
from wmo.common.rollouts import SimulationCellBinding
from wmo.common.tasks import TaskCase, TaskSet
from wmo.runtime.models import ResolvedModel
from wmo.simulation.engines.text.prompt import (
    WORLD_MODEL_TEXT_PROMPT_ID,
    WORLD_MODEL_TEXT_PROMPT_VERSION,
    text_prompt_sha256,
)
from wmo.simulation.specs import SimulationSpec


class SimulationResolution(ArtifactEnvelope):
    """One immutable resolution of all aliases and task evidence for a simulation spec."""

    resolution_id: ArtifactId
    simulation_id: ArtifactId
    simulation_spec_sha256: Sha256
    simulation_spec_input: ArtifactInput
    cell_bindings: tuple[SimulationCellBinding, ...] = Field(min_length=1)

    @field_validator("cell_bindings")
    @classmethod
    def _require_unique_cells(
        cls,
        value: tuple[SimulationCellBinding, ...],
    ) -> tuple[SimulationCellBinding, ...]:
        """Reject duplicate bound task and candidate repeat cells."""
        keys = tuple(
            (binding.task_sha256, binding.candidate_alias, binding.repeat) for binding in value
        )
        if len(set(keys)) != len(keys):
            raise ValueError("simulation resolution must not repeat a bound evaluation cell")
        return value


def make_cell_binding(
    *,
    spec: SimulationSpec,
    simulation_spec_input: ArtifactInput,
    evaluation_plan_input: ArtifactInput,
    task_set_input: ArtifactInput,
    task_set: TaskSet,
    cell: EvaluationCell,
    task: TaskCase,
    candidate: ResolvedModel,
    world_model: ResolvedModel,
) -> SimulationCellBinding:
    """Bind one selected cell to all immutable data and resolved model identities.

    Args:
        spec: Persisted text-simulation specification being executed.
        simulation_spec_input: Manifest identity of the persisted specification artifact.
        evaluation_plan_input: Manifest identity of the frozen evaluation plan.
        task_set_input: Manifest identity of the full selected task set.
        task_set: Digest-bearing task-set envelope that owns ``task``.
        cell: Explicit plan cell selected by the simulation specification.
        task: Loaded canonical task content for the cell.
        candidate: Candidate alias resolved before any provider call.
        world_model: World-model alias resolved before any provider call.

    Returns:
        A complete identity record used for rollout IDs, persistence, and resume checks.

    Raises:
        ValueError: The specification does not retain world-model settings.
    """
    settings = spec.world_model
    if settings is None:  # pragma: no cover - concrete text callers validate the mode first
        raise ValueError("text simulation binding requires world-model settings")
    return SimulationCellBinding(
        evaluation_plan_input=evaluation_plan_input,
        task_set_input=task_set_input,
        task_set_tasks_sha256=task_set.tasks_sha256,
        task_sha256=sha256_json(task),
        candidate_alias=cell.candidate_alias,
        candidate=candidate.snapshot,
        agent_id=spec.agent_id,
        repeat=cell.repeat,
        world_model_alias=settings.world_model_alias,
        world_model=world_model.snapshot,
        simulator_id="text-world-model-v1",
        prompt_id=WORLD_MODEL_TEXT_PROMPT_ID,
        prompt_version=WORLD_MODEL_TEXT_PROMPT_VERSION,
        prompt_sha256=text_prompt_sha256(),
        simulation_spec_input=simulation_spec_input,
        simulation_spec_sha256=sha256_json(spec),
        simulation_inputs_sha256=sha256_json(
            cast(JsonValue, [item.model_dump(mode="json") for item in spec.inputs])
        ),
    )


def make_resolution(
    *,
    spec: SimulationSpec,
    simulation_spec_input: ArtifactInput,
    evaluation_plan_input: ArtifactInput,
    task_set_input: ArtifactInput,
    bindings: Sequence[SimulationCellBinding],
    created_at: datetime,
) -> SimulationResolution:
    """Build one collision-detecting immutable resolution artifact for a simulation spec.

    Args:
        spec: Frozen specification whose aliases and inputs were resolved.
        simulation_spec_input: Verified persisted specification manifest reference.
        evaluation_plan_input: Verified immutable evaluation-plan reference.
        task_set_input: Verified immutable task-set reference.
        bindings: Complete cell bindings in selected-cell order.
        created_at: Timestamp for the immutable resolution envelope.

    Returns:
        An immutable resolution whose stable ID is deliberately independent of mutable aliases.
    """
    spec_digest = sha256_json(spec)
    resolution_id = stable_id(
        "simulation-resolution",
        {
            "simulation_id": spec.simulation_id,
            "simulation_spec_sha256": spec_digest,
            "simulation_spec_input": simulation_spec_input.model_dump(mode="json"),
        },
    )
    return SimulationResolution(
        schema_version=1,
        created_at=created_at,
        inputs=tuple(
            sorted(
                (evaluation_plan_input, task_set_input, simulation_spec_input),
                key=lambda item: item.artifact_id,
            )
        ),
        code_revision=spec.code_revision,
        resolution_id=resolution_id,
        simulation_id=spec.simulation_id,
        simulation_spec_sha256=spec_digest,
        simulation_spec_input=simulation_spec_input,
        cell_bindings=tuple(bindings),
    )


def binding_digest(binding: SimulationCellBinding) -> Sha256:
    """Return the full content digest used to name one immutable rollout cell.

    Args:
        binding: Complete cell identity persisted in a simulation-resolution artifact.

    Returns:
        SHA-256 digest of every pinned task, model, prompt, and input reference.
    """
    return sha256_json(binding)


def rollout_id_for_binding(binding: SimulationCellBinding) -> ArtifactId:
    """Return the deterministic immutable rollout ID for one full cell binding.

    Args:
        binding: Complete resolved cell identity.

    Returns:
        Stable rollout artifact ID that changes whenever any bound input or model changes.
    """
    return stable_id("rollout", {"binding_sha256": binding_digest(binding)})


def lease_id_for_binding(
    resolution: SimulationResolution,
    binding: SimulationCellBinding,
) -> ArtifactId:
    """Return the local durable paid-work claim ID for one resolved rollout cell.

    Args:
        resolution: Immutable resolution artifact that owns the binding.
        binding: Complete resolved cell identity.

    Returns:
        Stable lease filename identity scoped to this exact resolution and cell binding.
    """
    return stable_id(
        "text-cell-lease",
        {
            "resolution_id": resolution.resolution_id,
            "binding_sha256": binding_digest(binding),
        },
    )
