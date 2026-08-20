"""Immutable task, model, environment, and artifact bindings for sandbox cells."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import cast

from pydantic import Field, field_validator

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactId,
    ArtifactInput,
    JsonValue,
    Sha256,
    sha256_json,
    stable_id,
)
from exp.common.evaluations import EvaluationCell
from exp.common.models import ModelAlias, ModelClient, ModelSnapshot
from exp.common.rollouts import SandboxSimulationCellBinding
from exp.common.tasks import TaskCase, TaskSet
from exp.simulation.specs import SimulationSpec

SANDBOX_SIMULATOR_ID = "sandbox-v1"


@dataclass(frozen=True)
class CandidateBinding:
    """One candidate alias bound to a client, snapshot, and optional strict call reservation.

    Args:
        alias: Stable candidate alias selected by evaluation cells.
        client: Resolved model client injected into the customer agent.
        snapshot: Exact provider and model identity pinned by the evaluation plan.
        maximum_call_cost_usd: Proven upper bound for one candidate request. A finite run budget
            requires this value so admission can stop before dispatch instead of overspending.
        cost_is_observable: Whether successful responses are contractually guaranteed to report
            cost. Finite-budget runs reject candidates without this capability before dispatch.
    """

    alias: ModelAlias
    client: ModelClient
    snapshot: ModelSnapshot
    maximum_call_cost_usd: float | None = None
    cost_is_observable: bool = False

    def __post_init__(self) -> None:
        """Reject a nonpositive or nonfinite call reservation before provider work."""
        value = self.maximum_call_cost_usd
        if value is not None and (not math.isfinite(value) or value <= 0):
            raise ValueError("candidate maximum_call_cost_usd must be finite and positive")


@dataclass(frozen=True)
class EnvironmentCostBinding:
    """Preflight facts needed to enforce one finite environment budget.

    Args:
        maximum_episode_cost_usd: Proven upper bound for opening, using, and cleaning one
            environment. Zero explicitly proves the runtime has no billed sandbox cost.
        cost_is_observable: Whether nonzero environment cost is guaranteed to reach canonical
            observation economics. A finite run rejects an environment without this capability.
    """

    maximum_episode_cost_usd: float | None = None
    cost_is_observable: bool = False

    def __post_init__(self) -> None:
        """Reject a negative or nonfinite environment reservation."""
        value = self.maximum_episode_cost_usd
        if value is not None and (not math.isfinite(value) or value < 0):
            raise ValueError("environment maximum_episode_cost_usd must be finite and nonnegative")


class SandboxSimulationResolution(ArtifactEnvelope):
    """One frozen resolution of all executable cell aliases and evidence inputs."""

    resolution_id: ArtifactId
    simulation_id: ArtifactId
    simulation_spec_sha256: Sha256
    simulation_spec_input: ArtifactInput
    cell_bindings: tuple[SandboxSimulationCellBinding, ...] = Field(min_length=1)

    @field_validator("cell_bindings")
    @classmethod
    def _require_unique_cells(
        cls,
        value: tuple[SandboxSimulationCellBinding, ...],
    ) -> tuple[SandboxSimulationCellBinding, ...]:
        """Reject duplicate IDs or task, candidate, repeat, and purpose coordinates."""
        cell_ids = tuple(binding.cell_id for binding in value)
        if len(set(cell_ids)) != len(cell_ids):
            raise ValueError("sandbox resolution cell IDs must be unique")
        coordinates = tuple(
            (
                binding.task_id,
                binding.candidate_alias,
                binding.repeat,
                binding.purpose,
            )
            for binding in value
        )
        if len(set(coordinates)) != len(coordinates):
            raise ValueError("sandbox resolution must not repeat a bound evaluation cell")
        return value


def make_sandbox_cell_binding(
    *,
    spec: SimulationSpec,
    simulation_spec_input: ArtifactInput,
    evaluation_plan_input: ArtifactInput,
    task_set_input: ArtifactInput,
    task_set: TaskSet,
    cell: EvaluationCell,
    task: TaskCase,
    candidate: CandidateBinding,
    environment_cost: EnvironmentCostBinding,
) -> SandboxSimulationCellBinding:
    """Bind one selected cell to immutable task, candidate, environment, and spec identity.

    Args:
        spec: Persisted sandbox specification being executed.
        simulation_spec_input: Exact manifest identity for the persisted specification.
        evaluation_plan_input: Exact manifest identity for the evaluation plan.
        task_set_input: Exact manifest identity for the selected task set.
        task_set: Digest-bearing task-set envelope that owns ``task``.
        cell: Explicit simulated plan cell.
        task: Canonical task selected by the cell.
        candidate: Resolved candidate identity and optional strict cost reservation.

    Returns:
        Complete identity used for rollout IDs, leases, and resume validation.

    Raises:
        ValueError: The specification does not retain sandbox settings.
    """
    settings = spec.sandbox
    if settings is None:  # pragma: no cover - the concrete simulator validates its mode first
        raise ValueError("sandbox simulation binding requires sandbox settings")
    return SandboxSimulationCellBinding(
        cell_id=cell.cell_id,
        task_id=cell.task_id,
        purpose=cell.purpose,
        evaluation_plan_input=evaluation_plan_input,
        task_set_input=task_set_input,
        task_set_tasks_sha256=task_set.tasks_sha256,
        task_sha256=sha256_json(task),
        task_lineage_group_id=task.lineage_group_id,
        candidate_alias=cell.candidate_alias,
        candidate=candidate.snapshot,
        candidate_maximum_call_cost_usd=candidate.maximum_call_cost_usd,
        candidate_cost_is_observable=candidate.cost_is_observable,
        environment_maximum_episode_cost_usd=environment_cost.maximum_episode_cost_usd,
        environment_cost_is_observable=environment_cost.cost_is_observable,
        agent_id=spec.agent_id,
        repeat=cell.repeat,
        simulator_id=SANDBOX_SIMULATOR_ID,
        environment_id=settings.environment_id,
        environment_sha256=settings.environment_sha256,
        simulation_spec_input=simulation_spec_input,
        simulation_spec_sha256=sha256_json(spec),
        simulation_inputs_sha256=sha256_json(
            cast(JsonValue, [item.model_dump(mode="json") for item in spec.inputs])
        ),
    )


def make_sandbox_resolution(
    *,
    spec: SimulationSpec,
    simulation_spec_input: ArtifactInput,
    evaluation_plan_input: ArtifactInput,
    task_set_input: ArtifactInput,
    bindings: Sequence[SandboxSimulationCellBinding],
    created_at: datetime,
) -> SandboxSimulationResolution:
    """Build one collision-detecting immutable sandbox resolution artifact.

    Args:
        spec: Persisted sandbox specification being executed.
        simulation_spec_input: Exact manifest identity for the persisted specification.
        evaluation_plan_input: Exact manifest identity for the evaluation plan.
        task_set_input: Exact manifest identity for the selected task set.
        bindings: Immutable executable bindings selected for the simulation cells.
        created_at: Timestamp recorded in the resolution envelope.

    Returns:
        Immutable resolution that binds all selected cells to their exact inputs.
    """
    spec_digest = sha256_json(spec)
    resolution_id = stable_id(
        "sandbox-simulation-resolution",
        {
            "simulation_id": spec.simulation_id,
            "simulation_spec_sha256": spec_digest,
            "simulation_spec_input": simulation_spec_input.model_dump(mode="json"),
        },
    )
    return SandboxSimulationResolution(
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


def sandbox_binding_digest(binding: SandboxSimulationCellBinding) -> Sha256:
    """Return the complete immutable digest for one executable cell binding."""
    return sha256_json(binding)


def sandbox_rollout_id(binding: SandboxSimulationCellBinding) -> ArtifactId:
    """Return the deterministic rollout artifact ID for one exact sandbox binding."""
    return stable_id("rollout", {"sandbox_binding_sha256": sandbox_binding_digest(binding)})


def sandbox_lease_id(
    resolution: SandboxSimulationResolution,
    binding: SandboxSimulationCellBinding,
) -> ArtifactId:
    """Return the durable in-flight claim ID for one exact executable cell."""
    return stable_id(
        "sandbox-cell-lease",
        {
            "resolution_id": resolution.resolution_id,
            "binding_sha256": sandbox_binding_digest(binding),
        },
    )
