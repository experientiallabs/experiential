"""Canonical immutable simulation and rollout artifact contracts."""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Literal

from pydantic import Field, field_validator, model_validator

from wmo.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactId,
    ArtifactInput,
    ContractModel,
    Sha256,
    StructuredFailure,
    validate_artifact_file_path,
)
from wmo.common.models import (
    AssistantAction,
    EmbeddingCostReservation,
    ModelAlias,
    ModelSnapshot,
    OperationEconomics,
)
from wmo.common.rollouts.otel import (
    ProductionSimulatorSnapshot,
    RolloutSpan,
    SandboxSimulatorSnapshot,
    SimulatorSnapshot,
    WorldModelSimulatorSnapshot,
)


class SimulationMode(StrEnum):
    """Execution mode selected by an immutable simulation specification."""

    WORLD_MODEL = "world_model"
    SANDBOX = "sandbox"
    MIXED_REALITY = "mixed_reality"


class StopReason(StrEnum):
    """Terminal reason captured for an agent episode."""

    COMPLETED = "completed"
    AGENT_STOP = "agent_stop"
    MAXIMUM_STEPS = "maximum_steps"
    MAXIMUM_TIME = "maximum_time"
    MAXIMUM_COST = "maximum_cost"
    CONTEXT_OVERFLOW = "context_overflow"
    LENGTH = "length"
    FAILURE = "failure"
    CANCELLED = "cancelled"


class SimulationArtifact(ArtifactEnvelope):
    """Shared envelope emitted by a concrete simulator for one planned cell."""

    artifact_id: ArtifactId
    simulation_id: ArtifactId
    cell_id: ArtifactId
    mode: SimulationMode


class SimulationCellBinding(ContractModel):
    """Immutable identity needed to safely resume one simulated evaluation cell.

    A simulation specification names aliases and artifact inputs. This binding resolves those
    references once before provider work starts, then carries the exact task, model, prompt, and
    input identities into the persisted rollout. It prevents an alias or a same-ID local artifact
    from being silently rebound during a later resume.
    """

    evaluation_plan_input: ArtifactInput
    task_set_input: ArtifactInput
    fit_rag_input: ArtifactInput
    grounded_world_model_input: ArtifactInput
    task_set_tasks_sha256: Sha256
    task_sha256: Sha256
    candidate_alias: ModelAlias
    candidate: ModelSnapshot
    agent_id: str = Field(min_length=1, max_length=256)
    repeat: int = Field(ge=0)
    world_model_alias: ModelAlias
    world_model: ModelSnapshot
    simulator_id: str = Field(min_length=1, max_length=256)
    prompt_id: str = Field(min_length=1, max_length=256)
    prompt_version: str = Field(min_length=1, max_length=256)
    prompt_sha256: Sha256
    query_embedding: EmbeddingCostReservation
    simulation_spec_input: ArtifactInput
    simulation_spec_sha256: Sha256
    simulation_inputs_sha256: Sha256


class SandboxSimulationCellBinding(ContractModel):
    """Exact task, model, environment, and artifact identity for one sandbox cell."""

    cell_id: ArtifactId
    task_id: ArtifactId
    purpose: Literal["fit", "held_out", "fidelity"]
    evaluation_plan_input: ArtifactInput
    task_set_input: ArtifactInput
    task_set_tasks_sha256: Sha256
    task_sha256: Sha256
    task_lineage_group_id: ArtifactId
    candidate_alias: ModelAlias
    candidate: ModelSnapshot
    candidate_maximum_call_cost_usd: float | None = Field(default=None, gt=0)
    candidate_cost_is_observable: bool = False
    environment_maximum_episode_cost_usd: float | None = Field(default=None, ge=0)
    environment_cost_is_observable: bool = False
    agent_id: str = Field(min_length=1, max_length=256)
    repeat: int = Field(ge=0)
    simulator_id: str = Field(min_length=1, max_length=256)
    environment_id: ArtifactId
    environment_sha256: Sha256
    simulation_spec_input: ArtifactInput
    simulation_spec_sha256: Sha256
    simulation_inputs_sha256: Sha256

    @field_validator(
        "candidate_maximum_call_cost_usd",
        "environment_maximum_episode_cost_usd",
    )
    @classmethod
    def _require_finite_call_cost(cls, value: float | None) -> float | None:
        """Reject a reservation that cannot enforce a finite spend boundary."""
        if value is not None and not math.isfinite(value):
            raise ValueError("sandbox cost reservations must be finite")
        return value


class ProviderFreeSourceProvenance(ContractModel):
    """Typed evidence that a historical source trace records no model identity at all.

    Some real trace exports capture environment activity without provider or model attribution,
    so their normalized spans carry no model snapshot. That evidence still grounds human judge
    calibration, which reads task, action, and outcome content rather than generator identity.
    A production rollout imported from such a trace states the absence as immutable evidence
    instead of borrowing an identity from the judge model, the catalog, or a placeholder, and it
    stays unusable for candidate evaluation, pricing, routing, and attribution.
    """

    provenance_version: Literal["provider-free-source-v1"] = "provider-free-source-v1"
    reason: Literal["source_trace_records_no_model_identity"] = (
        "source_trace_records_no_model_identity"
    )
    checked_span_count: int = Field(gt=0)


class RolloutArtifact(SimulationArtifact):
    """The v1 simulation artifact subtype that preserves one full agent episode."""

    artifact_kind: Literal["rollout"] = "rollout"
    rollout_id: ArtifactId
    trace_id: str = Field(min_length=1, max_length=512)
    evidence_source: Literal["production", "world_model", "sandbox"]
    source_run_id: str = Field(min_length=1, max_length=512)
    task_id: ArtifactId
    candidate: ModelSnapshot | None = None
    provider_free_source: ProviderFreeSourceProvenance | None = None
    agent_id: str = Field(min_length=1, max_length=256)
    simulator: SimulatorSnapshot
    world_model: ModelSnapshot | None = None
    seed: int | None = None
    repeat: int = Field(ge=0)
    spans: tuple[RolloutSpan, ...]
    final_output: AssistantAction | None = None
    stop_reason: StopReason
    failure: StructuredFailure | None = None
    candidate_economics: OperationEconomics
    world_model_economics: OperationEconomics | None = None
    retrieval_economics: OperationEconomics | None = None
    sandbox_economics: OperationEconomics | None = None
    orchestration_economics: OperationEconomics | None = None
    simulation_spec_sha256: Sha256 | None = None
    simulation_binding: SimulationCellBinding | None = None
    sandbox_binding: SandboxSimulationCellBinding | None = None

    @field_validator("spans")
    @classmethod
    def _require_unique_span_ids(cls, value: tuple[RolloutSpan, ...]) -> tuple[RolloutSpan, ...]:
        if not value:
            raise ValueError("a rollout must contain at least one span")
        span_ids = tuple(span.span_id for span in value)
        if len(set(span_ids)) != len(span_ids):
            raise ValueError("rollout span IDs must be unique")
        return value

    @model_validator(mode="after")
    def _require_consistent_source_provenance(self) -> RolloutArtifact:
        """Validate source-specific simulator, binding, and economics provenance.

        Returns:
            The rollout after its evidence source agrees with all bound fields.

        Raises:
            ValueError: The rollout mixes production, world-model, or sandbox provenance.
        """
        self._require_exact_generator_provenance()
        if self.evidence_source == "production" and self.simulation_spec_sha256 is not None:
            raise ValueError("production rollouts must not name a simulation specification")
        if self.evidence_source != "production" and self.simulation_spec_sha256 is None:
            raise ValueError("simulated rollouts require a simulation specification digest")
        if self.stop_reason == StopReason.FAILURE and self.failure is None:
            raise ValueError("failed rollouts require a structured failure")
        if self.evidence_source == "production":
            if not isinstance(self.simulator, ProductionSimulatorSnapshot):
                raise ValueError("production rollouts require a production simulator snapshot")
            if self.world_model is not None:
                raise ValueError("production rollouts must not name a world model")
            if self.sandbox_binding is not None:
                raise ValueError("production rollouts must not contain a sandbox binding")
            if self.retrieval_economics is not None:
                raise ValueError("production rollouts must not contain retrieval economics")
        elif self.evidence_source == "world_model":
            if self.mode != SimulationMode.WORLD_MODEL:
                raise ValueError("world-model rollouts require world_model mode")
            if not isinstance(self.simulator, WorldModelSimulatorSnapshot):
                raise ValueError("world-model rollouts require a world-model simulator snapshot")
            if self.world_model != self.simulator.world_model:
                raise ValueError("world-model rollout identity must match its simulator snapshot")
            if self.sandbox_binding is not None:
                raise ValueError("world-model rollouts must not contain a sandbox binding")
            if self.retrieval_economics is None:
                raise ValueError("world-model rollouts require retrieval economics")
            _require_world_model_binding(self)
        elif self.evidence_source == "sandbox":
            if self.mode != SimulationMode.SANDBOX:
                raise ValueError("sandbox rollouts require sandbox mode")
            if not isinstance(self.simulator, SandboxSimulatorSnapshot):
                raise ValueError("sandbox rollouts require a sandbox simulator snapshot")
            if self.world_model is not None:
                raise ValueError("sandbox rollouts must not name a world model")
            if self.simulation_binding is not None:
                raise ValueError("sandbox rollouts must not contain a world-model binding")
            if self.retrieval_economics is not None:
                raise ValueError("sandbox rollouts must not contain retrieval economics")
            _require_sandbox_binding(self)
        return self

    def _require_exact_generator_provenance(self) -> None:
        """Require either an exact generator identity or explicit provider-free source evidence.

        Raises:
            ValueError: Generator identity is missing, doubled, or contradicted by span evidence.
        """
        provider_free = self.provider_free_source
        if (self.candidate is None) == (provider_free is None):
            raise ValueError(
                "rollouts record either a candidate model snapshot or explicit provider-free "
                "source provenance"
            )
        if provider_free is None:
            return
        if self.evidence_source != "production":
            raise ValueError("only production rollouts may record provider-free source evidence")
        if provider_free.checked_span_count != len(self.spans):
            raise ValueError("provider-free source evidence must cover every rollout span")
        if any(span.model is not None for span in self.spans):
            raise ValueError("provider-free rollouts must not contain span model identity")


def _require_world_model_binding(rollout: RolloutArtifact) -> None:
    """Verify the text-simulation binding agrees with the persisted rollout envelope.

    Args:
        rollout: World-model rollout whose complete immutable binding is required.

    Raises:
        ValueError: Any task, model, prompt, RAG, specification, or input pin differs.
    """
    binding = rollout.simulation_binding
    simulator = rollout.simulator
    if binding is None:
        raise ValueError("world-model rollouts require a complete simulation cell binding")
    if not isinstance(simulator, WorldModelSimulatorSnapshot):
        raise ValueError("world-model rollouts require a world-model simulator snapshot")
    if binding.simulation_spec_sha256 != rollout.simulation_spec_sha256:
        raise ValueError("world-model binding must match the rollout simulation specification")
    if binding.task_set_input.artifact_id == binding.evaluation_plan_input.artifact_id:
        raise ValueError("world-model binding task-set and evaluation-plan inputs must differ")
    if binding.candidate != rollout.candidate:
        raise ValueError("world-model binding candidate must match the rollout candidate")
    if binding.agent_id != rollout.agent_id:
        raise ValueError("world-model binding agent must match the rollout agent")
    if binding.repeat != rollout.repeat:
        raise ValueError("world-model binding repeat must match the rollout repeat")
    if binding.world_model != rollout.world_model:
        raise ValueError("world-model binding must match the rollout world model")
    if binding.simulator_id != simulator.simulator_id:
        raise ValueError("world-model binding simulator must match the rollout simulator")
    if binding.prompt_id != simulator.prompt_id:
        raise ValueError("world-model binding prompt must match the rollout simulator")
    if binding.prompt_version != simulator.prompt_version:
        raise ValueError("world-model binding prompt version must match the rollout simulator")
    if binding.prompt_sha256 != simulator.prompt_sha256:
        raise ValueError("world-model binding prompt digest must match the rollout simulator")
    required_inputs = {
        binding.evaluation_plan_input,
        binding.task_set_input,
        binding.fit_rag_input,
        binding.grounded_world_model_input,
        binding.simulation_spec_input,
    }
    if not required_inputs.issubset(set(rollout.inputs)):
        raise ValueError("world-model rollout inputs must retain every bound simulation input")


def _require_sandbox_binding(rollout: RolloutArtifact) -> None:
    """Verify an executable rollout agrees with its complete sandbox cell binding."""
    binding = rollout.sandbox_binding
    simulator = rollout.simulator
    if binding is None:
        raise ValueError("sandbox rollouts require a complete sandbox cell binding")
    if not isinstance(simulator, SandboxSimulatorSnapshot):
        raise ValueError("sandbox rollouts require a sandbox simulator snapshot")
    if binding.simulation_spec_sha256 != rollout.simulation_spec_sha256:
        raise ValueError("sandbox binding must match the rollout simulation specification")
    if binding.cell_id != rollout.cell_id or binding.task_id != rollout.task_id:
        raise ValueError("sandbox binding cell and task must match the rollout")
    if binding.task_set_input.artifact_id == binding.evaluation_plan_input.artifact_id:
        raise ValueError("sandbox binding task-set and evaluation-plan inputs must differ")
    if binding.candidate != rollout.candidate:
        raise ValueError("sandbox binding candidate must match the rollout candidate")
    if binding.agent_id != rollout.agent_id or binding.repeat != rollout.repeat:
        raise ValueError("sandbox binding agent and repeat must match the rollout")
    if binding.simulator_id != simulator.simulator_id:
        raise ValueError("sandbox binding simulator must match the rollout simulator")
    if (
        binding.environment_id != simulator.environment_id
        or binding.environment_sha256 != simulator.environment_sha256
    ):
        raise ValueError("sandbox binding environment must match the rollout simulator")
    required_inputs = {
        binding.evaluation_plan_input,
        binding.task_set_input,
        binding.simulation_spec_input,
    }
    if not required_inputs.issubset(set(rollout.inputs)):
        raise ValueError("sandbox rollout inputs must retain every bound simulation input")


class SimulationArtifactSet(ArtifactEnvelope):
    """A frozen set of completed artifacts emitted for one simulation specification."""

    artifact_set_id: ArtifactId
    simulation_id: ArtifactId
    artifact_ids: tuple[ArtifactId, ...]
    artifacts_path: str = Field(min_length=1)
    artifacts_sha256: Sha256

    @field_validator("artifacts_path")
    @classmethod
    def _require_safe_artifacts_path(cls, value: str) -> str:
        return validate_artifact_file_path(value).as_posix()

    @field_validator("artifact_ids")
    @classmethod
    def _require_unique_artifact_ids(cls, value: tuple[ArtifactId, ...]) -> tuple[ArtifactId, ...]:
        if not value:
            raise ValueError("a simulation artifact set must contain at least one artifact")
        if len(set(value)) != len(value):
            raise ValueError("artifact_ids must not contain duplicates")
        return value
