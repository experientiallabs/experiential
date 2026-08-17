"""Canonical rollout construction for grounded text-world-model execution."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from wmo.common.core.artifacts import (
    ArtifactInput,
    StructuredFailure,
    sha256_json,
    sorted_unique_inputs,
)
from wmo.common.evaluations import EvaluationCell
from wmo.common.models import AssistantAction, OperationEconomics
from wmo.common.rollouts import (
    RolloutArtifact,
    RolloutSpan,
    SimulationCellBinding,
    SimulationMode,
    StopReason,
    WorldModelSimulatorSnapshot,
)
from wmo.runtime.models import ResolvedModel
from wmo.simulation.engines.clock import timestamp
from wmo.simulation.engines.text.bindings import rollout_id_for_binding
from wmo.simulation.engines.text.prompt import (
    WORLD_MODEL_TEXT_PROMPT_ID,
)
from wmo.simulation.engines.text.redaction import redact_action, redact_failure
from wmo.simulation.specs import SimulationSpec

TEXT_WORLD_MODEL_SIMULATOR_ID = "text-world-model-v1"


@dataclass(frozen=True)
class GroundedRolloutBuilder:
    """Build immutable rollout evidence with the exact grounded artifact graph.

    Args:
        plan_input: Frozen evaluation-plan manifest pointer.
        task_set_input: Frozen task-set manifest pointer.
        fit_rag_input: Exact fit-only retrieval manifest pointer.
        redacted_field_names: Field labels removed from persistent evidence.
        clock: Time source for rollout creation timestamps.
    """

    plan_input: ArtifactInput
    task_set_input: ArtifactInput
    fit_rag_input: ArtifactInput
    redacted_field_names: frozenset[str]
    clock: Callable[[], datetime]

    def make(
        self,
        *,
        spec: SimulationSpec,
        cell: EvaluationCell,
        candidate: ResolvedModel,
        world_model: ResolvedModel,
        binding: SimulationCellBinding,
        resolution_input: ArtifactInput,
        stop_reason: StopReason,
        failure: StructuredFailure | None,
        final_output: AssistantAction | None,
        spans: tuple[RolloutSpan, ...],
        candidate_economics: OperationEconomics,
        world_model_economics: OperationEconomics,
        retrieval_economics: OperationEconomics,
        orchestration_economics: OperationEconomics,
    ) -> RolloutArtifact:
        """Compose one canonical rollout from bound execution evidence.

        Args:
            spec: Simulation specification owning the rollout.
            cell: Exact evaluation cell.
            candidate: Resolved candidate identity.
            world_model: Resolved simulator-model identity.
            binding: Complete immutable cell binding.
            resolution_input: Exact resolution manifest pointer.
            stop_reason: Canonical terminal reason.
            failure: Optional structured terminal failure.
            final_output: Optional visible candidate output.
            spans: Ordered redacted execution spans.
            candidate_economics: Combined candidate-call economics.
            world_model_economics: Combined world-model-call economics.
            retrieval_economics: Conservative query-embedding economics.
            orchestration_economics: Simulator-owned elapsed-time economics.

        Returns:
            Fully bound world-model rollout ready for immutable persistence.
        """
        rollout_id = rollout_id_for_binding(binding)
        return RolloutArtifact(
            schema_version=1,
            created_at=timestamp(self.clock),
            inputs=sorted_unique_inputs(
                self.plan_input,
                self.task_set_input,
                self.fit_rag_input,
                binding.grounded_world_model_input,
                binding.simulation_spec_input,
                resolution_input,
            ),
            code_revision=spec.code_revision,
            artifact_id=rollout_id,
            simulation_id=spec.simulation_id,
            cell_id=cell.cell_id,
            mode=SimulationMode.WORLD_MODEL,
            rollout_id=rollout_id,
            trace_id=sha256_json({"binding": binding.model_dump(mode="json")}),
            evidence_source="world_model",
            source_run_id=spec.simulation_id,
            task_id=cell.task_id,
            candidate=candidate.snapshot,
            agent_id=spec.agent_id,
            simulator=WorldModelSimulatorSnapshot(
                simulator_id=TEXT_WORLD_MODEL_SIMULATOR_ID,
                prompt_id=WORLD_MODEL_TEXT_PROMPT_ID,
                prompt_version=binding.prompt_version,
                prompt_sha256=binding.prompt_sha256,
                world_model=world_model.snapshot,
            ),
            world_model=world_model.snapshot,
            seed=spec.seed,
            repeat=cell.repeat,
            spans=spans,
            final_output=redact_action(final_output, self.redacted_field_names),
            stop_reason=stop_reason,
            failure=redact_failure(failure, self.redacted_field_names),
            candidate_economics=candidate_economics,
            world_model_economics=world_model_economics,
            retrieval_economics=retrieval_economics,
            orchestration_economics=orchestration_economics,
            simulation_spec_sha256=binding.simulation_spec_sha256,
            simulation_binding=binding,
        )
