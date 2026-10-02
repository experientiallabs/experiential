"""Explicit budget continuations with immutable parent lineage and retained paid work."""

from collections.abc import Mapping

from exp.common.core.artifacts import ArtifactInput, sorted_unique_inputs
from exp.common.evaluations import EvaluationCell
from exp.common.project import ArtifactStore, artifact_input
from exp.common.rollouts import RolloutArtifact, SimulationCellBinding, StopReason
from exp.simulation.engines.text.bindings import rollout_id_for_binding
from exp.simulation.engines.text.errors import SimulationResumeError
from exp.simulation.engines.text.resume import load_rollout
from exp.simulation.specs import SimulationSpec


def load_continuations(
    store: ArtifactStore,
    spec: SimulationSpec,
    cells: tuple[EvaluationCell, ...],
    bindings: Mapping[str, SimulationCellBinding],
) -> dict[str, RolloutArtifact]:
    """Validate the entire parent matrix before any child provider dispatch.

    Args:
        store: Immutable project evidence.
        spec: Child specification carrying an explicit parent manifest.
        cells: Requested child cells.
        bindings: Resolved child task, model, prompt and retrieval identities.

    Returns:
        Parent evidence keyed by child cell ID, with the newest attempt for each coordinate.

    Raises:
        SimulationResumeError: Budgets decrease, semantic pins drift, or a prefix is unsafe.
    """
    pointer = spec.continuation_of
    if pointer is None:
        return {}
    if artifact_input(store.read(pointer.artifact_id).manifest) != pointer:
        raise SimulationResumeError("continuation parent manifest changed")
    parent = SimulationSpec.model_validate_json(
        store.read_bytes(pointer.artifact_id, "simulation-spec.json")
    )
    if (
        parent.code_revision != spec.code_revision
        or parent.stop_on_overspend != spec.stop_on_overspend
        or parent.mode != spec.mode
        or parent.world_model != spec.world_model
        or parent.agent_id != spec.agent_id
        or parent.seed != spec.seed
        or spec.maximum_steps < parent.maximum_steps
        or spec.maximum_rollout_output_tokens < parent.maximum_rollout_output_tokens
        or (spec.maximum_cost_usd or 0) < (parent.maximum_cost_usd or 0)
    ):
        raise SimulationResumeError(
            "continuation must retain execution semantics and only increase budgets"
        )
    if (spec.maximum_steps, spec.maximum_rollout_output_tokens, spec.maximum_cost_usd) == (
        parent.maximum_steps,
        parent.maximum_rollout_output_tokens,
        parent.maximum_cost_usd,
    ):
        raise SimulationResumeError("increase an execution budget before continuing")
    by_coordinate: dict[tuple[str, str, int], RolloutArtifact] = {}
    for identity in store.list_ids():
        stored = store.read(identity)
        if stored.manifest.artifact_type != "rollout":
            continue
        rollout = load_rollout(store, identity)
        if rollout.simulation_id != parent.simulation_id or rollout.simulation_binding is None:
            continue
        key = (rollout.task_id, rollout.simulation_binding.candidate_alias, rollout.repeat)
        prior = by_coordinate.get(key)
        if prior is None or prior.retry_attempt < rollout.retry_attempt:
            by_coordinate[key] = rollout
    result = {}
    ignored = {
        "evaluation_plan_input",
        "simulation_spec_input",
        "simulation_spec_sha256",
        "simulation_inputs_sha256",
    }
    for cell in cells:
        rollout = by_coordinate.get((cell.task_id, cell.candidate_alias, cell.repeat))
        if rollout is None or rollout.simulation_binding is None:
            raise SimulationResumeError(
                "continuation requires a persisted parent for every selected cell"
            )
        if rollout.simulation_binding.model_dump(exclude=ignored) != bindings[
            cell.cell_id
        ].model_dump(exclude=ignored):
            raise SimulationResumeError(
                "continuation task, model, prompt or retrieval identity changed"
            )
        if rollout.stop_reason != StopReason.COMPLETED and (
            rollout.stop_reason
            not in {
                StopReason.MAXIMUM_STEPS,
                StopReason.MAXIMUM_OUTPUT_TOKENS,
                StopReason.MAXIMUM_COST,
            }
            or rollout.text_checkpoint is None
        ):
            raise SimulationResumeError(
                "parent has no safe turn-boundary checkpoint; start a fresh evaluation"
            )
        result[cell.cell_id] = rollout
    return result


def rebind_completed(
    store: ArtifactStore,
    parent: RolloutArtifact,
    spec: SimulationSpec,
    cell: EvaluationCell,
    binding: SimulationCellBinding,
    resolution_input: ArtifactInput,
) -> RolloutArtifact:
    """Carry completed evidence into the child matrix without another model call."""
    identity = rollout_id_for_binding(binding)
    return parent.model_copy(
        update={
            "artifact_id": identity,
            "rollout_id": identity,
            "retry_attempt": 0,
            "simulation_id": spec.simulation_id,
            "source_run_id": spec.simulation_id,
            "cell_id": cell.cell_id,
            "created_at": spec.created_at,
            "code_revision": spec.code_revision,
            "simulation_binding": binding,
            "simulation_spec_sha256": binding.simulation_spec_sha256,
            "continuation_of": artifact_input(store.read(parent.rollout_id).manifest),
            "inputs": sorted_unique_inputs(
                binding.evaluation_plan_input,
                binding.task_set_input,
                binding.fit_rag_input,
                binding.grounded_world_model_input,
                binding.simulation_spec_input,
                resolution_input,
                artifact_input(store.read(parent.rollout_id).manifest),
            ),
        }
    )


def retain_lineage(
    store: ArtifactStore, rollout: RolloutArtifact, parent: RolloutArtifact | None
) -> RolloutArtifact:
    """Link a continued rollout to its exact immutable parent evidence."""
    if parent is None:
        return rollout
    pointer = artifact_input(store.read(parent.rollout_id).manifest)
    return rollout.model_copy(
        update={
            "continuation_of": pointer,
            "inputs": sorted_unique_inputs(*rollout.inputs, pointer),
        }
    )
