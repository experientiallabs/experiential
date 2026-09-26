"""Runtime pin verification for explicit larger-budget evaluation continuations."""

from exp.common.core.artifacts import ArtifactEnvelope, ArtifactInput
from exp.common.evaluations import EvaluationPlan
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.prepare import PreparedModelEvaluation
from exp.runtime.agents import agent_factory_sha256
from exp.simulation.specs import SimulationSpec


class EvaluationRuntimeContract(ArtifactEnvelope):
    """Immutable runtime, judge and quote binding included in the evaluation identity.

    Attributes:
        contract_id: Immutable runtime contract identity.
        prepared: Exact authorized inputs and quote retained for replay and continuation.
    """

    contract_id: str
    prepared: PreparedModelEvaluation


def validate_continuation(project: ProjectStore, prepared: PreparedModelEvaluation) -> None:
    """Reject any non-budget execution change before resolving provider credentials.

    Args:
        project: Owner of the parent simulation and runtime contracts.
        prepared: Child preparation with an explicit parent specification pointer.

    Raises:
        ValueError: The prior runtime is absent, altered, custom, or semantically different.
    """
    pointer = prepared.setup.continuation_of
    if pointer is None:
        return
    config = project.load_project()
    if config.agent is not None:
        raise ValueError("continuation requires the built-in chat runtime")
    _verify_pointer(project, pointer, "simulation-spec")
    spec = SimulationSpec.model_validate_json(
        project.artifacts.read_bytes(pointer.artifact_id, "simulation-spec.json")
    )
    plan = EvaluationPlan.model_validate_json(
        project.artifacts.read_bytes(spec.evaluation_plan_id, "evaluation-plan.json")
    )
    contracts = []
    for item in plan.inputs:
        if (
            project.artifacts.read(item.artifact_id).manifest.artifact_type
            != "evaluation-runtime-contract"
        ):
            continue
        _verify_pointer(project, item, "evaluation-runtime-contract")
        contracts.append(
            EvaluationRuntimeContract.model_validate_json(
                project.artifacts.read_bytes(item.artifact_id, "runtime.json")
            )
        )
    if len(contracts) != 1:
        raise ValueError("continuation requires one exact parent evaluation runtime contract")
    parent = contracts[0].prepared
    mutable = {
        "maximum_steps",
        "maximum_rollout_output_tokens",
        "maximum_concurrency",
        "continuation_of",
        "run_id",
    }
    if parent.setup.model_dump(exclude=mutable) != prepared.setup.model_dump(exclude=mutable):
        raise ValueError("continuation must retain worker, world, judge and reservation settings")
    if (
        parent.judge_setup != prepared.judge_setup
        or parent.judge_request != prepared.judge_request
        or parent.embedder_alias != prepared.embedder_alias
        or parent.redacted_field_names != prepared.redacted_field_names
        or parent.agent_factory_sha256
        != agent_factory_sha256(
            config.agent,
            maximum_model_calls=parent.setup.maximum_steps,
            system_prompt=config.system.system_prompt if config.system else None,
        )
    ):
        raise ValueError("continuation agent, judge or redaction settings changed")


def _verify_pointer(project: ProjectStore, pointer: ArtifactInput, kind: str) -> None:
    """Verify the manifest digest and expected artifact type of a parent input."""
    manifest = project.artifacts.read(pointer.artifact_id).manifest
    if manifest.artifact_type != kind or artifact_input(manifest) != pointer:
        raise ValueError("continuation parent artifact identity changed")
