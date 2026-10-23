"""Publish related executable task candidates with preserved environment state and lineage."""

from datetime import datetime
from typing import Literal

from pydantic import Field, model_validator

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    ContractModel,
    sha256_json,
    stable_id,
)
from exp.common.core.locks import CoordinationLock
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.continuations.intervention import (
    ActionIntervention,
    saved_intervention,
)
from exp.optimize.evaluation.executable.scenarios import (
    RecordScenarioResult,
    RecordScenarioSpec,
    saved_record_scenario,
)
from exp.optimize.evaluation.executable.task import (
    ExecutableTask,
    load_executable_task,
    persist_executable_task,
)
from exp.runtime.agents.chat import ChatAgentRuntime
from exp.simulation.capability_briefs import load_capability_brief
from exp.simulation.continuations import (
    ContinuationCheckpoint,
    load_continuation_checkpoint,
    persist_continuation_checkpoint,
)
from exp.simulation.evaluation.tasks import TaskVerifier


class TaskVariantSpec(ContractModel):
    """One authored candidate for independent feasibility and verifier qualification.

    Attributes:
        base: Existing executable task whose world, tools and lineage remain authoritative.
        kind: Decision-point drill, wording variation, or state changed by a saved intervention.
        prompt: Complete new public instruction, without private grader text.
        intent: Useful outcome, required to match the supplied capability brief.
        verifier: Explicit candidate grader; never inherits independent acceptance evidence.
        brief: Versioned capability brief, including unresolved assumptions.
        intervention: Exact saved action result for state_variation only.
        scenario: Replay-verified setup result for scenario_variation. Its new world must
            preserve the original lineage and explicitly cite the base checkpoint.
        proposal: Optional original proposal response, retained as research provenance only.
    """

    base: ArtifactInput
    kind: Literal["drill", "rephrasing", "state_variation", "scenario_variation"]
    prompt: str = Field(min_length=1, max_length=20000)
    intent: str = Field(min_length=1, max_length=4000)
    verifier: TaskVerifier
    brief: ArtifactInput
    intervention: ArtifactInput | None = None
    scenario: ArtifactInput | None = None
    proposal: ArtifactInput | None = Field(default=None, exclude_if=lambda value: value is None)

    @model_validator(mode="after")
    def explicit_state_change(self) -> "TaskVariantSpec":
        """State variants require executed effects; wording and drills cannot smuggle in state."""
        if (self.kind == "state_variation") != (self.intervention is not None):
            raise ValueError("only state_variation requires an intervention receipt")
        if (self.kind == "scenario_variation") != (self.scenario is not None):
            raise ValueError("only scenario_variation requires a reached scenario receipt")
        return self

    def identity(self, code_revision: str) -> str:
        """Freeze the complete authored change and its source evidence."""
        return stable_id("task-variant", {"spec": sha256_json(self), "code": code_revision})


class TaskVariant(ContractModel):
    """A related development task candidate, never a newly independent task or qualified reward.

    Attributes:
        spec: Frozen authored change.
        task: Executable candidate consumed by the existing qualification and Harbor owners.
        checkpoint: Exact changed actor context paired with its separately bound environment.
        lineage: Unchanged source family shared by parent, drill and every variation.
        scope: Required independent checks; rephrasing equivalence is not presumed.
    """

    spec: TaskVariantSpec
    task: ArtifactInput
    checkpoint: ArtifactInput
    lineage: str
    scope: str = (
        "development candidate; related source family; requires fresh feasibility and verifier "
        "qualification; prompt equivalence and native-world fidelity are not established"
    )


def _sources(spec: TaskVariantSpec) -> tuple[ArtifactInput, ...]:
    """Bind all declared intent and executed-state dependencies, independent of presentation."""
    return tuple(
        sorted(
            {
                spec.base,
                spec.brief,
                *((spec.intervention,) if spec.intervention else ()),
                *((spec.scenario,) if spec.scenario else ()),
                *((spec.proposal,) if spec.proposal else ()),
            },
            key=lambda ref: ref.artifact_id,
        )
    )


def _parent(project: ProjectStore, spec: TaskVariantSpec) -> tuple[ExecutableTask, ArtifactInput]:
    """Load the exact state source, refusing altered tools or an unrelated intervention."""
    base = load_executable_task(project, spec.base)
    if (
        spec.proposal is not None
        and artifact_input(project.artifacts.read(spec.proposal.artifact_id).manifest)
        != spec.proposal
    ):
        raise ValueError("task proposal differs from its original evidence")
    if spec.kind == "rephrasing" and (
        spec.verifier != base.job.study.verifier or spec.intent != base.intent
    ):
        raise ValueError("rephrasing must keep the same intent and verifier")
    checkpoint = base.job.study.checkpoint
    if spec.intervention is not None:
        manifest = project.artifacts.read(spec.intervention.artifact_id).manifest
        if (
            artifact_input(manifest) != spec.intervention
            or manifest.artifact_type != "action-intervention"
        ):
            raise ValueError("state variation requires the exact saved intervention result")
        intervention = ActionIntervention.model_validate_json(
            project.artifacts.read_bytes(spec.intervention.artifact_id, "intervention.json")
        )
        # Contrast replicates have their own registration and count. Those measurements do
        # not change the actor/world/verifier policy, and none of their acceptance transfers.
        measurement_fields = {
            name: getattr(base.job.study, name)
            for name in (
                "repeats",
                "selector",
                "critic_prediction",
                "predicted_success",
                "selection_plan",
                "training_specification",
            )
        }
        effect_job = intervention.spec.base.model_copy(
            update={"study": intervention.spec.base.study.model_copy(update=measurement_fields)}
        )
        if (
            effect_job != base.job
            or spec.intervention.artifact_id
            != intervention.spec.identity(manifest.code_revision) + "-result"
            or saved_intervention(project, intervention.spec, code_revision=manifest.code_revision)
            != intervention
        ):
            raise ValueError("state intervention does not belong to this exact base task")
        checkpoint = intervention.checkpoint
    if spec.scenario is not None:
        manifest = project.artifacts.read(spec.scenario.artifact_id).manifest
        if (
            artifact_input(manifest) != spec.scenario
            or manifest.artifact_type != "record-scenario-result"
        ):
            raise ValueError("scenario variation requires an exact reached scenario result")
        reached = RecordScenarioResult.model_validate_json(
            project.artifacts.read_bytes(spec.scenario.artifact_id, "result.json")
        )
        scenario = RecordScenarioSpec.model_validate_json(
            project.artifacts.read_bytes(reached.specification.artifact_id, "specification.json")
        )
        if (
            spec.scenario.artifact_id != scenario.identity(manifest.code_revision) + "-result"
            or saved_record_scenario(project, scenario, code_revision=manifest.code_revision)
            != reached
        ):
            raise ValueError("scenario variation lost its replay-verified setup")
        parent = load_continuation_checkpoint(project.artifacts, checkpoint)
        child = load_continuation_checkpoint(project.artifacts, reached.checkpoint)
        if (
            checkpoint not in scenario.capture.sources
            or child.task.lineage_group_id != parent.task.lineage_group_id
            or set(child.task.source_trace_ids) != set(parent.task.source_trace_ids)
            or child.task.tools != parent.task.tools
            or scenario.capture.maximum_model_calls != base.job.maximum_model_calls
            or scenario.capture.system_prompt != base.job.system_prompt
        ):
            raise ValueError(
                "scenario variation must retain parent provenance, tools and actor policy"
            )
        base = base.model_copy(
            update={
                "job": base.job.model_copy(
                    update={
                        "runtime_file": scenario.capture.runtime_file,
                        "runtime_sha256": scenario.capture.runtime_sha256,
                        "configuration": scenario.capture.configuration,
                        "study": base.job.study.model_copy(
                            update={"probe_sha256": scenario.capture.probe_sha256}
                        ),
                    }
                )
            }
        )
        checkpoint = reached.checkpoint
    return base, checkpoint


def _capsule(
    project: ProjectStore, spec: TaskVariantSpec, registration: ArtifactInput | None
) -> tuple[ExecutableTask, ContinuationCheckpoint]:
    """Change actor wording while keeping the world initialized against its original task."""
    base, parent_ref = _parent(project, spec)
    parent = load_continuation_checkpoint(project.artifacts, parent_ref)
    actor = ChatAgentRuntime(
        maximum_model_calls=base.job.maximum_model_calls, system_prompt=base.job.system_prompt
    )
    prefix = actor.initial_checkpoint(parent.task).messages
    if parent.actor.messages[: len(prefix)] != prefix:
        raise ValueError("task variant cannot identify the exact initial instruction prefix")
    task = parent.task.model_copy(
        update={
            "task_id": stable_id("variant-task", {"spec": sha256_json(spec)}),
            "instruction": spec.prompt,
        }
    )
    load_capability_brief(project.artifacts, spec.brief).require_task(task, spec.intent)
    messages = actor.initial_checkpoint(task).messages + parent.actor.messages[len(prefix) :]
    state = parent.actor.model_copy(update={"messages": messages, "task_sha256": sha256_json(task)})
    return base, ContinuationCheckpoint(
        task=task,
        environment_task=parent.world_task,
        actor=state,
        world=parent.world,
        sources=(parent_ref,) + ((registration,) if registration else ()),
    )


def _task(base: ExecutableTask, spec: TaskVariantSpec, checkpoint: ArtifactInput) -> ExecutableTask:
    """Clear old predictions and training acceptance: a new task must earn its own evidence."""
    study = base.job.study.model_copy(
        update={
            "checkpoint": checkpoint,
            "verifier": spec.verifier,
            "critic_prediction": None,
            "predicted_success": None,
            "selection_plan": None,
            "training_specification": None,
            "selector": "authored task family " + sha256_json(spec),
        }
    )
    return base.model_copy(
        update={
            "intent": spec.intent,
            "capability_brief": spec.brief,
            "job": base.job.model_copy(update={"study": study}),
        }
    )


def load_task_variant(project: ProjectStore, reference: ArtifactInput) -> TaskVariant:
    """Reproduce the exact authored context/state change and every dependency without imports."""
    store = project.artifacts
    manifest = store.read(reference.artifact_id).manifest
    if artifact_input(manifest) != reference or manifest.artifact_type != "task-variant":
        raise ValueError("task variant reference differs")
    result = TaskVariant.model_validate_json(
        store.read_bytes(reference.artifact_id, "variant.json")
    )
    identity = result.spec.identity(manifest.code_revision)
    registration = store.read(identity).manifest
    if (
        reference.artifact_id != identity + "-result"
        or registration.inputs != _sources(result.spec)
        or TaskVariantSpec.model_validate_json(store.read_bytes(identity, "spec.json"))
        != result.spec
    ):
        raise ValueError("task variant changed its registered proposal")
    reg = artifact_input(registration)
    base, expected = _capsule(project, result.spec, reg)
    if (
        load_continuation_checkpoint(store, result.checkpoint) != expected
        or load_executable_task(project, result.task) != _task(base, result.spec, result.checkpoint)
        or result.lineage != expected.task.lineage_group_id
        or set(manifest.inputs) != {reg, result.task, result.checkpoint}
    ):
        raise ValueError("task variant changed more than its registered intervention")
    return result


def _publish_task_variant(
    project: ProjectStore, spec: TaskVariantSpec, *, created_at: datetime, code_revision: str
) -> ArtifactInput:
    """Publish an executable candidate without inference or native effects."""
    store = project.artifacts
    identity = spec.identity(code_revision)
    if store.exists(identity + "-result"):
        reference = artifact_input(store.read(identity + "-result").manifest)
        load_task_variant(project, reference)
        return reference
    _parent(project, spec)
    envelope = ArtifactEnvelope(
        schema_version=1, created_at=created_at, code_revision=code_revision, inputs=_sources(spec)
    )
    if not store.exists(identity):
        store.write_json(
            artifact_id=identity,
            artifact_type="task-variant-proposal",
            envelope=envelope,
            files={"spec.json": spec},
        )
    registration = artifact_input(store.read(identity).manifest)
    base, capsule = _capsule(project, spec, registration)
    checkpoint = persist_continuation_checkpoint(
        store, capsule, created_at=created_at, code_revision=code_revision
    )
    task = persist_executable_task(
        project, _task(base, spec, checkpoint), created_at=created_at, code_revision=code_revision
    )
    result = TaskVariant(
        spec=spec, task=task, checkpoint=checkpoint, lineage=capsule.task.lineage_group_id
    )
    store.write_json(
        artifact_id=identity + "-result",
        artifact_type="task-variant",
        envelope=envelope.model_copy(
            update={
                "inputs": tuple(
                    sorted({registration, task, checkpoint}, key=lambda ref: ref.artifact_id)
                )
            }
        ),
        files={"variant.json": result},
    )
    reference = artifact_input(store.read(identity + "-result").manifest)
    load_task_variant(project, reference)
    return reference


def preflight_task_variant(project: ProjectStore, spec: TaskVariantSpec) -> None:
    """Resolve sources and actor-prefix transformation without publishing or opening a world."""
    spec = TaskVariantSpec.model_validate_json(spec.model_dump_json())
    _capsule(project, spec, None)


def publish_task_variant(
    project: ProjectStore, spec: TaskVariantSpec, *, created_at: datetime, code_revision: str
) -> ArtifactInput:
    """Serialize candidate publication; identical re-entry never creates another task version."""
    spec = TaskVariantSpec.model_validate_json(spec.model_dump_json())
    with CoordinationLock(
        project.artifacts.project_directory / (spec.identity(code_revision) + ".lock"), timeout=0
    ):
        preflight_task_variant(project, spec)
        return _publish_task_variant(
            project, spec, created_at=created_at, code_revision=code_revision
        )
