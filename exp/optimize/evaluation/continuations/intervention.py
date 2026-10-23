"""Scripted action interventions on exact development checkpoints, with durable effect receipts."""

from datetime import UTC, datetime

from jsonschema import Draft202012Validator
from pydantic import Field

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    ContractModel,
    JsonObject,
    sha256_json,
    stable_id,
)
from exp.common.core.json_schema import expand_local_references
from exp.common.core.locks import CoordinationLock
from exp.common.models import AssistantAction, ModelMessage, ToolCall
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.continuations.job import (
    ContinuationJob,
    load_continuation_runtime,
    preflight_continuation_job,
)
from exp.optimize.evaluation.continuations.study import restore_continuation_world
from exp.runtime.environments.interface import Observation
from exp.simulation.continuations import (
    ContinuationCheckpoint,
    load_continuation_checkpoint,
    persist_continuation_checkpoint,
)
from exp.simulation.development_scope import require_development_task


class ActionInterventionSpec(ContractModel):
    """One deliberately imposed action, never presented as an actor's sampled decision.

    Attributes:
        base: Exact pre-action checkpoint, runtime, actor policy and later grading contract.
        action: Single task-visible call executed in an isolated restored world.
        rationale: Hypothesis tested by this intervention, not a causal conclusion.
        evidence: Saved observations supporting the hypothesis; no label authority implied.
    """

    base: ContinuationJob
    action: ToolCall
    rationale: str = Field(min_length=1, max_length=4000)
    evidence: tuple[ArtifactInput, ...] = ()

    def identity(self, code_revision: str) -> str:
        """Freeze the imposed decision and exact original world before any runtime import."""
        return stable_id("action-intervention", {"spec": sha256_json(self), "code": code_revision})


class ActionIntervention(ContractModel):
    """Saved tool response and resulting paired state for subsequent continuation measurement.

    Attributes:
        spec: Original hypothesis and exact imposed action.
        checkpoint: Resulting state and public context, including the actual tool reply.
        observation: Actual native or simulated observation, retaining its own authority tags.
        before: Evaluator-only projection before the action.
        after: Evaluator-only projection after the action.
        scope: Scripted prefix and sampling limits; no unassisted actor-success claim.
    """

    spec: ActionInterventionSpec
    checkpoint: ArtifactInput
    observation: Observation
    before: JsonObject
    after: JsonObject
    scope: str = (
        "development scripted action; actor inference allowance unchanged; resumed outcomes are "
        "conditional on this imposed action, not unassisted task success or native fidelity"
    )


def _context(
    original: ContinuationCheckpoint, action: ToolCall, observation: Observation
) -> tuple[ModelMessage, ...]:
    """Append an authored action and its actual reply without manufacturing a model response."""
    return original.actor.messages + (
        ModelMessage(role="assistant", assistant_action=AssistantAction(tool_calls=(action,))),
        ModelMessage(role="tool", content=observation.content, tool_call_id=action.call_id),
    )


def preflight_intervention(project: ProjectStore, spec: ActionInterventionSpec) -> None:
    """Check development eligibility, call identity and schema before any runtime or effect."""
    preflight_continuation_job(project, spec.base)
    original = load_continuation_checkpoint(project.artifacts, spec.base.study.checkpoint)
    tools = {tool.name: tool for tool in original.task.tools}
    if spec.action.name not in tools:
        raise ValueError("intervention action is not visible to this task")
    used = {
        call.call_id
        for message in original.actor.messages
        if message.assistant_action is not None
        for call in message.assistant_action.tool_calls
    }
    if spec.action.call_id in used:
        raise ValueError("intervention call ID collides with the original prefix")
    schema = expand_local_references(tools[spec.action.name].input_schema)
    Draft202012Validator.check_schema(schema)
    if not Draft202012Validator(schema).is_valid(spec.action.arguments):
        raise ValueError("intervention arguments do not satisfy the task tool schema")
    for reference in spec.evidence:
        if artifact_input(project.artifacts.read(reference.artifact_id).manifest) != reference:
            raise ValueError("intervention hypothesis evidence differs from its saved digest")


def saved_intervention(
    project: ProjectStore, spec: ActionInterventionSpec, *, code_revision: str
) -> ActionIntervention | None:
    """Replay saved action effects without loading a runtime or executing any tool again."""
    store = project.artifacts
    identity = spec.identity(code_revision)
    if not store.exists(identity + "-result"):
        return None
    result = ActionIntervention.model_validate_json(
        store.read_bytes(identity + "-result", "intervention.json")
    )
    registration = store.read(identity).manifest
    if (
        result.spec != spec
        or registration.artifact_type != "action-intervention-intent"
        or ActionInterventionSpec.model_validate_json(store.read_bytes(identity, "spec.json"))
        != spec
        or set(registration.inputs) != {spec.base.study.checkpoint, *spec.evidence}
    ):
        raise ValueError("action intervention differs from its original intent")
    reference = artifact_input(registration)
    original = load_continuation_checkpoint(store, spec.base.study.checkpoint)
    checkpoint = load_continuation_checkpoint(store, result.checkpoint)
    require_development_task(store, original.task)
    actor = original.actor.model_copy(
        update={"messages": _context(original, spec.action, result.observation)}
    )
    if (
        checkpoint.actor != actor
        or checkpoint.task != original.task
        or checkpoint.environment_task != original.environment_task
        or set(checkpoint.sources) != {spec.base.study.checkpoint, reference}
        or set(store.read(identity + "-result").manifest.inputs) != {reference, result.checkpoint}
    ):
        raise ValueError("action intervention changed its visible prefix or source lineage")
    return result


def capture_action_intervention(
    project: ProjectStore, spec: ActionInterventionSpec, *, code_revision: str
) -> ActionIntervention:
    """Restore, impose one action and save once; an unknown native outcome is never retried.

    No actor inference is purchased. A simulated tool can use its separately bounded runtime
    provider. Model-call allowance is unchanged: the scripted action is an external treatment,
    not a fictional model draw. Compare treatments with the same starting capsule and allowances.
    """
    store = project.artifacts
    identity = spec.identity(code_revision)
    with CoordinationLock(store.project_directory / (identity + ".lock"), timeout=0):
        saved = saved_intervention(project, spec, code_revision=code_revision)
        if saved is not None:
            return saved
        preflight_intervention(project, spec)
        if store.exists(identity):
            raise ValueError("action intervention has an unknown outcome; no implicit redispatch")
        envelope = ArtifactEnvelope(
            schema_version=1,
            created_at=datetime.now(UTC),
            code_revision=code_revision,
            inputs=tuple(
                sorted(
                    {spec.base.study.checkpoint, *spec.evidence}, key=lambda ref: ref.artifact_id
                )
            ),
        )
        registration = artifact_input(
            store.write_json(
                artifact_id=identity,
                artifact_type="action-intervention-intent",
                envelope=envelope,
                files={"spec.json": spec},
            )
        )
        original = load_continuation_checkpoint(store, spec.base.study.checkpoint)
        with (
            load_continuation_runtime(project, spec.base) as runtime,
            restore_continuation_world(runtime.world, original.world_task, original.world) as world,
        ):
            before = runtime.probe(world.checkpoint())
            observation = world.execute(spec.action)
            image = world.checkpoint()
            after = runtime.probe(image)
            actor = original.actor.model_copy(
                update={"messages": _context(original, spec.action, observation)}
            )
            checkpoint = persist_continuation_checkpoint(
                store,
                ContinuationCheckpoint(
                    task=original.task,
                    environment_task=original.environment_task,
                    actor=actor,
                    world=image,
                    sources=(spec.base.study.checkpoint, registration),
                ),
                created_at=datetime.now(UTC),
                code_revision=code_revision,
            )
        result = ActionIntervention(
            spec=spec,
            checkpoint=checkpoint,
            observation=observation,
            before=before,
            after=after,
        )
        store.write_json(
            artifact_id=identity + "-result",
            artifact_type="action-intervention",
            envelope=envelope.model_copy(
                update={
                    "inputs": tuple(
                        sorted((registration, checkpoint), key=lambda ref: ref.artifact_id)
                    )
                }
            ),
            files={"intervention.json": result},
        )
        return result
