"""Reconstruct checkpoint conversations from executed ancestors before positive supervision."""

from exp.common.core.artifacts import ArtifactInput
from exp.common.models import ModelMessage, ModelRequest, ModelResponse, ToolCall
from exp.common.project import ArtifactStore, artifact_input
from exp.optimize.evaluation.continuations.study import (
    ContinuationBranch,
    ContinuationStudySpec,
    saved_actor_call_evidence,
    study_identity,
)
from exp.runtime.agents.chat import task_messages
from exp.runtime.environments import Observation
from exp.simulation.continuations import ContinuationCheckpoint, load_continuation_checkpoint


def verify_training_prefix(store: ArtifactStore, reference: ArtifactInput) -> None:
    """Require initial instructions or an exact, recursively witnessed executed conversation.

    Authored conversation interventions remain usable for diagnostic studies, but cannot supply
    positive training context merely because a later continuation succeeded. This checks the
    trusted execution records; it does not establish that a simulator matches a native world.
    """
    current = reference
    seen: set[str] = set()
    while True:
        if current.artifact_id in seen:
            raise ValueError("continuation prefix contains a provenance cycle")
        seen.add(current.artifact_id)
        point = load_continuation_checkpoint(store, current)
        if point.actor.calls == 0:
            messages = point.actor.messages
            initial = task_messages(point.task)
            if messages != initial and not (
                len(messages) == len(initial) + 1
                and messages[0].role == "system"
                and messages[0].assistant_action is None
                and messages[0].tool_call_id is None
                and messages[1:] == initial
            ):
                raise ValueError("initial continuation prefix contains unexecuted conversation")
            return
        current = _executed_parent(store, current, point)


def _executed_parent(
    store: ArtifactStore, reference: ArtifactInput, point: ContinuationCheckpoint
) -> ArtifactInput:
    """Bind a captured boundary to its parent, registration, saved calls and observations."""
    parents = [
        ref
        for ref in point.sources
        if store.read(ref.artifact_id).manifest.artifact_type == "continuation-checkpoint"
    ]
    intents = [ref for ref in point.sources if ref.artifact_id.endswith("-intent")]
    if len(parents) != 1 or len(intents) != 1 or set(point.sources) != {*parents, *intents}:
        raise ValueError("continuation prefix lacks its executed parent and branch intent")
    parent_ref, intent = parents[0], intents[0]
    parent = load_continuation_checkpoint(store, parent_ref)
    spec = ContinuationStudySpec.model_validate_json(
        store.read_bytes(intent.artifact_id, "record.json")
    )
    revision = store.read(intent.artifact_id).manifest.code_revision
    registration_id = study_identity(spec, revision)
    registration = artifact_input(store.read(registration_id).manifest)
    branch_id = intent.artifact_id.removesuffix("-intent")
    branch = ContinuationBranch.model_validate_json(store.read_bytes(branch_id, "record.json"))
    if (
        spec.checkpoint != parent_ref
        or branch_id != f"{registration_id}-branch-{branch.index}"
        or store.read(intent.artifact_id).manifest.inputs != (registration,)
        or reference not in branch.checkpoints
        or point.task != parent.task
        or point.environment_task != parent.environment_task
        or point.actor.runtime_sha256 != parent.actor.runtime_sha256
        or point.actor.calls <= parent.actor.calls
        or branch.actor_calls != saved_actor_call_evidence(store, branch_id)
    ):
        raise ValueError("continuation prefix differs from its captured execution boundary")
    messages = list(parent.actor.messages)
    steps = branch.outcome.get("steps")
    if not isinstance(steps, list):
        raise ValueError("continuation prefix lacks saved tool observations")
    consumed = 0
    for index in range(point.actor.calls - parent.actor.calls):
        call_id = f"{branch_id}-call-{index}"
        request = ModelRequest.model_validate_json(
            store.read_bytes(call_id + "-intent", "request.json")
        )
        response = ModelResponse.model_validate_json(
            store.read_bytes(call_id + "-response", "response.json")
        )
        if (
            request.messages != tuple(messages)
            or request.tools != parent.task.tools
            or response.model != spec.actor_model
        ):
            raise ValueError("continuation prefix differs from its dispatched actor context")
        if not response.output.tool_calls:
            raise ValueError("continuation boundary follows a terminal actor response")
        messages.append(ModelMessage(role="assistant", assistant_action=response.output))
        for action in response.output.tool_calls:
            if consumed >= len(steps):
                raise ValueError("continuation prefix lacks an executed tool result")
            step = steps[consumed]
            consumed += 1
            if (
                not isinstance(step, dict)
                or step.get("status") != "valid"
                or ToolCall.model_validate(step.get("action")) != action
            ):
                raise ValueError("continuation prefix tool result belongs to a different action")
            reply = Observation.model_validate(step.get("reply"))
            messages.append(
                ModelMessage(role="tool", content=reply.content, tool_call_id=action.call_id)
            )
    if tuple(messages) != point.actor.messages:
        raise ValueError("continuation prefix contains an observation absent from execution")
    return parent_ref
