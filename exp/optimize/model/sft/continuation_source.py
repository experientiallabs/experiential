"""Witnessed continuation targets with exact visible prefixes and source-level exclusions."""

from exp.common.core.artifacts import ArtifactInput, JsonValue, sha256_json
from exp.common.models import AssistantAction, ModelMessage, ModelRequest, ModelResponse, ToolCall
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.continuations.admission import (
    ContinuationAdmission,
    admit_continuations,
)
from exp.optimize.evaluation.continuations.prefix import verify_training_prefix
from exp.optimize.evaluation.continuations.records import load_study
from exp.optimize.model.sft.contracts import (
    AssistantActionEvent,
    ContinuationExampleSource,
    SFTContextEvent,
    SFTExclusion,
    SFTMessage,
    SFTSourceReference,
    SFTTranscript,
    ToolEvent,
)
from exp.optimize.model.sft.sources import PreparedSFTSource, SFTSourceVerificationError
from exp.runtime.environments import Observation
from exp.simulation.continuations import load_continuation_checkpoint


def resolve_continuation_source(
    project: ProjectStore, reference: ArtifactInput
) -> tuple[tuple[PreparedSFTSource, ...], tuple[SFTExclusion, ...], tuple[SFTSourceReference, ...]]:
    """Re-derive mechanical admission and hydrate tool-only post-checkpoint targets.

    Observable goal and policy checks do not verify free-text explanations. Responses with prose
    stay verbatim in subsequent context, including prose accompanying a tool call. This path has
    no calibrated text-judgment authority and cannot turn those responses into supervised targets.
    Every target's complete tool batch must have matching, explicitly successful saved results.
    Error-producing actions remain in recovery context. This checks observed execution, not the
    usefulness of each action or the fidelity of a reconstructed environment.

    Args:
        project: Trusted project containing the study and its independent witnesses.
        reference: Exact admission report, never an asserted success flag.

    Returns:
        Accepted transcripts, exclusions and the complete planned source inventory.
    """
    store = project.artifacts
    manifest = store.read(reference.artifact_id).manifest
    if artifact_input(manifest) != reference or manifest.artifact_type != "continuation-admission":
        raise SFTSourceVerificationError("continuation admission differs from its saved artifact")
    admission = ContinuationAdmission.model_validate_json(
        store.read_bytes(reference.artifact_id, "admission.json")
    )
    verified = admit_continuations(
        store,
        admission.study,
        policy=admission.policy,
        created_at=manifest.created_at,
        code_revision=manifest.code_revision,
    )
    if verified != reference:
        raise SFTSourceVerificationError("continuation admission no longer re-derives")
    study = load_study(store, admission.study)
    checkpoint = load_continuation_checkpoint(store, study.spec.checkpoint)
    try:
        verify_training_prefix(store, study.spec.checkpoint)
    except ValueError as error:
        raise SFTSourceVerificationError(str(error)) from error
    prepared, exclusions, references = [], [], []
    for decision, branch in zip(admission.branches, study.branches, strict=True):
        identity = f"{admission.study.artifact_id}-branch-{branch.index}"
        references.append(
            SFTSourceReference(
                kind="continuation",
                source_id=identity,
                source_artifact=admission.study,
                source_record_sha256=sha256_json(branch),
                leakage_group_id=checkpoint.task.lineage_group_id,
                acceptance_evidence=reference,
                accepted=decision.admitted,
                exclusion_reason=None if decision.admitted else decision.reason,
            )
        )
        if not decision.admitted:
            exclusions.append(
                SFTExclusion(
                    source_kind="continuation",
                    source_id=identity,
                    reason="unapproved_action",
                    detail=decision.reason,
                )
            )
            continue
        calls = branch.outcome.get("model_calls")
        if not isinstance(calls, list) or not calls:
            raise SFTSourceVerificationError("admitted continuation has no saved model calls")
        steps = branch.outcome.get("steps")
        if not isinstance(steps, list):
            raise SFTSourceVerificationError("admitted continuation has no saved tool results")
        messages = list(checkpoint.actor.messages)
        events = list(_history(tuple(messages)))
        targets = []
        consumed = 0
        pending: tuple[ModelMessage, ...] = ()
        for call in calls:
            if not isinstance(call, dict):
                raise SFTSourceVerificationError("continuation model receipt is malformed")
            request = ModelRequest.model_validate(call.get("request"))
            response = ModelResponse.model_validate(call.get("response"))
            if response.model != study.spec.actor_model or request.tools != checkpoint.task.tools:
                raise SFTSourceVerificationError("continuation model or tool permissions changed")
            if request.messages[: len(messages)] != tuple(messages):
                raise SFTSourceVerificationError(
                    "continuation prefix changed across committed requests"
                )
            # Compare original observations, not merely the roles of intervening messages.
            suffix = request.messages[len(messages) :]
            if suffix != pending:
                raise SFTSourceVerificationError(
                    "continuation context differs from its saved tool results"
                )
            events.extend(_history(suffix))
            count = len(response.output.tool_calls)
            pending, result_failure = _tool_results(
                response.output.tool_calls, steps[consumed : consumed + count]
            )
            consumed += count
            if response.output.tool_calls and not response.output.content and not result_failure:
                targets.append(len(events))
            else:
                exclusions.append(
                    SFTExclusion(
                        source_kind="continuation",
                        source_id=identity,
                        action_index=len(events),
                        reason="unapproved_action",
                        detail=result_failure
                        or (
                            "mechanical outcome acceptance does not verify assistant prose; "
                            "the complete response remains context, never a rewritten target"
                        ),
                    )
                )
            events.append(AssistantActionEvent(action=response.output))
            messages = [
                *request.messages,
                ModelMessage(role="assistant", assistant_action=response.output),
            ]
        if consumed != len(steps):
            raise SFTSourceVerificationError("continuation has tool results without model actions")
        transcript = SFTTranscript(events=tuple(events))
        prepared.append(
            PreparedSFTSource(
                kind="continuation",
                source_id=identity,
                source_artifact=admission.study,
                source_record_sha256=sha256_json(branch),
                leakage_group_id=checkpoint.task.lineage_group_id,
                task=checkpoint.task.instruction,
                transcript_events=transcript.events,
                example_source=ContinuationExampleSource(
                    study=admission.study,
                    branch=branch.index,
                    acceptance_evidence=reference,
                    qualification=admission.evidence.qualification if admission.evidence else None,
                    qualification_scope=admission.evidence.qualification_scope
                    if admission.evidence
                    else None,
                ),
                score=None,
                direct_inputs=(admission.study, reference, study.spec.checkpoint),
                acceptance_rule_id=None,
                acceptance_evidence_id=reference.artifact_id,
                acceptance_evidence=reference,
                target_action_indexes=tuple(targets),
                tools=checkpoint.task.tools,
            )
        )
    return tuple(prepared), tuple(exclusions), tuple(references)


def _tool_results(
    actions: tuple[ToolCall, ...], steps: list[JsonValue]
) -> tuple[tuple[ModelMessage, ...], str | None]:
    """Bind ordered results to complete actions and withhold any unsuccessful tool batch.

    Args:
        actions: Complete, unmodified response batch, including call IDs and arguments.
        steps: Corresponding slice of the execution ledger, never matched by ID alone.

    Returns:
        Exact public observations and an exclusion reason if any result lacks success evidence.
    """
    if len(actions) != len(steps):
        raise SFTSourceVerificationError("continuation action lacks its saved tool result")
    messages, failures = [], []
    for action, step in zip(actions, steps, strict=True):
        if not isinstance(step, dict) or ToolCall.model_validate(step.get("action")) != action:
            raise SFTSourceVerificationError("continuation tool result names a different action")
        raw_reply = step.get("reply")
        if not isinstance(raw_reply, dict):
            raise SFTSourceVerificationError("continuation action lacks its saved observation")
        reply = Observation.model_validate(raw_reply)
        messages.append(
            ModelMessage(role="tool", content=reply.content, tool_call_id=action.call_id)
        )
        if step.get("status") != "valid" or raw_reply.get("is_error") is not False:
            failures.append(action.call_id)
    reason = (
        "tool result is an error or lacks explicit success for calls: "
        + ", ".join(failures)
        + "; the complete response remains context, never a rewritten target"
        if failures
        else None
    )
    return tuple(messages), reason


def _history(messages: tuple[ModelMessage, ...]) -> tuple[SFTContextEvent, ...]:
    """Preserve role and content without turning prefix assistant actions into targets."""
    events: list[SFTContextEvent] = []
    for message in messages:
        if message.content_parts:
            raise SFTSourceVerificationError(
                "multimodal continuation export needs a preserving renderer"
            )
        if message.role == "assistant":
            events.append(
                AssistantActionEvent(
                    action=message.assistant_action or AssistantAction(content=message.content)
                )
            )
        elif message.role == "tool":
            if message.tool_call_id is None or message.content is None:
                raise SFTSourceVerificationError("tool context lacks its exact call linkage")
            events.append(ToolEvent(tool_call_id=message.tool_call_id, content=message.content))
        else:
            if message.content is None:
                raise SFTSourceVerificationError("continuation context has no text")
            events.append(SFTMessage(role=message.role, content=message.content))
    return tuple(events)
