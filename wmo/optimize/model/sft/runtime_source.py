"""Resolve validated routed interactions into unfiltered immutable SFT sources."""

from __future__ import annotations

from dataclasses import dataclass

from wmo.common.core.artifacts import ArtifactInput, sha256_json, stable_id
from wmo.common.models import AssistantAction, ModelMessage
from wmo.common.project import ArtifactCorruptionError, ProjectStore, artifact_input
from wmo.optimize.model.sft.contracts import (
    AssistantActionEvent,
    RuntimeInteractionExampleSource,
    RuntimeSFTSource,
    SFTContextEvent,
    SFTExclusion,
    SFTMessage,
    SFTSourceReference,
    SFTTranscript,
    ToolEvent,
)
from wmo.optimize.model.sft.sources import PreparedSFTSource, SFTSourceVerificationError
from wmo.runtime.router import (
    RuntimeAcceptedEvent,
    RuntimeCompletedEvent,
    RuntimeTraceInteraction,
    load_runtime_trace_snapshot,
)


@dataclass(frozen=True)
class PreparedRuntimeSFTSnapshot:
    """Verified completed sources and excluded interactions from one runtime snapshot."""

    snapshot: ArtifactInput
    prepared: tuple[PreparedSFTSource, ...]
    references: tuple[SFTSourceReference, ...]
    exclusions: tuple[SFTExclusion, ...]


def resolve_runtime_source(
    store: ProjectStore,
    source: RuntimeSFTSource,
) -> PreparedRuntimeSFTSnapshot:
    """Resolve every completed routed interaction without applying a quality filter.

    Args:
        store: Project store containing the immutable runtime snapshot.
        source: Pointer to the exact snapshot whose completed interactions are requested.

    Returns:
        Completed interaction sources plus explicit failed and incomplete exclusions.

    Raises:
        SFTSourceVerificationError: The source is missing, corrupt, non-production, or cannot
            preserve its exact request and completed response.
    """
    try:
        loaded = load_runtime_trace_snapshot(store.artifacts, source.snapshot_id)
    except (ArtifactCorruptionError, ValueError) as exc:
        raise SFTSourceVerificationError(
            f"runtime SFT snapshot is unavailable or invalid: {source.snapshot_id}"
        ) from exc
    snapshot = loaded.snapshot
    if snapshot.project_id != store.paths.project_id:
        raise SFTSourceVerificationError("runtime SFT snapshot belongs to another project")
    if snapshot.source is None or snapshot.source.kind != "production":
        raise SFTSourceVerificationError("runtime SFT sources require production journal evidence")
    snapshot_input = artifact_input(loaded.manifest)
    prepared: list[PreparedSFTSource] = []
    references: list[SFTSourceReference] = []
    exclusions: list[SFTExclusion] = []
    for interaction in loaded.interactions:
        if interaction.completed_attempt_ordinal is None:
            reference, exclusion = _excluded_interaction(snapshot_input, interaction)
            references.append(reference)
            exclusions.append(exclusion)
            continue
        completed = _prepared_interaction(snapshot_input, interaction)
        prepared.append(completed)
        references.append(completed.reference())
    return PreparedRuntimeSFTSnapshot(
        snapshot=snapshot_input,
        prepared=tuple(prepared),
        references=tuple(references),
        exclusions=tuple(exclusions),
    )


def _prepared_interaction(
    snapshot: ArtifactInput,
    interaction: RuntimeTraceInteraction,
) -> PreparedSFTSource:
    """Convert one completed logical interaction into one eligible assistant target.

    Args:
        snapshot: Exact immutable snapshot manifest pointer.
        interaction: Verified logical interaction with one completed attempt.

    Returns:
        One source whose only eligible target is the completed routed response.

    Raises:
        SFTSourceVerificationError: Completion evidence or request history is ambiguous.
    """
    completed_attempt_ordinal = interaction.completed_attempt_ordinal
    if completed_attempt_ordinal is None:
        raise SFTSourceVerificationError("runtime SFT interaction has no completed target")
    completed_attempt = interaction.attempts[completed_attempt_ordinal - 1]
    completed = next(
        (
            event
            for event in completed_attempt.terminal_events
            if isinstance(event, RuntimeCompletedEvent)
        ),
        None,
    )
    if completed is None:
        raise SFTSourceVerificationError("runtime SFT completed attempt has no response event")
    accepted = completed_attempt.accepted
    history = _request_history(accepted.identity.request.messages)
    target_index = len(history)
    try:
        transcript = SFTTranscript(
            events=(
                *history,
                AssistantActionEvent(action=completed.response.output, approved=True),
            )
        ).events
    except ValueError as exc:
        raise SFTSourceVerificationError(
            "runtime SFT request and target do not form a valid transcript"
        ) from exc
    return PreparedSFTSource(
        kind="runtime_interaction",
        source_id=interaction.interaction_id,
        source_artifact=snapshot,
        source_record_sha256=sha256_json(interaction),
        leakage_group_id=stable_id(
            "sft-lineage",
            {
                "kind": "runtime_interaction",
                "source_lineage": accepted.identity.lineage_id,
            },
        ),
        task=_task_text(accepted),
        transcript_events=transcript,
        example_source=RuntimeInteractionExampleSource(
            snapshot=snapshot,
            interaction_id=interaction.interaction_id,
            accepted_ordinal=accepted.ordinal,
            completed_ordinal=completed.ordinal,
        ),
        score=None,
        direct_inputs=(snapshot,),
        acceptance_rule_id=None,
        acceptance_evidence_id=None,
        acceptance_evidence=None,
        target_action_indexes=(target_index,),
    )


def _excluded_interaction(
    snapshot: ArtifactInput,
    interaction: RuntimeTraceInteraction,
) -> tuple[SFTSourceReference, SFTExclusion]:
    """Describe why one non-completed runtime interaction emits no SFT target.

    Args:
        snapshot: Exact immutable snapshot manifest pointer.
        interaction: Verified failed or still-open logical interaction.

    Returns:
        Rejected source reference and matching inspection exclusion.
    """
    accepted = interaction.attempts[0].accepted
    failed = any(attempt.terminal_events for attempt in interaction.attempts)
    reason = "runtime_interaction_failed" if failed else "runtime_interaction_incomplete"
    detail = (
        "runtime interaction ended without a completed response"
        if failed
        else "runtime interaction has no completed response"
    )
    leakage_group_id = stable_id(
        "sft-lineage",
        {
            "kind": "runtime_interaction",
            "source_lineage": accepted.identity.lineage_id,
        },
    )
    reference = SFTSourceReference(
        kind="runtime_interaction",
        source_id=interaction.interaction_id,
        source_artifact=snapshot,
        source_record_sha256=sha256_json(interaction),
        leakage_group_id=leakage_group_id,
        acceptance_evidence=None,
        accepted=False,
        exclusion_reason=detail,
    )
    exclusion = SFTExclusion(
        source_kind="runtime_interaction",
        source_id=interaction.interaction_id,
        action_index=None,
        reason=reason,
        detail=detail,
    )
    return reference, exclusion


def _request_history(messages: tuple[ModelMessage, ...]) -> tuple[SFTContextEvent, ...]:
    """Convert exact request-visible messages into canonical SFT context.

    Args:
        messages: Ordered visible messages sent on the completed routed attempt.

    Returns:
        Canonical context preserving text, assistant actions, tool calls, and tool results.

    Raises:
        SFTSourceVerificationError: A message has ambiguous or incomplete canonical content.
    """
    history: list[SFTContextEvent] = []
    tool_names: dict[str, str] = {}
    for message in messages:
        if message.role == "system":
            if message.content is None:
                raise SFTSourceVerificationError("runtime SFT text message has no content")
            history.append(SFTMessage(role="system", content=message.content))
            continue
        if message.role == "user":
            if message.content is None:
                raise SFTSourceVerificationError("runtime SFT text message has no content")
            history.append(SFTMessage(role="user", content=message.content))
            continue
        if message.role == "assistant":
            action = _message_action(message)
            history.append(AssistantActionEvent(action=action, approved=True))
            for call in action.tool_calls:
                tool_names[call.call_id] = call.name
            continue
        if message.content is None or message.tool_call_id is None:
            raise SFTSourceVerificationError("runtime SFT tool message is incomplete")
        history.append(
            ToolEvent(
                tool_call_id=message.tool_call_id,
                content=message.content,
                tool_name=tool_names.get(message.tool_call_id),
            )
        )
    return tuple(history)


def _message_action(message: ModelMessage) -> AssistantAction:
    """Recover one unambiguous assistant action from request history.

    Args:
        message: Request-visible assistant message.

    Returns:
        Complete assistant action including text and ordered tool calls.

    Raises:
        SFTSourceVerificationError: Structured and plain text representations conflict.
    """
    action = message.assistant_action
    if action is None:
        if message.content is None:
            raise SFTSourceVerificationError("runtime SFT assistant message has no action")
        return AssistantAction(content=message.content)
    if message.content is None or message.content == action.content:
        return action
    if action.content is None:
        return action.model_copy(update={"content": message.content})
    raise SFTSourceVerificationError(
        "runtime SFT assistant message has conflicting text and structured action"
    )


def _task_text(accepted: RuntimeAcceptedEvent) -> str:
    """Choose the stable user-visible task for one routed request.

    Args:
        accepted: Accepted event containing the exact routed request.

    Returns:
        Last user text, first available message text, or a stable fallback.
    """
    for message in reversed(accepted.identity.request.messages):
        if message.role == "user" and message.content:
            return message.content
    for message in accepted.identity.request.messages:
        if message.content:
            return message.content
    return "Complete the routed model request."
