"""Lineage-safe observation stitching for immutable routed interaction snapshots."""

from __future__ import annotations

from collections import defaultdict

from pydantic import JsonValue, TypeAdapter

from exp.common.core.artifacts import ArtifactInput, JsonObject, stable_id
from exp.common.models import AssistantAction, ModelMessage, ToolCall
from exp.common.project import artifact_input
from exp.common.traces import Trace, TraceSpan
from exp.runtime.router import (
    PersistedRuntimeTraceExport,
    RuntimeAcceptedEvent,
    RuntimeCompletedEvent,
    RuntimeTraceInteraction,
)

_JSON_OBJECT_ADAPTER = TypeAdapter(JsonObject)


class RuntimeTraceStitchingError(ValueError):
    """Runtime interaction lineage cannot produce trustworthy observed transitions."""


def stitch_runtime_observations(
    export: PersistedRuntimeTraceExport,
) -> tuple[Trace, ...]:
    """Attach later real observations to completed outputs within each runtime lineage.

    The canonical snapshot traces remain present for future training. A completed assistant output
    gains retrieval-visible spans only when the next accepted interaction in the same lineage
    contains that exact output followed by a new user or matching tool-result message.

    Args:
        export: Verified runtime prefix snapshot and its canonical trace dataset.

    Returns:
        Snapshot traces in canonical order, augmented only with lineage-safe observed pairs.

    Raises:
        RuntimeTraceStitchingError: Snapshot traces, interaction IDs, lineage history, timestamps,
            or continued request content differ from the sealed evidence.
    """
    traces_by_id = {trace.trace_id: trace for trace in export.traces}
    if len(traces_by_id) != len(export.traces):
        raise RuntimeTraceStitchingError("runtime snapshot trace IDs are not unique")
    dataset_input = artifact_input(export.dataset_manifest)
    snapshot_input = artifact_input(export.snapshot_manifest)
    interactions_by_lineage: dict[str, list[RuntimeTraceInteraction]] = defaultdict(list)
    for interaction in export.interactions:
        accepted = interaction.attempts[0].accepted
        interactions_by_lineage[accepted.identity.lineage_id].append(interaction)
    stitched: dict[str, Trace] = dict(traces_by_id)
    for lineage_id, interactions in interactions_by_lineage.items():
        ordered = tuple(sorted(interactions, key=lambda item: item.attempts[0].accepted.ordinal))
        for index, interaction in enumerate(ordered):
            if interaction.completed_attempt_ordinal is None:
                continue
            trace = traces_by_id.get(interaction.interaction_id)
            if trace is None:
                raise RuntimeTraceStitchingError(
                    f"completed runtime interaction {interaction.interaction_id} has no trace"
                )
            if trace.conversation_id != lineage_id:
                raise RuntimeTraceStitchingError(
                    f"runtime interaction {interaction.interaction_id} crossed trace lineages"
                )
            for next_interaction in ordered[index + 1 :]:
                candidate = _stitch_interaction(
                    trace,
                    interaction,
                    next_interaction,
                    snapshot_input=snapshot_input,
                    dataset_input=dataset_input,
                    allow_missing_output=next_interaction.completed_attempt_ordinal is None,
                )
                if candidate is None:
                    continue
                stitched[trace.trace_id] = candidate
                break
    return tuple(stitched[trace.trace_id] for trace in export.traces)


def _stitch_interaction(
    trace: Trace,
    interaction: RuntimeTraceInteraction,
    next_interaction: RuntimeTraceInteraction,
    *,
    snapshot_input: ArtifactInput,
    dataset_input: ArtifactInput,
    allow_missing_output: bool,
) -> Trace | None:
    """Build retrieval-visible spans from one completed output and its next request.

    Args:
        trace: Canonical request-scoped trace for the completed interaction.
        interaction: Completed logical interaction that produced the assistant action.
        next_interaction: Next accepted logical interaction in the same lineage.
        snapshot_input: Exact runtime snapshot manifest pointer.
        dataset_input: Exact canonical runtime trace-dataset manifest pointer.
        allow_missing_output: Whether an incomplete candidate may be skipped when it does not
            preserve the completed output.

    Returns:
        The trace plus zero or more observed pairs, or None for a candidate accepted before the
        output existed or an incomplete candidate that omits it.

    Raises:
        RuntimeTraceStitchingError: The next request does not continue the same output, reuses an
            event ordinal, or its timestamps and lineage differ from the completed interaction.
    """
    completed_ordinal = interaction.completed_attempt_ordinal
    if completed_ordinal is None:
        raise RuntimeTraceStitchingError("runtime observation source has no completed attempt")
    completed_attempt = interaction.attempts[completed_ordinal - 1]
    completed = next(
        event
        for event in completed_attempt.terminal_events
        if isinstance(event, RuntimeCompletedEvent)
    )
    accepted = completed_attempt.accepted
    next_accepted = next_interaction.attempts[0].accepted
    if next_accepted.identity.lineage_id != accepted.identity.lineage_id:
        raise RuntimeTraceStitchingError("runtime observation stitching cannot cross lineages")
    if next_accepted.ordinal < completed.ordinal:
        return None
    if next_accepted.ordinal == completed.ordinal:
        raise RuntimeTraceStitchingError("runtime observation events repeat an ordinal")
    if next_accepted.received_at < completed.completed_at:
        raise RuntimeTraceStitchingError("runtime observation timestamp precedes its completion")
    later_messages = _messages_after_output(
        next_accepted,
        completed.response.output,
        allow_missing=allow_missing_output,
    )
    if later_messages is None:
        return None
    provenance = _provenance(
        accepted,
        completed,
        next_accepted,
        snapshot_input=snapshot_input,
        dataset_input=dataset_input,
    )
    spans: list[TraceSpan] = []
    if completed.response.output.content is not None:
        user_message = next((item for item in later_messages if item.role == "user"), None)
        if user_message is not None and user_message.content:
            spans.extend(
                _message_spans(
                    completed,
                    next_accepted,
                    user_message.content,
                    provenance=provenance,
                )
            )
    for tool_call in completed.response.output.tool_calls:
        tool_result = next(
            (
                item
                for item in later_messages
                if item.role == "tool" and item.tool_call_id == tool_call.call_id
            ),
            None,
        )
        if tool_result is not None and tool_result.content:
            spans.extend(
                _tool_spans(
                    completed,
                    next_accepted,
                    tool_call,
                    tool_result.content,
                    provenance=provenance,
                )
            )
    if not spans:
        return trace
    context = dict(trace.initial_context)
    context["runtime_observation_provenance"] = provenance
    return trace.model_copy(
        update={
            "initial_context": _JSON_OBJECT_ADAPTER.validate_python(context),
            "spans": (*trace.spans, *spans),
        }
    )


def _messages_after_output(
    accepted: RuntimeAcceptedEvent,
    output: AssistantAction,
    *,
    allow_missing: bool,
) -> tuple[ModelMessage, ...] | None:
    """Return new messages after the exact prior assistant output in a continued request.

    Args:
        accepted: Next interaction acceptance containing visible request history.
        output: Exact completed assistant action expected in that history.
        allow_missing: Whether an incomplete candidate can be skipped when history omits output.

    Returns:
        Messages following the last exact output, or None for a skippable incomplete candidate.

    Raises:
        RuntimeTraceStitchingError: The next same-lineage request omits the completed output.
    """
    matching_indexes = tuple(
        index
        for index, message in enumerate(accepted.identity.request.messages)
        if _message_matches_output(message, output)
    )
    if not matching_indexes:
        if allow_missing:
            return None
        raise RuntimeTraceStitchingError(
            f"runtime interaction {accepted.interaction_id} does not preserve the prior "
            "assistant output in its same-lineage request"
        )
    return accepted.identity.request.messages[matching_indexes[-1] + 1 :]


def _message_matches_output(message: ModelMessage, output: AssistantAction) -> bool:
    """Recognize the canonical structured or plain-text spelling of one assistant output.

    Args:
        message: Visible message from the next accepted request.
        output: Prior completed assistant action expected in its history.

    Returns:
        True only for an exact assistant action, or equivalent plain text without tool calls.
    """
    if message.role != "assistant":
        return False
    if message.assistant_action == output:
        return True
    return (
        not output.tool_calls
        and output.content is not None
        and message.assistant_action is None
        and message.content == output.content
    )


def _provenance(
    accepted: RuntimeAcceptedEvent,
    completed: RuntimeCompletedEvent,
    next_accepted: RuntimeAcceptedEvent,
    *,
    snapshot_input: ArtifactInput,
    dataset_input: ArtifactInput,
) -> JsonObject:
    """Render exact runtime identities consumed by one stitched observation.

    Args:
        accepted: Acceptance that produced the assistant output.
        completed: Completed response event for that acceptance.
        next_accepted: Later acceptance containing the real observation.
        snapshot_input: Exact runtime snapshot manifest pointer.
        dataset_input: Exact canonical runtime trace-dataset manifest pointer.

    Returns:
        Canonical JSON provenance copied into the trace and derived spans.
    """
    return _JSON_OBJECT_ADAPTER.validate_python(
        {
            "interaction_id": accepted.interaction_id,
            "accepted_ordinal": accepted.ordinal,
            "completed_ordinal": completed.ordinal,
            "lineage_id": accepted.identity.lineage_id,
            "next_interaction_id": next_accepted.interaction_id,
            "next_accepted_ordinal": next_accepted.ordinal,
            "runtime_snapshot_input": snapshot_input.model_dump(mode="json"),
            "runtime_trace_dataset_input": dataset_input.model_dump(mode="json"),
        }
    )


def _message_spans(
    completed: RuntimeCompletedEvent,
    next_accepted: RuntimeAcceptedEvent,
    observation: str,
    *,
    provenance: JsonObject,
) -> tuple[TraceSpan, TraceSpan]:
    """Create one assistant-message to later-user-message observed pair.

    Args:
        completed: Completed response that contains the assistant message.
        next_accepted: Later acceptance that contains the user observation.
        observation: Exact nonempty later user text.
        provenance: Immutable runtime source identities and ordinals.

    Returns:
        Ordered action and observation spans using supported GenAI attributes.
    """
    action_id = stable_id("runtime-rag-action", {"event_id": completed.event_id, "kind": "message"})
    observation_id = stable_id(
        "runtime-rag-observation",
        {"event_id": next_accepted.event_id, "action_span_id": action_id},
    )
    action_attributes = _JSON_OBJECT_ADAPTER.validate_python(
        {
            "gen_ai.output.messages": [
                {"role": "assistant", "content": completed.response.output.content}
            ],
            "runtime.observation_provenance": provenance,
        }
    )
    observation_attributes = _JSON_OBJECT_ADAPTER.validate_python(
        {
            "gen_ai.input.messages": [{"role": "user", "content": observation}],
            "runtime.observation_provenance": provenance,
        }
    )
    return (
        TraceSpan(
            span_id=action_id,
            name="exp.runtime.observed_assistant_message",
            started_at=completed.completed_at,
            ended_at=completed.completed_at,
            attributes=action_attributes,
            model=completed.response.model,
            usage=completed.response.economics.usage,
        ),
        TraceSpan(
            span_id=observation_id,
            parent_span_id=action_id,
            name="exp.runtime.observed_user_message",
            started_at=next_accepted.received_at,
            ended_at=next_accepted.received_at,
            attributes=observation_attributes,
        ),
    )


def _tool_spans(
    completed: RuntimeCompletedEvent,
    next_accepted: RuntimeAcceptedEvent,
    tool_call: ToolCall,
    observation: str,
    *,
    provenance: JsonObject,
) -> tuple[TraceSpan, TraceSpan]:
    """Create one assistant tool-call to matching later tool-result observed pair.

    Args:
        completed: Completed response that contains the tool call.
        next_accepted: Later acceptance that contains the tool result.
        tool_call: Exact assistant tool action being paired.
        observation: Matching nonempty tool-result text.
        provenance: Immutable runtime source identities and ordinals.

    Returns:
        Ordered tool action and tool-result spans using supported GenAI attributes.
    """
    action_id = stable_id(
        "runtime-rag-action",
        {"event_id": completed.event_id, "kind": "tool_call", "call_id": tool_call.call_id},
    )
    observation_id = stable_id(
        "runtime-rag-observation",
        {"event_id": next_accepted.event_id, "action_span_id": action_id},
    )
    common: dict[str, JsonValue] = {
        "gen_ai.tool.call.id": tool_call.call_id,
        "gen_ai.tool.name": tool_call.name,
        "runtime.observation_provenance": provenance,
    }
    action_attributes = _JSON_OBJECT_ADAPTER.validate_python(
        {**common, "gen_ai.tool.call.arguments": tool_call.arguments}
    )
    observation_attributes = _JSON_OBJECT_ADAPTER.validate_python(
        {
            **common,
            "gen_ai.operation.name": "execute_tool",
            "gen_ai.tool.message": observation,
        }
    )
    return (
        TraceSpan(
            span_id=action_id,
            name="exp.runtime.observed_tool_call",
            started_at=completed.completed_at,
            ended_at=completed.completed_at,
            attributes=action_attributes,
            model=completed.response.model,
            usage=completed.response.economics.usage,
        ),
        TraceSpan(
            span_id=observation_id,
            parent_span_id=action_id,
            name="exp.runtime.observed_tool_result",
            started_at=next_accepted.received_at,
            ended_at=next_accepted.received_at,
            attributes=observation_attributes,
        ),
    )
