"""Extraction and versioned key rendering for real observed trace transitions."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence

from pydantic import JsonValue

from wmo.common.core.artifacts import canonical_json_bytes, stable_id
from wmo.common.core.text import normalize_durable_text
from wmo.common.traces import Trace, TraceSpan
from wmo.simulation.retrieval.contracts import (
    RAG_KEY_SCHEMA_VERSION,
    RAGAction,
    RAGLineageBinding,
    RAGObservation,
    RAGTransition,
)

_TOOL_OPERATION = "execute_tool"


def extract_fit_transitions(
    traces: Sequence[Trace],
    lineage_bindings: Sequence[RAGLineageBinding],
) -> tuple[RAGTransition, ...]:
    """Extract only fit-side real action-to-following-observation transitions.

    Args:
        traces: Verified canonical traces from immutable real-source datasets.
        lineage_bindings: Frozen leakage-safe assignment for every supplied trace.

    Returns:
        Deterministically ordered fit transitions. Terminal assistant messages without a real
        subsequent user or environment observation are intentionally absent.

    Raises:
        ValueError: Trace identities and lineage bindings differ or repeat.
    """
    return extract_real_transitions(
        traces,
        lineage_bindings,
        included_partitions=frozenset({"fit"}),
    )


def extract_real_transitions(
    traces: Sequence[Trace],
    lineage_bindings: Sequence[RAGLineageBinding],
    *,
    included_partitions: frozenset[str],
) -> tuple[RAGTransition, ...]:
    """Extract deterministic observed transitions from selected frozen real partitions.

    Args:
        traces: Verified real traces.
        lineage_bindings: Complete trace-to-lineage partition assignments.
        included_partitions: Fit, held-out, or both partitions to include.

    Returns:
        Sorted immutable real transitions from only the selected partitions.
    """
    trace_ids = tuple(trace.trace_id for trace in traces)
    if len(set(trace_ids)) != len(trace_ids):
        raise ValueError("RAG source datasets repeat a trace ID")
    binding_by_trace = {binding.trace_id: binding for binding in lineage_bindings}
    if len(binding_by_trace) != len(lineage_bindings):
        raise ValueError("RAG lineage bindings repeat a trace ID")
    if set(binding_by_trace) != set(trace_ids):
        raise ValueError("RAG lineage bindings must cover exactly the verified source traces")
    transitions = []
    for trace in traces:
        binding = binding_by_trace[trace.trace_id]
        if binding.partition not in included_partitions:
            continue
        transitions.extend(_trace_transitions(trace, binding.lineage_id))
    ordered = tuple(sorted(transitions, key=lambda item: item.transition_id))
    ids = tuple(item.transition_id for item in ordered)
    if len(set(ids)) != len(ids):
        raise ValueError("real trace evidence produced duplicate RAG transition IDs")
    return ordered


def render_rag_key(
    *, task: str, initial_context: Mapping[str, JsonValue], action: RAGAction
) -> str:
    """Render the exact versioned fit and query key without using an observation.

    Args:
        task: Request-visible source task or query task.
        initial_context: Request-visible starting context.
        action: Visible agent action whose environment response is sought.

    Returns:
        Canonical JSON text shared by index construction and retrieval queries.
    """
    return canonical_json_bytes(
        {
            "action": action.model_dump(mode="json", exclude_none=False),
            "initial_context": dict(initial_context),
            "key_schema_version": RAG_KEY_SCHEMA_VERSION,
            "task": task,
        }
    ).decode("utf-8")


def _trace_transitions(trace: Trace, lineage_id: str) -> list[RAGTransition]:
    """Extract ordered message and tool transitions from one verified real trace."""
    spans = tuple(sorted(trace.spans, key=lambda item: (item.started_at, item.span_id)))
    transitions: list[RAGTransition] = []
    used_tool_observations: set[str] = set()
    for index, span in enumerate(spans):
        action = _action(span)
        if action is None:
            continue
        if action.kind == "tool_call":
            observation_pair = _tool_observation(
                span,
                spans[index + 1 :],
                used_tool_observations,
            )
        else:
            observation_pair = _message_observation(span, spans[index + 1 :])
        if observation_pair is None:
            continue
        observation_span, observation = observation_pair
        key_text = render_rag_key(
            task=trace.task,
            initial_context=trace.initial_context,
            action=action,
        )
        material: JsonValue = {
            "action": action.model_dump(mode="json", exclude_none=False),
            "action_span_id": span.span_id,
            "key_schema_version": RAG_KEY_SCHEMA_VERSION,
            "lineage_id": lineage_id,
            "observation": observation.model_dump(mode="json"),
            "observation_span_id": observation_span.span_id,
            "trace_id": trace.trace_id,
        }
        transitions.append(
            RAGTransition(
                transition_id=stable_id("rag-transition", material),
                trace_id=trace.trace_id,
                conversation_id=trace.conversation_id,
                lineage_id=lineage_id,
                action_span_id=span.span_id,
                observation_span_id=observation_span.span_id,
                task=trace.task,
                initial_context=trace.initial_context,
                action=action,
                observation=observation,
                key_text=key_text,
                key_sha256=hashlib.sha256(key_text.encode("utf-8")).hexdigest(),
            )
        )
    return transitions


def _action(span: TraceSpan) -> RAGAction | None:
    """Return a real visible agent action from one source span, if present."""
    attributes = span.attributes
    operation = attributes.get("gen_ai.operation.name")
    if operation == _TOOL_OPERATION:
        return None
    tool_name = attributes.get("gen_ai.tool.name")
    if isinstance(tool_name, str) and tool_name.strip():
        return RAGAction(
            kind="tool_call",
            tool_name=normalize_durable_text(tool_name.strip()),
            tool_arguments=_tool_arguments(attributes.get("gen_ai.tool.call.arguments")),
        )
    content = _assistant_text(attributes)
    if content is None:
        return None
    return RAGAction(kind="message", content=content)


def _tool_observation(
    action_span: TraceSpan,
    later_spans: Sequence[TraceSpan],
    used_span_ids: set[str],
) -> tuple[TraceSpan, RAGObservation] | None:
    """Find the exact subsequent observed result for one real tool call."""
    call_id = action_span.attributes.get("gen_ai.tool.call.id")
    tool_name = action_span.attributes.get("gen_ai.tool.name")
    for span in later_spans:
        if span.span_id in used_span_ids or span.started_at < action_span.ended_at:
            continue
        if span.attributes.get("gen_ai.operation.name") != _TOOL_OPERATION:
            continue
        if call_id is not None:
            if span.attributes.get("gen_ai.tool.call.id") != call_id:
                continue
        elif span.attributes.get("gen_ai.tool.name") != tool_name:
            continue
        content = _visible_text(span.attributes.get("gen_ai.tool.message"))
        if content is None:
            continue
        used_span_ids.add(span.span_id)
        return span, RAGObservation(kind="tool_result", content=content)
    return None


def _message_observation(
    action_span: TraceSpan,
    later_spans: Sequence[TraceSpan],
) -> tuple[TraceSpan, RAGObservation] | None:
    """Find a genuinely new user-visible message after one assistant response."""
    previous_user_text = _user_text(action_span.attributes)
    for span in later_spans:
        if span.started_at < action_span.ended_at:
            continue
        content = _user_text(span.attributes)
        if content is None or content == previous_user_text:
            continue
        return span, RAGObservation(kind="message", content=content)
    return None


def _assistant_text(attributes: Mapping[str, JsonValue]) -> str | None:
    """Read visible assistant output from supported GenAI semantic-convention attributes."""
    messages = _json_value(attributes.get("gen_ai.output.messages"))
    text = _last_role_message(messages, frozenset({"assistant", "model"}))
    if text is not None:
        return text
    return _visible_text(attributes.get("gen_ai.completion"))


def _user_text(attributes: Mapping[str, JsonValue]) -> str | None:
    """Read the latest visible user input from one model-call span."""
    messages = _json_value(attributes.get("gen_ai.input.messages"))
    text = _last_role_message(messages, frozenset({"user", "human"}))
    if text is not None:
        return text
    return _visible_text(attributes.get("gen_ai.prompt"))


def _last_role_message(value: JsonValue, roles: frozenset[str]) -> str | None:
    """Return the last text message whose role belongs to the requested set."""
    if not isinstance(value, list):
        return None
    for item in reversed(value):
        if not isinstance(item, dict) or item.get("role") not in roles:
            continue
        content = _message_content(item.get("content"))
        if content is not None:
            return content
    return None


def _message_content(value: JsonValue) -> str | None:
    """Read plain text from a GenAI message content value."""
    if isinstance(value, str):
        return _visible_text(value)
    if not isinstance(value, list):
        return None
    pieces = []
    for item in value:
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        if isinstance(text, str) and text.strip():
            pieces.append(text.strip())
    return _visible_text("\n".join(pieces)) if pieces else None


def _tool_arguments(value: JsonValue) -> JsonValue:
    """Preserve exact JSON tool arguments when present, otherwise retain visible text."""
    parsed = _json_value(value)
    if parsed is None:
        return {}
    return parsed


def _json_value(value: JsonValue) -> JsonValue:
    """Decode JSON-valued semantic attributes without guessing malformed text."""
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _visible_text(value: JsonValue) -> str | None:
    """Render a non-empty captured JSON value as stable visible text."""
    if isinstance(value, str):
        text = value.strip()
    elif value is None:
        return None
    else:
        text = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    if not text:
        return None
    return normalize_durable_text(text)
