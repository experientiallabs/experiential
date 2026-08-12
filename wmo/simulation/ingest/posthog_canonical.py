"""Focused PostHog LLM-observability conversion to canonical WMO trace contracts."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict, deque
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from pydantic import JsonValue

from wmo.common.core.artifacts import (
    FailureCode,
    JsonObject,
    SourceIdentity,
    StructuredFailure,
    canonical_json_bytes,
)
from wmo.common.core.text import normalize_durable_text
from wmo.common.models import ModelSnapshot
from wmo.common.tasks import ToolSchema
from wmo.common.traces import Trace, TraceOutcome, TraceSource, TraceSpan
from wmo.simulation.ingest.otlp import (
    GENAI_SEMANTIC_CONVENTION_VERSION,
    TraceNormalizationIssue,
    TraceNormalizationResult,
)

_TRACE_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
_SPAN_ID_PATTERN = re.compile(r"^[0-9a-f]{16}$")
_OUTCOME_STATUS_KEY = "wmo.outcome.status"
_OUTCOME_NAME_KEY = "wmo.outcome.name"
_OUTCOME_FAILURE_CODE_KEY = "wmo.outcome.failure.code"
_OUTCOME_FAILURE_MESSAGE_KEY = "wmo.outcome.failure.message"
_OUTCOME_FAILURE_RETRYABLE_KEY = "wmo.outcome.failure.retryable"


class PostHogPullError(ValueError):
    """Raised when an authorized PostHog HogQL pull cannot be validated or completed."""


@dataclass(frozen=True)
class _PostHogEvent:
    """One source event with deterministic source-order tie breaking."""

    event: JsonObject
    ordinal: int
    source_order_key: str


@dataclass(frozen=True)
class _PendingToolCall:
    """One emitted model tool call that awaits a matching PostHog tool result."""

    call_id: str
    parent_span_id: str


@dataclass(frozen=True)
class _SpanEmission:
    """One canonical span before source and paired parent relationships are resolved."""

    span: TraceSpan
    source_span_id: str
    source_parent_span_id: str | None
    paired_parent_span_id: str | None = None


def load_posthog_file(
    path: Path,
    *,
    semantic_convention_version: str = GENAI_SEMANTIC_CONVENTION_VERSION,
    source_id: str | None = None,
) -> TraceNormalizationResult:
    """Read PostHog LLM-observability JSON or JSONL through the focused source path.

    Args:
        path: Single event, event array, query wrapper, or JSONL export path.
        semantic_convention_version: Pinned GenAI semantic-convention version for resulting traces.
        source_id: Optional durable source label. The path is used when omitted.

    Returns:
        Canonical traces and explicit invalid event or trace exclusions.

    Raises:
        PostHogPullError: The local export cannot be read or decoded as UTF-8 JSON or JSONL.
    """
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise PostHogPullError(f"cannot read PostHog export {path}: {exc}") from exc
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PostHogPullError(f"PostHog export is not UTF-8: {path}") from exc
    source = SourceIdentity(
        kind="file",
        source_id=source_id or str(path),
        sha256=hashlib.sha256(payload).hexdigest(),
    )
    try:
        document: JsonValue = json.loads(text)
    except json.JSONDecodeError:
        return _normalize_jsonl(
            text,
            source=source,
            semantic_convention_version=semantic_convention_version,
        )
    return normalize_posthog_payload(
        document,
        source=source,
        semantic_convention_version=semantic_convention_version,
    )


def normalize_posthog_payload(
    payload: JsonValue,
    *,
    source: SourceIdentity,
    semantic_convention_version: str = GENAI_SEMANTIC_CONVENTION_VERSION,
    initial_issues: Sequence[TraceNormalizationIssue] = (),
) -> TraceNormalizationResult:
    """Normalize one PostHog event shape directly to canonical W2 trace evidence.

    Args:
        payload: Single event, event array, query wrapper, or normalized HogQL event rows.
        source: Immutable source identity for resulting trace records.
        semantic_convention_version: Pinned GenAI semantic-convention version for resulting traces.
        initial_issues: File-level JSONL parse exclusions to retain in the result.

    Returns:
        Valid canonical traces and explicit invalid event or trace exclusions.

    Raises:
        PostHogPullError: The semantic convention version is blank.
    """
    if not semantic_convention_version.strip():
        raise PostHogPullError("semantic convention version must not be blank")
    issues = list(initial_issues)
    try:
        events = _events_from_payload(payload)
    except PostHogPullError as exc:
        return TraceNormalizationResult(
            traces=(),
            issues=(*issues, TraceNormalizationIssue("posthog-payload", str(exc))),
        )
    by_trace: dict[str, list[_PostHogEvent]] = defaultdict(list)
    for event in events:
        try:
            source_trace_id = _source_trace_id(event.event)
        except PostHogPullError as exc:
            issues.append(TraceNormalizationIssue(f"event-{event.ordinal}", str(exc)))
            continue
        by_trace[source_trace_id].append(event)
    traces: list[Trace] = []
    for source_trace_id in sorted(by_trace):
        try:
            traces.append(
                _normalize_trace_events(
                    source_trace_id,
                    by_trace[source_trace_id],
                    source=source,
                    semantic_convention_version=semantic_convention_version,
                )
            )
        except PostHogPullError as exc:
            issues.append(TraceNormalizationIssue(f"trace-{source_trace_id}", str(exc)))
    traces.sort(key=lambda trace: (trace.spans[0].started_at, trace.trace_id))
    return TraceNormalizationResult(traces=tuple(traces), issues=tuple(issues))


def _normalize_jsonl(
    text: str,
    *,
    source: SourceIdentity,
    semantic_convention_version: str,
) -> TraceNormalizationResult:
    """Decode JSONL export records while retaining malformed lines for coverage reporting."""
    records: list[JsonValue] = []
    issues: list[TraceNormalizationIssue] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            issues.append(
                TraceNormalizationIssue(f"line-{line_number}", f"invalid JSONL record: {exc.msg}")
            )
    if not records and not issues:
        raise PostHogPullError("PostHog JSONL export contains no records")
    return normalize_posthog_payload(
        records,
        source=source,
        semantic_convention_version=semantic_convention_version,
        initial_issues=issues,
    )


def _events_from_payload(payload: JsonValue) -> tuple[_PostHogEvent, ...]:
    """Flatten supported PostHog export shapes without a generic vendor auto-detector."""
    events: list[JsonObject] = []

    def append(value: JsonValue) -> None:
        if isinstance(value, list):
            for item in value:
                append(item)
            return
        if not isinstance(value, dict):
            raise PostHogPullError("PostHog exports must contain event objects")
        if "event" in value or "properties" in value:
            events.append(value)
            return
        for wrapper_key in ("results", "events"):
            wrapped = value.get(wrapper_key)
            if isinstance(wrapped, list):
                append(wrapped)
                return
        raise PostHogPullError("PostHog export has no event, events, or results shape")

    append(payload)
    return tuple(
        _PostHogEvent(
            event=event,
            ordinal=index,
            source_order_key=_source_event_order_key(event, index),
        )
        for index, event in enumerate(events)
    )


def _normalize_trace_events(
    source_trace_id: str,
    events: Sequence[_PostHogEvent],
    *,
    source: SourceIdentity,
    semantic_convention_version: str,
) -> Trace:
    """Convert ordered PostHog root, generation, tool, and error events to one canonical trace."""
    trace_id = _w3c_trace_id(source_trace_id, kind="trace", namespace="trace")
    ordered = sorted(
        events,
        key=lambda event: (_event_timestamp(event.event), event.source_order_key, event.ordinal),
    )
    initial_request = _initial_request(ordered)
    if initial_request is None:
        raise PostHogPullError("PostHog trace has no initial user message or root task text")
    task, initial_event = initial_request
    attributes_by_event = tuple(_mapped_extensions(_properties(event.event)) for event in ordered)
    initial_extensions = _mapped_extensions(_properties(initial_event.event))
    initial_context = _consistent_json_object((initial_extensions,), "wmo.request.context")
    conversation_id = _consistent_text(
        attributes_by_event, ("wmo.conversation.id", "gen_ai.conversation.id")
    )
    tools = _collect_tools((initial_extensions,))
    spans: list[_SpanEmission] = []
    pending_calls: dict[str, deque[_PendingToolCall]] = defaultdict(deque)
    errors: list[StructuredFailure] = []
    for event in ordered:
        event_name = _event_name(event.event)
        properties = _properties(event.event)
        extensions = _mapped_extensions(properties)
        timestamp = _event_timestamp(event.event)
        if event_name == "$ai_generation":
            generated, call_ids = _generation_spans(
                source_trace_id,
                event,
                timestamp,
                task,
                extensions,
            )
            spans.extend(generated)
            errors.extend(
                emission.span.failure for emission in generated if emission.span.failure is not None
            )
            for tool_name, pending_call in call_ids:
                pending_calls[tool_name].append(pending_call)
        elif event_name == "$ai_span":
            span = _tool_span(
                source_trace_id,
                event,
                timestamp,
                extensions,
                pending_calls,
            )
            spans.append(span)
            if span.span.failure is not None:
                errors.append(span.span.failure)
        elif event_name == "$ai_trace":
            span = _root_span(source_trace_id, event, timestamp, task, extensions)
            spans.append(span)
            if span.span.failure is not None:
                errors.append(span.span.failure)
        elif _is_error(properties):
            span = _event_error_span(source_trace_id, event, timestamp, extensions)
            spans.append(span)
            if span.span.failure is not None:
                errors.append(span.span.failure)
    if not spans:
        raise PostHogPullError("PostHog trace has no $ai_generation, $ai_span, or $ai_trace event")
    unmatched_calls = tuple(
        f"{tool_name}:{pending_call.call_id}"
        for tool_name in sorted(pending_calls)
        for pending_call in pending_calls[tool_name]
    )
    if unmatched_calls:
        raise PostHogPullError(
            "unmatched generated PostHog tool calls: " + ", ".join(unmatched_calls)
        )
    outcome = _collect_outcome(attributes_by_event, errors)
    return Trace(
        trace_id=trace_id,
        conversation_id=conversation_id,
        task=task,
        initial_context=initial_context,
        tools=tools,
        spans=_resolve_parent_links(spans),
        outcome=outcome,
        source=TraceSource(
            identity=source,
            semantic_convention_version=semantic_convention_version,
        ),
    )


def _generation_spans(
    source_trace_id: str,
    event: _PostHogEvent,
    timestamp: datetime,
    task: str,
    extensions: JsonObject,
) -> tuple[tuple[_SpanEmission, ...], tuple[tuple[str, _PendingToolCall], ...]]:
    """Map one PostHog generation and all of its tool calls to canonical model-call spans."""
    properties = _properties(event.event)
    choices = properties.get("$ai_output_choices")
    tool_calls = _tool_calls(choices)
    model = _model_snapshot(properties)
    base_attributes = dict(extensions)
    base_attributes["gen_ai.operation.name"] = "chat"
    base_attributes["gen_ai.prompt"] = task
    input_messages = properties.get("$ai_input")
    if input_messages is not None:
        base_attributes["gen_ai.input.messages"] = input_messages
    spans: list[_SpanEmission] = []
    call_ids: list[tuple[str, _PendingToolCall]] = []
    if tool_calls:
        for call_index, call in enumerate(tool_calls):
            tool_name, arguments, raw_call_id = _tool_call_details(call)
            call_id = raw_call_id or f"posthog-call-{event.ordinal}-{call_index}"
            attributes = dict(base_attributes)
            attributes.update(
                {
                    "gen_ai.tool.name": tool_name,
                    "gen_ai.tool.call.arguments": arguments,
                    "gen_ai.tool.call.id": call_id,
                }
            )
            span = _span(
                source_trace_id,
                event,
                timestamp,
                name="agent.model_call",
                attributes=attributes,
                model=model,
                suffix=f"tool-{call_index}",
            )
            spans.append(span)
            call_ids.append(
                (tool_name, _PendingToolCall(call_id=call_id, parent_span_id=span.span.span_id))
            )
    else:
        attributes = dict(base_attributes)
        attributes["gen_ai.completion"] = _choices_text(choices)
        spans.append(
            _span(
                source_trace_id,
                event,
                timestamp,
                name="agent.model_call",
                attributes=attributes,
                model=model,
                suffix="completion",
            )
        )
    return tuple(spans), tuple(call_ids)


def _tool_span(
    source_trace_id: str,
    event: _PostHogEvent,
    timestamp: datetime,
    extensions: JsonObject,
    pending_calls: dict[str, deque[_PendingToolCall]],
) -> _SpanEmission:
    """Map a PostHog tool event and pair it to an earlier generation call when possible."""
    properties = _properties(event.event)
    tool_name = _required_text(properties.get("$ai_span_name"), "PostHog $ai_span_name")
    attributes = dict(extensions)
    attributes.update(
        {
            "gen_ai.operation.name": "execute_tool",
            "gen_ai.tool.name": tool_name,
            "gen_ai.tool.call.arguments": _json_text(properties.get("$ai_input_state")),
            "gen_ai.tool.message": _json_text(properties.get("$ai_output_state")),
        }
    )
    call_id = _explicit_tool_call_id(properties)
    matched_call: _PendingToolCall | None = None
    if call_id is None:
        if pending_calls[tool_name]:
            matched_call = pending_calls[tool_name].popleft()
    else:
        matched_call = _consume_pending_call(pending_calls[tool_name], call_id)
        if matched_call is None:
            raise PostHogPullError(f"unmatched explicit PostHog tool result: {tool_name}:{call_id}")
    if matched_call is not None:
        attributes["gen_ai.tool.call.id"] = matched_call.call_id
    return _span(
        source_trace_id,
        event,
        timestamp,
        name="agent.tool_call",
        attributes=attributes,
        model=None,
        suffix="tool-result",
        paired_parent_span_id=None if matched_call is None else matched_call.parent_span_id,
    )


def _root_span(
    source_trace_id: str,
    event: _PostHogEvent,
    timestamp: datetime,
    task: str,
    extensions: JsonObject,
) -> _SpanEmission:
    """Retain a PostHog trace-root event as canonical trace-level evidence."""
    attributes = dict(extensions)
    attributes["gen_ai.operation.name"] = "invoke_agent"
    attributes["gen_ai.prompt"] = task
    properties = _properties(event.event)
    if properties.get("$ai_output_state") is not None:
        attributes["gen_ai.completion"] = _json_text(properties.get("$ai_output_state"))
    return _span(
        source_trace_id,
        event,
        timestamp,
        name="agent.trace",
        attributes=attributes,
        model=None,
        suffix="root",
    )


def _event_error_span(
    source_trace_id: str,
    event: _PostHogEvent,
    timestamp: datetime,
    extensions: JsonObject,
) -> _SpanEmission:
    """Retain an errored unrecognized PostHog AI event as structured failure evidence."""
    attributes = dict(extensions)
    attributes["gen_ai.operation.name"] = "invoke_agent"
    return _span(
        source_trace_id,
        event,
        timestamp,
        name="agent.error",
        attributes=attributes,
        model=None,
        suffix="error",
    )


def _span(
    source_trace_id: str,
    event: _PostHogEvent,
    timestamp: datetime,
    *,
    name: str,
    attributes: JsonObject,
    model: ModelSnapshot | None,
    suffix: str,
    paired_parent_span_id: str | None = None,
) -> _SpanEmission:
    """Build one canonical span with deterministic W3C-shaped IDs and PostHog error mapping."""
    properties = _properties(event.event)
    source_span_id = _source_span_id(event)
    span_id = _w3c_trace_id(
        f"{source_trace_id}\0{source_span_id}\0{suffix}", kind="span", namespace="span"
    )
    failure = _event_failure(properties)
    return _SpanEmission(
        span=TraceSpan(
            span_id=span_id,
            parent_span_id=None,
            name=name,
            started_at=timestamp,
            ended_at=timestamp,
            attributes=attributes,
            model=model,
            failure=failure,
        ),
        source_span_id=source_span_id,
        source_parent_span_id=_source_parent_span_id(properties),
        paired_parent_span_id=paired_parent_span_id,
    )


def _resolve_parent_links(emissions: Sequence[_SpanEmission]) -> tuple[TraceSpan, ...]:
    """Resolve source and WMO-created parents only to canonical spans emitted in this trace."""
    emitted_by_source_id: dict[str, list[str]] = defaultdict(list)
    emitted_span_ids = {emission.span.span_id for emission in emissions}
    for emission in emissions:
        emitted_by_source_id[emission.source_span_id].append(emission.span.span_id)
    resolved: list[TraceSpan] = []
    for emission in emissions:
        parent_span_id = _resolved_parent_span_id(
            emission,
            emitted_by_source_id,
            emitted_span_ids,
        )
        resolved.append(emission.span.model_copy(update={"parent_span_id": parent_span_id}))
    return tuple(resolved)


def _resolved_parent_span_id(
    emission: _SpanEmission,
    emitted_by_source_id: dict[str, list[str]],
    emitted_span_ids: set[str],
) -> str | None:
    """Prefer one unambiguous source parent, then a validated paired model-call parent."""
    source_candidates = tuple(
        span_id
        for span_id in emitted_by_source_id.get(emission.source_parent_span_id or "", ())
        if span_id != emission.span.span_id
    )
    if len(source_candidates) == 1:
        return source_candidates[0]
    if emission.paired_parent_span_id in source_candidates:
        return emission.paired_parent_span_id
    paired_parent_span_id = emission.paired_parent_span_id
    if (
        paired_parent_span_id is not None
        and paired_parent_span_id != emission.span.span_id
        and paired_parent_span_id in emitted_span_ids
    ):
        return paired_parent_span_id
    return None


def _initial_request(
    events: Sequence[_PostHogEvent],
) -> tuple[str, _PostHogEvent] | None:
    """Read the first user message and retain only its request-visible event evidence."""
    for event in events:
        input_value = _properties(event.event).get("$ai_input")
        task = _first_user_text(input_value)
        if task is not None:
            return task, event
        if isinstance(input_value, str) and input_value.strip():
            return normalize_durable_text(input_value.strip()), event
    return None


def _first_user_text(value: JsonValue | None) -> str | None:
    """Extract first user or human message text from PostHog's LLM input message list."""
    if not isinstance(value, list):
        return None
    for message in value:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if not isinstance(role, str) or role.casefold() not in {"user", "human"}:
            continue
        text = _content_text(message.get("content"))
        if text:
            return text
    return None


def _content_text(value: JsonValue | None) -> str:
    """Read text from plain or normalized multi-part PostHog message content."""
    if isinstance(value, str):
        return normalize_durable_text(value)
    if not isinstance(value, list):
        return ""
    texts = [
        item["text"].strip()
        for item in value
        if isinstance(item, dict)
        and item.get("type") == "text"
        and isinstance(item.get("text"), str)
        and item["text"].strip()
    ]
    return normalize_durable_text("\n".join(texts))


def _tool_calls(value: JsonValue | None) -> tuple[JsonObject, ...]:
    """Read normalized content-part and raw OpenAI tool calls from output choices."""
    if not isinstance(value, list):
        return ()
    calls: list[JsonObject] = []
    for choice in value:
        if not isinstance(choice, dict):
            continue
        raw_calls = choice.get("tool_calls")
        if isinstance(raw_calls, list):
            calls.extend(call for call in raw_calls if isinstance(call, dict))
        content = choice.get("content")
        if isinstance(content, list):
            calls.extend(
                item
                for item in content
                if isinstance(item, dict) and item.get("type") == "function"
            )
    return tuple(calls)


def _tool_call_details(call: JsonObject) -> tuple[str, str, str | None]:
    """Extract a normalized or raw OpenAI tool-call name, arguments, and optional call ID."""
    function = call.get("function")
    candidate = function if isinstance(function, dict) else call
    name = _required_text(candidate.get("name"), "PostHog tool call name")
    arguments = _json_text(candidate.get("arguments"))
    raw_call_id = call.get("id")
    return name, arguments, raw_call_id if isinstance(raw_call_id, str) and raw_call_id else None


def _explicit_tool_call_id(properties: JsonObject) -> str | None:
    """Read one consistent explicit PostHog tool-result call identity when supplied."""
    values = tuple(
        _required_text(properties[key], f"PostHog {key}")
        for key in ("$ai_tool_call_id", "$ai_call_id")
        if key in properties
    )
    if not values:
        return None
    if len(set(values)) != 1:
        raise PostHogPullError(
            "PostHog $ai_tool_call_id and $ai_call_id must agree when both are supplied"
        )
    return values[0]


def _choices_text(value: JsonValue | None) -> str:
    """Read the first assistant text choice, retaining empty completion evidence when absent."""
    if isinstance(value, list):
        for choice in value:
            if isinstance(choice, dict):
                text = _content_text(choice.get("content"))
                if text:
                    return text
    return _json_text(value)


def _mapped_extensions(properties: JsonObject) -> JsonObject:
    """Copy only approved request and outcome WMO extensions into canonical span attributes."""
    keys = (
        "wmo.customer.id",
        "wmo.conversation.id",
        "wmo.request.context",
        "wmo.request.tags",
        "wmo.request.tools",
        "wmo.outcome.escalated",
        _OUTCOME_STATUS_KEY,
        _OUTCOME_NAME_KEY,
        _OUTCOME_FAILURE_CODE_KEY,
        _OUTCOME_FAILURE_MESSAGE_KEY,
        _OUTCOME_FAILURE_RETRYABLE_KEY,
    )
    return {key: properties[key] for key in keys if key in properties}


def _consistent_json_object(attributes: Sequence[JsonObject], key: str) -> JsonObject:
    """Return one repeated JSON-object request extension or reject inconsistent event copies."""
    values: list[JsonObject] = []
    for item in attributes:
        value = _json_value(item.get(key))
        if value is None:
            continue
        if not isinstance(value, dict):
            raise PostHogPullError(f"{key} must be a JSON object")
        values.append(value)
    if not values:
        return {}
    if any(value != values[0] for value in values[1:]):
        raise PostHogPullError(f"{key} differs across one PostHog trace")
    return values[0]


def _consistent_text(attributes: Sequence[JsonObject], keys: tuple[str, ...]) -> str | None:
    """Return one repeated source extension string or reject ambiguous event values."""
    values: list[str] = []
    for item in attributes:
        for key in keys:
            value = item.get(key)
            if value is None:
                continue
            if not isinstance(value, str) or not value.strip():
                raise PostHogPullError(f"{key} must be non-empty text")
            values.append(normalize_durable_text(value.strip()))
            break
    if not values:
        return None
    if any(value != values[0] for value in values[1:]):
        raise PostHogPullError(f"{keys[0]} differs across one PostHog trace")
    return values[0]


def _collect_tools(attributes: Sequence[JsonObject]) -> tuple[ToolSchema, ...]:
    """Convert optional PostHog request tool definitions to canonical visible tools."""
    by_name: dict[str, ToolSchema] = {}
    for item in attributes:
        value = _json_value(item.get("wmo.request.tools"))
        if value is None:
            continue
        if not isinstance(value, list):
            raise PostHogPullError("wmo.request.tools must be a JSON array")
        for raw_tool in value:
            if not isinstance(raw_tool, dict):
                raise PostHogPullError("PostHog tool definitions must be objects")
            function = raw_tool.get("function") if raw_tool.get("type") == "function" else raw_tool
            if not isinstance(function, dict):
                raise PostHogPullError("PostHog function tool must contain a function object")
            name = _required_text(function.get("name"), "PostHog tool definition name")
            description = function.get("description")
            schema = function.get(
                "input_schema", function.get("parameters", function.get("schema"))
            )
            if not isinstance(schema, dict):
                raise PostHogPullError(f"PostHog tool {name!r} needs an object input schema")
            tool = ToolSchema(
                name=name,
                description=(
                    normalize_durable_text(description.strip())
                    if isinstance(description, str) and description.strip()
                    else "No description captured."
                ),
                input_schema=schema,
            )
            if name in by_name and by_name[name] != tool:
                raise PostHogPullError(f"PostHog tool {name!r} has conflicting definitions")
            by_name[name] = tool
    return tuple(by_name[name] for name in sorted(by_name))


def _collect_outcome(
    attributes: Sequence[JsonObject], errors: Sequence[StructuredFailure]
) -> TraceOutcome | None:
    """Map WMO extensions or source error events to canonical terminal trace outcome evidence."""
    status = _consistent_text(attributes, (_OUTCOME_STATUS_KEY,))
    outcome_name = _consistent_text(attributes, (_OUTCOME_NAME_KEY,))
    failure_code = _consistent_text(attributes, (_OUTCOME_FAILURE_CODE_KEY,))
    failure_message = _consistent_text(attributes, (_OUTCOME_FAILURE_MESSAGE_KEY,))
    retryable = _consistent_bool(attributes, _OUTCOME_FAILURE_RETRYABLE_KEY)
    if status is None:
        if errors:
            return TraceOutcome(status="failure", failure=errors[0])
        if any(
            value is not None for value in (outcome_name, failure_code, failure_message, retryable)
        ):
            raise PostHogPullError("PostHog WMO outcome details require wmo.outcome.status")
        return None
    if status != "failure":
        if status not in {"success", "abandoned", "unknown"}:
            raise PostHogPullError(
                "wmo.outcome.status must be success, failure, abandoned, or unknown"
            )
        if any(value is not None for value in (failure_code, failure_message, retryable)):
            raise PostHogPullError("PostHog failure outcome details require failure status")
        if status == "success":
            return TraceOutcome(status="success", outcome_name=outcome_name)
        if status == "abandoned":
            return TraceOutcome(status="abandoned", outcome_name=outcome_name)
        return TraceOutcome(status="unknown", outcome_name=outcome_name)
    if failure_code is None or failure_message is None:
        if errors:
            return TraceOutcome(status="failure", outcome_name=outcome_name, failure=errors[0])
        raise PostHogPullError("PostHog failure outcome needs code and message extensions")
    try:
        code = FailureCode(failure_code)
    except ValueError as exc:
        raise PostHogPullError("PostHog failure outcome has an unsupported failure code") from exc
    return TraceOutcome(
        status="failure",
        outcome_name=outcome_name,
        failure=StructuredFailure(code=code, message=failure_message, retryable=retryable or False),
    )


def _consistent_bool(attributes: Sequence[JsonObject], key: str) -> bool | None:
    """Return one repeated boolean extension or reject inconsistent source event values."""
    values: list[bool] = []
    for item in attributes:
        value = item.get(key)
        if value is None:
            continue
        if not isinstance(value, bool):
            raise PostHogPullError(f"{key} must be boolean")
        values.append(value)
    if not values:
        return None
    if any(value != values[0] for value in values[1:]):
        raise PostHogPullError(f"{key} differs across one PostHog trace")
    return values[0]


def _model_snapshot(properties: JsonObject) -> ModelSnapshot | None:
    """Capture PostHog model identity only when both provider and model are present."""
    model_id = _first_property_text(properties, ("$ai_model", "$ai_model_id", "model"))
    provider = _first_property_text(properties, ("$ai_provider", "$ai_system", "provider"))
    if model_id is None and provider is None:
        return None
    if model_id is None:
        raise PostHogPullError("PostHog model identity has a provider but no model")
    if provider is None:
        raise PostHogPullError("PostHog model identity has a model but no provider")
    revision = _first_property_text(properties, ("$ai_model_revision", "model_revision"))
    digest = hashlib.sha256(f"{provider}\0{model_id}\0{revision or ''}".encode()).hexdigest()
    return ModelSnapshot(
        provider=provider,
        model_id=model_id,
        revision=revision,
        capabilities_sha256=digest,
    )


def _event_failure(properties: JsonObject) -> StructuredFailure | None:
    """Map PostHog's current error flag and message fields to structured span failure evidence."""
    if not _is_error(properties):
        return None
    message = _first_property_text(properties, ("$ai_error", "error", "$exception_message"))
    return StructuredFailure(
        code=FailureCode.INTERNAL,
        message=message or "PostHog AI event marked as an error",
    )


def _consume_pending_call(
    pending: deque[_PendingToolCall], call_id: str
) -> _PendingToolCall | None:
    """Remove and return one matching pending call without accepting a mismatched result."""
    for candidate in pending:
        if candidate.call_id == call_id:
            pending.remove(candidate)
            return candidate
    return None


def _source_trace_id(event: JsonObject) -> str:
    """Resolve PostHog's trace ID with the documented span or event fallback."""
    properties = _properties(event)
    for key in ("$ai_trace_id", "$ai_span_id"):
        value = properties.get(key)
        if isinstance(value, str) and value:
            return value
    for key in ("id", "uuid"):
        value = event.get(key)
        if isinstance(value, str) and value:
            return value
    raise PostHogPullError("PostHog AI event needs $ai_trace_id, $ai_span_id, id, or uuid")


def _source_span_id(event: _PostHogEvent) -> str:
    """Resolve one PostHog source span identity used for stable IDs and parent conversion."""
    properties = _properties(event.event)
    candidates = (
        properties.get("$ai_span_id"),
        event.event.get("uuid"),
        event.event.get("id"),
    )
    for candidate in candidates:
        if isinstance(candidate, str) and candidate:
            return candidate
    return f"ordinal:{event.ordinal}"


def _source_parent_span_id(properties: JsonObject) -> str | None:
    """Read the PostHog parent span identity when this source event supplies one."""
    value = properties.get("$ai_parent_id")
    if value is None:
        return None
    return _required_text(value, "PostHog $ai_parent_id")


def _source_event_order_key(event: JsonObject, ordinal: int) -> str:
    """Return a stable event identity for timestamp-tie ordering, or source ordinal fallback."""
    for key in ("uuid", "id"):
        value = event.get(key)
        if isinstance(value, str) and value:
            return f"{key}:{value}"
    return f"ordinal:{ordinal:020d}"


def _event_timestamp(event: JsonObject) -> datetime:
    """Parse a timezone-aware PostHog event timestamp for stable trace ordering."""
    raw = event.get("timestamp")
    if not isinstance(raw, str) or not raw.strip():
        raise PostHogPullError("PostHog AI event needs a timezone-aware timestamp")
    try:
        timestamp = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise PostHogPullError(f"PostHog event timestamp is invalid: {raw!r}") from exc
    if timestamp.tzinfo is None:
        raise PostHogPullError("PostHog event timestamp must include a timezone")
    return timestamp.astimezone(UTC)


def _event_name(event: JsonObject) -> str:
    """Return one PostHog event name or an empty string for non-AI records."""
    value = event.get("event")
    return value if isinstance(value, str) else ""


def _properties(event: JsonObject) -> JsonObject:
    """Return PostHog's properties object or an empty object when it is absent."""
    value = event.get("properties")
    return value if isinstance(value, dict) else {}


def _is_error(properties: JsonObject) -> bool:
    """Interpret PostHog error flags without treating ordinary values as errors."""
    value = properties.get("$ai_is_error")
    if isinstance(value, bool):
        return value
    error = properties.get("$ai_error")
    return isinstance(error, str) and bool(error.strip())


def _json_value(value: JsonValue | None) -> JsonValue | None:
    """Decode JSON-encoded PostHog property values while preserving plain text values."""
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _json_text(value: JsonValue | None) -> str:
    """Render a PostHog property as stable compact JSON text or plain text."""
    if isinstance(value, str):
        return normalize_durable_text(value)
    if value is None:
        return ""
    return canonical_json_bytes(value).decode("utf-8")


def _required_text(value: JsonValue | None, label: str) -> str:
    """Require one non-empty durable source text value."""
    if not isinstance(value, str) or not value.strip():
        raise PostHogPullError(f"{label} must be non-empty text")
    return normalize_durable_text(value.strip())


def _first_property_text(properties: JsonObject, keys: tuple[str, ...]) -> str | None:
    """Return the first non-empty text property from an ordered PostHog property key list."""
    for key in keys:
        value = properties.get(key)
        if isinstance(value, str) and value.strip():
            return normalize_durable_text(value.strip())
    return None


def _w3c_trace_id(value: str, *, kind: str, namespace: str) -> str:
    """Keep valid W3C IDs and deterministically hash PostHog opaque IDs into W3C width."""
    normalized = value.casefold()
    pattern = _TRACE_ID_PATTERN if kind == "trace" else _SPAN_ID_PATTERN
    if pattern.fullmatch(normalized) and set(normalized) != {"0"}:
        return normalized
    width = 32 if kind == "trace" else 16
    return hashlib.sha256(f"posthog\0{namespace}\0{value}".encode()).hexdigest()[:width]
