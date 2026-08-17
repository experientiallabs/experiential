"""Focused PostHog LLM-observability conversion to canonical WMO trace contracts.

PostHog captures LLM observability as flat ``$ai_*`` events: ``$ai_generation`` for a model call
and the tool calls its output requests, ``$ai_span`` for an executed tool, and ``$ai_trace`` for
the trace root. This module owns only that format knowledge: it reads each event, declares what
the event says as a :class:`~wmo.simulation.ingest.vendor_observations.VendorObservation`, and
hands canonical trace construction to :mod:`wmo.simulation.ingest.vendor_trace` with strict tool
pairing, because PostHog generations enumerate every requested call.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

from pydantic import JsonValue

from wmo.common.core.artifacts import JsonObject, SourceIdentity
from wmo.simulation.ingest.otlp import (
    GENAI_SEMANTIC_CONVENTION_VERSION,
    TraceNormalizationIssue,
    TraceNormalizationResult,
)
from wmo.simulation.ingest.trace_extensions import (
    OUTCOME_FAILURE_CODE_KEY,
    OUTCOME_FAILURE_MESSAGE_KEY,
    OUTCOME_FAILURE_RETRYABLE_KEY,
    OUTCOME_NAME_KEY,
    OUTCOME_STATUS_KEY,
    REQUEST_CONTEXT_KEY,
    REQUEST_TOOLS_KEY,
)
from wmo.simulation.ingest.vendor_observations import (
    VendorModelIdentity,
    VendorObservation,
    declared_tool_calls,
)
from wmo.simulation.ingest.vendor_records import (
    VendorTraceFormatError,
    first_text,
    first_user_text,
    flatten_records,
    json_text,
    message_text,
    read_vendor_export,
    required_text,
)
from wmo.simulation.ingest.vendor_trace import approved_extensions, build_vendor_traces

VENDOR = "posthog"

_REQUEST_VISIBLE_KEYS = (REQUEST_CONTEXT_KEY, "wmo.request.tags", REQUEST_TOOLS_KEY)

_TRACE_LEVEL_KEYS = (
    "wmo.customer.id",
    "wmo.conversation.id",
    *_REQUEST_VISIBLE_KEYS,
    "wmo.outcome.escalated",
    OUTCOME_STATUS_KEY,
    OUTCOME_NAME_KEY,
    OUTCOME_FAILURE_CODE_KEY,
    OUTCOME_FAILURE_MESSAGE_KEY,
    OUTCOME_FAILURE_RETRYABLE_KEY,
)


class PostHogPullError(VendorTraceFormatError):
    """Raised when an authorized PostHog HogQL pull cannot be validated or completed."""


@dataclass(frozen=True)
class _PostHogEvent:
    """One source event with deterministic source-order tie breaking.

    Args:
        event: Decoded PostHog event object.
        ordinal: Source order position across the export.
        source_order_key: Stable event identity used for timestamp-tie ordering.
    """

    event: JsonObject
    ordinal: int
    source_order_key: str


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
    export = read_vendor_export(
        path,
        vendor="PostHog",
        source_id=source_id or str(path),
        error_type=PostHogPullError,
    )
    return normalize_posthog_payload(
        list(export.payloads),
        source=export.source,
        semantic_convention_version=semantic_convention_version,
        initial_issues=export.issues,
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
    except VendorTraceFormatError as exc:
        return TraceNormalizationResult(
            traces=(),
            issues=(*issues, TraceNormalizationIssue("posthog-payload", str(exc))),
        )
    by_trace: dict[str, list[_PostHogEvent]] = defaultdict(list)
    for event in events:
        try:
            source_trace_id = _source_trace_id(event.event)
        except VendorTraceFormatError as exc:
            issues.append(TraceNormalizationIssue(f"event-{event.ordinal}", str(exc)))
            continue
        by_trace[source_trace_id].append(event)
    observations: list[VendorObservation] = []
    for source_trace_id in sorted(by_trace):
        try:
            observations.extend(_trace_observations(source_trace_id, by_trace[source_trace_id]))
        except VendorTraceFormatError as exc:
            issues.append(TraceNormalizationIssue(f"trace-{source_trace_id}", str(exc)))
    return build_vendor_traces(
        observations,
        vendor=VENDOR,
        source=source,
        semantic_convention_version=semantic_convention_version,
        initial_issues=issues,
        strict_tool_pairing=True,
    )


def _events_from_payload(payload: JsonValue) -> tuple[_PostHogEvent, ...]:
    """Flatten supported PostHog export shapes without a generic vendor auto-detector.

    Args:
        payload: Single event, event array, query wrapper, or normalized HogQL event rows.

    Returns:
        Ordered source events with deterministic tie-break keys.

    Raises:
        VendorTraceFormatError: The payload is not a supported wrapper or event shape.
    """
    events = flatten_records(
        payload,
        vendor="PostHog",
        wrapper_keys=("results", "events"),
        record_keys=("event", "properties"),
    )
    return tuple(
        _PostHogEvent(
            event=event,
            ordinal=index,
            source_order_key=_source_event_order_key(event, index),
        )
        for index, event in enumerate(events)
    )


def _trace_observations(
    source_trace_id: str,
    events: Sequence[_PostHogEvent],
) -> tuple[VendorObservation, ...]:
    """Declare ordered vendor observations for one PostHog trace.

    PostHog producers may declare trace-level WMO facts, such as the outcome, conversation, or
    customer identity, on events that emit no span evidence, such as ``$ai_metric`` or
    ``$ai_feedback``. Those facts are collected from every event and folded onto the first
    emitted observation so the shared canonical conversion still sees and cross-checks them.

    Args:
        source_trace_id: PostHog trace key shared by these events.
        events: Source events of this trace, in export order.

    Returns:
        Declared observations in canonical order, with the request text on the first.

    Raises:
        VendorTraceFormatError: An event is invalid, the trace declares no user request, or a
            trace-level extension fact disagrees across events.
    """
    ordered = sorted(
        events,
        key=lambda event: (_event_timestamp(event.event), event.source_order_key, event.ordinal),
    )
    initial_request = _initial_request(ordered)
    if initial_request is None:
        raise PostHogPullError("PostHog trace has no initial user message or root task text")
    task, initial_event = initial_request
    observations: list[VendorObservation] = []
    folded: JsonObject = {}
    for event in ordered:
        observation = _event_observation(
            source_trace_id,
            event,
            ordinal=len(observations),
            initial=event is initial_event,
        )
        if observation is None:
            extensions = _event_extensions(_properties(event.event), initial=event is initial_event)
            _fold_trace_level_extensions(folded, extensions)
            continue
        if not observations:
            observation = replace(observation, request_text=task)
        observations.append(observation)
    if folded and observations:
        first = observations[0]
        _fold_trace_level_extensions(folded, first.extensions)
        observations[0] = replace(first, extensions={**first.extensions, **folded})
    return tuple(observations)


def _event_extensions(properties: JsonObject, *, initial: bool) -> JsonObject:
    """Collect approved WMO extensions from one event with PostHog's null leniency.

    Args:
        properties: PostHog event properties.
        initial: Whether this event carries the trace's initial request and may keep
            request-visible extensions.

    Returns:
        Approved extension attributes with null-valued keys dropped.
    """
    extensions: JsonObject = {
        key: value for key, value in approved_extensions(properties).items() if value is not None
    }
    if not initial:
        for key in _REQUEST_VISIBLE_KEYS:
            extensions.pop(key, None)
    return extensions


def _fold_trace_level_extensions(target: JsonObject, extensions: JsonObject) -> None:
    """Fold one event's trace-level extension facts, rejecting cross-event disagreement.

    Args:
        target: Accumulated trace-level extension facts, updated in place.
        extensions: One event's approved extensions.

    Raises:
        PostHogPullError: A trace-level fact differs from an earlier event's declaration.
    """
    for key in _TRACE_LEVEL_KEYS:
        if key not in extensions:
            continue
        if key in target and target[key] != extensions[key]:
            raise PostHogPullError(f"{key} differs across events in one PostHog trace")
        target[key] = extensions[key]


def _event_observation(
    source_trace_id: str,
    event: _PostHogEvent,
    *,
    ordinal: int,
    initial: bool,
) -> VendorObservation | None:
    """Declare what one PostHog event observes, or nothing for non-evidence events.

    Args:
        source_trace_id: PostHog trace key of this event.
        event: Source event with its order metadata.
        ordinal: Per-trace emission position used for stable tie breaking.
        initial: Whether this event carries the trace's initial request and may keep
            request-visible extensions.

    Returns:
        Declared observation, or ``None`` when the event carries no span evidence.

    Raises:
        VendorTraceFormatError: The event declares invalid timing, identity, or tool fields.
    """
    event_name = _event_name(event.event)
    properties = _properties(event.event)
    timestamp = _event_timestamp(event.event)
    extensions = _event_extensions(properties, initial=initial)
    if event_name == "$ai_generation":
        choices = properties.get("$ai_output_choices")
        tool_calls = declared_tool_calls(choices)
        return VendorObservation(
            source_trace_id=source_trace_id,
            source_span_id=_source_span_id(event),
            ordinal=ordinal,
            started_at=timestamp,
            ended_at=timestamp,
            kind="model",
            source_parent_span_id=_source_parent_span_id(properties),
            input_messages=properties.get("$ai_input"),
            completion_text=None if tool_calls else _choices_text(choices),
            tool_calls=tool_calls,
            model=_model_identity(properties),
            failure_message=_failure_message(properties),
            extensions=extensions,
        )
    if event_name == "$ai_span":
        return VendorObservation(
            source_trace_id=source_trace_id,
            source_span_id=_source_span_id(event),
            ordinal=ordinal,
            started_at=timestamp,
            ended_at=timestamp,
            kind="tool_result",
            source_parent_span_id=_source_parent_span_id(properties),
            tool_name=required_text(properties.get("$ai_span_name"), "PostHog $ai_span_name"),
            tool_arguments=json_text(properties.get("$ai_input_state")),
            tool_message=json_text(properties.get("$ai_output_state")),
            tool_call_id=_explicit_tool_call_id(properties),
            failure_message=_failure_message(properties),
            extensions=extensions,
        )
    if event_name == "$ai_trace" or _is_error(properties):
        output_state = properties.get("$ai_output_state") if event_name == "$ai_trace" else None
        return VendorObservation(
            source_trace_id=source_trace_id,
            source_span_id=_source_span_id(event),
            ordinal=ordinal,
            started_at=timestamp,
            ended_at=timestamp,
            kind="agent",
            source_parent_span_id=_source_parent_span_id(properties),
            completion_text=None if output_state is None else json_text(output_state),
            failure_message=_failure_message(properties),
            extensions=extensions,
        )
    return None


def _initial_request(
    events: Sequence[_PostHogEvent],
) -> tuple[str, _PostHogEvent] | None:
    """Read the first user request and the event whose request-visible extensions are kept.

    Args:
        events: Source events of one trace in canonical order.

    Returns:
        Request text and its declaring event, or ``None`` when no event carries one.
    """
    for event in events:
        task = first_user_text(_properties(event.event).get("$ai_input"))
        if task is not None:
            return task, event
    return None


def _choices_text(value: JsonValue | None) -> str:
    """Read the first assistant text choice, retaining empty completion evidence when absent.

    Args:
        value: Declared ``$ai_output_choices`` value.

    Returns:
        First readable choice text, falling back to compact JSON for structured output.
    """
    if isinstance(value, list):
        for choice in value:
            if isinstance(choice, dict):
                text = message_text(choice.get("content"))
                if text:
                    return text
    return json_text(value)


def _model_identity(properties: JsonObject) -> VendorModelIdentity | None:
    """Capture PostHog model identity only when both provider and model are present.

    Args:
        properties: PostHog event properties.

    Returns:
        Declared model identity, or ``None`` when the event declares neither field.

    Raises:
        PostHogPullError: The event declares only one of provider and model.
    """
    model_id = first_text(properties, ("$ai_model", "$ai_model_id", "model"))
    provider = first_text(properties, ("$ai_provider", "$ai_system", "provider"))
    if model_id is None and provider is None:
        return None
    if model_id is None:
        raise PostHogPullError("PostHog model identity has a provider but no model")
    if provider is None:
        raise PostHogPullError("PostHog model identity has a model but no provider")
    return VendorModelIdentity(
        provider=provider,
        model_id=model_id,
        revision=first_text(properties, ("$ai_model_revision", "model_revision")),
    )


def _explicit_tool_call_id(properties: JsonObject) -> str | None:
    """Read one consistent explicit PostHog tool-result call identity when supplied.

    Args:
        properties: PostHog event properties.

    Returns:
        The declared call identity, or ``None`` when the event declares none.

    Raises:
        PostHogPullError: The declared identity fields are blank or disagree.
    """
    values = tuple(
        required_text(properties[key], f"PostHog {key}")
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


def _failure_message(properties: JsonObject) -> str | None:
    """Map PostHog's current error flag and message fields to declared failure text.

    Args:
        properties: PostHog event properties.

    Returns:
        Declared or fallback failure text, or ``None`` when the event declares no error.
    """
    if not _is_error(properties):
        return None
    message = first_text(properties, ("$ai_error", "error", "$exception_message"))
    return message or "PostHog AI event marked as an error"


def _source_trace_id(event: JsonObject) -> str:
    """Resolve PostHog's trace ID with the documented span or event fallback.

    Args:
        event: Decoded PostHog event object.

    Returns:
        The trace key grouping this event.

    Raises:
        PostHogPullError: The event declares no usable identity field.
    """
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
    """Resolve one PostHog source span identity used for stable IDs and parent conversion.

    Args:
        event: Source event with its order metadata.

    Returns:
        Declared span, uuid, or id value, or the source ordinal fallback.
    """
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
    """Read the PostHog parent span identity when this source event supplies one.

    Args:
        properties: PostHog event properties.

    Returns:
        Declared parent span key, or ``None`` when the event declares none.

    Raises:
        VendorTraceFormatError: The declared parent identity is blank or not text.
    """
    value = properties.get("$ai_parent_id")
    if value is None:
        return None
    return required_text(value, "PostHog $ai_parent_id")


def _source_event_order_key(event: JsonObject, ordinal: int) -> str:
    """Return a stable event identity for timestamp-tie ordering, or source ordinal fallback.

    Args:
        event: Decoded PostHog event object.
        ordinal: Source order position across the export.

    Returns:
        Stable tie-break key for this event.
    """
    for key in ("uuid", "id"):
        value = event.get(key)
        if isinstance(value, str) and value:
            return f"{key}:{value}"
    return f"ordinal:{ordinal:020d}"


def _event_timestamp(event: JsonObject) -> datetime:
    """Parse a timezone-aware PostHog event timestamp for stable trace ordering.

    Args:
        event: Decoded PostHog event object.

    Returns:
        Timezone-aware UTC event instant.

    Raises:
        PostHogPullError: The timestamp is absent, malformed, or naive.
    """
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
    """Return one PostHog event name or an empty string for non-AI records.

    Args:
        event: Decoded PostHog event object.

    Returns:
        Declared event name, empty when absent.
    """
    value = event.get("event")
    return value if isinstance(value, str) else ""


def _properties(event: JsonObject) -> JsonObject:
    """Return PostHog's properties object or an empty object when it is absent.

    Args:
        event: Decoded PostHog event object.

    Returns:
        Declared properties object, empty when absent or malformed.
    """
    value = event.get("properties")
    return value if isinstance(value, dict) else {}


def _is_error(properties: JsonObject) -> bool:
    """Interpret PostHog error flags without treating ordinary values as errors.

    Args:
        properties: PostHog event properties.

    Returns:
        Whether the event declares itself failed.
    """
    value = properties.get("$ai_is_error")
    if isinstance(value, bool):
        return value
    error = properties.get("$ai_error")
    return isinstance(error, str) and bool(error.strip())
