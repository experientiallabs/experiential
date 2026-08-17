"""Normalize Arize Phoenix and OpenInference exports into canonical trace evidence.

Phoenix publishes the same OpenInference span semantics through three shapes, and all three are
accepted here:

- an OTLP JSON envelope whose span attributes carry OpenInference keys,
- native Phoenix span objects with a ``context`` identity object and nested attributes,
- flat span rows whose column names are dotted paths such as ``attributes.llm.model_name``.

OpenInference span kinds map to canonical evidence by what they observe. ``LLM`` becomes a model
call, including the tool calls its output messages request. ``TOOL`` becomes the tool result paired
with the earlier requesting call. ``CHAIN``, ``AGENT``, ``RETRIEVER``, ``EMBEDDING``, ``RERANKER``,
``GUARDRAIL``, and ``EVALUATOR`` observe orchestration or scoring rather than a visible agent step
and are not converted.

Model identity is retained only when a span declares both an OpenInference provider or system and a
model name. Token accounting comes from ``llm.token_count.prompt`` and
``llm.token_count.completion``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from pydantic import JsonValue

from wmo.common.core.artifacts import JsonObject
from wmo.simulation.ingest.otlp import decode_otlp_attributes
from wmo.simulation.ingest.vendor_observations import (
    VendorObservation,
    VendorTokenUsage,
    declared_completion_text,
    declared_model_identity,
    declared_tool_calls,
)
from wmo.simulation.ingest.vendor_records import (
    VendorTraceFormatError,
    dotted_lookup,
    first_user_text,
    flatten_records,
    json_text,
    message_text,
    required_text,
    source_timestamp,
)
from wmo.simulation.ingest.vendor_source import VendorSource
from wmo.simulation.ingest.vendor_trace import approved_extensions

VENDOR = "phoenix"

_MODEL_KINDS = frozenset({"llm"})
_TOOL_KINDS = frozenset({"tool"})
_KIND_KEYS = ("openinference.span.kind", "span_kind", "spanKind")
_MODEL_KEYS = ("llm.model_name", "model_name", "llm.model")
_PROVIDER_KEYS = ("llm.provider", "llm.system", "provider")
_PROMPT_TOKEN_KEY = "llm.token_count.prompt"
_COMPLETION_TOKEN_KEY = "llm.token_count.completion"
_WRAPPER_KEYS = ("spans", "data", "results", "items", "traces")
_RECORD_KEYS = ("context", "span_id", "context.span_id", "spanId")


@dataclass(frozen=True)
class _PhoenixRecord:
    """One undecoded Phoenix span record and the export shape that declared it.

    Args:
        raw: Undecoded span record.
        otlp: Whether the record came from an OTLP envelope rather than a native export.
    """

    raw: JsonObject
    otlp: bool


@dataclass(frozen=True)
class _PhoenixSpan:
    """One Phoenix span reduced to the identity, timing, and attributes WMO reads.

    Args:
        trace_id: Source trace identity.
        span_id: Source span identity.
        parent_id: Declared source parent span identity, if any.
        name: Declared span name.
        attributes: OpenInference attributes keyed by their dotted names.
        started_at: Source start instant.
        ended_at: Source end instant.
        error: Declared error message, if the span declares a failure.
    """

    trace_id: str
    span_id: str
    parent_id: str | None
    name: str
    attributes: JsonObject
    started_at: datetime
    ended_at: datetime
    error: str | None


def _record_observations(record: _PhoenixRecord, ordinal: int) -> tuple[VendorObservation, ...]:
    """Convert one Phoenix record through its declared shape into observations.

    Args:
        record: Undecoded Phoenix span record marked with its declared export shape.
        ordinal: Source order position for the emitted observation.

    Returns:
        Declared observation, or nothing for orchestration-only span kinds.

    Raises:
        VendorTraceFormatError: The record lacks identity, timing, or tool evidence.
        OtlpTraceFormatError: An OTLP record declares malformed attributes.
    """
    span = _otlp_span(record.raw) if record.otlp else _native_span(record.raw)
    converted = _span_observation(span, ordinal)
    return () if converted is None else (converted,)


def _payload_records(payload: JsonValue) -> tuple[_PhoenixRecord, ...]:
    """Read every Phoenix span record declared by one decoded document.

    Args:
        payload: Decoded Phoenix document, OTLP envelope, span array, or span row.

    Returns:
        Undecoded span records in source order, each marked with its declared shape.

    Raises:
        VendorTraceFormatError: The document is not a supported Phoenix shape.
        OtlpTraceFormatError: An OTLP envelope declares a malformed span array.
    """
    if isinstance(payload, dict) and ("resourceSpans" in payload or "resource_spans" in payload):
        return _otlp_records(payload)
    records = flatten_records(
        payload,
        vendor=VENDOR,
        wrapper_keys=_WRAPPER_KEYS,
        record_keys=_RECORD_KEYS,
    )
    return tuple(_PhoenixRecord(raw=record, otlp=False) for record in records)


def _otlp_records(payload: JsonObject) -> tuple[_PhoenixRecord, ...]:
    """Read undecoded Phoenix span records from one OTLP JSON envelope.

    Args:
        payload: OTLP envelope with resource and scope span arrays.

    Returns:
        Undecoded OTLP span records in source order.

    Raises:
        VendorTraceFormatError: The envelope structure is malformed.
        OtlpTraceFormatError: A span declares malformed OTLP attributes.
    """
    resource_spans = payload.get("resourceSpans", payload.get("resource_spans"))
    if not isinstance(resource_spans, list):
        raise VendorTraceFormatError("Phoenix OTLP envelope needs a resourceSpans array")
    spans: list[_PhoenixRecord] = []
    for resource_span in resource_spans:
        if not isinstance(resource_span, dict):
            raise VendorTraceFormatError("Phoenix OTLP resourceSpans entries must be objects")
        scope_spans = resource_span.get("scopeSpans", resource_span.get("scope_spans"))
        if not isinstance(scope_spans, list):
            raise VendorTraceFormatError("Phoenix OTLP resource span needs a scopeSpans array")
        for scope_span in scope_spans:
            if not isinstance(scope_span, dict):
                raise VendorTraceFormatError("Phoenix OTLP scopeSpans entries must be objects")
            raw_spans = scope_span.get("spans")
            if not isinstance(raw_spans, list):
                raise VendorTraceFormatError("Phoenix OTLP scope span needs a spans array")
            for raw_span in raw_spans:
                if not isinstance(raw_span, dict):
                    raise VendorTraceFormatError("Phoenix OTLP spans entries must be objects")
                spans.append(_PhoenixRecord(raw=raw_span, otlp=True))
    return tuple(spans)


def _otlp_span(raw: JsonObject) -> _PhoenixSpan:
    """Convert one OTLP span object into a Phoenix span.

    Args:
        raw: OTLP span object.

    Returns:
        Phoenix span with decoded OpenInference attributes.

    Raises:
        VendorTraceFormatError: Identity or timing is absent.
        OtlpTraceFormatError: The span declares malformed OTLP attributes.
    """
    attributes = decode_otlp_attributes(raw.get("attributes"), label="Phoenix span attributes")
    parent = raw.get("parentSpanId", raw.get("parent_span_id"))
    return _PhoenixSpan(
        trace_id=required_text(raw.get("traceId", raw.get("trace_id")), "Phoenix traceId"),
        span_id=required_text(raw.get("spanId", raw.get("span_id")), "Phoenix spanId"),
        parent_id=parent if isinstance(parent, str) and parent.strip() else None,
        name=required_text(raw.get("name"), "Phoenix span name"),
        attributes=attributes,
        started_at=_nano_timestamp(
            raw.get("startTimeUnixNano", raw.get("start_time_unix_nano")),
            "Phoenix startTimeUnixNano",
        ),
        ended_at=_nano_timestamp(
            raw.get("endTimeUnixNano", raw.get("end_time_unix_nano")),
            "Phoenix endTimeUnixNano",
        ),
        error=_otlp_error(raw.get("status")),
    )


def _nano_timestamp(value: JsonValue | None, label: str) -> datetime:
    """Parse one OTLP epoch-nanosecond instant, accepting the string form vendors emit.

    Args:
        value: Epoch nanoseconds as an integer or digit text, or an ISO-8601 timestamp.
        label: Field label used in the validation message.

    Returns:
        Timezone-aware UTC instant.

    Raises:
        VendorTraceFormatError: The value is absent or is not a supported instant.
    """
    if isinstance(value, str) and value.strip().isdigit():
        value = int(value.strip())
    if isinstance(value, bool) or not isinstance(value, int):
        return source_timestamp(value, label)
    try:
        return datetime.fromtimestamp(value / 1_000_000_000, UTC)
    except (OSError, OverflowError, ValueError) as exc:
        raise VendorTraceFormatError(f"{label} is outside the supported range") from exc


def _otlp_error(status: JsonValue | None) -> str | None:
    """Read the declared error message from one OTLP span status.

    Args:
        status: OTLP span status object, if declared.

    Returns:
        Declared error message when the status declares an error, otherwise ``None``.
    """
    if not isinstance(status, dict):
        return None
    code = status.get("code")
    is_error = code == 2 or (isinstance(code, str) and code.casefold().endswith("error"))
    if not is_error:
        return None
    message = status.get("message")
    return message if isinstance(message, str) and message.strip() else "span reported an error"


def _native_span(record: JsonObject) -> _PhoenixSpan:
    """Convert one native or flat Phoenix span record into a Phoenix span.

    Args:
        record: Native Phoenix span object or flat dotted span row.

    Returns:
        Phoenix span with dotted OpenInference attributes.

    Raises:
        VendorTraceFormatError: Identity or timing is absent.
    """
    attributes = _record_attributes(record)
    parent = dotted_lookup(record, ("parent_id", "parentId", "parent_span_id", "parentSpanId"))
    return _PhoenixSpan(
        trace_id=required_text(
            dotted_lookup(record, ("context.trace_id", "trace_id", "traceId")), "Phoenix trace_id"
        ),
        span_id=required_text(
            dotted_lookup(record, ("context.span_id", "span_id", "spanId")), "Phoenix span_id"
        ),
        parent_id=parent if isinstance(parent, str) and parent.strip() else None,
        name=required_text(dotted_lookup(record, ("name", "span_name")), "Phoenix span name"),
        attributes=attributes,
        started_at=source_timestamp(
            dotted_lookup(record, ("start_time", "startTime")), "Phoenix start_time"
        ),
        ended_at=source_timestamp(
            dotted_lookup(record, ("end_time", "endTime")), "Phoenix end_time"
        ),
        error=_native_error(record),
    )


def _native_error(record: JsonObject) -> str | None:
    """Read the declared error message from one native Phoenix span record.

    Args:
        record: Native Phoenix span object or flat dotted span row.

    Returns:
        Declared error message when the record declares an error status, otherwise ``None``.
    """
    code = dotted_lookup(record, ("status_code", "statusCode", "status.code"))
    if not isinstance(code, str) or code.casefold() != "error":
        return None
    message = dotted_lookup(record, ("status_message", "statusMessage", "status.message"))
    return message if isinstance(message, str) and message.strip() else "span reported an error"


def _record_attributes(record: JsonObject) -> JsonObject:
    """Collect OpenInference attributes from one native or flat Phoenix span record.

    Args:
        record: Native Phoenix span object or flat dotted span row.

    Returns:
        Attributes keyed by their dotted OpenInference names.
    """
    attributes: JsonObject = {}
    declared = record.get("attributes")
    if isinstance(declared, dict):
        _flatten_into(declared, "", attributes)
    for key, value in record.items():
        if key.startswith("attributes."):
            attributes[key[len("attributes.") :]] = value
        elif "." in key and key.split(".", 1)[0] in {"llm", "tool", "input", "output", "message"}:
            attributes[key] = value
        elif key in {"openinference.span.kind", "span_kind", "spanKind"}:
            attributes[key] = value
    return attributes


def _flatten_into(node: JsonObject, prefix: str, target: JsonObject) -> None:
    """Flatten one nested attribute object into dotted keys.

    Args:
        node: Nested attribute object.
        prefix: Dotted prefix accumulated so far.
        target: Accumulator for flattened attributes.
    """
    for key, value in node.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            target[path] = value
            _flatten_into(value, f"{path}.", target)
        else:
            target[path] = value


def _span_observation(span: _PhoenixSpan, ordinal: int) -> VendorObservation | None:
    """Convert one Phoenix span to a declared model or tool-result observation.

    Args:
        span: Phoenix span.
        ordinal: Source order position for the emitted observation.

    Returns:
        Declared observation, or ``None`` for orchestration-only span kinds.

    Raises:
        VendorTraceFormatError: The span lacks tool identity or declares invalid usage.
    """
    kind = _span_kind(span.attributes)
    if kind not in _MODEL_KINDS | _TOOL_KINDS:
        return None
    extensions = approved_extensions(span.attributes)
    if kind in _TOOL_KINDS:
        return VendorObservation(
            source_trace_id=span.trace_id,
            source_span_id=span.span_id,
            ordinal=ordinal,
            started_at=span.started_at,
            ended_at=span.ended_at,
            kind="tool_result",
            source_parent_span_id=span.parent_id,
            tool_name=_tool_name(span),
            tool_arguments=_tool_arguments(span.attributes),
            tool_message=_output_text(span.attributes),
            tool_call_id=_text(
                span.attributes, ("tool_call.id", "tool.call_id", "message.tool_call_id")
            ),
            failure_message=span.error,
            extensions=extensions,
        )
    messages = _messages(span.attributes.get("llm.input_messages"))
    outputs = _messages(span.attributes.get("llm.output_messages"))
    model, declared_model = declared_model_identity(
        span.attributes,
        model_keys=_MODEL_KEYS,
        provider_keys=_PROVIDER_KEYS,
    )
    return VendorObservation(
        source_trace_id=span.trace_id,
        source_span_id=span.span_id,
        ordinal=ordinal,
        started_at=span.started_at,
        ended_at=span.ended_at,
        kind="model",
        source_parent_span_id=span.parent_id,
        request_text=first_user_text(messages) or _input_text(span.attributes),
        input_messages=messages,
        completion_text=declared_completion_text(outputs) or _output_text(span.attributes) or None,
        tool_calls=declared_tool_calls(outputs),
        model=model,
        usage=_usage(span.attributes),
        failure_message=span.error,
        declared_attributes=(
            {} if declared_model is None else {"gen_ai.request.model": declared_model}
        ),
        extensions=extensions,
    )


def _span_kind(attributes: JsonObject) -> str:
    """Read the declared OpenInference span kind.

    Args:
        attributes: Dotted OpenInference attributes.

    Returns:
        Lowercase declared span kind, empty when the span declares none.
    """
    return (_text(attributes, _KIND_KEYS) or "").casefold()


def _tool_name(span: _PhoenixSpan) -> str:
    """Read the executed tool name of one OpenInference tool span.

    Args:
        span: Phoenix span of kind tool.

    Returns:
        Declared tool name, falling back to the span name.

    Raises:
        VendorTraceFormatError: Neither the attributes nor the span name declare a tool name.
    """
    declared = _text(span.attributes, ("tool.name", "tool_name"))
    if declared is not None:
        return declared
    return required_text(span.name, "Phoenix tool span name")


def _tool_arguments(attributes: JsonObject) -> str:
    """Read declared tool arguments from one OpenInference tool span.

    Args:
        attributes: Dotted OpenInference attributes.

    Returns:
        Declared arguments as durable JSON text.
    """
    for key in ("tool.parameters", "tool.arguments", "input.value"):
        value = attributes.get(key)
        if value is not None:
            return json_text(value)
    return "{}"


def _input_text(attributes: JsonObject) -> str | None:
    """Read the declared request text from an OpenInference input value.

    Args:
        attributes: Dotted OpenInference attributes.

    Returns:
        Declared request text, or ``None`` when the span declares none.
    """
    value = attributes.get("input.value")
    if value is None:
        return None
    text = message_text(value)
    return text or None


def _output_text(attributes: JsonObject) -> str:
    """Read the declared output text from an OpenInference output value.

    Args:
        attributes: Dotted OpenInference attributes.

    Returns:
        Declared output text, empty when the span declares none.
    """
    value = attributes.get("output.value")
    return message_text(value) if value is not None else ""


def _messages(value: JsonValue | None) -> JsonValue | None:
    """Unwrap OpenInference message and tool-call envelopes into plain message objects.

    Args:
        value: Declared ``llm.input_messages`` or ``llm.output_messages`` value.

    Returns:
        Plain message objects, or ``None`` when the span declares no message list.
    """
    if not isinstance(value, list):
        return None
    messages: list[JsonValue] = []
    for item in value:
        message = item.get("message") if isinstance(item, dict) else None
        source = message if isinstance(message, dict) else item
        if not isinstance(source, dict):
            continue
        unwrapped: JsonObject = {
            key: source[key] for key in source if key not in {"tool_calls", "toolCalls"}
        }
        calls = source.get("tool_calls", source.get("toolCalls"))
        if isinstance(calls, list):
            unwrapped["tool_calls"] = [_tool_call(call) for call in calls]
        messages.append(unwrapped)
    return messages


def _tool_call(call: JsonValue) -> JsonValue:
    """Unwrap one OpenInference tool-call envelope.

    Args:
        call: Declared tool-call entry.

    Returns:
        The inner tool-call object when the entry wraps one, otherwise the entry itself.
    """
    if isinstance(call, dict):
        inner = call.get("tool_call", call.get("toolCall"))
        if isinstance(inner, dict):
            return inner
    return call


def _usage(attributes: JsonObject) -> VendorTokenUsage | None:
    """Read declared token accounting from OpenInference token counts.

    Args:
        attributes: Dotted OpenInference attributes.

    Returns:
        Declared usage, or ``None`` when the span declares no complete accounting.

    Raises:
        VendorTraceFormatError: A declared token count is not a non-negative integer.
    """
    prompt = attributes.get(_PROMPT_TOKEN_KEY)
    completion = attributes.get(_COMPLETION_TOKEN_KEY)
    if prompt is None or completion is None:
        return None
    if not isinstance(prompt, int) or isinstance(prompt, bool) or prompt < 0:
        raise VendorTraceFormatError(f"Phoenix {_PROMPT_TOKEN_KEY} must be a non-negative integer")
    if not isinstance(completion, int) or isinstance(completion, bool) or completion < 0:
        raise VendorTraceFormatError(
            f"Phoenix {_COMPLETION_TOKEN_KEY} must be a non-negative integer"
        )
    return VendorTokenUsage(input_tokens=prompt, output_tokens=completion)


def _text(attributes: JsonObject, keys: Sequence[str]) -> str | None:
    """Read the first declared non-empty text attribute.

    Args:
        attributes: Dotted OpenInference attributes.
        keys: Candidate attribute names in preference order.

    Returns:
        First declared non-empty text value, or ``None`` when none is declared.
    """
    for key in keys:
        value = attributes.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


PHOENIX_SOURCE: VendorSource[_PhoenixRecord] = VendorSource(
    vendor=VENDOR,
    records=_payload_records,
    convert=_record_observations,
)
