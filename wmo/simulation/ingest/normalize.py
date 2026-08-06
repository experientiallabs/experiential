"""Shared span-to-`Trace` normalizer — the one core every span-based adapter reuses.

Most agent-observability providers (Arize/Phoenix, Langfuse, LangSmith, Braintrust) export spans
that follow either the OpenTelemetry **GenAI** semantic conventions (`gen_ai.*`) or the
**OpenInference** conventions (`llm.*`, `tool.*`, `input.value`/`output.value`, `openinference.span.
kind`). Rather than write a bespoke parser per provider, an adapter normalizes its raw export into a
flat list of `SpanRecord`s and calls `spans_to_traces()` here. The attribute *keys* differ across
conventions, so the field extractors below look in both vocabularies (GenAI first, OpenInference as
fallback).

Pipeline:
  raw OTLP/OpenInference payload --(collect_spans / a provider's own transform)--> list[SpanRecord]
  list[SpanRecord] --(spans_to_traces)--> list[Trace]

`spans_to_traces` groups by `trace_id`, orders each group by start time, and pairs each LLM/agent
span (an Action) with the following tool/execution span (its Observation), mirroring how a real
agent step reads: `(state, action) -> observation`. The optional `wmo.*` enrichment attributes
(`wmo.state.*` -> `Step.state_before`, `wmo.trace.metadata` -> `Trace.metadata`) are honored on any
span, so a faithfully captured trace round-trips for open-loop replay.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

from pydantic import BaseModel, Field, JsonValue

from wmo.core.types import (
    Action,
    ActionKind,
    EnvState,
    ErrorClass,
    JsonObject,
    Observation,
    Step,
    StepAttribution,
    Trace,
)

# --- attribute vocabularies (GenAI semconv first, OpenInference fallback) ---------------------

# Operation/kind markers.
_LLM_OPS = frozenset({"chat", "text_completion", "invoke_agent", "generate_content"})
_TOOL_OPS = frozenset({"execute_tool"})
# OpenInference span kinds (attribute `openinference.span.kind`).
_OI_LLM_KINDS = frozenset({"LLM", "AGENT", "CHAIN"})
_OI_TOOL_KINDS = frozenset({"TOOL"})

# A tool call's name, in priority order across conventions.
_TOOL_NAME_KEYS = ("gen_ai.tool.name", "tool.name", "tool_call.function.name")
# A tool call's serialized arguments, in priority order.
_TOOL_ARG_KEYS = (
    "gen_ai.tool.call.arguments",
    "gen_ai.tool.arguments",
    "gen_ai.tool.input",
    "gen_ai.request.arguments",
    "tool_call.function.arguments",
    "input.value",
    "input",
)
# A tool execution's output, in priority order.
_TOOL_OUTPUT_KEYS = (
    "gen_ai.tool.message",
    "gen_ai.tool.output",
    "gen_ai.tool.call.result",
    "gen_ai.tool.result",
    "gen_ai.completion",
    "output.value",
    "output",
)
# The originating prompt / task, in priority order.
_PROMPT_KEYS = ("gen_ai.prompt", "input.value", "llm.input_messages")
# An LLM message completion, in priority order.
_COMPLETION_KEYS = ("gen_ai.completion", "output.value", "llm.output_messages")
# Presence of any of these marks a span as an LLM/agent span when no explicit op/kind is set.
_LLM_PRESENCE_KEYS = (
    "gen_ai.request.model",
    "gen_ai.completion",
    "gen_ai.prompt",
    "llm.model_name",
    "llm.input_messages",
)

# Optional `wmo.*` enrichment keys (a strict superset of any semconv).
_STATE_STRUCTURED_KEY = "wmo.state.structured"
_STATE_SCRATCHPAD_KEY = "wmo.state.scratchpad"
_TRACE_METADATA_KEY = "wmo.trace.metadata"
# The pre-rename spelling of the key above. Corpora emitted before the wmh -> wmo rename still
# carry it — including anything produced by the published `environment-capture` until its own
# rename ships — and dropping their metadata silently would lose benchmark/task_id/reward.
_LEGACY_TRACE_METADATA_KEY = "wmh.trace.metadata"
_ATTRIBUTION_KEY = "wmo.attribution"

# Per-step attribution vocabularies (GenAI semconv first, OpenInference fallback). The response
# model is preferred over the request model: with provider-side routing/fallback they differ, and
# attribution wants the model that actually answered.
_MODEL_KEYS = ("gen_ai.response.model", "gen_ai.request.model", "llm.model_name")
_PROVIDER_KEYS = ("gen_ai.system", "gen_ai.provider.name", "llm.provider")
_INPUT_TOKEN_KEYS = (
    "gen_ai.usage.input_tokens",
    "gen_ai.usage.prompt_tokens",
    "llm.token_count.prompt",
)
_OUTPUT_TOKEN_KEYS = (
    "gen_ai.usage.output_tokens",
    "gen_ai.usage.completion_tokens",
    "llm.token_count.completion",
)
_COST_KEYS = ("gen_ai.usage.cost", "llm.usage.total_cost")
# `gen_ai.request.*` keys that are NOT sampling/config knobs: the model has its own field, and
# "arguments" is a legacy tool-args location (see _TOOL_ARG_KEYS), not an LLM parameter.
_NON_CONFIG_REQUEST_KEYS = frozenset({"gen_ai.request.model", "gen_ai.request.arguments"})
_REQUEST_PREFIX = "gen_ai.request."


class SpanRecord(BaseModel):
    """A flattened span with attributes decoded to plain JSON — the normalizer's input unit.

    Adapters either build these directly (from a provider's own event shape) or via
    `collect_spans` (from an OTLP/OpenInference-JSON payload).
    """

    trace_id: str
    span_id: str = ""
    parent_span_id: str = ""
    name: str = ""
    start_nano: int = 0
    end_nano: int = 0
    attributes: JsonObject = Field(default_factory=dict)
    status_error: bool = False


class SpanEmitter:
    """Ordered `SpanRecord` builder for one trace, shared by the row-shaped adapters.

    Providers that log rows rather than OTLP spans (Langfuse, LangSmith, Braintrust, Mastra,
    PostHog) all synthesize spans in the GenAI vocabulary the same way: a per-trace ordinal that
    is both the span id suffix and the start-time sort key, `chat` for an action span and
    `execute_tool` for a tool-result span, and trace-level attributes seeded onto the first span
    only. This class owns that shape so the adapters keep only their own row mapping.

    Args:
        trace_id: Trace every emitted span belongs to.
        first_attributes: Attributes seeded onto the first span emitted (the trace task and
            metadata). Seeded with `setdefault`, so a span carrying its own value keeps it.
    """

    def __init__(self, trace_id: str, first_attributes: JsonObject | None = None) -> None:
        self.trace_id = trace_id
        self.spans: list[SpanRecord] = []
        self._first_attributes = first_attributes or {}

    def emit(self, attrs: JsonObject, *, tool: bool, error: bool = False) -> None:
        """Append one span for this trace: `execute_tool` when `tool`, else `chat`."""
        ordinal = len(self.spans)
        if ordinal == 0:
            for key, value in self._first_attributes.items():
                attrs.setdefault(key, value)
        name = "execute_tool" if tool else "chat"
        self.spans.append(
            SpanRecord(
                trace_id=self.trace_id,
                span_id=f"{self.trace_id[:12]}{ordinal:06x}{'t' if tool else 'a'}",
                name=name,
                start_nano=ordinal,
                attributes={"gen_ai.operation.name": name, **attrs},
                status_error=error,
            )
        )


# --- value coercion ---------------------------------------------------------------------------


def iso_to_ordinal(value: object, fallback: int) -> int:
    """Map an ISO-8601 timestamp (or a `datetime`) to epoch microseconds; `fallback` if unusable.

    Accepts a string OR a `datetime`/pandas `Timestamp` (Phoenix's `get_spans_dataframe` yields
    datetimes, not ISO strings). A naive value (no tz offset — e.g. LangSmith's `2026-01-01T00:00`)
    is treated as **UTC**, not the machine's local zone, so ordering is reproducible across hosts.
    Only monotonicity within a trace matters (spans_to_traces sorts by start_nano), so microsecond
    precision and a list-index fallback are plenty. Shared by every row adapter's timestamp order.
    """
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str) and value:
        text = value[:-1] + "+00:00" if value.endswith("Z") else value
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return fallback
    else:
        return fallback
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    try:
        return int(parsed.timestamp() * 1_000_000)
    except (ValueError, OverflowError, OSError):
        return fallback


def to_int(value: JsonValue) -> int:
    """Coerce an OTLP numeric/string to int; bool (an int subclass) is treated as non-numeric."""
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return 0
    return 0


def as_text(value: JsonValue) -> str:
    """Render a value as a JSON-clean string: strings pass through, else compact JSON (no repr)."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _as_str(value: JsonValue) -> str:
    return value if isinstance(value, str) else ""


# --- OpenAI chat-completion tool-call shape ---------------------------------------------------


def openai_tool_calls(output: JsonValue) -> list[JsonObject]:
    """Extract OpenAI-style tool calls from an `output` (a message object or a message list)."""
    if isinstance(output, dict):
        raw = output.get("tool_calls")
        if isinstance(raw, list):
            return [tc for tc in raw if isinstance(tc, dict)]
    if isinstance(output, list):
        calls: list[JsonObject] = []
        for message in output:
            if isinstance(message, dict):
                raw = message.get("tool_calls")
                if isinstance(raw, list):
                    calls.extend(tc for tc in raw if isinstance(tc, dict))
        return calls
    return []


def openai_call_name_args(tool_call: JsonObject) -> tuple[str, str]:
    """(name, raw-arguments-json) from a tool call in OpenAI-nested or flattened shape.

    Arguments are usually a JSON *string* (OpenAI) but may be an object; either way the returned
    value is a string the span carries, which the normalizer's `_tool_args` re-parses.
    """
    fn = tool_call.get("function")
    if isinstance(fn, dict):
        name = fn.get("name")
        args = fn.get("arguments")
    else:
        name = tool_call.get("name")
        args = tool_call.get("arguments")
    name_s = name if isinstance(name, str) else ""
    args_s = args if isinstance(args, str) else as_text(args)
    return name_s, args_s


# --- OTLP / OpenInference AnyValue decoding ---------------------------------------------------


def any_value(value: JsonValue) -> JsonValue:
    """Decode an OTLP `AnyValue` (`{"stringValue": ...}` etc.) to a plain JSON value."""
    if not isinstance(value, dict):
        return value
    if "stringValue" in value:
        return value["stringValue"]
    if "intValue" in value:
        return to_int(value["intValue"])
    if "doubleValue" in value:
        return value["doubleValue"]
    if "boolValue" in value:
        return value["boolValue"]
    if "arrayValue" in value:
        arr = value["arrayValue"]
        values = arr.get("values") if isinstance(arr, dict) else None
        return [any_value(v) for v in values] if isinstance(values, list) else []
    if "kvlistValue" in value:
        kv = value["kvlistValue"]
        values = kv.get("values") if isinstance(kv, dict) else None
        return attrs_to_dict(values) if isinstance(values, list) else {}
    return value


def attrs_to_dict(attrs: JsonValue) -> JsonObject:
    """Turn an OTLP attribute list (`[{"key", "value": <AnyValue>}, ...]`) into a flat dict.

    Also accepts an already-flat `{key: value}` mapping (some providers export that shape), in which
    case values are returned as-is.
    """
    out: JsonObject = {}
    if isinstance(attrs, dict):
        for key, val in attrs.items():
            if isinstance(key, str):
                out[key] = val
        return out
    if not isinstance(attrs, list):
        return out
    for attr in attrs:
        if isinstance(attr, dict):
            key = attr.get("key")
            if isinstance(key, str):
                out[key] = any_value(attr.get("value"))
    return out


def parse_span(raw: JsonValue) -> SpanRecord | None:
    """Parse one OTLP-JSON span object into a `SpanRecord` (None if it lacks a trace id)."""
    if not isinstance(raw, dict):
        return None
    trace_id = raw.get("traceId")
    if not isinstance(trace_id, str) or not trace_id:
        return None
    status = raw.get("status")
    status_error = False
    if isinstance(status, dict):
        code = status.get("code")
        status_error = code in (2, "STATUS_CODE_ERROR")
    return SpanRecord(
        trace_id=trace_id,
        span_id=_as_str(raw.get("spanId")),
        parent_span_id=_as_str(raw.get("parentSpanId")),
        name=_as_str(raw.get("name")),
        start_nano=to_int(raw.get("startTimeUnixNano")),
        end_nano=to_int(raw.get("endTimeUnixNano")),
        attributes=attrs_to_dict(raw.get("attributes")),
        status_error=status_error,
    )


def collect_spans(obj: JsonValue) -> list[SpanRecord]:
    """Walk an OTLP-JSON payload, a list of payloads/spans, or a bare span into `SpanRecord`s."""
    spans: list[SpanRecord] = []
    if isinstance(obj, list):
        for item in obj:
            spans.extend(collect_spans(item))
        return spans
    if not isinstance(obj, dict):
        return spans
    if "resourceSpans" in obj:
        resource_spans = obj["resourceSpans"]
        if isinstance(resource_spans, list):
            for resource_span in resource_spans:
                spans.extend(_spans_in_resource(resource_span))
        return spans
    parsed = parse_span(obj)
    if parsed is not None:
        spans.append(parsed)
    return spans


def _spans_in_resource(resource_span: JsonValue) -> list[SpanRecord]:
    spans: list[SpanRecord] = []
    if not isinstance(resource_span, dict):
        return spans
    scope_spans = resource_span.get("scopeSpans")
    if not isinstance(scope_spans, list):
        return spans
    for scope_span in scope_spans:
        if not isinstance(scope_span, dict):
            continue
        raw_spans = scope_span.get("spans")
        if not isinstance(raw_spans, list):
            continue
        for raw in raw_spans:
            parsed = parse_span(raw)
            if parsed is not None:
                spans.append(parsed)
    return spans


# --- classification + field extraction --------------------------------------------------------


def _first(attrs: JsonObject, keys: tuple[str, ...]) -> JsonValue:
    for key in keys:
        value = attrs.get(key)
        if value is not None:
            return value
    return None


def _operation(span: SpanRecord) -> str:
    op = span.attributes.get("gen_ai.operation.name")
    return op if isinstance(op, str) else ""


def _oi_kind(span: SpanRecord) -> str:
    kind = span.attributes.get("openinference.span.kind")
    return kind if isinstance(kind, str) else ""


def is_tool_span(span: SpanRecord) -> bool:
    op = _operation(span)
    if op in _TOOL_OPS:
        return True
    if op in _LLM_OPS:
        return False
    kind = _oi_kind(span)
    if kind in _OI_TOOL_KINDS:
        return True
    if kind in _OI_LLM_KINDS:
        return False
    return span.name.startswith("execute_tool")


def is_llm_span(span: SpanRecord) -> bool:
    op = _operation(span)
    if op in _LLM_OPS:
        return True
    if op in _TOOL_OPS:
        return False
    kind = _oi_kind(span)
    if kind in _OI_LLM_KINDS:
        return True
    if kind in _OI_TOOL_KINDS:
        return False
    return any(span.attributes.get(key) is not None for key in _LLM_PRESENCE_KEYS)


def _coerce_args(raw: JsonValue) -> JsonObject:
    """Coerce a tool call's arguments (a dict, a JSON string, or a scalar) to an arguments dict."""
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        if not raw.strip():
            return {}  # empty/blank serialized args (e.g. a no-arg tool call) -> no arguments
        try:
            parsed: JsonValue = json.loads(raw)
        except json.JSONDecodeError:
            return {"value": raw}
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    return {"value": raw}


def _tool_args(attrs: JsonObject) -> JsonObject:
    return _coerce_args(_first(attrs, _TOOL_ARG_KEYS))


# OpenInference flattens an LLM-emitted tool call across INDEXED attribute keys, e.g.
#   llm.output_messages.0.message.tool_calls.0.tool_call.function.name      = "get_user"
#   llm.output_messages.0.message.tool_calls.0.tool_call.function.arguments = '{"id": "u1"}'
# so the tool call lives on the LLM span itself, not a static `tool.name` key.
_OI_TOOL_NAME_SUFFIX = ".tool_call.function.name"
_OI_TOOL_ARGS_SUFFIX = ".tool_call.function.arguments"

# Arize's `export_model_to_df` flattens the SAME tool call differently: `llm.output_messages` stays
# a nested list whose inner keys are dotted, so NO top-level key ends in the suffixes above, e.g.
#   llm.output_messages = [{"message.role": "assistant",
#                           "message.tool_calls": [{"tool_call.function.name": "read_file",
#                                                   "tool_call.function.arguments": "{...}"}]}]
# Every level is shape-checked rather than assumed: the dataframe path fills an absent column with
# a float NaN, and a raw OTLP export carries the same field as a JSON string. Neither is a list, so
# both fall through to the caller's existing behavior instead of being guessed at.
_OI_MESSAGES_KEY = "llm.output_messages"
_OI_TOOL_CALLS_KEY = "message.tool_calls"
_OI_CALL_NAME_KEY = "tool_call.function.name"
_OI_CALL_ARGS_KEY = "tool_call.function.arguments"


def _nested_openinference_tool_call(attrs: JsonObject) -> tuple[str, JsonValue] | None:
    """The first tool call inside a nested `llm.output_messages` list, if any."""
    messages = attrs.get(_OI_MESSAGES_KEY)
    if not isinstance(messages, list):
        return None
    for message in messages:
        if not isinstance(message, dict):
            continue
        calls = message.get(_OI_TOOL_CALLS_KEY)
        if not isinstance(calls, list):
            continue
        for call in calls:
            if not isinstance(call, dict):
                continue
            name = call.get(_OI_CALL_NAME_KEY)
            if isinstance(name, str) and name:
                return name, call.get(_OI_CALL_ARGS_KEY)
    return None


def _openinference_tool_call(attrs: JsonObject) -> tuple[str, JsonValue] | None:
    """The first OpenInference tool call `(name, raw_args)` on an LLM span, if any.

    Reads either flattening of the same span: Phoenix's indexed top-level keys first, then Arize's
    nested `llm.output_messages` list. A turn with parallel calls yields only its first here, the
    one paired with this span; the rest arrive as their own steps from their tool spans.
    """
    name_keys = sorted(k for k in attrs if k.endswith(_OI_TOOL_NAME_SUFFIX))
    for name_key in name_keys:
        name = attrs.get(name_key)
        if isinstance(name, str) and name:
            args_key = name_key[: -len(_OI_TOOL_NAME_SUFFIX)] + _OI_TOOL_ARGS_SUFFIX
            return name, attrs.get(args_key)
    return _nested_openinference_tool_call(attrs)


def action_from_llm_span(span: SpanRecord) -> Action:
    attrs = span.attributes
    tool_name = _first(attrs, _TOOL_NAME_KEYS)
    if isinstance(tool_name, str) and tool_name:
        return Action(kind=ActionKind.TOOL_CALL, name=tool_name, arguments=_tool_args(attrs))
    oi_call = _openinference_tool_call(attrs)
    if oi_call is not None:
        name, raw_args = oi_call
        return Action(kind=ActionKind.TOOL_CALL, name=name, arguments=_coerce_args(raw_args))
    completion = _first(attrs, _COMPLETION_KEYS)
    content = _first(attrs, _PROMPT_KEYS) if completion is None else completion
    return Action(kind=ActionKind.MESSAGE, content=as_text(content))


def tool_call_action_from_tool_span(span: SpanRecord) -> Action:
    name = _first(span.attributes, _TOOL_NAME_KEYS)
    return Action(
        kind=ActionKind.TOOL_CALL,
        name=name if isinstance(name, str) and name else None,
        arguments=_tool_args(span.attributes),
    )


def observation_from_tool_span(span: SpanRecord) -> Observation:
    content = as_text(_first(span.attributes, _TOOL_OUTPUT_KEYS))
    return Observation(content=content, is_error=span.status_error)


def _trace_task(spans: list[SpanRecord]) -> str | None:
    for span in spans:
        prompt = _first(span.attributes, _PROMPT_KEYS)
        if prompt is not None:
            return as_text(prompt)
    return None


def _state_before(span: SpanRecord) -> EnvState:
    """Read an optional `wmo.state.*` snapshot off a span (empty when absent)."""
    attrs = span.attributes
    structured = attrs.get(_STATE_STRUCTURED_KEY)
    if isinstance(structured, str):
        try:
            decoded: JsonValue = json.loads(structured)
        except json.JSONDecodeError:
            decoded = {}
        structured = decoded
    scratchpad = attrs.get(_STATE_SCRATCHPAD_KEY)
    return EnvState(
        structured=structured if isinstance(structured, dict) else {},
        scratchpad=scratchpad if isinstance(scratchpad, str) else "",
    )


def _trace_metadata(spans: list[SpanRecord]) -> JsonObject:
    """First `wmo.trace.metadata` object across a trace's spans (legacy `wmh.` key accepted)."""
    for span in spans:
        raw = span.attributes.get(_TRACE_METADATA_KEY)
        if raw is None:
            raw = span.attributes.get(_LEGACY_TRACE_METADATA_KEY)
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str):
            try:
                decoded: JsonValue = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(decoded, dict):
                return decoded
    return {}


def _opt_int(value: JsonValue) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float, str)):
        coerced = to_int(value)
        return coerced if coerced or value in (0, "0", 0.0) else None
    return None


def _opt_float(value: JsonValue) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _opt_str(value: JsonValue) -> str | None:
    return value if isinstance(value, str) and value else None


# `start_nano` scale sniffing for latency: adapters put three kinds of value in start/end:
# real OTLP epoch NANOseconds (~2e18 today), `iso_to_ordinal` epoch MICROseconds (~2e15 today),
# and synthetic ordinals (list indexes, the writer's i*10 stamps: tiny). Only the first two are
# real clocks a latency can be derived from; synthetic ordinals must not masquerade as durations.
_EPOCH_NANO_FLOOR = 10**17  # ≥ 1973 when read as ns
_EPOCH_MICRO_FLOOR = 10**14  # ≥ 1973 when read as µs


def _span_latency_ms(start: int, end: int) -> float | None:
    if start >= _EPOCH_NANO_FLOOR:
        return (end - start) / 1_000_000
    if start >= _EPOCH_MICRO_FLOOR:
        return (end - start) / 1_000
    return None


def _attribution(
    action_span: SpanRecord | None, tool_span: SpanRecord | None
) -> StepAttribution | None:
    """Best-effort per-step attribution from the action span's attributes.

    An explicit `wmo.attribution` JSON attribute (what `otel_writer` emits, and what a capture
    system can stamp directly) round-trips verbatim. Otherwise the fields are derived from the
    GenAI/OpenInference vocabularies plus span timing/status. Returns None when nothing is known,
    so sources without attribution stay clean rather than carrying an all-empty object.
    """
    attrs = action_span.attributes if action_span is not None else {}
    explicit = attrs.get(_ATTRIBUTION_KEY)
    if isinstance(explicit, str) and explicit:
        try:
            return StepAttribution.model_validate_json(explicit)
        except ValueError:
            pass
    error_class: ErrorClass | None = None
    if action_span is not None and action_span.status_error:
        error_class = ErrorClass.CONTROLLABLE
    elif tool_span is not None and tool_span.status_error:
        error_class = ErrorClass.ENVIRONMENTAL
    latency_ms: float | None = None
    if action_span is not None and 0 < action_span.start_nano < action_span.end_nano:
        latency_ms = _span_latency_ms(action_span.start_nano, action_span.end_nano)
    config = {
        key[len(_REQUEST_PREFIX) :]: value
        for key, value in attrs.items()
        if key.startswith(_REQUEST_PREFIX) and key not in _NON_CONFIG_REQUEST_KEYS
    }
    attribution = StepAttribution(
        model=_opt_str(_first(attrs, _MODEL_KEYS)),
        provider=_opt_str(_first(attrs, _PROVIDER_KEYS)),
        config=config,
        input_tokens=_opt_int(_first(attrs, _INPUT_TOKEN_KEYS)),
        output_tokens=_opt_int(_first(attrs, _OUTPUT_TOKEN_KEYS)),
        cost_usd=_opt_float(_first(attrs, _COST_KEYS)),
        latency_ms=latency_ms,
        error_class=error_class,
    )
    # exclude_defaults so an empty config dict does not make an otherwise-unknown step
    # carry an all-empty attribution object.
    return attribution if attribution.model_dump(exclude_defaults=True) else None


def _build_steps(spans: list[SpanRecord]) -> list[Step]:
    """Pair ordered Action spans with their following Observation spans into Steps."""
    task = _trace_task(spans)
    steps: list[Step] = []
    pending: Action | None = None
    pending_span: SpanRecord | None = None
    pending_ids: list[str] = []
    pending_state = EnvState()

    def flush(
        action: Action,
        observation: Observation,
        span_ids: list[str],
        state: EnvState,
        action_span: SpanRecord | None,
        tool_span: SpanRecord | None,
    ) -> None:
        steps.append(
            Step(
                action=action,
                observation=observation,
                state_before=state,
                task=task,
                raw_span_ids=span_ids,
                attribution=_attribution(action_span, tool_span),
            )
        )

    for span in spans:
        if is_tool_span(span):
            observation = observation_from_tool_span(span)
            if pending is None:
                action = tool_call_action_from_tool_span(span)
                flush(action, observation, [span.span_id], _state_before(span), None, span)
            else:
                # The LLM span usually carries the call's name/args; backfill from the tool span
                # only when it didn't. Derive the tool-span action once to avoid re-parsing.
                if pending.kind == ActionKind.TOOL_CALL and (
                    not pending.arguments or pending.name is None
                ):
                    from_tool = tool_call_action_from_tool_span(span)
                    if not pending.arguments:
                        pending.arguments = from_tool.arguments
                    if pending.name is None:
                        pending.name = from_tool.name
                flush(
                    pending,
                    observation,
                    [*pending_ids, span.span_id],
                    pending_state,
                    pending_span,
                    span,
                )
            pending, pending_span, pending_ids, pending_state = None, None, [], EnvState()
        elif is_llm_span(span):
            if pending is not None:
                flush(
                    pending, Observation(content=""), pending_ids, pending_state, pending_span, None
                )
            pending, pending_span, pending_ids = action_from_llm_span(span), span, [span.span_id]
            pending_state = _state_before(span)
        # Non-agent spans are ignored.

    if pending is not None:
        flush(pending, Observation(content=""), pending_ids, pending_state, pending_span, None)
    return steps


def group_spans(spans: list[SpanRecord]) -> list[list[SpanRecord]]:
    """Group spans by trace id and order: spans within a group by start time, groups likewise.

    Splitting this from `trace_from_group` lets the streaming ingest count traces up front
    (the `detected` event) and normalize group-by-group (the `progress` events).
    """
    by_trace: dict[str, list[SpanRecord]] = {}
    for span in spans:
        by_trace.setdefault(span.trace_id, []).append(span)
    # Sorting each group by (start_nano, span_id) leaves group[0] as that trace's earliest span, so
    # we reuse it as the inter-trace sort key rather than re-scanning.
    groups = list(by_trace.values())
    for group in groups:
        group.sort(key=lambda s: (s.start_nano, s.span_id))
    groups.sort(key=lambda group: group[0].start_nano)
    return groups


def trace_from_group(group: list[SpanRecord], *, source: str) -> Trace:
    """Build one `Trace` from one already-ordered same-trace-id span group."""
    return Trace(
        trace_id=group[0].trace_id,
        steps=_build_steps(group),
        source=source,
        metadata=_trace_metadata(group),
    )


def spans_to_traces(spans: list[SpanRecord], *, source: str) -> list[Trace]:
    """Group spans by trace id, order each group by start time, and build `Trace`s.

    This is the shared tail every span-based adapter calls after producing `SpanRecord`s.
    """
    return [trace_from_group(group, source=source) for group in group_spans(spans)]
