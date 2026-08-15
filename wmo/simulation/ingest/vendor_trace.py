"""Convert declared vendor observations into canonical trace and span contracts.

Vendor exports describe an agent run as ordered observations: a model call that may request tools,
a tool result, or an agent-level record. A vendor module reads its own export shape and declares
those observations with :class:`VendorObservation`; this module owns the single canonical
conversion shared by every vendor source.

The conversion is deliberately narrow:

- Identity is derived, never fabricated. Opaque vendor keys map to deterministic W3C-shaped IDs,
  and every canonical span retains its exact source trace and span keys as attributes.
- Model identity is retained only when the export declares both a provider and a model.
- Canonical spans record each observation at its source start instant so ordering and tool pairing
  follow the export's own causality. The full source interval is retained as span attributes.
- A tool result pairs with an earlier model tool call by explicit vendor call ID, otherwise by tool
  name in call order. An unpaired call keeps its span but asserts no pairing.
- Invalid records and traces are excluded with an explicit issue instead of being repaired.
"""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from pydantic import JsonValue

from wmo.common.core.artifacts import (
    FailureCode,
    JsonObject,
    SourceIdentity,
    StructuredFailure,
)
from wmo.common.core.text import normalize_durable_text
from wmo.common.models import ModelSnapshot, Usage
from wmo.common.tasks import ToolSchema
from wmo.common.traces import Trace, TraceOutcome, TraceSource, TraceSpan
from wmo.simulation.ingest.model_identity import (
    CAPABILITIES_DIGEST_ATTRIBUTE,
    CONNECTION_DIGEST_ATTRIBUTE,
    normalized_capabilities_sha256,
    normalized_connection_sha256,
    normalized_model_identity_evidence,
)
from wmo.simulation.ingest.otlp import (
    GENAI_SEMANTIC_CONVENTION_VERSION,
    TraceNormalizationIssue,
    TraceNormalizationResult,
)
from wmo.simulation.ingest.vendor_observations import (
    VendorObservation,
)
from wmo.simulation.ingest.vendor_records import (
    VendorTraceFormatError,
    json_value,
    required_text,
    vendor_w3c_id,
)

VENDOR_ATTRIBUTE = "wmo.source.vendor"
SOURCE_TRACE_ATTRIBUTE = "wmo.source.trace.id"
SOURCE_SPAN_ATTRIBUTE = "wmo.source.span.id"
SOURCE_STARTED_ATTRIBUTE = "wmo.source.span.started_at"
SOURCE_ENDED_ATTRIBUTE = "wmo.source.span.ended_at"
SYNTHETIC_TIME_ATTRIBUTE = "wmo.source.time.synthetic"

_OUTCOME_STATUS_KEY = "wmo.outcome.status"
_NONFAILURE_STATUSES: dict[str, Literal["success", "abandoned", "unknown"]] = {
    "success": "success",
    "abandoned": "abandoned",
    "unknown": "unknown",
}
_OUTCOME_NAME_KEY = "wmo.outcome.name"
_OUTCOME_FAILURE_CODE_KEY = "wmo.outcome.failure.code"
_OUTCOME_FAILURE_MESSAGE_KEY = "wmo.outcome.failure.message"
_OUTCOME_FAILURE_RETRYABLE_KEY = "wmo.outcome.failure.retryable"
_TOOLS_KEY = "wmo.request.tools"
_CONTEXT_KEY = "wmo.request.context"
_CONVERSATION_KEYS = ("wmo.conversation.id", "gen_ai.conversation.id")

APPROVED_EXTENSION_KEYS = (
    "wmo.customer.id",
    "wmo.conversation.id",
    _CONTEXT_KEY,
    "wmo.request.tags",
    _TOOLS_KEY,
    "wmo.trace.metadata",
    CAPABILITIES_DIGEST_ATTRIBUTE,
    CONNECTION_DIGEST_ATTRIBUTE,
    _OUTCOME_STATUS_KEY,
    _OUTCOME_NAME_KEY,
    _OUTCOME_FAILURE_CODE_KEY,
    _OUTCOME_FAILURE_MESSAGE_KEY,
    _OUTCOME_FAILURE_RETRYABLE_KEY,
)


@dataclass(frozen=True)
class _PendingToolCall:
    """One emitted model tool call awaiting a matching vendor tool result.

    Args:
        call_id: Canonical call identity stamped on the model-call span.
        span_id: Canonical span ID of the model call.
    """

    call_id: str
    span_id: str


@dataclass(frozen=True)
class _SpanEmission:
    """One canonical span before parent and pairing relationships resolve.

    Args:
        span: Canonical span with attributes already mapped.
        source_span_id: Vendor span key that produced this span.
        source_parent_span_id: Vendor parent span key declared by the source record.
        paired_call_id: Canonical tool-call identity carried by this span, when any.
        paired_parent_span_id: Canonical model-call span paired with a tool result.
    """

    span: TraceSpan
    source_span_id: str
    source_parent_span_id: str | None
    paired_call_id: str | None = None
    paired_parent_span_id: str | None = None


def approved_extensions(record: JsonObject) -> JsonObject:
    """Copy only approved WMO extension attributes from one source mapping.

    Args:
        record: Source record, property bag, or attribute mapping.

    Returns:
        Approved extension attributes in canonical attribute form.
    """
    return {key: record[key] for key in APPROVED_EXTENSION_KEYS if key in record}


def build_vendor_traces(
    observations: Sequence[VendorObservation],
    *,
    vendor: str,
    source: SourceIdentity,
    semantic_convention_version: str = GENAI_SEMANTIC_CONVENTION_VERSION,
    initial_issues: Sequence[TraceNormalizationIssue] = (),
) -> TraceNormalizationResult:
    """Convert declared vendor observations into canonical traces and explicit exclusions.

    Args:
        observations: Declared vendor observations in source order.
        vendor: Vendor label retained on every canonical span.
        source: Immutable identity of the source bytes or transport result.
        semantic_convention_version: Pinned GenAI semantic-convention version for the traces.
        initial_issues: Parse exclusions collected before observation mapping.

    Returns:
        Valid canonical traces and every retained validation exclusion.

    Raises:
        VendorTraceFormatError: The semantic convention version is blank.
    """
    if not semantic_convention_version.strip():
        raise VendorTraceFormatError("semantic convention version must not be blank")
    issues = list(initial_issues)
    by_trace: dict[str, list[VendorObservation]] = defaultdict(list)
    for observation in observations:
        by_trace[observation.source_trace_id].append(observation)
    traces: list[Trace] = []
    for source_trace_id in sorted(by_trace):
        try:
            traces.append(
                _build_trace(
                    source_trace_id,
                    by_trace[source_trace_id],
                    vendor=vendor,
                    source=source,
                    semantic_convention_version=semantic_convention_version,
                )
            )
        except (VendorTraceFormatError, ValueError) as exc:
            issues.append(TraceNormalizationIssue(f"trace-{source_trace_id}", str(exc)))
    traces.sort(key=lambda trace: (trace.spans[0].started_at, trace.trace_id))
    normalized = tuple(traces)
    return TraceNormalizationResult(
        traces=normalized,
        issues=tuple(issues),
        identity_evidence=normalized_model_identity_evidence(normalized),
    )


def _build_trace(
    source_trace_id: str,
    observations: Sequence[VendorObservation],
    *,
    vendor: str,
    source: SourceIdentity,
    semantic_convention_version: str,
) -> Trace:
    """Build one canonical trace from the observations sharing a vendor trace key.

    Args:
        source_trace_id: Vendor trace key.
        observations: Observations belonging to this trace.
        vendor: Vendor label retained on every span.
        source: Immutable source identity.
        semantic_convention_version: Pinned GenAI semantic-convention version.

    Returns:
        One canonical trace with ordered spans and resolved parents.

    Raises:
        VendorTraceFormatError: The trace has no request text or no convertible observation.
    """
    ordered = sorted(observations, key=lambda item: (item.started_at, item.ordinal))
    task = next(
        (item.request_text for item in ordered if item.request_text),
        None,
    )
    if task is None:
        raise VendorTraceFormatError(f"{vendor} trace has no user request text")
    trace_id = vendor_w3c_id(source_trace_id, vendor=vendor, kind="trace", namespace="trace")
    emissions: list[_SpanEmission] = []
    pending: dict[str, deque[_PendingToolCall]] = defaultdict(deque)
    failures: list[StructuredFailure] = []
    for observation in ordered:
        if observation.kind == "model":
            emitted = _model_spans(
                observation,
                trace_id=trace_id,
                vendor=vendor,
                task=task,
                pending=pending,
            )
        elif observation.kind == "tool_result":
            emitted = (
                _tool_result_span(
                    observation,
                    trace_id=trace_id,
                    vendor=vendor,
                    pending=pending,
                ),
            )
        else:
            emitted = (_agent_span(observation, trace_id=trace_id, vendor=vendor, task=task),)
        emissions.extend(emitted)
        failures.extend(
            emission.span.failure for emission in emitted if emission.span.failure is not None
        )
    if not emissions:
        raise VendorTraceFormatError(f"{vendor} trace has no convertible observations")
    unmatched = {
        call.call_id for calls in pending.values() for call in calls
    }  # Unpaired calls keep their span without asserting a pairing.
    attributes_by_span = tuple(emission.span.attributes for emission in emissions)
    return Trace(
        trace_id=trace_id,
        conversation_id=_consistent_text(attributes_by_span, _CONVERSATION_KEYS),
        task=task,
        initial_context=_initial_context(attributes_by_span),
        tools=_collect_tools(attributes_by_span),
        spans=_resolve_spans(emissions, unmatched),
        outcome=_collect_outcome(attributes_by_span, failures),
        source=TraceSource(
            identity=source,
            semantic_convention_version=semantic_convention_version,
        ),
    )


def _model_spans(
    observation: VendorObservation,
    *,
    trace_id: str,
    vendor: str,
    task: str,
    pending: dict[str, deque[_PendingToolCall]],
) -> tuple[_SpanEmission, ...]:
    """Map one model observation to one span per requested tool call, or one completion span.

    Args:
        observation: Declared model observation.
        trace_id: Canonical trace identity.
        vendor: Vendor label retained on every span.
        task: Canonical trace request text.
        pending: Tool-call queues by tool name, extended with each emitted call.

    Returns:
        Canonical span emissions for this observation.
    """
    base = _base_attributes(observation, vendor=vendor)
    base["gen_ai.operation.name"] = "chat"
    base["gen_ai.prompt"] = task
    if observation.input_messages is not None:
        base["gen_ai.input.messages"] = observation.input_messages
    model = _model_snapshot(observation)
    usage = _usage(observation)
    if not observation.tool_calls:
        attributes = dict(base)
        attributes["gen_ai.completion"] = observation.completion_text or ""
        return (
            _emission(
                observation,
                trace_id=trace_id,
                vendor=vendor,
                name="agent.model_call",
                attributes=attributes,
                suffix="completion",
                model=model,
                usage=usage,
            ),
        )
    emissions: list[_SpanEmission] = []
    for index, tool_call in enumerate(observation.tool_calls):
        call_id = tool_call.call_id or (
            f"{vendor}-call-{observation.source_span_id}-{index}"  # Deterministic per source span.
        )
        attributes = dict(base)
        attributes["gen_ai.tool.name"] = tool_call.name
        attributes["gen_ai.tool.call.arguments"] = tool_call.arguments
        attributes["gen_ai.tool.call.id"] = call_id
        if observation.completion_text:
            attributes["gen_ai.completion"] = observation.completion_text
        emission = _emission(
            observation,
            trace_id=trace_id,
            vendor=vendor,
            name="agent.model_call",
            attributes=attributes,
            suffix=f"tool-{index}",
            model=model,
            usage=usage,
            paired_call_id=call_id,
        )
        emissions.append(emission)
        pending[tool_call.name].append(
            _PendingToolCall(call_id=call_id, span_id=emission.span.span_id)
        )
    return tuple(emissions)


def _tool_result_span(
    observation: VendorObservation,
    *,
    trace_id: str,
    vendor: str,
    pending: dict[str, deque[_PendingToolCall]],
) -> _SpanEmission:
    """Map one tool-result observation and pair it with an earlier model tool call.

    Args:
        observation: Declared tool-result observation.
        trace_id: Canonical trace identity.
        vendor: Vendor label retained on every span.
        pending: Tool-call queues by tool name, consumed on an exact match.

    Returns:
        One canonical tool-result span emission.

    Raises:
        VendorTraceFormatError: The observation declares no tool name.
    """
    tool_name = required_text(observation.tool_name, f"{vendor} tool result name")
    attributes = _base_attributes(observation, vendor=vendor)
    attributes["gen_ai.operation.name"] = "execute_tool"
    attributes["gen_ai.tool.name"] = tool_name
    attributes["gen_ai.tool.call.arguments"] = observation.tool_arguments or ""
    attributes["gen_ai.tool.message"] = observation.tool_message or ""
    matched = _match_pending_call(pending[tool_name], observation.tool_call_id)
    if matched is not None:
        attributes["gen_ai.tool.call.id"] = matched.call_id
    return _emission(
        observation,
        trace_id=trace_id,
        vendor=vendor,
        name="agent.tool_call",
        attributes=attributes,
        suffix="tool-result",
        model=None,
        usage=None,
        paired_call_id=None if matched is None else matched.call_id,
        paired_parent_span_id=None if matched is None else matched.span_id,
    )


def _agent_span(
    observation: VendorObservation,
    *,
    trace_id: str,
    vendor: str,
    task: str,
) -> _SpanEmission:
    """Retain one agent-level vendor record as canonical trace evidence.

    Args:
        observation: Declared agent-level observation.
        trace_id: Canonical trace identity.
        vendor: Vendor label retained on every span.
        task: Canonical trace request text.

    Returns:
        One canonical agent-level span emission.
    """
    attributes = _base_attributes(observation, vendor=vendor)
    attributes["gen_ai.operation.name"] = "invoke_agent"
    attributes["gen_ai.prompt"] = task
    if observation.completion_text:
        attributes["gen_ai.completion"] = observation.completion_text
    return _emission(
        observation,
        trace_id=trace_id,
        vendor=vendor,
        name="agent.trace",
        attributes=attributes,
        suffix="agent",
        model=None,
        usage=None,
    )


def _base_attributes(observation: VendorObservation, *, vendor: str) -> JsonObject:
    """Build the provenance attributes every canonical vendor span carries.

    Args:
        observation: Declared vendor observation.
        vendor: Vendor label retained on every span.

    Returns:
        Approved extensions plus exact source identity and interval provenance.
    """
    attributes: JsonObject = dict(observation.extensions)
    attributes.update(observation.declared_attributes)
    attributes[VENDOR_ATTRIBUTE] = vendor
    attributes[SOURCE_TRACE_ATTRIBUTE] = observation.source_trace_id
    attributes[SOURCE_SPAN_ATTRIBUTE] = observation.source_span_id
    if observation.synthetic_time:
        attributes[SYNTHETIC_TIME_ATTRIBUTE] = True
    else:
        attributes[SOURCE_STARTED_ATTRIBUTE] = observation.started_at.isoformat()
        attributes[SOURCE_ENDED_ATTRIBUTE] = observation.ended_at.isoformat()
    return attributes


def _emission(
    observation: VendorObservation,
    *,
    trace_id: str,
    vendor: str,
    name: str,
    attributes: JsonObject,
    suffix: str,
    model: ModelSnapshot | None,
    usage: Usage | None,
    paired_call_id: str | None = None,
    paired_parent_span_id: str | None = None,
) -> _SpanEmission:
    """Build one canonical span recorded at the observation's source start instant.

    Args:
        observation: Declared vendor observation.
        trace_id: Canonical trace identity.
        vendor: Vendor label included in deterministic span identity.
        name: Canonical span name.
        attributes: Canonical span attributes.
        suffix: Stable role suffix distinguishing spans from one source record.
        model: Retained model identity, when the export declares one.
        usage: Retained token accounting, when the export declares complete counts.
        paired_call_id: Canonical tool-call identity carried by this span.
        paired_parent_span_id: Canonical model-call span paired with a tool result.

    Returns:
        One canonical span emission with unresolved parent links.
    """
    span_id = vendor_w3c_id(
        f"{trace_id}\0{observation.source_span_id}\0{suffix}",
        vendor=vendor,
        kind="span",
        namespace="span",
    )
    failure = (
        None
        if observation.failure_message is None
        else StructuredFailure(
            code=FailureCode.INTERNAL,
            message=normalize_durable_text(observation.failure_message),
        )
    )
    return _SpanEmission(
        span=TraceSpan(
            span_id=span_id,
            parent_span_id=None,
            name=name,
            started_at=observation.started_at,
            ended_at=observation.started_at,
            attributes=attributes,
            model=model,
            usage=usage,
            failure=failure,
        ),
        source_span_id=observation.source_span_id,
        source_parent_span_id=observation.source_parent_span_id,
        paired_call_id=paired_call_id,
        paired_parent_span_id=paired_parent_span_id,
    )


def _match_pending_call(
    pending: deque[_PendingToolCall], call_id: str | None
) -> _PendingToolCall | None:
    """Consume the tool call this result pairs with, without accepting a mismatch.

    Args:
        pending: Outstanding calls for one tool name in emission order.
        call_id: Explicit vendor call identity declared by the result, when any.

    Returns:
        The matching pending call, or ``None`` when the export declares no pairing.
    """
    if call_id is not None:
        for candidate in pending:
            if candidate.call_id == call_id:
                pending.remove(candidate)
                return candidate
        return None
    if pending:
        return pending.popleft()
    return None


def _resolve_spans(
    emissions: Sequence[_SpanEmission],
    unmatched_call_ids: set[str],
) -> tuple[TraceSpan, ...]:
    """Resolve parent links and drop tool-call identities that never paired.

    Args:
        emissions: Canonical span emissions in source order.
        unmatched_call_ids: Canonical call identities with no matching tool result.

    Returns:
        Ordered canonical spans with resolved parents and asserted pairings only.
    """
    emitted_by_source: dict[str, list[str]] = defaultdict(list)
    emitted_ids = {emission.span.span_id for emission in emissions}
    for emission in emissions:
        emitted_by_source[emission.source_span_id].append(emission.span.span_id)
    resolved: list[TraceSpan] = []
    for emission in emissions:
        attributes = emission.span.attributes
        if emission.paired_call_id is not None and emission.paired_call_id in unmatched_call_ids:
            attributes = {
                key: value for key, value in attributes.items() if key != "gen_ai.tool.call.id"
            }
        resolved.append(
            emission.span.model_copy(
                update={
                    "parent_span_id": _parent_span_id(emission, emitted_by_source, emitted_ids),
                    "attributes": attributes,
                }
            )
        )
    return tuple(resolved)


def _parent_span_id(
    emission: _SpanEmission,
    emitted_by_source: dict[str, list[str]],
    emitted_ids: set[str],
) -> str | None:
    """Prefer one unambiguous source parent, then a paired model-call parent.

    Args:
        emission: One canonical span emission.
        emitted_by_source: Canonical span IDs emitted for each vendor span key.
        emitted_ids: Every canonical span ID emitted for this trace.

    Returns:
        Resolved canonical parent span ID, or ``None`` when none applies.
    """
    candidates = tuple(
        span_id
        for span_id in emitted_by_source.get(emission.source_parent_span_id or "", ())
        if span_id != emission.span.span_id
    )
    if len(candidates) == 1:
        return candidates[0]
    paired = emission.paired_parent_span_id
    if paired is not None and paired != emission.span.span_id and paired in emitted_ids:
        return paired
    return None


def _model_snapshot(observation: VendorObservation) -> ModelSnapshot | None:
    """Retain declared vendor model identity with explicit digest provenance.

    Args:
        observation: Declared model observation.

    Returns:
        Model snapshot when the export declares provider and model, otherwise ``None``.
    """
    identity = observation.model
    if identity is None:
        return None
    return ModelSnapshot(
        provider=identity.provider,
        model_id=identity.model_id,
        revision=identity.revision,
        capabilities_sha256=normalized_capabilities_sha256(
            observation.extensions,
            identity.provider,
            identity.model_id,
            identity.revision,
            error_type=VendorTraceFormatError,
        ),
        connection_sha256=normalized_connection_sha256(
            observation.extensions,
            identity.provider,
            error_type=VendorTraceFormatError,
        ),
    )


def _usage(observation: VendorObservation) -> Usage | None:
    """Retain declared complete token accounting for one model observation.

    Args:
        observation: Declared model observation.

    Returns:
        Canonical usage when the export declares complete counts, otherwise ``None``.
    """
    usage = observation.usage
    if usage is None:
        return None
    return Usage(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cached_input_tokens=usage.cached_input_tokens,
    )


def _initial_context(attributes_by_span: Sequence[JsonObject]) -> JsonObject:
    """Return one consistent declared request context for the trace.

    Args:
        attributes_by_span: Canonical span attributes in trace order.

    Returns:
        Declared request context, empty when the export declares none.

    Raises:
        VendorTraceFormatError: The context is not an object or differs across spans.
    """
    values: list[JsonObject] = []
    for attributes in attributes_by_span:
        value = json_value(attributes.get(_CONTEXT_KEY))
        if value is None:
            continue
        if not isinstance(value, dict):
            raise VendorTraceFormatError(f"{_CONTEXT_KEY} must be a JSON object")
        values.append(value)
    if not values:
        return {}
    if any(value != values[0] for value in values[1:]):
        raise VendorTraceFormatError(f"{_CONTEXT_KEY} differs across spans in one trace")
    return values[0]


def _consistent_text(attributes_by_span: Sequence[JsonObject], keys: Sequence[str]) -> str | None:
    """Return one repeated text extension across a trace or reject ambiguity.

    Args:
        attributes_by_span: Canonical span attributes in trace order.
        keys: Ordered candidate attribute keys.

    Returns:
        The single declared value, or ``None`` when no span declares one.

    Raises:
        VendorTraceFormatError: A declared value is blank or spans disagree.
    """
    values: list[str] = []
    for attributes in attributes_by_span:
        for key in keys:
            if key not in attributes:
                continue
            values.append(required_text(attributes[key], key))
            break
    if not values:
        return None
    if any(value != values[0] for value in values[1:]):
        raise VendorTraceFormatError(f"{keys[0]} differs across spans in one trace")
    return values[0]


def _collect_tools(attributes_by_span: Sequence[JsonObject]) -> tuple[ToolSchema, ...]:
    """Convert declared request tool definitions to canonical visible tools.

    Args:
        attributes_by_span: Canonical span attributes in trace order.

    Returns:
        Deterministically ordered tool schemas declared by the export.

    Raises:
        VendorTraceFormatError: A definition list, schema, or name is invalid or conflicting.
    """
    by_name: dict[str, ToolSchema] = {}
    for attributes in attributes_by_span:
        value = json_value(attributes.get(_TOOLS_KEY))
        if value is None:
            continue
        if not isinstance(value, list):
            raise VendorTraceFormatError(f"{_TOOLS_KEY} must be a JSON array")
        for raw_tool in value:
            tool = _tool_schema(raw_tool)
            if tool.name in by_name and by_name[tool.name] != tool:
                raise VendorTraceFormatError(f"tool {tool.name!r} has conflicting definitions")
            by_name[tool.name] = tool
    return tuple(by_name[name] for name in sorted(by_name))


def _tool_schema(raw_tool: JsonValue) -> ToolSchema:
    """Convert one declared function tool definition to the canonical tool contract.

    Args:
        raw_tool: One declared tool definition.

    Returns:
        Canonical visible tool schema.

    Raises:
        VendorTraceFormatError: The definition is not an object with a name and object schema.
    """
    if not isinstance(raw_tool, dict):
        raise VendorTraceFormatError(f"{_TOOLS_KEY} entries must be objects")
    candidate = raw_tool.get("function") if raw_tool.get("type") == "function" else raw_tool
    if not isinstance(candidate, dict):
        raise VendorTraceFormatError("function tool definitions need a function object")
    name = required_text(candidate.get("name"), "tool definition name")
    description = candidate.get("description")
    schema = candidate.get("input_schema", candidate.get("parameters", candidate.get("schema")))
    if not isinstance(schema, dict):
        raise VendorTraceFormatError(f"tool {name!r} needs an object input schema")
    return ToolSchema(
        name=name,
        description=(
            normalize_durable_text(description.strip())
            if isinstance(description, str) and description.strip()
            else "No description captured."
        ),
        input_schema=schema,
    )


def _collect_outcome(
    attributes_by_span: Sequence[JsonObject],
    failures: Sequence[StructuredFailure],
) -> TraceOutcome | None:
    """Map declared WMO outcome extensions or source errors to terminal trace evidence.

    Args:
        attributes_by_span: Canonical span attributes in trace order.
        failures: Structured span failures observed in this trace, in order.

    Returns:
        Canonical trace outcome, or ``None`` when the export declares none.

    Raises:
        VendorTraceFormatError: Outcome extensions are incomplete or contradictory.
    """
    status = _consistent_text(attributes_by_span, (_OUTCOME_STATUS_KEY,))
    outcome_name = _consistent_text(attributes_by_span, (_OUTCOME_NAME_KEY,))
    failure_code = _consistent_text(attributes_by_span, (_OUTCOME_FAILURE_CODE_KEY,))
    failure_message = _consistent_text(attributes_by_span, (_OUTCOME_FAILURE_MESSAGE_KEY,))
    retryable = _consistent_bool(attributes_by_span, _OUTCOME_FAILURE_RETRYABLE_KEY)
    if status is None:
        if failures:
            return TraceOutcome(status="failure", failure=failures[0])
        if any(
            value is not None for value in (outcome_name, failure_code, failure_message, retryable)
        ):
            raise VendorTraceFormatError(f"outcome details require {_OUTCOME_STATUS_KEY}")
        return None
    nonfailure_status = _NONFAILURE_STATUSES.get(status)
    if nonfailure_status is not None:
        if any(value is not None for value in (failure_code, failure_message, retryable)):
            raise VendorTraceFormatError("outcome failure details require failure status")
        return TraceOutcome(status=nonfailure_status, outcome_name=outcome_name)
    if status != "failure":
        raise VendorTraceFormatError(
            f"{_OUTCOME_STATUS_KEY} must be success, failure, abandoned, or unknown"
        )
    if failure_code is None or failure_message is None:
        if failures:
            return TraceOutcome(status="failure", outcome_name=outcome_name, failure=failures[0])
        raise VendorTraceFormatError("failure outcomes need a failure code and message")
    try:
        code = FailureCode(failure_code)
    except ValueError as exc:
        valid = ", ".join(item.value for item in FailureCode)
        raise VendorTraceFormatError(
            f"{_OUTCOME_FAILURE_CODE_KEY} must be one of: {valid}"
        ) from exc
    return TraceOutcome(
        status="failure",
        outcome_name=outcome_name,
        failure=StructuredFailure(code=code, message=failure_message, retryable=retryable or False),
    )


def _consistent_bool(attributes_by_span: Sequence[JsonObject], key: str) -> bool | None:
    """Return one repeated boolean extension or reject inconsistent span values.

    Args:
        attributes_by_span: Canonical span attributes in trace order.
        key: Extension attribute key.

    Returns:
        The single declared boolean, or ``None`` when no span declares one.

    Raises:
        VendorTraceFormatError: A declared value is not boolean or spans disagree.
    """
    values: list[bool] = []
    for attributes in attributes_by_span:
        value = attributes.get(key)
        if value is None:
            continue
        if not isinstance(value, bool):
            raise VendorTraceFormatError(f"{key} must be boolean")
        values.append(value)
    if not values:
        return None
    if any(value != values[0] for value in values[1:]):
        raise VendorTraceFormatError(f"{key} differs across spans in one trace")
    return values[0]
