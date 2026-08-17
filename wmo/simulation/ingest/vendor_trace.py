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

from wmo.common.core.artifacts import (
    FailureCode,
    JsonObject,
    SourceIdentity,
    StructuredFailure,
)
from wmo.common.core.text import normalize_durable_text
from wmo.common.models import ModelSnapshot, Usage
from wmo.common.traces import Trace, TraceSource, TraceSpan
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
from wmo.simulation.ingest.trace_extensions import (
    CONVERSATION_ID_KEYS,
    OUTCOME_FAILURE_CODE_KEY,
    OUTCOME_FAILURE_MESSAGE_KEY,
    OUTCOME_FAILURE_RETRYABLE_KEY,
    OUTCOME_NAME_KEY,
    OUTCOME_STATUS_KEY,
    REQUEST_CONTEXT_KEY,
    REQUEST_TOOLS_KEY,
    collect_outcome,
    collect_tools,
    consistent_json_object,
    consistent_text,
)
from wmo.simulation.ingest.vendor_observations import (
    VendorObservation,
)
from wmo.simulation.ingest.vendor_records import (
    VendorTraceFormatError,
    required_text,
    vendor_w3c_id,
)

VENDOR_ATTRIBUTE = "wmo.source.vendor"
SOURCE_TRACE_ATTRIBUTE = "wmo.source.trace.id"
SOURCE_SPAN_ATTRIBUTE = "wmo.source.span.id"
SOURCE_STARTED_ATTRIBUTE = "wmo.source.span.started_at"
SOURCE_ENDED_ATTRIBUTE = "wmo.source.span.ended_at"
SYNTHETIC_TIME_ATTRIBUTE = "wmo.source.time.synthetic"

APPROVED_EXTENSION_KEYS = (
    "wmo.customer.id",
    "wmo.conversation.id",
    REQUEST_CONTEXT_KEY,
    "wmo.request.tags",
    REQUEST_TOOLS_KEY,
    "wmo.trace.metadata",
    CAPABILITIES_DIGEST_ATTRIBUTE,
    CONNECTION_DIGEST_ATTRIBUTE,
    "wmo.outcome.escalated",
    OUTCOME_STATUS_KEY,
    OUTCOME_NAME_KEY,
    OUTCOME_FAILURE_CODE_KEY,
    OUTCOME_FAILURE_MESSAGE_KEY,
    OUTCOME_FAILURE_RETRYABLE_KEY,
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
    strict_tool_pairing: bool = False,
) -> TraceNormalizationResult:
    """Convert declared vendor observations into canonical traces and explicit exclusions.

    Args:
        observations: Declared vendor observations in source order.
        vendor: Vendor label retained on every canonical span.
        source: Immutable identity of the source bytes or transport result.
        semantic_convention_version: Pinned GenAI semantic-convention version for the traces.
        initial_issues: Parse exclusions collected before observation mapping.
        strict_tool_pairing: Whether an unpaired tool call or explicit tool result excludes
            its trace instead of keeping the span without an asserted pairing.

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
                    strict_tool_pairing=strict_tool_pairing,
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
    strict_tool_pairing: bool,
) -> Trace:
    """Build one canonical trace from the observations sharing a vendor trace key.

    Args:
        source_trace_id: Vendor trace key.
        observations: Observations belonging to this trace.
        vendor: Vendor label retained on every span.
        source: Immutable source identity.
        semantic_convention_version: Pinned GenAI semantic-convention version.
        strict_tool_pairing: Whether an unpaired tool call or explicit tool result rejects
            the trace.

    Returns:
        One canonical trace with ordered spans and resolved parents.

    Raises:
        VendorTraceFormatError: The trace has no request text, no convertible observation, or
            an unpaired tool call under strict pairing.
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
                    strict=strict_tool_pairing,
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
    if strict_tool_pairing and any(pending.values()):
        unmatched_calls = ", ".join(
            f"{name}:{call.call_id}" for name in sorted(pending) for call in pending[name]
        )
        raise VendorTraceFormatError(f"unmatched generated {vendor} tool calls: {unmatched_calls}")
    unmatched = {
        call.call_id for calls in pending.values() for call in calls
    }  # Unpaired calls keep their span without asserting a pairing.
    attributes_by_span = tuple(emission.span.attributes for emission in emissions)
    return Trace(
        trace_id=trace_id,
        conversation_id=consistent_text(
            attributes_by_span, CONVERSATION_ID_KEYS, error_type=VendorTraceFormatError
        ),
        task=task,
        initial_context=consistent_json_object(
            attributes_by_span, REQUEST_CONTEXT_KEY, error_type=VendorTraceFormatError
        ),
        tools=collect_tools(
            attributes_by_span, keys=(REQUEST_TOOLS_KEY,), error_type=VendorTraceFormatError
        ),
        spans=_resolve_spans(emissions, unmatched),
        outcome=collect_outcome(
            attributes_by_span, failures=failures, error_type=VendorTraceFormatError
        ),
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
    strict: bool,
) -> _SpanEmission:
    """Map one tool-result observation and pair it with an earlier model tool call.

    Args:
        observation: Declared tool-result observation.
        trace_id: Canonical trace identity.
        vendor: Vendor label retained on every span.
        pending: Tool-call queues by tool name, consumed on an exact match.
        strict: Whether an unmatched explicit call identity rejects the trace.

    Returns:
        One canonical tool-result span emission.

    Raises:
        VendorTraceFormatError: The observation declares no tool name, or its explicit call
            identity matches no earlier tool call under strict pairing.
    """
    tool_name = required_text(observation.tool_name, f"{vendor} tool result name")
    attributes = _base_attributes(observation, vendor=vendor)
    attributes["gen_ai.operation.name"] = "execute_tool"
    attributes["gen_ai.tool.name"] = tool_name
    attributes["gen_ai.tool.call.arguments"] = observation.tool_arguments or ""
    attributes["gen_ai.tool.message"] = observation.tool_message or ""
    matched = _match_pending_call(pending[tool_name], observation.tool_call_id)
    if strict and matched is None and observation.tool_call_id is not None:
        raise VendorTraceFormatError(
            f"unmatched explicit {vendor} tool result: {tool_name}:{observation.tool_call_id}"
        )
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
