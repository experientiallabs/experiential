"""Declared vendor observations and the message readers vendor sources share.

A vendor module reads its own export shape and declares what the export actually says with
:class:`VendorObservation`: when the record happened, whether it is a model call, a tool result, or
agent-level evidence, and which request, completion, tool, model, and usage facts it carries.
:mod:`exp.simulation.ingest.vendor_trace` owns the single canonical conversion of those
observations, so no vendor module builds canonical spans itself.

The readers here cover the assistant-output shapes that recur across vendors: the OpenAI
``tool_calls`` shape, the Vercel AI SDK ``toolCalls`` shape, Anthropic-style ``tool_use`` content
parts, and the common ``message``, ``response``, and ``choices`` nestings around them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

from pydantic import JsonValue

from exp.common.core.artifacts import JsonObject
from exp.simulation.ingest.vendor_records import (
    VendorTraceFormatError,
    first_text,
    json_text,
    message_text,
    required_text,
)

VendorObservationKind = Literal["model", "tool_result", "agent"]

_TOOL_CALL_KEYS = ("tool_calls", "toolCalls")
_NESTING_KEYS = (
    "message",
    "response",
    "output",
    "data",
    "generations",
    "kwargs",
    "additional_kwargs",
)
_NAME_KEYS = ("name", "toolName", "tool_name")
_ARGUMENT_KEYS = ("arguments", "input", "args", "parameters")
_CALL_ID_KEYS = ("id", "toolCallId", "tool_call_id", "call_id")
_TOOL_CONTENT_TYPES = frozenset({"function", "tool_use", "tool_call"})


@dataclass(frozen=True)
class VendorModelIdentity:
    """Provider and model identity exactly as the vendor export declares it.

    Args:
        provider: Declared provider or system name.
        model_id: Declared request or response model identifier.
        revision: Declared model revision, when the export carries one.
    """

    provider: str
    model_id: str
    revision: str | None = None


@dataclass(frozen=True)
class VendorTokenUsage:
    """Complete token accounting declared by a vendor export.

    Args:
        input_tokens: Declared prompt token count.
        output_tokens: Declared completion token count.
        cached_input_tokens: Declared cached prompt token count, when present.
    """

    input_tokens: int
    output_tokens: int
    cached_input_tokens: int | None = None


@dataclass(frozen=True)
class VendorToolCall:
    """One tool call requested by a vendor model observation.

    Args:
        name: Declared tool name.
        arguments: Declared arguments as durable text or compact JSON.
        call_id: Explicit vendor call identity used for exact result pairing.
    """

    name: str
    arguments: str
    call_id: str | None = None


@dataclass(frozen=True)
class VendorObservation:
    """One vendor record ready for canonical conversion.

    Args:
        source_trace_id: Vendor trace key grouping this observation.
        source_span_id: Vendor span, observation, or row key.
        ordinal: Source order position used for stable tie breaking.
        started_at: Source start instant of the observation.
        ended_at: Source end instant, equal to ``started_at`` for point records.
        kind: Whether the record is a model call, a tool result, or agent-level evidence.
        source_parent_span_id: Vendor parent span key, when the export declares one.
        request_text: User-visible request text carried by this record.
        input_messages: Declared input messages retained as canonical GenAI evidence.
        completion_text: Assistant-visible output text for a model or agent record.
        tool_calls: Tool calls requested by a model record.
        tool_name: Executed tool name for a tool-result record.
        tool_arguments: Executed tool arguments for a tool-result record.
        tool_message: Tool output text for a tool-result record.
        tool_call_id: Explicit vendor call identity declared by a tool-result record.
        model: Declared provider and model identity for a model record.
        usage: Declared token accounting for a model record.
        failure_message: Vendor error text marking this record as failed.
        declared_attributes: Canonical GenAI attributes the vendor module read from the record,
            such as a declared model name that carries no provider evidence.
        extensions: Approved EXP extension attributes read from the source record.
        synthetic_time: Whether timestamps are EXP-assigned because the export carries none.
    """

    source_trace_id: str
    source_span_id: str
    ordinal: int
    started_at: datetime
    ended_at: datetime
    kind: VendorObservationKind
    source_parent_span_id: str | None = None
    request_text: str | None = None
    input_messages: JsonValue | None = None
    completion_text: str | None = None
    tool_calls: tuple[VendorToolCall, ...] = ()
    tool_name: str | None = None
    tool_arguments: str | None = None
    tool_message: str | None = None
    tool_call_id: str | None = None
    model: VendorModelIdentity | None = None
    usage: VendorTokenUsage | None = None
    failure_message: str | None = None
    declared_attributes: JsonObject = field(default_factory=dict)
    extensions: JsonObject = field(default_factory=dict)
    synthetic_time: bool = False


def declared_tool_calls(output: JsonValue | None) -> tuple[VendorToolCall, ...]:
    """Read the tool calls one assistant output declares, in source order.

    Args:
        output: Assistant output object, message list, or content parts.

    Returns:
        Declared tool calls, empty when the output requests none.

    Raises:
        VendorTraceFormatError: A declared tool call has no readable name.
    """
    return tuple(_tool_call(raw_call) for raw_call in _raw_tool_calls(output))


def _raw_tool_calls(output: JsonValue | None) -> tuple[JsonObject, ...]:
    """Collect raw tool-call objects from the supported assistant-output nestings.

    Args:
        output: Assistant output object, message list, or content parts.

    Returns:
        Raw tool-call objects in source order.
    """
    calls: list[JsonObject] = []
    for candidate in _output_candidates(output):
        for key in _TOOL_CALL_KEYS:
            raw = candidate.get(key)
            if isinstance(raw, list):
                calls.extend(item for item in raw if isinstance(item, dict))
        content = candidate.get("content")
        if isinstance(content, list):
            calls.extend(
                item
                for item in content
                if isinstance(item, dict) and item.get("type") in _TOOL_CONTENT_TYPES
            )
    return tuple(calls)


def _output_candidates(output: JsonValue | None) -> tuple[JsonObject, ...]:
    """Return every object that may carry tool calls for one assistant output.

    Args:
        output: Assistant output object, message list, or content parts.

    Returns:
        Candidate objects including declared message, response, and choice nestings.
    """
    if isinstance(output, list):
        candidates: list[JsonObject] = []
        for item in output:
            candidates.extend(_output_candidates(item))
        return tuple(candidates)
    if not isinstance(output, dict):
        return ()
    candidates = [output]
    for key in _NESTING_KEYS:
        nested = output.get(key)
        if nested is not None:
            candidates.extend(_output_candidates(nested))
    choices = output.get("choices")
    if isinstance(choices, list):
        candidates.extend(_output_candidates(choices))
    return tuple(candidates)


def _tool_call(raw_call: JsonObject) -> VendorToolCall:
    """Convert one raw tool call in a supported shape to a declared tool call.

    Args:
        raw_call: Raw OpenAI, AI SDK, or content-part tool call.

    Returns:
        Declared tool call with normalized name, arguments, and optional identity.

    Raises:
        VendorTraceFormatError: The tool call declares no readable name.
    """
    function = raw_call.get("function")
    candidate = function if isinstance(function, dict) else raw_call
    name = first_text(candidate, _NAME_KEYS) or first_text(raw_call, _NAME_KEYS)
    if name is None:
        raise VendorTraceFormatError("tool calls need a declared tool name")
    arguments: JsonValue | None = None
    for key in _ARGUMENT_KEYS:
        if key in candidate:
            arguments = candidate[key]
            break
    return VendorToolCall(
        name=name,
        arguments=json_text(arguments),
        call_id=first_text(raw_call, _CALL_ID_KEYS),
    )


def declared_completion_text(output: JsonValue | None) -> str:
    """Read visible assistant text from one declared model output.

    Args:
        output: Assistant output object, message list, plain text, or content parts.

    Returns:
        Visible assistant text, falling back to compact JSON for structured output.
    """
    if isinstance(output, str):
        return json_text(output)
    if isinstance(output, dict):
        for key in ("text", "content", "output_text"):
            text = message_text(output.get(key))
            if text:
                return text
        for key in _NESTING_KEYS:
            nested = output.get(key)
            if nested is not None:
                text = declared_completion_text(nested)
                if text:
                    return text
    if isinstance(output, list):
        text = message_text(output)
        if text:
            return text
        for item in reversed(output):
            nested_text = declared_completion_text(item)
            if nested_text:
                return nested_text
    return json_text(output)


def declared_usage(usage: JsonValue | None) -> VendorTokenUsage | None:
    """Read complete declared token accounting from a vendor usage object.

    Args:
        usage: Vendor usage object with input and output token counts.

    Returns:
        Declared usage, or ``None`` when the export declares no complete accounting.

    Raises:
        VendorTraceFormatError: A declared token count is not a non-negative integer.
    """
    if not isinstance(usage, dict):
        return None
    input_tokens = _token_count(usage, ("input", "input_tokens", "promptTokens", "prompt_tokens"))
    output_tokens = _token_count(
        usage, ("output", "output_tokens", "completionTokens", "completion_tokens")
    )
    if input_tokens is None or output_tokens is None:
        return None
    cached = _token_count(usage, ("cached_input_tokens", "cache_read_input_tokens"))
    return VendorTokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_input_tokens=cached,
    )


def _token_count(usage: JsonObject, keys: tuple[str, ...]) -> int | None:
    """Read one non-negative integral token count from an ordered key list.

    Args:
        usage: Vendor usage object.
        keys: Ordered candidate keys.

    Returns:
        The declared count, or ``None`` when no key declares one.

    Raises:
        VendorTraceFormatError: A declared count is not a non-negative integer.
    """
    for key in keys:
        if key not in usage:
            continue
        value = usage[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise VendorTraceFormatError(f"usage {key} must be a non-negative integer")
        return value
    return None


def declared_model_identity(
    record: JsonObject,
    *,
    model_keys: tuple[str, ...],
    provider_keys: tuple[str, ...],
    revision_keys: tuple[str, ...] = ("model_revision", "modelRevision", "revision"),
) -> tuple[VendorModelIdentity | None, str | None]:
    """Read declared model identity and the declared model name separately.

    Model identity is retained only when the export declares both a provider and a model, because
    EXP never infers a provider. A model name declared without a provider is still returned so the
    vendor module can retain it as source evidence.

    Args:
        record: Source record or attribute mapping.
        model_keys: Ordered candidate model-name keys.
        provider_keys: Ordered candidate provider keys.
        revision_keys: Ordered candidate model-revision keys.

    Returns:
        The retained identity when complete, and the declared model name when present.
    """
    model_id = first_text(record, model_keys)
    provider = first_text(record, provider_keys)
    if model_id is None or provider is None:
        return None, model_id
    return (
        VendorModelIdentity(
            provider=provider,
            model_id=model_id,
            revision=first_text(record, revision_keys),
        ),
        model_id,
    )


def declared_error_message(
    record: JsonObject,
    *,
    keys: tuple[str, ...],
    label: str,
) -> str | None:
    """Read a vendor error message from text or structured error fields.

    Args:
        record: Source record.
        keys: Ordered candidate error keys.
        label: Field label used in the validation message.

    Returns:
        Normalized error text, or ``None`` when the record declares no error.

    Raises:
        VendorTraceFormatError: A declared error object carries no readable content.
    """
    for key in keys:
        if key not in record:
            continue
        value = record[key]
        if value is None or value == "" or value == {} or value == []:
            continue
        if isinstance(value, str):
            return required_text(value, f"{label} {key}")
        return json_text(value)
    return None
