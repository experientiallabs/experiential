"""Shared readers for WMO extension attributes on normalized trace spans.

Every canonical trace source (OTLP, PostHog, and the declared vendor exports) accepts the same
WMO extension attributes for trace-level facts: one conversation identity, one request context,
one visible tool list, and one terminal outcome. This module owns the single copy of the rules
those sources share: a fact declared on several spans must agree exactly, a declared value must
be well formed, and an invalid declaration excludes the trace instead of being repaired. Every
reader raises the caller's source-specific error type so exclusions keep their source labels.
The one exception is the lenient text mode that the OTLP conversation-id read opts into: foreign
exporters populate identity attributes with non-text values, so those are ignored rather than
excluding the trace.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Literal

from pydantic import JsonValue

from wmo.common.core.artifacts import FailureCode, JsonObject, StructuredFailure
from wmo.common.core.text import normalize_durable_text
from wmo.common.tasks import ToolSchema
from wmo.common.traces import TraceOutcome

CONVERSATION_ID_KEYS = ("wmo.conversation.id", "gen_ai.conversation.id")
REQUEST_CONTEXT_KEY = "wmo.request.context"
REQUEST_TOOLS_KEY = "wmo.request.tools"
OUTCOME_STATUS_KEY = "wmo.outcome.status"
OUTCOME_NAME_KEY = "wmo.outcome.name"
OUTCOME_FAILURE_CODE_KEY = "wmo.outcome.failure.code"
OUTCOME_FAILURE_MESSAGE_KEY = "wmo.outcome.failure.message"
OUTCOME_FAILURE_RETRYABLE_KEY = "wmo.outcome.failure.retryable"

_NONFAILURE_STATUSES: dict[str, Literal["success", "abandoned", "unknown"]] = {
    "success": "success",
    "abandoned": "abandoned",
    "unknown": "unknown",
}


def json_value(value: JsonValue | None) -> JsonValue | None:
    """Decode a JSON-encoded string while leaving native JSON and plain text unchanged.

    Args:
        value: Raw source value.

    Returns:
        Decoded JSON when the text parses, otherwise the original value.
    """
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def required_text(value: JsonValue | None, label: str, *, error_type: type[ValueError]) -> str:
    """Require one non-empty durable text value.

    Args:
        value: Raw source value.
        label: Field label used in the validation message.
        error_type: Source-specific validation exception type.

    Returns:
        Normalized durable text.

    Raises:
        ValueError: The declared ``error_type`` when the value is absent, not text, or blank.
    """
    if not isinstance(value, str) or not value.strip():
        raise error_type(f"{label} must be non-empty text")
    return normalize_durable_text(value.strip())


def consistent_text(
    attributes_by_span: Sequence[JsonObject],
    keys: Sequence[str],
    *,
    error_type: type[ValueError],
    lenient: bool = False,
) -> str | None:
    """Return one repeated text extension across a trace or reject ambiguity.

    Args:
        attributes_by_span: Canonical span attributes in trace order.
        keys: Ordered candidate attribute keys; the first declared key wins per span.
        error_type: Source-specific validation exception type.
        lenient: Whether a declared non-text or blank value is skipped, falling through to
            the next candidate key, instead of excluding the trace. Only the OTLP
            conversation-id read opts in; every other extension read stays strict.

    Returns:
        The single declared value, or ``None`` when no span declares one.

    Raises:
        ValueError: The declared ``error_type`` when a strictly read declared value is blank,
            not text, or an explicit null, or when spans disagree.
    """
    values: list[str] = []
    for attributes in attributes_by_span:
        value = _span_text(attributes, keys, error_type=error_type, lenient=lenient)
        if value is not None:
            values.append(value)
    if not values:
        return None
    if any(value != values[0] for value in values[1:]):
        raise error_type(f"{keys[0]} differs across spans in one trace")
    return values[0]


def _span_text(
    attributes: JsonObject,
    keys: Sequence[str],
    *,
    error_type: type[ValueError],
    lenient: bool,
) -> str | None:
    """Read one span's declared text value from an ordered candidate key list.

    Args:
        attributes: Canonical attributes of one span.
        keys: Ordered candidate attribute keys.
        error_type: Source-specific validation exception type.
        lenient: Whether a declared non-text or blank value is skipped instead of raising.

    Returns:
        The first valid declared text, or ``None`` when the span declares none.

    Raises:
        ValueError: The declared ``error_type`` when a strictly read declared value is not
            non-empty text. Strict reads treat a present key as a declaration, so an explicit
            null is rejected rather than read as undeclared.
    """
    for key in keys:
        if key not in attributes:
            continue
        value = attributes[key]
        if not lenient:
            return required_text(value, key, error_type=error_type)
        if isinstance(value, str) and value.strip():
            return normalize_durable_text(value.strip())
    return None


def consistent_bool(
    attributes_by_span: Sequence[JsonObject],
    key: str,
    *,
    error_type: type[ValueError],
) -> bool | None:
    """Return one repeated boolean extension or reject inconsistent span values.

    Args:
        attributes_by_span: Canonical span attributes in trace order.
        key: Extension attribute key.
        error_type: Source-specific validation exception type.

    Returns:
        The single declared boolean, or ``None`` when no span declares one.

    Raises:
        ValueError: The declared ``error_type`` when a declared value is not boolean or spans
            disagree.
    """
    values: list[bool] = []
    for attributes in attributes_by_span:
        value = attributes.get(key)
        if value is None:
            continue
        if not isinstance(value, bool):
            raise error_type(f"{key} must be boolean")
        values.append(value)
    if not values:
        return None
    if any(value != values[0] for value in values[1:]):
        raise error_type(f"{key} differs across spans in one trace")
    return values[0]


def consistent_json_object(
    attributes_by_span: Sequence[JsonObject],
    key: str,
    *,
    error_type: type[ValueError],
) -> JsonObject:
    """Return one repeated JSON-object extension or reject conflicting copies.

    Args:
        attributes_by_span: Canonical span attributes in trace order.
        key: Extension attribute key holding a JSON object or its JSON-encoded text.
        error_type: Source-specific validation exception type.

    Returns:
        The single declared object, empty when no span declares one.

    Raises:
        ValueError: The declared ``error_type`` when a declared value is not an object or spans
            disagree.
    """
    values: list[JsonObject] = []
    for attributes in attributes_by_span:
        value = json_value(attributes.get(key))
        if value is None:
            continue
        if not isinstance(value, dict):
            raise error_type(f"{key} must be a JSON object")
        values.append(value)
    if not values:
        return {}
    if any(value != values[0] for value in values[1:]):
        raise error_type(f"{key} differs across spans in one trace")
    return values[0]


def collect_tools(
    attributes_by_span: Sequence[JsonObject],
    *,
    keys: Sequence[str],
    error_type: type[ValueError],
) -> tuple[ToolSchema, ...]:
    """Convert declared request tool definitions to canonical visible tools.

    Args:
        attributes_by_span: Canonical span attributes in trace order.
        keys: Ordered candidate attribute keys; the first non-null key wins per span.
        error_type: Source-specific validation exception type.

    Returns:
        Deterministically ordered tool schemas declared by the source.

    Raises:
        ValueError: The declared ``error_type`` when a definition list, schema, or name is
            invalid or conflicting.
    """
    by_name: dict[str, ToolSchema] = {}
    for attributes in attributes_by_span:
        declared: JsonValue | None = None
        declared_key = keys[0]
        for key in keys:
            value = json_value(attributes.get(key))
            if value is not None:
                declared, declared_key = value, key
                break
        if declared is None:
            continue
        if not isinstance(declared, list):
            raise error_type(f"{declared_key} must be a JSON array")
        for raw_tool in declared:
            tool = _tool_schema(raw_tool, error_type=error_type)
            if tool.name in by_name and by_name[tool.name] != tool:
                raise error_type(f"tool {tool.name!r} has conflicting definitions")
            by_name[tool.name] = tool
    return tuple(by_name[name] for name in sorted(by_name))


def _tool_schema(raw_tool: JsonValue, *, error_type: type[ValueError]) -> ToolSchema:
    """Convert one declared function tool definition to the canonical tool contract.

    Args:
        raw_tool: One declared tool definition.
        error_type: Source-specific validation exception type.

    Returns:
        Canonical visible tool schema.

    Raises:
        ValueError: The declared ``error_type`` when the definition is not an object with a
            name and an object input schema.
    """
    if not isinstance(raw_tool, dict):
        raise error_type("tool definitions must be objects")
    candidate = raw_tool.get("function") if raw_tool.get("type") == "function" else raw_tool
    if not isinstance(candidate, dict):
        raise error_type("function tool definitions need a function object")
    name = required_text(candidate.get("name"), "tool definition name", error_type=error_type)
    description = candidate.get("description")
    schema = candidate.get("input_schema", candidate.get("parameters", candidate.get("schema")))
    if not isinstance(schema, dict):
        raise error_type(f"tool {name!r} needs an object input schema")
    return ToolSchema(
        name=name,
        description=(
            normalize_durable_text(description.strip())
            if isinstance(description, str) and description.strip()
            else "No description captured."
        ),
        input_schema=schema,
    )


def collect_outcome(
    attributes_by_span: Sequence[JsonObject],
    *,
    failures: Sequence[StructuredFailure],
    error_type: type[ValueError],
) -> TraceOutcome | None:
    """Map declared WMO outcome extensions or source failures to terminal trace evidence.

    Args:
        attributes_by_span: Canonical span attributes in trace order.
        failures: Structured span failures observed in this trace, in order. Sources without
            span-failure fallback semantics pass an empty sequence.
        error_type: Source-specific validation exception type.

    Returns:
        Canonical trace outcome, or ``None`` when the source declares none.

    Raises:
        ValueError: The declared ``error_type`` when outcome extensions are incomplete or
            contradictory.
    """
    status = consistent_text(attributes_by_span, (OUTCOME_STATUS_KEY,), error_type=error_type)
    outcome_name = consistent_text(attributes_by_span, (OUTCOME_NAME_KEY,), error_type=error_type)
    failure_code = consistent_text(
        attributes_by_span, (OUTCOME_FAILURE_CODE_KEY,), error_type=error_type
    )
    failure_message = consistent_text(
        attributes_by_span, (OUTCOME_FAILURE_MESSAGE_KEY,), error_type=error_type
    )
    retryable = consistent_bool(
        attributes_by_span, OUTCOME_FAILURE_RETRYABLE_KEY, error_type=error_type
    )
    if status is None:
        if failures:
            return TraceOutcome(status="failure", failure=failures[0])
        if any(
            value is not None for value in (outcome_name, failure_code, failure_message, retryable)
        ):
            raise error_type(f"outcome details require {OUTCOME_STATUS_KEY}")
        return None
    nonfailure_status = _NONFAILURE_STATUSES.get(status)
    if nonfailure_status is not None:
        if any(value is not None for value in (failure_code, failure_message, retryable)):
            raise error_type("outcome failure details require failure status")
        return TraceOutcome(status=nonfailure_status, outcome_name=outcome_name)
    if status != "failure":
        raise error_type(f"{OUTCOME_STATUS_KEY} must be success, failure, abandoned, or unknown")
    if failure_code is None or failure_message is None:
        if failures:
            return TraceOutcome(status="failure", outcome_name=outcome_name, failure=failures[0])
        raise error_type("failure outcomes need a failure code and message")
    try:
        code = FailureCode(failure_code)
    except ValueError as exc:
        valid = ", ".join(item.value for item in FailureCode)
        raise error_type(f"{OUTCOME_FAILURE_CODE_KEY} must be one of: {valid}") from exc
    return TraceOutcome(
        status="failure",
        outcome_name=outcome_name,
        failure=StructuredFailure(code=code, message=failure_message, retryable=retryable or False),
    )
