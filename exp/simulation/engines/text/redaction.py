"""Project-configured redaction applied before text simulation evidence is persisted."""

from __future__ import annotations

import re

from pydantic import JsonValue

from exp.common.core.artifacts import StructuredFailure, redact_secret_json, redact_secret_text
from exp.common.models import AssistantAction, ToolCall
from exp.common.rollouts import RolloutArtifact, RolloutSpan

_REDACTED = "[REDACTED]"


def redact_json(value: JsonValue, field_names: frozenset[str]) -> JsonValue:
    """Recursively remove configured field values from JSON-safe evidence.

    Args:
        value: JSON-safe value about to enter an immutable simulation artifact.
        field_names: Case-insensitive field names selected in the project configuration.

    Returns:
        A structurally equivalent value whose configured fields and matching inline labels are
        replaced with a fixed redaction marker.
    """
    if isinstance(value, dict):
        return {
            key: _REDACTED if key.casefold() in field_names else redact_json(item, field_names)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_json(item, field_names) for item in value]
    if isinstance(value, str):
        return redact_text(value, field_names)
    return value


def redact_text(value: str, field_names: frozenset[str]) -> str:
    """Replace simple ``field: value`` and JSON ``field`` labels inside visible text.

    Args:
        value: Visible text that may include serialized structured customer context.
        field_names: Case-insensitive local project field names to remove.

    Returns:
        Text with recognizable configured labels redacted. Unlabeled natural-language content is
        preserved so a configured field name does not erase an unrelated response.
    """
    redacted = value
    for field_name in sorted(field_names, key=len, reverse=True):
        if not field_name:
            continue
        escaped = re.escape(field_name)
        json_pattern = re.compile(
            rf"((?:\"{escaped}\"|'{escaped}')\s*:\s*)(?:\"(?:\\.|[^\"\\])*\"|[^,}}\n]+)",
            re.IGNORECASE,
        )
        label_pattern = re.compile(rf"(\b{escaped}\b\s*[=:]\s*)([^,\n;}}]+)", re.IGNORECASE)
        redacted = json_pattern.sub(rf"\1\"{_REDACTED}\"", redacted)
        redacted = label_pattern.sub(rf"\1{_REDACTED}", redacted)
    return redacted


def redact_action(
    action: AssistantAction | None,
    field_names: frozenset[str],
) -> AssistantAction | None:
    """Return a visible assistant action with configured labels removed from its text.

    Args:
        action: Optional terminal action emitted by the customer agent.
        field_names: Case-insensitive local project field names to remove.

    Returns:
        The unchanged action when it has no text, otherwise a redacted immutable copy.
    """
    if action is None or action.content is None:
        return action
    return action.model_copy(update={"content": redact_text(action.content, field_names)})


def redact_failure(
    failure: StructuredFailure | None, field_names: frozenset[str]
) -> StructuredFailure | None:
    """Redact structured failure message and details before artifact persistence.

    Args:
        failure: Optional structured boundary failure.
        field_names: Case-insensitive local project field names to remove.

    Returns:
        A redacted immutable failure copy, or ``None``.

    Raises:
        TypeError: Redaction does not retain a JSON object for structured failure details.
    """
    if failure is None:
        return None
    details = redact_json(failure.details, field_names)
    if not isinstance(details, dict):  # pragma: no cover - JsonObject input always stays an object
        raise TypeError("structured failure details must remain a JSON object")
    return failure.model_copy(
        update={"message": redact_text(failure.message, field_names), "details": details}
    )


def redact_span(span: RolloutSpan, field_names: frozenset[str]) -> RolloutSpan:
    """Redact one existing agent event while retaining its identity and ordering.

    Args:
        span: Existing agent event to preserve in a new immutable rollout artifact.
        field_names: Case-insensitive local project field names to remove.

    Returns:
        A redacted immutable span copy.

    Raises:
        TypeError: Redaction does not retain a JSON object for the span payload.
    """
    payload = redact_json(span.payload, field_names)
    if not isinstance(payload, dict):  # pragma: no cover - JsonObject input always stays an object
        raise TypeError("rollout span payload must remain a JSON object")
    return span.model_copy(
        update={
            "payload": payload,
            "failure": redact_failure(span.failure, field_names),
        }
    )


def redact_rollout_secrets(rollout: RolloutArtifact) -> RolloutArtifact:
    """Replace secret-like substrings in one simulated rollout before immutable persistence.

    Simulated model output can contain generated credential-shaped strings that the
    immutable-artifact secret boundary rejects. This replaces those substrings in span
    payloads, span failures, the terminal failure, and the visible final output with the
    fixed deterministic placeholder, and records the replacement count on the rollout.

    Args:
        rollout: Fully bound rollout about to enter the immutable store.

    Returns:
        The unchanged rollout when nothing matched, otherwise a copy whose
        ``secret_redaction_count`` records how many substrings were replaced.
    """
    spans: list[RolloutSpan] = []
    total = 0
    for span in rollout.spans:
        redacted_span, span_count = _redact_span_secrets(span)
        spans.append(redacted_span)
        total += span_count
    failure, failure_count = _redact_failure_secrets(rollout.failure)
    total += failure_count
    final_output, output_count = _redact_action_secrets(rollout.final_output)
    total += output_count
    if total == 0:
        return rollout
    return rollout.model_copy(
        update={
            "spans": tuple(spans),
            "failure": failure,
            "final_output": final_output,
            "secret_redaction_count": rollout.secret_redaction_count + total,
        }
    )


def _redact_span_secrets(span: RolloutSpan) -> tuple[RolloutSpan, int]:
    """Redact secret-like substrings in one span payload and failure.

    Args:
        span: Existing execution span about to persist.

    Returns:
        The span, copied only when changed, and the number of replaced substrings.

    Raises:
        TypeError: Redaction does not retain a JSON object for the span payload.
    """
    payload, payload_count = redact_secret_json(span.payload)
    if not isinstance(payload, dict):  # pragma: no cover - JsonObject input always stays an object
        raise TypeError("rollout span payload must remain a JSON object")
    failure, failure_count = _redact_failure_secrets(span.failure)
    total = payload_count + failure_count
    if total == 0:
        return span, 0
    return span.model_copy(update={"payload": payload, "failure": failure}), total


def _redact_failure_secrets(
    failure: StructuredFailure | None,
) -> tuple[StructuredFailure | None, int]:
    """Redact secret-like substrings in one structured failure message and details.

    Args:
        failure: Optional structured boundary failure.

    Returns:
        The failure, copied only when changed, and the number of replaced substrings.

    Raises:
        TypeError: Redaction does not retain a JSON object for structured failure details.
    """
    if failure is None:
        return None, 0
    message, message_count = redact_secret_text(failure.message)
    details, details_count = redact_secret_json(failure.details)
    if not isinstance(details, dict):  # pragma: no cover - JsonObject input always stays an object
        raise TypeError("structured failure details must remain a JSON object")
    total = message_count + details_count
    if total == 0:
        return failure, 0
    return failure.model_copy(update={"message": message, "details": details}), total


def _redact_action_secrets(
    action: AssistantAction | None,
) -> tuple[AssistantAction | None, int]:
    """Redact secret-like substrings in one visible assistant action.

    Args:
        action: Optional terminal action emitted by the customer agent.

    Returns:
        The action, copied only when changed, and the number of replaced substrings.

    Raises:
        TypeError: Redaction does not retain a JSON object for tool-call arguments.
    """
    if action is None:
        return None, 0
    total = 0
    content = action.content
    if content is not None:
        content, content_count = redact_secret_text(content)
        total += content_count
    tool_calls: list[ToolCall] = []
    for tool_call in action.tool_calls:
        arguments, argument_count = redact_secret_json(tool_call.arguments)
        if not isinstance(arguments, dict):  # pragma: no cover - JsonObject stays an object
            raise TypeError("tool-call arguments must remain a JSON object")
        total += argument_count
        tool_calls.append(
            tool_call.model_copy(update={"arguments": arguments}) if argument_count else tool_call
        )
    if total == 0:
        return action, 0
    return action.model_copy(update={"content": content, "tool_calls": tuple(tool_calls)}), total


def redacted_field_set(field_names: tuple[str, ...]) -> frozenset[str]:
    """Normalize configured field names once for all calls in one text simulation.

    Args:
        field_names: Project-configured labels whose values must not persist in artifacts.

    Returns:
        Trimmed case-insensitive labels, excluding empty configuration values.
    """
    return frozenset(name.strip().casefold() for name in field_names if name.strip())
