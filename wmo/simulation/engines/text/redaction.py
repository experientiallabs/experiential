"""Project-configured redaction applied before text simulation evidence is persisted."""

from __future__ import annotations

import re

from pydantic import JsonValue

from wmo.common.core.artifacts import StructuredFailure
from wmo.common.models import AssistantAction
from wmo.common.rollouts import RolloutSpan

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


def redacted_field_set(field_names: tuple[str, ...]) -> frozenset[str]:
    """Normalize configured field names once for all calls in one text simulation.

    Args:
        field_names: Project-configured labels whose values must not persist in artifacts.

    Returns:
        Trimmed case-insensitive labels, excluding empty configuration values.
    """
    return frozenset(name.strip().casefold() for name in field_names if name.strip())
