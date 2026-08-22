"""Normalize native data-plane settlement payloads for durable accounting."""

from __future__ import annotations

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.contracts import (
    GatewayEvent,
    GatewayEventKind,
    GatewayFailure,
    GatewayFailureClass,
    GatewayUsage,
)

_TERMINAL_KINDS = {
    "completed": GatewayEventKind.COMPLETED,
    "incomplete": GatewayEventKind.INCOMPLETE,
    "failed": GatewayEventKind.FAILED,
}


def terminal_from_settlement(
    data: JsonObject,
) -> tuple[GatewayEvent, GatewayFailure | None]:
    """Build a durable terminal event from one native settlement payload.

    Args:
        data: Parsed outcome, usage, tool names, and optional failure.

    Returns:
        The normalized terminal event and optional failure.
    """
    raw_usage = data.get("usage")
    raw_tool_names = data.get("tool_names")
    usage = _usage_from_payload(
        raw_usage if isinstance(raw_usage, dict) else None,
        [str(name) for name in raw_tool_names] if isinstance(raw_tool_names, list) else [],
    )
    failure_payload = data.get("failure")
    failure = None
    if isinstance(failure_payload, dict):
        failure = GatewayFailure(
            failure_class=GatewayFailureClass(str(failure_payload["failure_class"])),
            safe_message=str(failure_payload["safe_message"]),
        )
    kind = _TERMINAL_KINDS[str(data["outcome"])]
    terminal = GatewayEvent(
        kind=kind,
        sequence_number=0,
        usage=usage,
        failure=failure if kind == GatewayEventKind.FAILED else None,
    )
    return terminal, failure


def _usage_from_payload(
    payload: JsonObject | None,
    tool_names: list[str],
) -> GatewayUsage | None:
    """Build normalized usage from settlement scalars and tool names."""
    names = tuple(str(name) for name in tool_names)
    if payload is None or payload.get("input_tokens") is None:
        return GatewayUsage(tool_names=names) if names else None
    return GatewayUsage(
        input_tokens=_optional_count(payload.get("input_tokens")),
        output_tokens=_optional_count(payload.get("output_tokens")),
        cached_input_tokens=_optional_count(payload.get("cached_input_tokens")),
        reasoning_tokens=_optional_count(payload.get("reasoning_tokens")),
        tool_names=names,
    )


def _optional_count(value: object) -> int | None:
    """Return one integer settlement token count or ``None``."""
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value
