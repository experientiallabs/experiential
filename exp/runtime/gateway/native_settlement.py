"""Settlement wire parsing shared by the native gateway control plane.

The native data plane calls back into the python control plane with plain
JSON strings; these helpers turn the settlement and admission payloads into
the same typed gateway contracts the embedded python engine already writes
to the ledger, so both engines produce identical durable rows.
"""

from __future__ import annotations

from exp.common.core.artifacts import JsonObject, stable_id
from exp.runtime.gateway.contracts import (
    GatewayEvent,
    GatewayEventKind,
    GatewayFailure,
    GatewayFailureClass,
    GatewayUsage,
)
from exp.runtime.gateway.routing import GatewayRoute

TERMINAL_KINDS = {
    "completed": GatewayEventKind.COMPLETED,
    "incomplete": GatewayEventKind.INCOMPLETE,
    "failed": GatewayEventKind.FAILED,
}


def deployment_operation_key(route: GatewayRoute) -> str:
    """Derive the stable per-deployment idempotency key used by dispatch.

    Mirrors the executor's provider-operation identity so retried physical
    dispatches of the same deployment reuse one caller operation.

    Args:
        route: Resolved single-deployment route.

    Returns:
        Stable content-addressed operation identity.
    """
    authorization = route.snapshot.authorization
    return stable_id(
        "gateway-provider-operation",
        {
            "request_id": authorization.request_id,
            "catalog_sha256": authorization.catalog_sha256,
            "deployment_id": route.deployment.deployment_id,
            "connection_sha256": route.deployment.connection_sha256,
        },
    )


def optional_text(value: object) -> str | None:
    """Return one optional boundary string value or ``None``."""
    return value if isinstance(value, str) else None


def terminal_from_settlement(
    data: JsonObject,
) -> tuple[GatewayEvent, GatewayFailure | None]:
    """Build the durable terminal event from one settlement payload.

    Args:
        data: Parsed settlement with ``outcome``, optional ``usage``,
            ``tool_names``, and ``failure``.

    Returns:
        The terminal event and the optional normalized failure.
    """
    raw_usage = data.get("usage")
    raw_tool_names = data.get("tool_names")
    usage = usage_from_payload(
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
    kind = TERMINAL_KINDS[str(data["outcome"])]
    terminal = GatewayEvent(
        kind=kind,
        sequence_number=0,
        usage=usage,
        failure=failure if kind == GatewayEventKind.FAILED else None,
    )
    return terminal, failure


def usage_from_payload(
    payload: JsonObject | None,
    tool_names: list[str],
) -> GatewayUsage | None:
    """Build normalized usage from settlement scalars.

    Args:
        payload: Optional token totals observed by the data plane.
        tool_names: Invoked tool names in first-use order.

    Returns:
        Normalized usage, or ``None`` when nothing was observed.
    """
    names = tuple(str(name) for name in tool_names)
    if payload is None or payload.get("input_tokens") is None:
        if not names:
            return None
        return GatewayUsage(tool_names=names)
    return GatewayUsage(
        input_tokens=optional_count(payload.get("input_tokens")),
        output_tokens=optional_count(payload.get("output_tokens")),
        cached_input_tokens=optional_count(payload.get("cached_input_tokens")),
        reasoning_tokens=optional_count(payload.get("reasoning_tokens")),
        tool_names=names,
    )


def optional_count(value: object) -> int | None:
    """Return one non-negative settlement token count or ``None``."""
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value
