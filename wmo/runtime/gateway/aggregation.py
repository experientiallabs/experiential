"""Bounded semantic event retention for non-streaming and continuation assembly."""

from __future__ import annotations

from wmo.runtime.gateway.contracts import GatewayEvent, GatewayEventKind

_MAXIMUM_EVENTS = 100_000
_MAXIMUM_BYTES = 64 * 1024 * 1024
_DELTA_EVENT_KINDS = {
    GatewayEventKind.TEXT_DELTA,
    GatewayEventKind.REFUSAL_DELTA,
    GatewayEventKind.TOOL_ARGUMENTS_DELTA,
}
_DELTA_EVENT_OVERHEAD_BYTES = 64


class GatewayAggregationOverflowError(ValueError):
    """Provider output exceeded the launch data-plane retention ceiling."""


class BoundedGatewayEvents:
    """Retain normalized events under fixed count and accounted-byte ceilings."""

    def __init__(self) -> None:
        """Create empty bounded event state."""
        self._events: list[GatewayEvent] = []
        self._bytes = 0

    def append(self, event: GatewayEvent) -> None:
        """Retain one event or fail before process-local state becomes unbounded.

        Hot delta events are accounted from their payload text plus a fixed envelope
        constant, while the rare non-delta kinds use their exact serialized size.
        """
        if event.kind in _DELTA_EVENT_KINDS:
            event_bytes = (
                len((event.text_delta or "").encode("utf-8"))
                + len((event.raw_arguments_delta or "").encode("utf-8"))
                + _DELTA_EVENT_OVERHEAD_BYTES
            )
        else:
            event_bytes = len(event.model_dump_json().encode("utf-8"))
        if len(self._events) >= _MAXIMUM_EVENTS or self._bytes + event_bytes > _MAXIMUM_BYTES:
            raise GatewayAggregationOverflowError(
                "provider output exceeded the bounded gateway aggregation limit"
            )
        self._events.append(event)
        self._bytes += event_bytes

    def snapshot(self) -> tuple[GatewayEvent, ...]:
        """Return the retained immutable event sequence."""
        return tuple(self._events)
