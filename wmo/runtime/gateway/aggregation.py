"""Bounded semantic event retention for non-streaming and continuation assembly."""

from __future__ import annotations

from wmo.runtime.gateway.contracts import GatewayEvent

_MAXIMUM_EVENTS = 100_000
_MAXIMUM_BYTES = 64 * 1024 * 1024


class GatewayAggregationOverflowError(ValueError):
    """Provider output exceeded the launch data-plane retention ceiling."""


class BoundedGatewayEvents:
    """Retain normalized events under fixed count and serialized-byte ceilings."""

    def __init__(self) -> None:
        """Create empty bounded event state."""
        self._events: list[GatewayEvent] = []
        self._bytes = 0

    def append(self, event: GatewayEvent) -> None:
        """Retain one event or fail before process-local state becomes unbounded."""
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
