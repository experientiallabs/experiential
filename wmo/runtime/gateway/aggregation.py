"""Bounded semantic event retention for non-streaming and continuation assembly."""

from __future__ import annotations

from wmo.runtime.gateway.contracts import GatewayEvent


class GatewayAggregationOverflowError(ValueError):
    """Provider output exceeded the launch data-plane retention ceiling."""


class BoundedGatewayEvents:
    """Retain normalized events under fixed count and serialized-byte ceilings."""

    def __init__(
        self,
        *,
        maximum_events: int = 100_000,
        maximum_bytes: int = 64 * 1024 * 1024,
    ) -> None:
        """Create empty bounded event state.

        Args:
            maximum_events: Maximum normalized events retained for one response.
            maximum_bytes: Maximum UTF-8 serialized event bytes retained.
        """
        if maximum_events < 1 or maximum_bytes < 1:
            raise ValueError("gateway event bounds must be positive")
        self._maximum_events = maximum_events
        self._maximum_bytes = maximum_bytes
        self._events: list[GatewayEvent] = []
        self._bytes = 0

    def append(self, event: GatewayEvent) -> None:
        """Retain one event or fail before process-local state becomes unbounded."""
        event_bytes = len(event.model_dump_json().encode("utf-8"))
        if (
            len(self._events) >= self._maximum_events
            or self._bytes + event_bytes > self._maximum_bytes
        ):
            raise GatewayAggregationOverflowError(
                "provider output exceeded the bounded gateway aggregation limit"
            )
        self._events.append(event)
        self._bytes += event_bytes

    def snapshot(self) -> tuple[GatewayEvent, ...]:
        """Return the retained immutable event sequence."""
        return tuple(self._events)
