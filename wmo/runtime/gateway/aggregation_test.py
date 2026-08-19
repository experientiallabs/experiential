"""Tests for bounded gateway event retention and its byte accounting."""

from __future__ import annotations

import pytest

from wmo.runtime.gateway.aggregation import (
    BoundedGatewayEvents,
    GatewayAggregationOverflowError,
)
from wmo.runtime.gateway.contracts import GatewayEvent, GatewayEventKind


def _delta(sequence_number: int, text: str) -> GatewayEvent:
    """Return one text delta event carrying the given payload."""
    return GatewayEvent(
        kind=GatewayEventKind.TEXT_DELTA,
        sequence_number=sequence_number,
        text_delta=text,
    )


def test_delta_bytes_overflow_still_raises_under_proxy_accounting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Payload text plus the fixed per-event envelope still trips the byte ceiling."""
    monkeypatch.setattr("wmo.runtime.gateway.aggregation._MAXIMUM_BYTES", 300)
    events = BoundedGatewayEvents()
    events.append(_delta(0, "a" * 100))

    with pytest.raises(GatewayAggregationOverflowError, match="bounded gateway aggregation"):
        events.append(_delta(1, "b" * 200))
    assert len(events.snapshot()) == 1


def test_event_count_overflow_and_terminal_serialized_accounting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The count bound applies to every kind and terminals use exact serialized bytes."""
    monkeypatch.setattr("wmo.runtime.gateway.aggregation._MAXIMUM_EVENTS", 1)
    events = BoundedGatewayEvents()
    events.append(_delta(0, "hello"))
    with pytest.raises(GatewayAggregationOverflowError):
        events.append(GatewayEvent(kind=GatewayEventKind.COMPLETED, sequence_number=1))

    terminal = GatewayEvent(kind=GatewayEventKind.COMPLETED, sequence_number=0)
    monkeypatch.setattr(
        "wmo.runtime.gateway.aggregation._MAXIMUM_BYTES",
        len(terminal.model_dump_json().encode("utf-8")) - 1,
    )
    monkeypatch.setattr("wmo.runtime.gateway.aggregation._MAXIMUM_EVENTS", 10)
    bounded = BoundedGatewayEvents()
    with pytest.raises(GatewayAggregationOverflowError):
        bounded.append(terminal)
    assert bounded.snapshot() == ()
