"""Test support: drive an SSE encoder over a deterministic provider-event fixture."""

from __future__ import annotations

from collections.abc import Iterable

from exp.runtime.gateway.contracts import GatewayEvent
from exp.runtime.openai_protocol.streaming import ChatSseEncoder, ResponsesSseEncoder


def encode_events(
    encoder: ChatSseEncoder | ResponsesSseEncoder, events: Iterable[GatewayEvent]
) -> tuple[str, ...]:
    """Encode one complete event fixture into its SSE frame sequence."""
    frames = list(encoder.start())
    for event in events:
        frames.extend(encoder.feed(event))
    return tuple(frames)
