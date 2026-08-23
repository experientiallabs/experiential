"""Collect a winning stream and apply one output-chain callback."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from exp.runtime.gateway.aggregation import BoundedGatewayEvents
from exp.runtime.gateway.contracts import GatewayEvent
from exp.runtime.gateway.execution import GatewayExecutionStream
from exp.runtime.gateway.guardrails.completion import (
    apply_text_replacement,
    completion_from_events,
)
from exp.runtime.gateway.guardrails.contracts import GuardrailPolicy
from exp.runtime.gateway.guardrails.enforcement import GuardrailEngine


async def collect_and_enforce_output(
    *,
    engine: GuardrailEngine,
    policy: GuardrailPolicy,
    stream: GatewayExecutionStream,
    deadline_monotonic: float,
) -> tuple[GatewayEvent, ...]:
    """Buffer the winning stream, then run the output chain once.

    Args:
        engine: Assigned guardrail engine.
        policy: Identity policy that requested output checks.
        stream: Winning accounted provider stream.
        deadline_monotonic: Remaining request-wide deadline.

    Returns:
        The validated or text-rewritten event sequence.

    Raises:
        GuardrailRejected: The output chain blocked or fail-closed.
        GatewayAggregationOverflowError: The buffered stream exceeded bounds.
    """
    events = BoundedGatewayEvents()
    async for event in stream:
        events.append(event)
    retained = events.snapshot()
    if not policy.output_checks:
        return retained
    original = completion_from_events(retained)
    rewritten = await asyncio.to_thread(
        engine.enforce_output,
        policy=policy,
        completion=original,
        deadline_monotonic=deadline_monotonic,
    )
    if rewritten.text == original.text:
        return retained
    return apply_text_replacement(retained, rewritten.text)


def requires_output_buffer(policy: GuardrailPolicy | None) -> bool:
    """Return whether caller delivery must wait for one output-chain callback."""
    return policy is not None and bool(policy.output_checks)


async def iter_events(events: tuple[GatewayEvent, ...]) -> AsyncIterator[GatewayEvent]:
    """Yield an already-buffered event sequence as an async iterator."""
    for event in events:
        yield event
