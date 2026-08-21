"""Tests for cancellation-safe attempt reservation handling in gateway execution."""

from __future__ import annotations

import asyncio

import pytest

from exp.runtime.gateway.execution import _abandoned_reservation_outcome
from exp.runtime.gateway.ledger import GatewayLedgerError


def test_abandoned_reservation_returns_committed_attempt_id() -> None:
    """A cancellation-abandoned reservation still yields its durable attempt ID."""

    async def scenario() -> None:
        """Cancel the waiter repeatedly while the reservation keeps running."""
        started = asyncio.Event()

        async def reserve() -> str:
            """Simulate a shielded durable write that outlives the caller."""
            started.set()
            await asyncio.sleep(0.01)
            return "attempt-durable"

        reservation = asyncio.ensure_future(reserve())
        await started.wait()

        async def waiter() -> str | None:
            """Recover the abandoned reservation outcome."""
            return await _abandoned_reservation_outcome(reservation)

        waiting = asyncio.ensure_future(waiter())
        await asyncio.sleep(0)
        waiting.cancel()
        assert await waiting == "attempt-durable"

    asyncio.run(scenario())


def test_abandoned_reservation_returns_none_when_write_failed() -> None:
    """A reservation that raised committed nothing, so no attempt needs settling."""

    async def scenario() -> None:
        """Observe a failed reservation through the abandoned-outcome path."""

        async def reserve() -> str:
            """Simulate a rolled-back reservation write."""
            raise GatewayLedgerError("attempt reservation unavailable")

        reservation = asyncio.ensure_future(reserve())
        assert await _abandoned_reservation_outcome(reservation) is None

    asyncio.run(scenario())


def test_abandoned_reservation_waits_out_pending_write() -> None:
    """The outcome helper blocks until the in-flight write actually resolves."""

    async def scenario() -> None:
        """Resolve the reservation only after the helper starts waiting."""
        gate: asyncio.Future[str] = asyncio.get_running_loop().create_future()

        async def reserve() -> str:
            """Simulate a write pending on the group-commit batch."""
            return await gate

        reservation = asyncio.ensure_future(reserve())
        await asyncio.sleep(0)
        outcome = asyncio.ensure_future(_abandoned_reservation_outcome(reservation))
        await asyncio.sleep(0)
        assert not outcome.done()
        gate.set_result("attempt-late")
        assert await outcome == "attempt-late"

    asyncio.run(scenario())


def test_abandoned_reservation_cancel_absorption_raises_nothing() -> None:
    """Cancelling the helper's waiter does not surface once the write resolves."""

    async def scenario() -> None:
        """Cancel the helper while it waits, then confirm the durable result."""
        gate: asyncio.Future[str] = asyncio.get_running_loop().create_future()

        async def reserve() -> str:
            """Simulate a write pending on the group-commit batch."""
            return await gate

        reservation = asyncio.ensure_future(reserve())
        await asyncio.sleep(0)
        outcome = asyncio.ensure_future(_abandoned_reservation_outcome(reservation))
        await asyncio.sleep(0)
        outcome.cancel()
        gate.set_result("attempt-after-cancel")
        assert await outcome == "attempt-after-cancel"

    asyncio.run(scenario())


def test_cancelled_reservation_task_yields_none() -> None:
    """A reservation task cancelled before running reports no durable attempt."""

    async def scenario() -> None:
        """Cancel the reservation itself and confirm a None outcome."""

        async def reserve() -> str:
            """Simulate a write that never starts."""
            await asyncio.sleep(60)
            return "attempt-unreachable"

        reservation = asyncio.ensure_future(reserve())
        reservation.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.gather(reservation)
        assert await _abandoned_reservation_outcome(reservation) is None

    asyncio.run(scenario())
