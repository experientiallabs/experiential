"""Bounded async classifier execution tests."""

from __future__ import annotations

import asyncio

import pytest

from exp.runtime.gateway.guardrails.bounded import (
    BoundedInspect,
    ClassifierTimeoutError,
    run_on_private_loop,
)


def test_bounded_inspect_returns_the_inspect_result() -> None:
    """A timely inspect result is returned to the caller."""

    async def scenario() -> None:
        """Await one immediate inspect."""
        bound = BoundedInspect()

        async def inspect() -> int:
            """Return a constant."""
            return 7

        assert await bound.run(inspect, 0.5) == 7

    asyncio.run(scenario())


def test_bounded_inspect_cancels_a_hung_inspect_and_releases_the_slot() -> None:
    """A hung inspect times out, then a later inspect can take the same slot."""

    async def scenario() -> None:
        """Fill one slot with a never-returning inspect, then reuse it."""
        bound = BoundedInspect(max_inflight=1)
        entered = asyncio.Event()

        async def hang() -> int:
            """Wait forever after marking entry."""
            entered.set()
            await asyncio.Event().wait()
            return 1

        started = asyncio.get_running_loop().time()
        with pytest.raises(ClassifierTimeoutError):
            await bound.run(hang, 0.05)
        assert asyncio.get_running_loop().time() - started < 1.0
        assert entered.is_set()

        async def healthy() -> int:
            """Return immediately."""
            return 3

        assert await bound.run(healthy, 0.5) == 3

    asyncio.run(scenario())


def test_bounded_inspect_propagates_inspect_errors() -> None:
    """Adapter exceptions surface after the coroutine finishes inside the budget."""

    async def scenario() -> None:
        """Raise from a timely inspect."""
        bound = BoundedInspect()

        async def boom() -> int:
            """Fail immediately."""
            raise RuntimeError("classifier unavailable")

        with pytest.raises(RuntimeError, match="classifier unavailable"):
            await bound.run(boom, 0.5)

    asyncio.run(scenario())


def test_repeated_timeouts_do_not_exhaust_later_inspects() -> None:
    """Cancelling past the inflight cap still leaves capacity for a healthy inspect."""

    async def scenario() -> None:
        """Time out more inspects than the cap, then run a healthy inspect."""
        bound = BoundedInspect(max_inflight=2)

        async def hang() -> int:
            """Wait until cancelled."""
            await asyncio.Event().wait()
            return 1

        for _ in range(6):
            with pytest.raises(ClassifierTimeoutError):
                await bound.run(hang, 0.03)

        async def healthy() -> int:
            """Return immediately."""
            return 9

        started = asyncio.get_running_loop().time()
        assert await bound.run(healthy, 0.5) == 9
        assert asyncio.get_running_loop().time() - started < 0.2

    asyncio.run(scenario())


def test_run_on_private_loop_executes_without_a_running_loop() -> None:
    """Native callbacks create a private loop on the worker thread."""

    async def inspect() -> int:
        """Return a constant."""
        return 4

    assert run_on_private_loop(inspect()) == 4


def test_run_on_private_loop_refuses_a_running_event_loop() -> None:
    """A private loop is not nested onto the Python gateway loop."""

    async def scenario() -> None:
        """Call the native bridge helper from an already-running loop."""

        async def inspect() -> int:
            """Return a constant."""
            return 1

        coro = inspect()
        try:
            with pytest.raises(RuntimeError, match="already owns an event loop"):
                run_on_private_loop(coro)
        finally:
            coro.close()

    asyncio.run(scenario())
