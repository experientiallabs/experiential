"""Bounded async classifier execution tests."""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from exp.runtime.gateway.guardrails.bounded import (
    BoundedInspect,
    ClassifierTimeoutError,
    run_on_native_loop,
)


def test_bounded_inspect_returns_the_inspect_result() -> None:
    """A timely inspect result is returned to the caller."""

    async def scenario() -> None:
        """Await one immediate inspect."""
        bound = BoundedInspect()

        async def inspect() -> int:
            """Return a constant."""
            return 7

        assert await bound.run(inspect, 0.5, adapter_id="healthy") == 7

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
            await bound.run(hang, 0.05, adapter_id="hung")
        assert asyncio.get_running_loop().time() - started < 1.0
        assert entered.is_set()

        async def healthy() -> int:
            """Return immediately."""
            return 3

        assert await bound.run(healthy, 0.5, adapter_id="healthy") == 3

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
            await bound.run(boom, 0.5, adapter_id="boom")

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
                await bound.run(hang, 0.03, adapter_id="hung")

        async def healthy() -> int:
            """Return immediately."""
            return 9

        started = asyncio.get_running_loop().time()
        assert await bound.run(healthy, 0.5, adapter_id="healthy") == 9
        assert asyncio.get_running_loop().time() - started < 0.2

    asyncio.run(scenario())


def test_suppressed_cancellation_quarantines_only_that_adapter() -> None:
    """A classifier that swallows CancelledError cannot retain capacity or spawn tasks."""

    async def scenario() -> None:
        """Time out a cancel-swallowing inspect, retry it, then run a healthy inspect."""
        bound = BoundedInspect(max_inflight=1)
        hold = asyncio.Event()

        async def swallow() -> int:
            """Ignore cancellation and wait until the test releases the hold."""
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                await hold.wait()
                return 1
            return 0

        started = asyncio.get_running_loop().time()
        with pytest.raises(ClassifierTimeoutError):
            await bound.run(swallow, 0.05, adapter_id="rogue")
        assert asyncio.get_running_loop().time() - started < 0.5
        assert bound.detached_inspect_count() == 1
        assert bound.quarantined_adapter_ids() == frozenset({"rogue"})

        started = asyncio.get_running_loop().time()
        with pytest.raises(ClassifierTimeoutError, match="quarantined"):
            await bound.run(swallow, 0.5, adapter_id="rogue")
        assert asyncio.get_running_loop().time() - started < 0.2
        assert bound.detached_inspect_count() == 1

        async def healthy() -> int:
            """Return immediately."""
            return 4

        started = asyncio.get_running_loop().time()
        assert await bound.run(healthy, 0.5, adapter_id="healthy") == 4
        assert asyncio.get_running_loop().time() - started < 0.2
        assert bound.detached_inspect_count() == 1
        hold.set()
        await asyncio.sleep(0)
        assert bound.detached_inspect_count() == 0

    asyncio.run(scenario())


def test_concurrent_timeouts_keep_quarantine_until_every_detached_task_finishes() -> None:
    """One finished rogue inspect cannot lift quarantine while another is still live."""

    async def scenario() -> None:
        """Time out two inspects on one adapter, then release them one at a time."""
        bound = BoundedInspect(max_inflight=2)
        holds: list[asyncio.Event] = []

        async def swallow() -> int:
            """Ignore cancellation and wait on a per-inspect hold."""
            hold = asyncio.Event()
            holds.append(hold)
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                await hold.wait()
                return 1
            return 0

        async def rogue() -> None:
            """Run one cancel-swallowing inspect past its timeout."""
            with pytest.raises(ClassifierTimeoutError):
                await bound.run(swallow, 0.08, adapter_id="rogue")

        await asyncio.gather(rogue(), rogue())
        assert len(holds) == 2
        assert bound.detached_inspect_count() == 2
        assert bound.quarantined_adapter_ids() == frozenset({"rogue"})

        holds[0].set()
        for _ in range(10):
            if bound.detached_inspect_count() == 1:
                break
            await asyncio.sleep(0)
        assert bound.detached_inspect_count() == 1
        assert bound.quarantined_adapter_ids() == frozenset({"rogue"})

        started = asyncio.get_running_loop().time()
        with pytest.raises(ClassifierTimeoutError, match="quarantined"):
            await bound.run(swallow, 0.5, adapter_id="rogue")
        assert asyncio.get_running_loop().time() - started < 0.2
        assert bound.detached_inspect_count() == 1

        async def healthy() -> int:
            """Return immediately."""
            return 5

        assert await bound.run(healthy, 0.5, adapter_id="healthy") == 5
        holds[1].set()
        for _ in range(10):
            if bound.detached_inspect_count() == 0:
                break
            await asyncio.sleep(0)
        assert bound.detached_inspect_count() == 0

    asyncio.run(scenario())


def test_external_cancellation_propagates_and_releases_the_slot() -> None:
    """Caller cancellation is not converted into a classifier timeout."""

    async def scenario() -> None:
        """Cancel the waiting run, then reuse the slot on another adapter."""
        bound = BoundedInspect(max_inflight=1)
        entered = asyncio.Event()

        async def hang() -> int:
            """Wait until cancelled."""
            entered.set()
            await asyncio.Event().wait()
            return 1

        task = asyncio.create_task(bound.run(hang, 5.0, adapter_id="hung"))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        async def healthy() -> int:
            """Return immediately."""
            return 2

        assert await bound.run(healthy, 0.5, adapter_id="healthy") == 2

    asyncio.run(scenario())


def test_run_on_native_loop_executes_without_a_running_loop() -> None:
    """Native callbacks submit work onto the shared daemon loop."""

    async def inspect() -> int:
        """Return a constant."""
        return 4

    assert run_on_native_loop(inspect()) == 4


def test_run_on_native_loop_refuses_a_running_event_loop() -> None:
    """A native callback is not nested onto the Python gateway loop."""

    async def scenario() -> None:
        """Call the native helper from an already-running loop."""

        async def inspect() -> int:
            """Return a constant."""
            return 1

        coro = inspect()
        with pytest.raises(RuntimeError, match="already owns an event loop"):
            run_on_native_loop(coro)

    asyncio.run(scenario())


def test_native_loop_returns_while_a_quarantined_adapter_still_runs() -> None:
    """The Rust worker is not blocked on an adapter that ignores cancellation."""
    bound = BoundedInspect(max_inflight=1)
    hold = threading.Event()
    entries = 0

    async def swallow() -> int:
        """Ignore cancellation and wait until teardown releases the hold."""
        nonlocal entries
        entries += 1
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            while not hold.is_set():
                await asyncio.sleep(0.05)
            return 1
        return 0

    async def healthy() -> int:
        """Return immediately."""
        return 8

    async def rogue_once() -> int:
        """Run one cancel-swallowing inspect."""
        return await bound.run(swallow, 0.05, adapter_id="rogue")

    async def rogue_retry() -> int:
        """Retry the quarantined adapter."""
        return await bound.run(swallow, 0.5, adapter_id="rogue")

    async def healthy_once() -> int:
        """Run one healthy inspect on the same limiter."""
        return await bound.run(healthy, 0.5, adapter_id="healthy")

    try:
        started = time.monotonic()
        with pytest.raises(ClassifierTimeoutError):
            run_on_native_loop(rogue_once())
        assert time.monotonic() - started < 0.5
        assert bound.detached_inspect_count() == 1
        assert entries == 1

        started = time.monotonic()
        with pytest.raises(ClassifierTimeoutError, match="quarantined"):
            run_on_native_loop(rogue_retry())
        assert time.monotonic() - started < 0.2
        assert bound.detached_inspect_count() == 1
        assert entries == 1

        started = time.monotonic()
        assert run_on_native_loop(healthy_once()) == 8
        assert time.monotonic() - started < 0.2
    finally:
        hold.set()
        for _ in range(50):
            if bound.detached_inspect_count() == 0:
                break
            time.sleep(0.02)
        assert bound.detached_inspect_count() == 0
