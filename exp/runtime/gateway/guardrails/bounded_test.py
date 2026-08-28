"""Bounded async classifier execution tests."""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Callable

import httpx
import pytest

from exp.runtime.gateway.guardrails.bounded import (
    BoundedInspect,
    ClassifierTimeoutError,
    _IsolationWorker,
    _NativeCallbackRunner,
    run_on_native_loop,
)
from exp.runtime.gateway.guardrails.http_json import shared_http_json_client


async def _wait_until(predicate: Callable[[], bool], *, timeout: float = 10.0) -> None:
    """Poll ``predicate`` without blocking the caller event loop.

    The deadline is sized for the worst loaded CI worker: polling returns on
    the first success, so healthy runs stay fast, and only genuinely broken
    code waits out the budget before the assertion fails.
    """
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() >= deadline:
            break
        await asyncio.sleep(0.01)
    assert predicate()


def _wait_sync(predicate: Callable[[], bool], *, timeout: float = 10.0) -> None:
    """Poll ``predicate`` from synchronous test code, then assert it holds.

    Same worst-loaded-worker sizing as :func:`_wait_until`: detached-task
    bookkeeping is scheduled asynchronously, so asserting it after a fixed
    short sleep races the scheduler on a busy runner.
    """
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert predicate()


async def _wait_hold(hold: threading.Event, *, timeout: float = 5.0) -> None:
    """Wait for a test to release an abandoned inspect, then give up."""
    deadline = time.monotonic() + timeout
    while not hold.is_set() and time.monotonic() < deadline:
        await asyncio.sleep(0.02)


def _block_until(hold: threading.Event, *, timeout: float = 5.0) -> None:
    """Block one isolated inspect until the test releases it, or give up."""
    hold.wait(timeout=timeout)


def _wait_flag(flag: threading.Event, *, timeout: float = 5.0) -> None:
    """Wait for a worker-thread flag without spinning on the caller."""
    assert flag.wait(timeout=timeout)


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


def test_isolation_workers_are_daemon_threads() -> None:
    """Isolation workers must not keep the interpreter alive after shutdown."""

    async def scenario() -> None:
        """Inspect on a worker and assert the running thread is daemon."""
        bound = BoundedInspect(max_inflight=1)
        observed: list[bool] = []

        async def inspect() -> int:
            """Record the isolation thread daemon flag."""
            observed.append(threading.current_thread().daemon)
            return 1

        assert await bound.run(inspect, 1.0, adapter_id="daemon") == 1
        assert observed == [True]
        workers = [
            thread
            for thread in threading.enumerate()
            if thread.name.startswith("exp-guardrail-isolate")
        ]
        assert workers
        assert all(thread.daemon for thread in workers)

    asyncio.run(scenario())


def _delay_isolation_worker_start(
    monkeypatch: pytest.MonkeyPatch,
    release_start: threading.Event,
    started_thread: threading.Event,
) -> None:
    """Hold isolation-loop startup until ``release_start`` is set."""
    original_run_forever = _IsolationWorker._run_forever

    def delayed_run_forever(self: _IsolationWorker) -> None:
        """Wait for the test gate, then own the isolation loop."""
        started_thread.set()
        if not release_start.wait(timeout=5.0):
            self._signal_started(RuntimeError("isolation worker start was not released"))
            return
        original_run_forever(self)

    monkeypatch.setattr(_IsolationWorker, "_run_forever", delayed_run_forever)


async def _timeout_while_isolation_start_is_held(bound: BoundedInspect) -> int:
    """Time out one inspect and count caller-loop ticks during the wait."""
    ticks = 0

    async def ticker() -> None:
        """Advance while isolation-worker startup is blocked."""
        nonlocal ticks
        while True:
            ticks += 1
            await asyncio.sleep(0.01)

    async def inspect() -> int:
        """Return immediately if a worker is ever admitted."""
        return 1

    tick_task = asyncio.create_task(ticker())
    try:
        started = asyncio.get_running_loop().time()
        with pytest.raises(ClassifierTimeoutError):
            await bound.run(inspect, 0.08, adapter_id="start")
        assert asyncio.get_running_loop().time() - started < 0.5
        return ticks
    finally:
        tick_task.cancel()
        await asyncio.gather(tick_task, return_exceptions=True)


def test_delayed_isolation_worker_start_does_not_block_the_caller_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A slow isolation-worker start must not stall the Python gateway loop."""
    release_start = threading.Event()
    started_thread = threading.Event()
    _delay_isolation_worker_start(monkeypatch, release_start, started_thread)

    async def scenario() -> None:
        """Time out startup, then reuse the late-started worker."""
        bound = BoundedInspect(max_inflight=1)

        async def inspect() -> int:
            """Return a constant once a worker is running."""
            return 7

        try:
            ticks = await _timeout_while_isolation_start_is_held(bound)
            assert ticks >= 2
            _wait_flag(started_thread)
            assert bound.isolation_worker_count() == 1
            assert bound.quarantined_adapter_ids() == frozenset()
        finally:
            release_start.set()
        assert await bound.run(inspect, 1.0, adapter_id="later") == 7
        assert bound.isolation_worker_count() == 1

    try:
        asyncio.run(scenario())
    finally:
        release_start.set()


def test_delayed_isolation_worker_start_does_not_block_the_native_callback_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A slow isolation-worker start must not stall native guardrail callbacks."""
    release_start = threading.Event()
    started_thread = threading.Event()
    _delay_isolation_worker_start(monkeypatch, release_start, started_thread)
    bound = BoundedInspect(max_inflight=1)

    async def timed_out_start() -> int:
        """Run the delayed-start timeout on the native callback loop."""
        return await _timeout_while_isolation_start_is_held(bound)

    async def later() -> int:
        """Use the parked worker after startup is released."""

        async def inspect() -> int:
            """Return a constant once a worker is running."""
            return 9

        return await bound.run(inspect, 1.0, adapter_id="later")

    try:
        ticks = run_on_native_loop(timed_out_start())
        assert ticks >= 2
        _wait_flag(started_thread)
        assert bound.isolation_worker_count() == 1
        assert bound.quarantined_adapter_ids() == frozenset()
    finally:
        release_start.set()
    assert run_on_native_loop(later()) == 9
    assert bound.isolation_worker_count() == 1


def test_bounded_inspect_cancels_a_hung_inspect_and_releases_the_slot() -> None:
    """A hung inspect times out, then a later inspect can take the same slot."""

    async def scenario() -> None:
        """Fill one slot with a never-returning inspect, then reuse it."""
        bound = BoundedInspect(max_inflight=1)
        entered = threading.Event()

        async def hang() -> int:
            """Wait forever after marking entry."""
            entered.set()
            await asyncio.Event().wait()
            return 1

        started = asyncio.get_running_loop().time()
        with pytest.raises(ClassifierTimeoutError):
            await bound.run(hang, 0.05, adapter_id="hung")
        assert asyncio.get_running_loop().time() - started < 1.0
        _wait_flag(entered)
        await _wait_until(lambda: bound.detached_inspect_count() == 0)

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
            await _wait_until(lambda: bound.detached_inspect_count() == 0)

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
        bound = BoundedInspect(max_inflight=2)
        hold = threading.Event()

        async def swallow() -> int:
            """Ignore cancellation and wait until the test releases the hold."""
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                await _wait_hold(hold)
                return 1
            return 0

        try:
            started = asyncio.get_running_loop().time()
            with pytest.raises(ClassifierTimeoutError):
                await bound.run(swallow, 0.05, adapter_id="rogue")
            assert asyncio.get_running_loop().time() - started < 0.5
            await _wait_until(lambda: bound.detached_inspect_count() == 1)
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
        finally:
            hold.set()
            await _wait_until(lambda: bound.detached_inspect_count() == 0)

    asyncio.run(scenario())


def test_concurrent_timeouts_keep_quarantine_until_every_detached_task_finishes() -> None:
    """One finished rogue inspect cannot lift quarantine while another is still live."""

    async def scenario() -> None:
        """Time out two inspects on one adapter, then release them one at a time."""
        bound = BoundedInspect(max_inflight=2)
        holds: list[threading.Event] = []

        async def swallow() -> int:
            """Ignore cancellation and wait on a per-inspect hold."""
            hold = threading.Event()
            holds.append(hold)
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                await _wait_hold(hold)
                return 1
            return 0

        async def rogue() -> None:
            """Run one cancel-swallowing inspect past its timeout."""
            with pytest.raises(ClassifierTimeoutError):
                await bound.run(swallow, 0.08, adapter_id="rogue")

        try:
            await asyncio.gather(rogue(), rogue())
            await _wait_until(lambda: len(holds) == 2)
            await _wait_until(lambda: bound.detached_inspect_count() == 2)
            assert bound.quarantined_adapter_ids() == frozenset({"rogue"})

            holds[0].set()
            await _wait_until(lambda: bound.detached_inspect_count() == 1)
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
        finally:
            for hold in holds:
                hold.set()
            await _wait_until(lambda: bound.detached_inspect_count() == 0)

    asyncio.run(scenario())


def test_external_cancellation_propagates_and_releases_the_slot() -> None:
    """Caller cancellation is not converted into a classifier timeout."""

    async def scenario() -> None:
        """Cancel the waiting run, then reuse the slot on another adapter."""
        bound = BoundedInspect(max_inflight=1)
        entered = threading.Event()

        async def hang() -> int:
            """Wait until cancelled."""
            entered.set()
            await asyncio.Event().wait()
            return 1

        task = asyncio.create_task(bound.run(hang, 5.0, adapter_id="hung"))
        await asyncio.to_thread(entered.wait, 1.0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await _wait_until(lambda: bound.detached_inspect_count() == 0)

        async def healthy() -> int:
            """Return immediately."""
            return 2

        assert await bound.run(healthy, 0.5, adapter_id="healthy") == 2

    asyncio.run(scenario())


def test_delayed_cancel_does_not_cancel_the_next_inspect_on_a_reused_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A late cancel for one inspect cannot hit the next inspect on the same worker."""

    async def scenario() -> None:
        """Complete and reuse a worker before the first inspect's cancel runs."""
        bound = BoundedInspect(max_inflight=1)
        first_entered = threading.Event()
        first_hold = threading.Event()
        second_entered = threading.Event()
        second_hold = threading.Event()
        delay_cancel = threading.Event()
        original_request_cancel = _IsolationWorker.request_cancel

        def delayed_request_cancel(self: _IsolationWorker, generation: int) -> None:
            """Hold cancellation until the test reuses the worker."""

            def enqueue() -> None:
                """Forward the original cancel after the reuse gate opens."""
                if not delay_cancel.wait(timeout=5.0):
                    return
                original_request_cancel(self, generation)

            threading.Thread(
                target=enqueue,
                name="exp-test-delay-cancel",
                daemon=True,
            ).start()

        monkeypatch.setattr(_IsolationWorker, "request_cancel", delayed_request_cancel)

        async def first() -> int:
            """Finish after timeout without waiting for cancellation."""
            first_entered.set()
            await _wait_hold(first_hold)
            return 1

        async def second() -> int:
            """Stay live while the delayed first-inspect cancel is released."""
            second_entered.set()
            await _wait_hold(second_hold)
            return 2

        second_task: asyncio.Task[int] | None = None
        try:
            with pytest.raises(ClassifierTimeoutError):
                await bound.run(first, 0.08, adapter_id="first")
            _wait_flag(first_entered)
            first_hold.set()
            await _wait_until(lambda: bound.detached_inspect_count() == 0)
            second_task = asyncio.create_task(bound.run(second, 2.0, adapter_id="second"))
            assert await asyncio.to_thread(second_entered.wait, 1.0)
            delay_cancel.set()
            await asyncio.sleep(0.05)
            second_hold.set()
            assert await second_task == 2
            assert bound.isolation_worker_count() == 1
        finally:
            first_hold.set()
            second_hold.set()
            delay_cancel.set()
            if second_task is not None and not second_task.done():
                second_task.cancel()
                await asyncio.gather(second_task, return_exceptions=True)
            await _wait_until(lambda: bound.detached_inspect_count() == 0)

    asyncio.run(scenario())


def test_blocking_before_first_await_times_out_without_freezing_the_caller_loop() -> None:
    """Synchronous work before the first await cannot stall timeout enforcement."""

    async def scenario() -> None:
        """Time out a blocking inspect while another task on the caller still runs."""
        bound = BoundedInspect(max_inflight=1)
        entered = threading.Event()
        hold = threading.Event()

        async def block() -> int:
            """Block the isolation worker before yielding."""
            entered.set()
            _block_until(hold)
            return 1

        progressed = False

        async def marker() -> None:
            """Flip after a delay shorter than the inspect hang."""
            nonlocal progressed
            await asyncio.sleep(0.02)
            progressed = True

        marker_task = asyncio.create_task(marker())
        try:
            started = asyncio.get_running_loop().time()
            with pytest.raises(ClassifierTimeoutError):
                await bound.run(block, 0.08, adapter_id="blocked")
            assert asyncio.get_running_loop().time() - started < 0.5
            await marker_task
            assert progressed
            _wait_flag(entered)
            assert bound.detached_inspect_count() == 1
            assert bound.isolation_worker_count() == 1
            assert bound.quarantined_adapter_ids() == frozenset({"blocked"})
        finally:
            hold.set()
            await _wait_until(lambda: bound.detached_inspect_count() == 0)

    asyncio.run(scenario())


def test_repeated_blocking_before_await_does_not_start_another_worker() -> None:
    """A blocked adapter is quarantined instead of accumulating isolation workers."""

    async def scenario() -> None:
        """Time out one blocked inspect, then fail closed on the retry."""
        bound = BoundedInspect(max_inflight=2)
        entered = threading.Event()
        hold = threading.Event()
        calls = 0

        async def block() -> int:
            """Block before yielding and count admissions."""
            nonlocal calls
            calls += 1
            entered.set()
            _block_until(hold)
            return 1

        try:
            with pytest.raises(ClassifierTimeoutError):
                await bound.run(block, 0.08, adapter_id="blocked")
            _wait_flag(entered)
            assert calls == 1
            assert bound.isolation_worker_count() == 1

            started = asyncio.get_running_loop().time()
            with pytest.raises(ClassifierTimeoutError, match="quarantined"):
                await bound.run(block, 0.5, adapter_id="blocked")
            assert asyncio.get_running_loop().time() - started < 0.2
            assert calls == 1
            assert bound.isolation_worker_count() == 1
            hold.set()
            await _wait_until(lambda: bound.detached_inspect_count() == 0)
            assert await bound.run(block, 0.5, adapter_id="blocked") == 1
            assert calls == 2
        finally:
            hold.set()
            deadline = asyncio.get_running_loop().time() + 1.0
            while bound.detached_inspect_count() != 0:
                if asyncio.get_running_loop().time() >= deadline:
                    break
                await asyncio.sleep(0.01)

    asyncio.run(scenario())


def test_concurrent_blocking_before_await_inspects_time_out_independently() -> None:
    """Two blocked adapters each occupy one worker and leave a third free."""

    async def scenario() -> None:
        """Time out two blocked inspects, then succeed on a healthy adapter."""
        bound = BoundedInspect(max_inflight=3)
        holds = (threading.Event(), threading.Event())

        async def block_one() -> int:
            """Block the first isolation worker."""
            holds[0].wait(timeout=5.0)
            return 1

        async def block_two() -> int:
            """Block the second isolation worker."""
            holds[1].wait(timeout=5.0)
            return 2

        async def rogue(adapter_id: str) -> None:
            """Run one blocked inspect past its timeout."""
            factory = block_one if adapter_id == "one" else block_two
            with pytest.raises(ClassifierTimeoutError):
                await bound.run(factory, 0.08, adapter_id=adapter_id)

        try:
            await asyncio.gather(rogue("one"), rogue("two"))
            await _wait_until(lambda: bound.detached_inspect_count() == 2)
            assert bound.isolation_worker_count() == 2
            assert bound.quarantined_adapter_ids() == frozenset({"one", "two"})

            async def healthy() -> int:
                """Return immediately on the remaining worker."""
                return 9

            started = asyncio.get_running_loop().time()
            assert await bound.run(healthy, 0.5, adapter_id="healthy") == 9
            assert asyncio.get_running_loop().time() - started < 0.2
        finally:
            holds[0].set()
            holds[1].set()
            await _wait_until(lambda: bound.detached_inspect_count() == 0)

    asyncio.run(scenario())


def test_blocking_before_await_exhausts_isolation_capacity() -> None:
    """A full isolation pool fails closed without starting another worker."""

    async def scenario() -> None:
        """Fill both workers with blocked inspects, then refuse a third."""
        bound = BoundedInspect(max_inflight=2)
        hold = threading.Event()
        entered = threading.Semaphore(0)

        async def block() -> int:
            """Block after announcing that this inspect occupies a worker."""
            entered.release()
            _block_until(hold)
            return 1

        async def rogue(adapter_id: str) -> None:
            """Occupy one isolation worker past its timeout."""
            with pytest.raises(ClassifierTimeoutError):
                await bound.run(block, 0.2, adapter_id=adapter_id)

        first = asyncio.create_task(rogue("one"))
        second = asyncio.create_task(rogue("two"))
        try:
            assert await asyncio.to_thread(entered.acquire, True, 1.0)
            assert await asyncio.to_thread(entered.acquire, True, 1.0)
            await _wait_until(lambda: bound.isolation_worker_count() == 2)
            started_wait = asyncio.get_running_loop().time()
            with pytest.raises(ClassifierTimeoutError):
                await bound.run(block, 0.08, adapter_id="three")
            assert asyncio.get_running_loop().time() - started_wait < 0.4
            assert bound.isolation_worker_count() == 2
            await first
            await second
        finally:
            hold.set()
            for task in (first, second):
                if not task.done():
                    task.cancel()
            await asyncio.gather(first, second, return_exceptions=True)
            await _wait_until(lambda: bound.detached_inspect_count() == 0)

    asyncio.run(scenario())


def test_http_json_reuses_the_client_bound_to_the_isolation_loop() -> None:
    """Keep-alive reuse stays on the worker loop that runs the inspect."""

    async def scenario() -> None:
        """Run two sequential inspects on one worker and compare client identity."""
        bound = BoundedInspect(max_inflight=1)
        clients: list[httpx.AsyncClient] = []

        async def inspect() -> int:
            """Capture the loop-local client used by this isolated inspect."""
            clients.append(shared_http_json_client())
            return 1

        assert await bound.run(inspect, 1.0, adapter_id="http") == 1
        assert await bound.run(inspect, 1.0, adapter_id="http") == 1
        assert len(clients) == 2
        assert clients[0] is clients[1]
        assert clients[0] is not shared_http_json_client()

    asyncio.run(scenario())


def test_fresh_native_runner_shares_one_loop_across_concurrent_first_calls() -> None:
    """Concurrent first callers start exactly one daemon loop and one thread."""
    runner = _NativeCallbackRunner()
    workers = 8
    barrier = threading.Barrier(workers)
    loop_ids: list[int] = []
    lock = threading.Lock()

    def worker() -> None:
        """Submit one inspect as soon as every caller is ready."""
        barrier.wait(timeout=2.0)

        async def inspect() -> int:
            """Return the running loop identity."""
            return id(asyncio.get_running_loop())

        loop_id = runner.submit(inspect())
        with lock:
            loop_ids.append(loop_id)

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(workers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(2.0)
        assert not thread.is_alive()

    assert len(loop_ids) == workers
    assert loop_ids == [loop_ids[0]] * workers
    assert runner._start_count == 1
    assert runner._thread is not None
    assert runner._thread.is_alive()


def test_native_runner_retries_after_a_dead_startup() -> None:
    """A later caller can start the loop after the first thread dies."""

    class _FailOnce(_NativeCallbackRunner):
        """Die on the first start, then run the normal daemon loop."""

        def __init__(self) -> None:
            """Use a short ready wait so the failed start does not stall."""
            super().__init__(start_timeout=0.2)
            self._attempts = 0

        def _run_forever(self) -> None:
            """Return immediately once, then own a loop."""
            self._attempts += 1
            if self._attempts == 1:
                return
            super()._run_forever()

    runner = _FailOnce()
    with pytest.raises(RuntimeError, match="failed to start"):
        runner.loop()
    assert runner._thread is None

    async def inspect() -> int:
        """Return a constant."""
        return 1

    assert runner.submit(inspect()) == 1
    assert runner._attempts == 2
    assert runner._start_count == 2
    assert runner._thread is not None
    assert runner._thread.is_alive()


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
    bound = BoundedInspect(max_inflight=2)
    hold = threading.Event()
    entries = 0

    async def swallow() -> int:
        """Ignore cancellation and wait until teardown releases the hold."""
        nonlocal entries
        entries += 1
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await _wait_hold(hold)
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
        _wait_sync(lambda: bound.detached_inspect_count() == 1)
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
        _wait_sync(lambda: bound.detached_inspect_count() == 0)


def test_native_callback_returns_when_inspect_blocks_before_first_await() -> None:
    """A native worker returns at the timeout even if the inspect never yields."""
    bound = BoundedInspect(max_inflight=1)
    hold = threading.Event()
    entered = threading.Event()

    async def block() -> int:
        """Block the isolation worker before the first await."""
        entered.set()
        _block_until(hold)
        return 1

    async def rogue() -> int:
        """Run one blocking inspect through the native callback loop."""
        return await bound.run(block, 0.08, adapter_id="blocked")

    try:
        started = time.monotonic()
        with pytest.raises(ClassifierTimeoutError):
            run_on_native_loop(rogue())
        assert time.monotonic() - started < 0.5
        assert entered.wait(timeout=5.0)
        assert bound.detached_inspect_count() == 1
        assert bound.isolation_worker_count() == 1
    finally:
        hold.set()
        _wait_sync(lambda: bound.detached_inspect_count() == 0)
