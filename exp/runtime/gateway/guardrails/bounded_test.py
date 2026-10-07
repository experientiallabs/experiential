"""Bounded async classifier execution tests."""

from __future__ import annotations

import asyncio
import socket
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

from exp.runtime.gateway.contracts import GatewayApiSurface, GatewayMessage, GatewayRequest
from exp.runtime.gateway.guardrails.bounded import (
    BoundedInspect,
    ClassifierTimeoutError,
    _IsolationWorker,
    _NativeCallbackRunner,
    run_on_native_loop,
)
from exp.runtime.gateway.guardrails.classifiers import ClassifierRegistry, ScriptedClassifier
from exp.runtime.gateway.guardrails.client import DirectClassifierClient
from exp.runtime.gateway.guardrails.contracts import (
    ClassifierVerdict,
    GuardrailAction,
    GuardrailCapabilityKind,
    GuardrailCheck,
    GuardrailCheckStage,
    GuardrailPolicy,
)
from exp.runtime.gateway.guardrails.enforcement import GuardrailEngine
from exp.runtime.gateway.guardrails.http_json import (
    HttpJsonClassifier,
    _CookieFreeTransport,
    close_shared_http_json_client,
    shared_http_json_client,
)
from exp.runtime.gateway.guardrails.store import MappingGuardrailStore
from exp.runtime.gateway.guardrails.subjects import observation_subject_bytes


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


class _ShutdownClassifier(ScriptedClassifier):
    """Expose the actual worker thread and optionally hold cancellation-resistant work."""

    def __init__(self, release: threading.Event | None = None) -> None:
        """Create a thread receipt and an optional synchronous inspection barrier."""
        super().__init__()
        self.release = release
        self.started = threading.Event()
        self.threads: list[threading.Thread] = []

    async def inspect_input(
        self, *, request: GatewayRequest, check: GuardrailCheck
    ) -> ClassifierVerdict:
        """Record this worker and finish only after an optional blocking barrier releases."""
        del request, check
        self.threads.append(threading.current_thread())
        self.started.set()
        if self.release is not None:
            assert self.release.wait(5)
        return ClassifierVerdict(flagged=False)


def _shutdown_engine(
    classifier: _ShutdownClassifier, inspects: BoundedInspect | None = None
) -> tuple[GuardrailEngine, GuardrailPolicy, GatewayRequest]:
    """Compose the real engine's two worker pools around a synthetic classifier."""
    policy = GuardrailPolicy(
        policy_id="shutdown-observer",
        mode="observe",
        protected=True,
        checks=(
            GuardrailCheck(
                check_id="input",
                adapter_id="fixture",
                capability=GuardrailCapabilityKind.CONTENT_SAFETY,
                stage=GuardrailCheckStage.INPUT,
                action=GuardrailAction.BLOCK,
                timeout_ms=5000,
            ),
        ),
    )
    engine = GuardrailEngine(
        store=MappingGuardrailStore((policy,)),
        client=DirectClassifierClient(ClassifierRegistry({"fixture": classifier})),
        monotonic=time.monotonic,
        inspects=inspects,
        max_observations=1,
    )
    request = GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(GatewayMessage(role="user", content="synthetic shutdown probe"),),
    )
    return engine, policy, request


def test_repeated_engine_close_stops_owned_observation_and_enforcement_workers() -> None:
    """Each closed SDK engine releases its idle pool threads instead of accumulating them."""
    workers: set[threading.Thread] = set()
    for _ in range(3):
        classifier = _ShutdownClassifier()
        engine, policy, request = _shutdown_engine(classifier)
        try:
            engine.observe_input(
                policy=policy, request=request, deadline_monotonic=time.monotonic() + 5
            )
            assert classifier.started.wait(2)
            asyncio.run(
                engine.enforce_input(
                    policy=policy.model_copy(update={"mode": "enforce"}),
                    request=request,
                    deadline_monotonic=time.monotonic() + 5,
                )
            )
        finally:
            engine.close(timeout_seconds=1)
        workers.update(classifier.threads)
        assert len(set(classifier.threads)) == 2
        assert all(not worker.is_alive() for worker in workers)


def test_engine_close_preserves_a_caller_owned_enforcement_pool() -> None:
    """An injected pool remains usable after the engine closes its own observer pool."""
    shared = BoundedInspect(max_inflight=1)
    classifier = _ShutdownClassifier()
    engine, policy, request = _shutdown_engine(classifier, shared)

    async def enforce() -> GatewayRequest:
        """Use the supplied pool through the real engine before shutting down the host."""
        return await engine.enforce_input(
            policy=policy.model_copy(update={"mode": "enforce"}),
            request=request,
            deadline_monotonic=time.monotonic() + 5,
        )

    async def reuse() -> int:
        """Prove the caller still owns pool admission and its running worker."""

        async def inspect() -> int:
            """Return on the caller's reusable pool."""
            return 7

        return await shared.run(inspect, 1, adapter_id="independent-owner")

    try:
        assert asyncio.run(enforce()) == request
        engine.close(timeout_seconds=1)
        assert asyncio.run(reuse()) == 7
        assert classifier.threads[0].is_alive()
    finally:
        engine.close(timeout_seconds=1)
        shared.close(timeout_seconds=1)
    assert not classifier.threads[0].is_alive()


def test_pool_shutdown_preserves_stubborn_observation_ownership_until_actual_exit() -> None:
    """A bounded close cannot release bytes or destroy a classifier still using its subject."""
    release = threading.Event()
    classifier = _ShutdownClassifier(release)
    engine, policy, request = _shutdown_engine(classifier)
    try:
        engine.observe_input(
            policy=policy, request=request, deadline_monotonic=time.monotonic() + 5
        )
        assert classifier.started.wait(2)
        started = time.monotonic()
        engine.close(timeout_seconds=0.02)
        assert time.monotonic() - started < 0.5
        assert classifier.threads[0].is_alive()
        assert engine._observations._bytes == len(observation_subject_bytes(request))
        assert len(engine._observations._jobs) == 1

        async def refused() -> None:
            """A closed pool must not invoke even a different, nonquarantined adapter."""

            async def inspect() -> None:
                """Make accidental admission a visible failure."""
                pytest.fail("closed observation pool invoked a classifier")

            with pytest.raises(RuntimeError, match="closed"):
                await engine._observation_inspects.run(inspect, 1, adapter_id="later")

        asyncio.run(refused())
        release.set()
        _wait_sync(lambda: not classifier.threads[0].is_alive())
        assert engine._observations._idle.wait(2)
        assert engine._observations._bytes == 0
    finally:
        release.set()
        engine.close(timeout_seconds=2)


def test_pool_close_wakes_queued_acquisition_and_refuses_new_admission() -> None:
    """Shutdown wakes pending admission without interrupting a still-running inspect."""
    release = threading.Event()
    entered = threading.Event()
    workers: list[threading.Thread] = []
    bound = BoundedInspect(max_inflight=1)

    async def scenario() -> None:
        """Close a full pool while another adapter is waiting for its sole worker."""

        async def blocked() -> int:
            """Keep the worker busy until the test releases it."""
            workers.append(threading.current_thread())
            entered.set()
            assert release.wait(5)
            return 3

        async def unexpected() -> int:
            """Fail if queued or new work is admitted during shutdown."""
            pytest.fail("closed pool admitted another classifier")

        active = asyncio.create_task(bound.run(blocked, 5, adapter_id="active"))
        await _wait_until(entered.is_set)
        waiting = asyncio.create_task(bound.run(unexpected, 5, adapter_id="queued"))
        await _wait_until(lambda: bool(bound._pool._waiters))
        started = time.monotonic()
        bound.close(timeout_seconds=0.02)
        assert time.monotonic() - started < 0.5
        with pytest.raises(RuntimeError, match="closed"):
            await waiting
        with pytest.raises(RuntimeError, match="closed"):
            await bound.run(unexpected, 1, adapter_id="new")
        assert not active.done()
        release.set()
        assert await active == 3
        await _wait_until(lambda: not workers[0].is_alive())

    try:
        asyncio.run(scenario())
    finally:
        release.set()
        bound.close(timeout_seconds=2)


def test_pool_close_stops_a_worker_whose_startup_finishes_after_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A timed-out reserved worker cannot become a parked thread after its owner closes."""
    release_start = threading.Event()
    entered = threading.Event()
    _delay_isolation_worker_start(monkeypatch, release_start, entered)
    before = set(threading.enumerate())
    bound = BoundedInspect(max_inflight=1)
    try:
        asyncio.run(_timeout_while_isolation_start_is_held(bound))
        assert entered.wait(2)
        workers = [
            thread
            for thread in threading.enumerate()
            if thread not in before and thread.name == "exp-guardrail-isolate"
        ]
        assert len(workers) == 1
        bound.close(timeout_seconds=0.01)
        assert workers[0].is_alive()
        release_start.set()
        _wait_sync(lambda: not workers[0].is_alive())
    finally:
        release_start.set()
        bound.close(timeout_seconds=2)


@contextmanager
def _keepalive_classifier() -> Iterator[tuple[str, list[threading.Event]]]:
    """Serve real loopback HTTP and expose peer disconnects without retaining request data."""
    connections: list[socket.socket] = []
    disconnected: list[threading.Event] = []

    class Handler(BaseHTTPRequestHandler):
        """Return an allow verdict over a persistent HTTP/1.1 connection."""

        protocol_version = "HTTP/1.1"

        def setup(self) -> None:
            """Track the socket so a failing regression can still clean it up."""
            super().setup()
            self.disconnected = threading.Event()
            connections.append(self.connection)
            disconnected.append(self.disconnected)

        def do_POST(self) -> None:  # noqa: N802 - stdlib HTTP handler contract
            """Consume one synthetic request and send a bounded classifier verdict."""
            self.rfile.read(int(self.headers["Content-Length"]))
            body = b'{"flagged":false}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def finish(self) -> None:
            """Signal when the client actually closes its keepalive connection."""
            try:
                super().finish()
            finally:
                self.disconnected.set()

        def log_message(self, format: str, *args: object) -> None:
            """Keep the synthetic HTTP fixture silent."""
            del format, args

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(
            target=lambda: server.serve_forever(poll_interval=0.01), daemon=True
        )
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}/inspect", disconnected
        finally:
            server.shutdown()
            for connection in connections:
                try:
                    connection.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                connection.close()
            thread.join(2)
            assert not thread.is_alive()
            assert all(event.wait(2) for event in disconnected)


def test_repeated_pool_close_releases_real_http_keepalive_connections() -> None:
    """Closed owners must close pooled sockets on their loop before that loop disappears."""
    request = GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(GatewayMessage(role="user", content="synthetic HTTP shutdown probe"),),
    )
    check = GuardrailCheck(
        check_id="http-shutdown",
        adapter_id="http",
        capability=GuardrailCapabilityKind.CONTENT_SAFETY,
        stage=GuardrailCheckStage.INPUT,
        action=GuardrailAction.BLOCK,
        timeout_ms=2000,
    )
    clients: list[httpx.AsyncClient] = []
    workers: list[threading.Thread] = []
    with _keepalive_classifier() as (url, disconnected):
        classifier = HttpJsonClassifier(adapter_id="http", url=url)
        for count in range(1, 4):
            bound = BoundedInspect(max_inflight=1)

            async def inspect() -> ClassifierVerdict:
                """Use the actual adapter and retain receipts beyond worker-loop teardown."""
                clients.append(shared_http_json_client())
                workers.append(threading.current_thread())
                return await classifier.inspect_input(request=request, check=check)

            try:
                assert not asyncio.run(bound.run(inspect, 2, adapter_id="http")).flagged
                assert len(disconnected) == count
                assert not disconnected[-1].is_set()
                bound.close(timeout_seconds=1)
                assert clients[-1].is_closed
                assert not workers[-1].is_alive()
                assert disconnected[-1].wait(1)
            finally:
                bound.close(timeout_seconds=1)
        assert len({id(client) for client in clients}) == 3
        assert all(client.is_closed for client in clients)


def test_pool_close_waits_for_stubborn_inspect_before_closing_its_http_client() -> None:
    """An abandoned inspect keeps its usable loop-local client until the actual work exits."""
    bound = BoundedInspect(max_inflight=1)
    release = threading.Event()
    entered = threading.Event()
    clients: list[httpx.AsyncClient] = []
    workers: list[threading.Thread] = []

    async def inspect() -> None:
        """Keep using the worker after the caller times out and requests shutdown."""
        clients.append(shared_http_json_client())
        workers.append(threading.current_thread())
        entered.set()
        assert release.wait(5)
        assert not clients[-1].is_closed

    async def scenario() -> None:
        """Detach the blocking work before closing its owning pool."""
        active = asyncio.create_task(bound.run(inspect, 0.2, adapter_id="stubborn-http"))
        await _wait_until(entered.is_set)
        with pytest.raises(ClassifierTimeoutError):
            await active
        started = time.monotonic()
        bound.close(timeout_seconds=0.02)
        assert time.monotonic() - started < 0.5
        assert workers[0].is_alive()
        assert not clients[0].is_closed
        assert bound.detached_inspect_count() == 1
        release.set()
        await _wait_until(lambda: not workers[0].is_alive())
        assert clients[0].is_closed
        assert bound.detached_inspect_count() == 0

    try:
        asyncio.run(scenario())
    finally:
        release.set()
        bound.close(timeout_seconds=2)


def test_slow_http_cleanup_keeps_its_owner_without_extending_close_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A slow transport close continues on its owned daemon and never reopens admission."""
    release = threading.Event()
    closing = threading.Event()
    closed = threading.Event()
    workers: list[threading.Thread] = []
    original_close = _CookieFreeTransport.aclose
    bound = BoundedInspect(max_inflight=1)

    async def delayed_close(transport: _CookieFreeTransport) -> None:
        """Hold transport cleanup until the caller's finite join budget has expired."""
        closing.set()
        await _wait_hold(release)
        await original_close(transport)
        closed.set()

    async def inspect() -> None:
        """Create one client whose transport belongs only to this worker."""
        workers.append(threading.current_thread())
        shared_http_json_client()

    async def refused() -> None:
        """Reject new inspection while cleanup still owns its closing loop."""
        with pytest.raises(RuntimeError, match="closed"):
            await bound.run(inspect, 1, adapter_id="later")

    monkeypatch.setattr(_CookieFreeTransport, "aclose", delayed_close)
    try:
        asyncio.run(bound.run(inspect, 1, adapter_id="http"))
        started = time.monotonic()
        bound.close(timeout_seconds=0.02)
        assert time.monotonic() - started < 0.5
        assert closing.wait(1)
        assert workers[0].is_alive()
        assert not closed.is_set()
        asyncio.run(refused())
        bound.close(timeout_seconds=0.02)
        assert workers[0].is_alive()
        assert not closed.is_set()
        release.set()
        bound.close(timeout_seconds=2)
        assert closed.is_set()
        assert not workers[0].is_alive()
    finally:
        release.set()
        bound.close(timeout_seconds=2)


def test_pool_close_preserves_other_loops_and_injected_http_clients() -> None:
    """One closing worker cannot close another owner or an explicitly injected transport."""
    first = BoundedInspect(max_inflight=1)
    second = BoundedInspect(max_inflight=1)
    clients: list[httpx.AsyncClient] = []

    async def capture() -> httpx.AsyncClient:
        """Return a strong receipt for this worker's loop-local client."""
        client = shared_http_json_client()
        clients.append(client)
        return client

    async def scenario() -> None:
        """Keep the other owners usable throughout the first pool's shutdown."""
        caller_client = shared_http_json_client()

        def allow(_request: httpx.Request) -> httpx.Response:
            """Return a synthetic response without owning a real connection pool."""
            return httpx.Response(200, json={"flagged": False})

        async with httpx.AsyncClient(transport=httpx.MockTransport(allow)) as injected:
            classifier = HttpJsonClassifier(
                adapter_id="injected",
                url="https://classifier.example.invalid/inspect",
                client=injected,
            )

            async def use_injected() -> ClassifierVerdict:
                """Use a caller-supplied client from the worker that will shut down."""
                return await classifier.inspect_input(
                    request=GatewayRequest(
                        surface=GatewayApiSurface.CHAT_COMPLETIONS,
                        messages=(GatewayMessage(role="user", content="synthetic owner probe"),),
                    ),
                    check=GuardrailCheck(
                        check_id="owner",
                        adapter_id="injected",
                        capability=GuardrailCapabilityKind.CONTENT_SAFETY,
                        stage=GuardrailCheckStage.INPUT,
                        action=GuardrailAction.BLOCK,
                        timeout_ms=1000,
                    ),
                )

            try:
                first_client = await first.run(capture, 1, adapter_id="first")
                second_client = await second.run(capture, 1, adapter_id="second")
                assert not (await first.run(use_injected, 1, adapter_id="injected")).flagged
                first.close(timeout_seconds=1)
                assert first_client.is_closed
                assert not second_client.is_closed
                assert not caller_client.is_closed
                assert not injected.is_closed
                assert await second.run(capture, 1, adapter_id="second") is second_client
                assert not (await use_injected()).flagged
            finally:
                first.close(timeout_seconds=1)
                second.close(timeout_seconds=1)
                await close_shared_http_json_client()
        assert injected.is_closed
        assert caller_client.is_closed
        assert all(client.is_closed for client in clients)

    asyncio.run(scenario())


def test_http_cleanup_failure_logs_no_exception_content_and_stops_its_worker(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Failed cleanup is visible without logging transport exceptions or retaining a dead loop."""
    bound = BoundedInspect(max_inflight=1)
    workers: list[threading.Thread] = []
    original_close = _CookieFreeTransport.aclose

    async def failed_close(transport: _CookieFreeTransport) -> None:
        """Close the real transport before raising an intentionally sensitive fake error."""
        await original_close(transport)
        raise RuntimeError("synthetic private payload or credential")

    async def inspect() -> None:
        """Create the client that exercises the worker's failing finalizer."""
        workers.append(threading.current_thread())
        shared_http_json_client()

    monkeypatch.setattr(_CookieFreeTransport, "aclose", failed_close)
    try:
        asyncio.run(bound.run(inspect, 1, adapter_id="http"))
        bound.close(timeout_seconds=1)
        assert not workers[0].is_alive()
        assert "guardrail HTTP client cleanup failed" in caplog.text
        assert "synthetic private" not in caplog.text
        assert all(record.exc_info is None for record in caplog.records)
    finally:
        bound.close(timeout_seconds=1)
