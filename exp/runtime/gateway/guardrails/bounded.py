"""Isolate classifier inspects from the caller's timeout-watching event loop."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections.abc import Awaitable, Callable, Coroutine
from concurrent.futures import CancelledError as FutureCancelledError
from concurrent.futures import Future
from typing import Protocol, cast

from exp.runtime.gateway.guardrails.http_json import close_shared_http_json_client

MAX_INFLIGHT_ASYNC_CLASSIFIER_CALLS = 32
_WORKER_START_TIMEOUT_SECONDS = 5.0

_logger = logging.getLogger(__name__)


class ClassifierTimeoutError(TimeoutError):
    """A classifier exceeded its per-check timeout, wait, or quarantine."""


class InspectionRetention(Protocol):
    """Retain a subject's bounded admission until the isolated invocation actually exits."""

    def retain(self) -> None:
        """Acquire one subject reference before submitting isolated work."""
        ...

    def release(self) -> None:
        """Release that reference only after the actual isolated work completes."""
        ...


def _absorb_abandoned(task: Future[object]) -> None:
    """Retrieve an abandoned inspect so its exception is not unhandled."""
    if not task.cancelled():
        task.exception()


class _IsolationWorker:
    """One daemon thread that owns a private event loop for isolated inspects."""

    def __init__(self) -> None:
        """Start the worker thread without waiting for its loop."""
        self._ready = threading.Event()
        self._start_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._closing = False
        self._busy = False
        self._start_error: BaseException | None = None
        self._started_callbacks: list[Callable[[BaseException | None], None]] = []
        self._loop: asyncio.AbstractEventLoop | None = None
        self._generation = 0
        self._active_generation = 0
        self._pending_cancels: set[int] = set()
        self._task: asyncio.Task[object] | None = None
        self._thread = threading.Thread(
            target=self._run_forever,
            name="exp-guardrail-isolate",
            daemon=True,
        )
        self._thread.start()

    def submit[T](self, fn: Callable[[], Coroutine[object, object, T]]) -> tuple[Future[T], int]:
        """Schedule ``fn`` on this worker and return its future plus generation.

        Args:
            fn: Zero-argument coroutine factory for one inspect.

        Returns:
            A future that completes when the isolated inspect exits, and the
            generation that ``request_cancel`` must name.
        """
        loop = self._loop
        if loop is None:
            raise RuntimeError("classifier isolation loop is not running")
        result: Future[T] = Future()

        def start() -> None:
            """Create the inspect task on the worker loop."""
            if result.cancelled():
                self._pending_cancels.discard(generation)
                self._busy = False
                self._stop_if_idle()
                return
            try:
                task = loop.create_task(fn())
            except Exception as exc:  # noqa: BLE001 - factory errors must complete the waiter
                self._busy = False
                result.set_exception(exc)
                self._stop_if_idle()
                return
            self._task = task
            self._active_generation = generation
            if generation in self._pending_cancels:
                self._pending_cancels.discard(generation)
                task.cancel()

            def finish(done: asyncio.Task[T]) -> None:
                """Copy the inspect outcome onto the cross-thread future."""
                if self._task is done and self._active_generation == generation:
                    self._task = None
                    self._busy = False
                try:
                    if result.done():
                        return
                    if done.cancelled():
                        result.cancel()
                        return
                    error = done.exception()
                    if error is not None:
                        result.set_exception(error)
                        return
                    result.set_result(done.result())
                finally:
                    self._stop_if_idle()

            task.add_done_callback(finish)

        with self._state_lock:
            if self._closing:
                raise RuntimeError("classifier isolation worker is closed")
            self._generation += 1
            generation = self._generation
            self._busy = True
            try:
                loop.call_soon_threadsafe(start)
            except RuntimeError:
                self._busy = False
                raise
        return result, generation

    def stop_when_idle(self) -> None:
        """Reject submission and stop this loop only after its actual inspect finishes."""
        with self._state_lock:
            self._closing = True
            loop = self._loop
            if loop is not None:
                try:
                    loop.call_soon_threadsafe(self._stop_if_idle)
                except RuntimeError:
                    pass

    def _stop_if_idle(self) -> None:
        """Stop on the worker loop without destroying pending or cancellation-resistant work."""
        with self._state_lock:
            if self._closing and not self._busy and self._loop is not None:
                self._loop.stop()

    def wait_stopped(self, timeout: float) -> None:
        """Join within the caller's remaining budget, never joining this worker from itself."""
        if self._thread is not threading.current_thread():
            self._thread.join(timeout)

    def request_cancel(self, generation: int) -> None:
        """Queue cancellation for one submitted inspect without waiting.

        Args:
            generation: Inspect generation returned by ``submit``. A delayed
                callback is a no-op when this worker already started a later
                inspect.
        """
        loop = self._loop
        if loop is None:
            return

        def cancel() -> None:
            """Cancel only the inspect that still owns ``generation``."""
            if self._active_generation == generation:
                self._pending_cancels.discard(generation)
                task = self._task
                if task is not None and not task.done():
                    task.cancel()
                return
            if generation < self._active_generation:
                self._pending_cancels.discard(generation)
                return
            self._pending_cancels.add(generation)

        try:
            loop.call_soon_threadsafe(cancel)
        except RuntimeError:
            return

    def on_started(self, callback: Callable[[BaseException | None], None]) -> None:
        """Invoke ``callback`` once this worker has started or failed.

        Args:
            callback: Receives ``None`` on success or the startup error.
                Already-started workers invoke it immediately on this thread.
        """
        with self._start_lock:
            if not self._ready.is_set():
                self._started_callbacks.append(callback)
                return
            error = self._start_error
        callback(error)

    def attach_ready_waiter(self) -> asyncio.Future[None]:
        """Return a future that completes when this worker starts or fails.

        Returns:
            A future bound to the caller event loop. It is already done when
            the worker has started, so a zero-timeout wait can still succeed.
        """
        loop = asyncio.get_running_loop()
        future: asyncio.Future[None] = loop.create_future()

        def done(error: BaseException | None) -> None:
            """Complete ``future`` on the caller loop."""

            def complete() -> None:
                """Copy the startup outcome onto ``future``."""
                if future.done():
                    return
                if error is not None:
                    future.set_exception(error)
                    return
                future.set_result(None)

            try:
                if loop is asyncio.get_running_loop():
                    complete()
                    return
            except RuntimeError:
                pass
            try:
                loop.call_soon_threadsafe(complete)
            except RuntimeError:
                return

        self.on_started(done)
        return future

    def assert_running(self) -> None:
        """Raise when this worker's loop is not available for inspects.

        Raises:
            RuntimeError: Startup failed or the daemon loop is not running.
        """
        loop = self._loop
        if (
            self._start_error is not None
            or loop is None
            or not loop.is_running()
            or not self._thread.is_alive()
        ):
            raise RuntimeError("classifier isolation loop failed to start")

    def _announce_running(self) -> None:
        """Mark startup complete after the daemon loop is actually running."""
        self._signal_started(None)
        self._stop_if_idle()

    def _signal_started(self, error: BaseException | None) -> None:
        """Unblock waiters exactly once with ``error`` or success."""
        with self._start_lock:
            if self._ready.is_set():
                return
            self._start_error = error
            self._ready.set()
            callbacks = list(self._started_callbacks)
            self._started_callbacks.clear()
        for callback in callbacks:
            callback(error)

    def _run_forever(self) -> None:
        """Own one event loop for the life of this worker."""
        loop: asyncio.AbstractEventLoop | None = None
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._loop = loop
            loop.call_soon(self._announce_running)
            loop.run_forever()
        except BaseException as exc:  # noqa: BLE001 - startup failure must unblock waiters
            self._signal_started(exc)
            raise
        finally:
            if not self._ready.is_set():
                self._signal_started(RuntimeError("classifier isolation loop failed to start"))
            if loop is not None:
                with self._state_lock:
                    self._loop = None
                try:
                    loop.run_until_complete(close_shared_http_json_client())
                except Exception:  # noqa: BLE001 - cleanup failure cannot retain a dead loop
                    _logger.warning("guardrail HTTP client cleanup failed")
                finally:
                    loop.close()


class _IsolationPool:
    """Bounded set of isolation workers shared by every caller event loop."""

    def __init__(self, size: int) -> None:
        """Create an empty pool that will not grow past ``size``.

        Args:
            size: Maximum isolation workers this pool may start.
        """
        self._size = size
        self._lock = threading.Lock()
        self._idle: list[_IsolationWorker] = []
        self._created = 0
        self._waiters: list[asyncio.Future[_IsolationWorker | None]] = []
        self._workers: set[_IsolationWorker] = set()
        self._closed = False

    def close(self, timeout_seconds: float) -> None:
        """Stop admission and join owned workers within one finite shutdown budget."""
        if timeout_seconds < 0:
            raise ValueError("classifier shutdown timeout cannot be negative")
        deadline = time.monotonic() + timeout_seconds
        with self._lock:
            self._closed = True
            workers = tuple(self._workers)
            waiters, self._waiters = self._waiters, []
            self._idle.clear()
        for waiter in waiters:
            try:
                waiter.get_loop().call_soon_threadsafe(self._reject_closed, waiter)
            except RuntimeError:
                continue
        for worker in workers:
            worker.stop_when_idle()
        for worker in workers:
            worker.wait_stopped(max(0.0, deadline - time.monotonic()))

    @staticmethod
    def _reject_closed(waiter: asyncio.Future[_IsolationWorker | None]) -> None:
        """Reject pending admission on its caller's event loop."""
        if not waiter.done():
            waiter.set_exception(RuntimeError("classifier isolation pool is closed"))

    @property
    def worker_count(self) -> int:
        """Return how many isolation workers have been started."""
        with self._lock:
            return self._created

    async def acquire(self, timeout: float) -> _IsolationWorker | None:
        """Return a free worker, or ``None`` when ``timeout`` elapses first.

        Args:
            timeout: Seconds the caller can wait for a free worker.

        Returns:
            An idle or newly started worker, or ``None`` on timeout.

        Raises:
            asyncio.CancelledError: The caller task was cancelled while waiting.
            RuntimeError: A reserved isolation worker failed to start.
        """
        idle, reserved = self._claim()
        if idle is not None:
            return idle
        if reserved:
            return await self._start_reserved_worker(timeout)
        if timeout <= 0:
            return None
        loop = asyncio.get_running_loop()
        waiter: asyncio.Future[_IsolationWorker | None] = loop.create_future()
        with self._lock:
            if self._closed:
                raise RuntimeError("classifier isolation pool is closed")
            worker = self._take_idle()
            if worker is not None:
                return worker
            if self._created < self._size:
                self._created += 1
                reserved = True
            else:
                reserved = False
                self._waiters.append(waiter)
        if reserved:
            return await self._start_reserved_worker(timeout)
        handle = loop.call_later(timeout, self._expire, waiter)
        try:
            return await waiter
        except asyncio.CancelledError:
            self._drop_waiter(waiter)
            raise
        finally:
            handle.cancel()

    def release(self, worker: _IsolationWorker) -> None:
        """Return ``worker`` to a waiter or to the idle set.

        Args:
            worker: Isolation worker whose inspect has actually exited.
        """
        with self._lock:
            if not self._closed:
                while self._waiters:
                    waiter = self._waiters.pop(0)
                    if waiter.done():
                        continue
                    try:
                        waiter.get_loop().call_soon_threadsafe(self._deliver, waiter, worker)
                    except RuntimeError:
                        continue
                    return
                self._idle.append(worker)
                return
        worker.stop_when_idle()

    def _claim(self) -> tuple[_IsolationWorker | None, bool]:
        """Take an idle worker or reserve capacity to start one."""
        with self._lock:
            if self._closed:
                raise RuntimeError("classifier isolation pool is closed")
            worker = self._take_idle()
            if worker is not None:
                return worker, False
            if self._created >= self._size:
                return None, False
            self._created += 1
            return None, True

    def _take_idle(self) -> _IsolationWorker | None:
        """Pop one idle worker. The caller must hold ``_lock``."""
        if not self._idle:
            return None
        return self._idle.pop()

    async def _start_reserved_worker(self, timeout: float) -> _IsolationWorker | None:
        """Start a reserved worker without blocking the caller event loop.

        Args:
            timeout: Remaining seconds the caller can wait for startup.

        Returns:
            The running worker, or ``None`` when ``timeout`` elapses first.

        Raises:
            asyncio.CancelledError: The caller task was cancelled while waiting.
            RuntimeError: The reserved worker failed to start.
        """
        try:
            worker = _IsolationWorker()
        except BaseException:  # noqa: BLE001 - thread start failure must restore pool capacity
            with self._lock:
                self._created -= 1
            raise
        with self._lock:
            self._workers.add(worker)
            closed = self._closed
        if closed:
            worker.stop_when_idle()
            raise RuntimeError("classifier isolation pool is closed")
        ready = worker.attach_ready_waiter()
        start_wait = min(max(0.0, timeout), _WORKER_START_TIMEOUT_SECONDS)
        try:
            await asyncio.wait_for(asyncio.shield(ready), timeout=start_wait)
        except TimeoutError:
            return self._adopt_or_defer(worker, ready)
        except asyncio.CancelledError:
            taken = self._adopt_or_defer(worker, ready)
            if taken is not None:
                self.release(taken)
            raise
        except Exception:  # noqa: BLE001 - startup errors must restore pool capacity
            self._discard(worker)
            raise
        try:
            worker.assert_running()
        except RuntimeError:
            self._discard(worker)
            raise
        return worker

    def _discard(self, worker: _IsolationWorker) -> None:
        """Release a failed startup reservation and stop any surviving worker loop."""
        with self._lock:
            self._created -= 1
            self._workers.discard(worker)
        worker.stop_when_idle()

    def _adopt_or_defer(
        self,
        worker: _IsolationWorker,
        ready: asyncio.Future[None],
    ) -> _IsolationWorker | None:
        """Take ``worker`` if it is already up, otherwise park it when it is.

        Args:
            worker: Isolation worker whose startup is still in flight.
            ready: Caller-loop future attached to that startup.

        Returns:
            ``worker`` when it is already running, otherwise ``None``.
        """
        if ready.done():
            if ready.cancelled() or ready.exception() is not None:
                self._discard(worker)
                return None
            try:
                worker.assert_running()
            except RuntimeError:
                self._discard(worker)
                return None
            return worker

        def park(error: BaseException | None) -> None:
            """Release or drop the worker once startup finishes."""
            self._finish_late_start(worker, error)

        worker.on_started(park)
        return None

    def _finish_late_start(self, worker: _IsolationWorker, error: BaseException | None) -> None:
        """Park a worker that became ready after its acquirer timed out.

        Args:
            worker: Isolation worker whose startup finished late.
            error: Startup failure, or ``None`` on success.
        """
        if error is not None:
            self._discard(worker)
            return
        try:
            worker.assert_running()
        except RuntimeError:
            self._discard(worker)
            return
        self.release(worker)

    def _expire(self, waiter: asyncio.Future[_IsolationWorker | None]) -> None:
        """Complete a timed-out acquire with ``None``."""
        self._drop_waiter(waiter)
        if not waiter.done():
            waiter.set_result(None)

    def _drop_waiter(self, waiter: asyncio.Future[_IsolationWorker | None]) -> None:
        """Remove ``waiter`` from the FIFO if it is still queued."""
        with self._lock:
            self._waiters = [item for item in self._waiters if item is not waiter]

    def _deliver(
        self,
        waiter: asyncio.Future[_IsolationWorker | None],
        worker: _IsolationWorker,
    ) -> None:
        """Give ``worker`` to ``waiter`` on that waiter's event loop."""
        with self._lock:
            closed = self._closed
        if closed:
            self.release(worker)
            self._reject_closed(waiter)
            return
        if waiter.done():
            self.release(worker)
            return
        waiter.set_result(worker)


class BoundedInspect:
    """Run inspects off the caller loop so blocking work cannot freeze timeouts.

    Each inspect is admitted to a bounded isolation worker. The caller waits
    on a cross-thread future and returns when the tighter deadline elapses,
    even if the inspect blocks before its first await. The worker stays
    occupied until that isolated invocation actually exits. An adapter whose
    inspect was abandoned is quarantined until every abandoned invocation
    for it finishes. Further calls to that adapter fail immediately. Other
    adapters keep any remaining isolation workers.
    """

    def __init__(self, max_inflight: int = MAX_INFLIGHT_ASYNC_CLASSIFIER_CALLS) -> None:
        """Bind one isolation-worker cap shared by every caller event loop.

        Args:
            max_inflight: Maximum isolation workers this limiter may start.

        Raises:
            ValueError: ``max_inflight`` is not a positive integer.
        """
        if max_inflight < 1:
            raise ValueError("max_inflight must be a positive integer")
        self._pool = _IsolationPool(max_inflight)
        self._lock = threading.Lock()
        self._abandoned: dict[str, set[Future[object]]] = {}

    def close(self, *, timeout_seconds: float = 1.0) -> None:
        """Stop owned workers within a budget while live inspects keep their actual leases."""
        self._pool.close(timeout_seconds)

    def isolation_worker_count(self) -> int:
        """Return how many isolation workers this limiter has started."""
        return self._pool.worker_count

    def detached_inspect_count(self) -> int:
        """Return live abandoned inspects that still occupy isolation workers."""
        with self._lock:
            return sum(1 for tasks in self._abandoned.values() for task in tasks if not task.done())

    def quarantined_adapter_ids(self) -> frozenset[str]:
        """Return adapter identities with a live abandoned inspect."""
        with self._lock:
            return frozenset(
                adapter_id
                for adapter_id, tasks in self._abandoned.items()
                if any(not task.done() for task in tasks)
            )

    def _quarantined(self, adapter_id: str) -> bool:
        """Return whether ``adapter_id`` has a live abandoned inspect."""
        with self._lock:
            tasks = self._abandoned.get(adapter_id)
            return tasks is not None and any(not task.done() for task in tasks)

    def _track_abandoned(self, adapter_id: str, task: Future[object]) -> None:
        """Remember one abandoned inspect until it actually exits."""
        with self._lock:
            self._abandoned.setdefault(adapter_id, set()).add(task)
        _logger.info("guardrail adapter quarantined adapter_id=%s", adapter_id)

    def _finish_detached(self, adapter_id: str, task: Future[object]) -> None:
        """Absorb one abandoned inspect and lift quarantine when none remain."""
        _absorb_abandoned(task)
        with self._lock:
            tasks = self._abandoned.get(adapter_id)
            if tasks is None:
                return
            tasks.discard(task)
            if not tasks:
                del self._abandoned[adapter_id]

    def _abandon(
        self,
        adapter_id: str,
        worker: _IsolationWorker,
        task: Future[object],
        generation: int,
    ) -> None:
        """Stop waiting, keep the worker, and quarantine until ``task`` exits."""
        if task.done():
            _absorb_abandoned(task)
            return
        self._track_abandoned(adapter_id, task)
        worker.request_cancel(generation)

    async def _await_isolated[T](self, task: Future[T], timeout: float) -> T:
        """Wait for ``task`` on the caller loop without cancelling it.

        Args:
            task: Isolated inspect future.
            timeout: Remaining seconds the caller can wait.

        Returns:
            The inspect result.

        Raises:
            ClassifierTimeoutError: ``timeout`` elapsed while ``task`` is live.
            asyncio.CancelledError: The caller task was cancelled.
            Exception: Whatever the inspect raised.
        """
        loop = asyncio.get_running_loop()
        finished = asyncio.Event()

        def poke(_done: Future[T]) -> None:
            """Wake the caller loop when the isolated inspect exits."""
            loop.call_soon_threadsafe(finished.set)

        task.add_done_callback(poke)
        if task.done():
            finished.set()
        try:
            await asyncio.wait_for(finished.wait(), timeout=timeout)
        except TimeoutError as exc:
            if not task.done():
                raise ClassifierTimeoutError("classifier exceeded its per-check timeout") from exc
        return _isolated_result(task)

    async def run[T](
        self,
        fn: Callable[[], Awaitable[T]],
        timeout: float,
        *,
        adapter_id: str,
        retention: InspectionRetention | None = None,
    ) -> T:
        """Await ``fn`` on an isolation worker and abandon it when ``timeout`` elapses.

        Args:
            fn: Zero-argument coroutine factory for one inspect.
            timeout: Positive seconds budget, including worker wait.
            adapter_id: Policy adapter identity used for quarantine.
            retention: Optional subject owner held beyond timeout until the worker actually exits.

        Returns:
            The inspect result.

        Raises:
            ClassifierTimeoutError: The budget elapsed, the adapter is
                quarantined, or the timeout is not positive.
            asyncio.CancelledError: The caller task was cancelled.
            Exception: Whatever ``fn`` raised.
        """
        if timeout <= 0:
            raise ClassifierTimeoutError("classifier timeout is not positive")
        if self._quarantined(adapter_id):
            raise ClassifierTimeoutError("adapter is quarantined after ignoring cancellation")
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        worker = await self._pool.acquire(max(0.0, deadline - loop.time()))
        if worker is None:
            raise ClassifierTimeoutError("classifier exceeded its per-check timeout")
        if self._quarantined(adapter_id):
            self._pool.release(worker)
            raise ClassifierTimeoutError("adapter is quarantined after ignoring cancellation")
        remaining = deadline - loop.time()
        if remaining <= 0:
            self._pool.release(worker)
            raise ClassifierTimeoutError("classifier exceeded its per-check timeout")

        async def isolated() -> T:
            """Run the inspect on this isolation worker."""
            return await fn()

        if retention is not None:
            retention.retain()
        try:
            pending, generation = worker.submit(isolated)
        except BaseException:
            if retention is not None:
                retention.release()
            self._pool.release(worker)
            raise
        pending.add_done_callback(
            lambda done: self._reclaim(adapter_id, worker, cast(Future[object], done), retention)
        )
        try:
            return await self._await_isolated(pending, remaining)
        except ClassifierTimeoutError:
            if pending.done():
                return _isolated_result(pending)
            self._abandon(adapter_id, worker, cast(Future[object], pending), generation)
            raise
        except asyncio.CancelledError:
            self._abandon(adapter_id, worker, cast(Future[object], pending), generation)
            raise

    def _reclaim(
        self,
        adapter_id: str,
        worker: _IsolationWorker,
        task: Future[object],
        retention: InspectionRetention | None,
    ) -> None:
        """Return the worker after the isolated inspect actually exits."""
        try:
            self._pool.release(worker)
            if task.done():
                self._finish_detached(adapter_id, task)
        finally:
            if retention is not None:
                retention.release()


class _NativeCallbackRunner:
    """One lazily started daemon loop shared by native guardrail callbacks."""

    def __init__(self, *, start_timeout: float = 5.0) -> None:
        """Create an unstarted runner.

        Args:
            start_timeout: Seconds to wait for the daemon loop to become ready.

        Raises:
            ValueError: ``start_timeout`` is not positive.
        """
        if start_timeout <= 0:
            raise ValueError("start_timeout must be positive")
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._start_count = 0
        self._start_timeout = start_timeout

    def loop(self) -> asyncio.AbstractEventLoop:
        """Return the running daemon loop, starting it on first use.

        Exactly one caller may create the runner thread. Concurrent first
        callers wait on the same ready event outside the lock. A dead or
        failed startup is cleared so a later call can retry.
        """
        with self._lock:
            current = self._ready_or_start()
        if current is not None:
            return current
        if not self._ready.wait(timeout=self._start_timeout):
            self._reset_dead_startup()
            raise RuntimeError("native guardrail callback loop failed to start")
        started = self._loop
        thread = self._thread
        if started is None or not started.is_running() or thread is None or not thread.is_alive():
            self._reset_dead_startup()
            raise RuntimeError("native guardrail callback loop failed to start")
        return started

    def ready_loop(self) -> asyncio.AbstractEventLoop | None:
        """Start the shared daemon if needed, without waiting for its lock or readiness."""
        if not self._lock.acquire(blocking=False):
            return None
        try:
            return self._ready_or_start()
        finally:
            self._lock.release()

    def _ready_or_start(self) -> asyncio.AbstractEventLoop | None:
        """Under the runner lock, reuse a live loop or start one subject-free daemon."""
        current, thread = self._loop, self._thread
        if (
            current is not None
            and current.is_running()
            and thread is not None
            and thread.is_alive()
        ):
            return current
        if thread is None or not thread.is_alive():
            self._ready.clear()
            self._loop = None
            self._thread = threading.Thread(
                target=self._run_forever, name="exp-guardrail-native", daemon=True
            )
            self._start_count += 1
            self._thread.start()
        return None

    def _reset_dead_startup(self) -> None:
        """Clear a failed start so a later caller can create one new thread."""
        with self._lock:
            thread = self._thread
            if thread is not None and thread.is_alive():
                return
            self._thread = None
            self._loop = None
            self._ready.clear()

    def _run_forever(self) -> None:
        """Own one event loop for the life of the process."""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._loop = loop
        except Exception:  # noqa: BLE001 - startup failure must unblock the constructor
            self._ready.set()
            raise
        loop.call_soon(self._ready.set)
        loop.run_forever()

    def submit[T](self, coro: Coroutine[object, object, T]) -> T:
        """Run ``coro`` on the daemon loop and return when it finishes.

        Detached quarantined inspects stay on isolation workers after
        ``coro`` returns, so the Rust worker is not blocked on them.

        Args:
            coro: Coroutine produced by ``enforce_input`` or ``enforce_output``.

        Returns:
            The coroutine result.

        Raises:
            Exception: Whatever the coroutine raised.
        """
        return asyncio.run_coroutine_threadsafe(coro, self.loop()).result()


_NATIVE_RUNNER = _NativeCallbackRunner()


def start_on_native_loop[T](coro: Coroutine[object, object, T]) -> Future[T]:
    """Start bounded engine work without occupying a native bridge worker.

    Args:
        coro: Engine coroutine whose classifier calls use BoundedInspect.

    Returns:
        Request-owned future. Its owner cancels it when the request terminates.
    """
    return asyncio.run_coroutine_threadsafe(coro, _NATIVE_RUNNER.loop())


def try_start_on_native_loop[T](coro: Coroutine[object, object, T]) -> Future[T] | None:
    """Submit optional work only when the shared callback loop is already ready.

    This can initiate the one subject-free daemon, but never waits for loop
    initialization or another caller's startup lock. None leaves coroutine ownership
    with the caller, which must close it; exceptions also transfer no ownership.
    """
    loop = _NATIVE_RUNNER.ready_loop()
    return None if loop is None else asyncio.run_coroutine_threadsafe(coro, loop)


def run_on_native_loop[T](coro: Coroutine[object, object, T]) -> T:
    """Submit one coroutine to the shared native-callback event loop.

    Rust invokes control-plane methods on worker threads that do not own the
    Python gateway loop. The shared daemon loop lets those callbacks return
    as soon as enforcement finishes, even when a quarantined adapter is
    still occupying an isolation worker.

    Args:
        coro: Coroutine produced by ``enforce_input`` or ``enforce_output``.

    Returns:
        The coroutine result.

    Raises:
        RuntimeError: This thread already has a running event loop.
        Exception: Whatever the coroutine raised.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return _NATIVE_RUNNER.submit(coro)
    coro.close()
    raise RuntimeError(
        "native guardrail callbacks cannot run on a thread that already owns an event loop"
    )


def _isolated_result[T](task: Future[T]) -> T:
    """Return an isolated inspect result, mapping worker cancellation."""
    try:
        return task.result()
    except FutureCancelledError as exc:
        raise asyncio.CancelledError from exc
