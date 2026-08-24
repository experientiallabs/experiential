"""Isolate classifier inspects from the caller's timeout-watching event loop."""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Awaitable, Callable, Coroutine
from concurrent.futures import CancelledError as FutureCancelledError
from concurrent.futures import Future
from typing import cast

MAX_INFLIGHT_ASYNC_CLASSIFIER_CALLS = 32
_WORKER_START_TIMEOUT_SECONDS = 5.0

_logger = logging.getLogger(__name__)


class ClassifierTimeoutError(TimeoutError):
    """A classifier exceeded its per-check timeout, wait, or quarantine."""


def _absorb_abandoned(task: Future[object]) -> None:
    """Retrieve an abandoned inspect so its exception is not unhandled."""
    if not task.cancelled():
        task.exception()


class _IsolationWorker:
    """One daemon thread that owns a private event loop for isolated inspects."""

    def __init__(self, *, start_timeout: float = _WORKER_START_TIMEOUT_SECONDS) -> None:
        """Start the worker loop before returning.

        Args:
            start_timeout: Seconds to wait for the daemon loop to become ready.

        Raises:
            RuntimeError: The worker loop did not start in time.
        """
        self._ready = threading.Event()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._task: asyncio.Task[object] | None = None
        self._thread = threading.Thread(
            target=self._run_forever,
            name="exp-guardrail-isolate",
            daemon=True,
        )
        self._thread.start()
        if not self._ready.wait(timeout=start_timeout):
            raise RuntimeError("classifier isolation loop failed to start")
        loop = self._loop
        if loop is None or not loop.is_running() or not self._thread.is_alive():
            raise RuntimeError("classifier isolation loop failed to start")

    def submit[T](self, fn: Callable[[], Coroutine[object, object, T]]) -> Future[T]:
        """Schedule ``fn`` on this worker and return a cross-thread future.

        Args:
            fn: Zero-argument coroutine factory for one inspect.

        Returns:
            A future that completes when the isolated inspect exits.
        """
        loop = self._loop
        if loop is None:
            raise RuntimeError("classifier isolation loop is not running")
        result: Future[T] = Future()

        def start() -> None:
            """Create the inspect task on the worker loop."""
            if result.cancelled():
                return
            try:
                task = loop.create_task(fn())
            except Exception as exc:  # noqa: BLE001 - factory errors must complete the waiter
                result.set_exception(exc)
                return
            self._task = task

            def finish(done: asyncio.Task[T]) -> None:
                """Copy the inspect outcome onto the cross-thread future."""
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

            task.add_done_callback(finish)

        loop.call_soon_threadsafe(start)
        return result

    def request_cancel(self) -> None:
        """Queue cancellation for the running inspect without waiting."""
        loop = self._loop
        if loop is None:
            return

        def cancel() -> None:
            """Cancel the live inspect if it is still running."""
            task = self._task
            if task is not None and not task.done():
                task.cancel()

        loop.call_soon_threadsafe(cancel)

    def _run_forever(self) -> None:
        """Own one event loop for the life of this worker."""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._loop = loop
        except Exception:  # noqa: BLE001 - startup failure must unblock the constructor
            self._ready.set()
            raise
        self._ready.set()
        loop.run_forever()


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
        """
        worker = self._try_take()
        if worker is not None:
            return worker
        if timeout <= 0:
            return None
        loop = asyncio.get_running_loop()
        waiter: asyncio.Future[_IsolationWorker | None] = loop.create_future()
        with self._lock:
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
            return self._start_reserved_worker()
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
            while self._waiters:
                waiter = self._waiters.pop(0)
                if waiter.done():
                    continue
                waiter.get_loop().call_soon_threadsafe(self._deliver, waiter, worker)
                return
            self._idle.append(worker)

    def _try_take(self) -> _IsolationWorker | None:
        """Take an idle worker or reserve capacity to start one."""
        with self._lock:
            worker = self._take_idle()
            if worker is not None:
                return worker
            if self._created >= self._size:
                return None
            self._created += 1
        return self._start_reserved_worker()

    def _take_idle(self) -> _IsolationWorker | None:
        """Pop one idle worker. The caller must hold ``_lock``."""
        if not self._idle:
            return None
        return self._idle.pop()

    def _start_reserved_worker(self) -> _IsolationWorker:
        """Start a worker after ``_created`` has already been incremented."""
        try:
            return _IsolationWorker()
        except BaseException:
            with self._lock:
                self._created -= 1
            raise

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
        self._max_inflight = max_inflight
        self._pool = _IsolationPool(max_inflight)
        self._lock = threading.Lock()
        self._abandoned: dict[str, set[Future[object]]] = {}

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

    def _abandon(self, adapter_id: str, worker: _IsolationWorker, task: Future[object]) -> None:
        """Stop waiting, keep the worker, and quarantine until ``task`` exits."""
        if task.done():
            _absorb_abandoned(task)
            return
        self._track_abandoned(adapter_id, task)
        worker.request_cancel()

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
    ) -> T:
        """Await ``fn`` on an isolation worker and abandon it when ``timeout`` elapses.

        Args:
            fn: Zero-argument coroutine factory for one inspect.
            timeout: Positive seconds budget, including worker wait.
            adapter_id: Policy adapter identity used for quarantine.

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

        pending = worker.submit(isolated)
        pending.add_done_callback(
            lambda done: self._reclaim(adapter_id, worker, cast(Future[object], done))
        )
        try:
            return await self._await_isolated(pending, remaining)
        except ClassifierTimeoutError:
            if pending.done():
                return _isolated_result(pending)
            self._abandon(adapter_id, worker, cast(Future[object], pending))
            raise
        except asyncio.CancelledError:
            self._abandon(adapter_id, worker, cast(Future[object], pending))
            raise

    def _reclaim(
        self,
        adapter_id: str,
        worker: _IsolationWorker,
        task: Future[object],
    ) -> None:
        """Return the worker after the isolated inspect actually exits."""
        self._pool.release(worker)
        if not task.done():
            return
        self._finish_detached(adapter_id, task)


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
            current = self._loop
            thread = self._thread
            if (
                current is not None
                and current.is_running()
                and thread is not None
                and thread.is_alive()
            ):
                return current
            starting = thread is not None and thread.is_alive()
            if not starting:
                self._ready.clear()
                self._loop = None
                self._thread = threading.Thread(
                    target=self._run_forever,
                    name="exp-guardrail-native",
                    daemon=True,
                )
                self._start_count += 1
                self._thread.start()
        if not self._ready.wait(timeout=self._start_timeout):
            self._reset_dead_startup()
            raise RuntimeError("native guardrail callback loop failed to start")
        started = self._loop
        thread = self._thread
        if started is None or not started.is_running() or thread is None or not thread.is_alive():
            self._reset_dead_startup()
            raise RuntimeError("native guardrail callback loop failed to start")
        return started

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
        self._ready.set()
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
