"""Cancellable async classifier execution under a per-loop inflight cap."""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Awaitable, Callable, Coroutine
from typing import cast
from weakref import WeakKeyDictionary

MAX_INFLIGHT_ASYNC_CLASSIFIER_CALLS = 32

_logger = logging.getLogger(__name__)


class ClassifierTimeoutError(TimeoutError):
    """A classifier exceeded its per-check timeout, wait, or quarantine."""


def _absorb_abandoned(task: asyncio.Future[object]) -> None:
    """Retrieve an abandoned inspect so its exception is not unhandled."""
    if not task.cancelled():
        task.exception()


class BoundedInspect:
    """Run inspect tasks under a deadline without waiting for cancellation.

    Each event loop has its own semaphore. ``asyncio.wait`` times out without
    cancelling the inspect. This runner then cancels the task, releases the
    slot immediately, and returns. An adapter that ignores ``CancelledError``
    is quarantined on that loop until its detached task finishes. Further
    calls to that adapter fail immediately and do not create another task.
    Other adapters keep their full share of the inflight cap.
    """

    def __init__(self, max_inflight: int = MAX_INFLIGHT_ASYNC_CLASSIFIER_CALLS) -> None:
        """Bind one concurrency cap, realized separately on each event loop.

        Args:
            max_inflight: Maximum concurrent async inspects per event loop.

        Raises:
            ValueError: ``max_inflight`` is not a positive integer.
        """
        if max_inflight < 1:
            raise ValueError("max_inflight must be a positive integer")
        self._max_inflight = max_inflight
        self._lock = threading.Lock()
        self._slots_by_loop: WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore] = (
            WeakKeyDictionary()
        )
        self._detached_by_loop: WeakKeyDictionary[
            asyncio.AbstractEventLoop, dict[str, set[asyncio.Future[object]]]
        ] = WeakKeyDictionary()

    def detached_inspect_count(self) -> int:
        """Return live abandoned inspects across every bound event loop."""
        with self._lock:
            return sum(
                1
                for mapping in self._detached_by_loop.values()
                for tasks in mapping.values()
                for task in tasks
                if not task.done()
            )

    def quarantined_adapter_ids(self) -> frozenset[str]:
        """Return adapter identities with a live abandoned inspect."""
        with self._lock:
            return frozenset(
                adapter_id
                for mapping in self._detached_by_loop.values()
                for adapter_id, tasks in mapping.items()
                if any(not task.done() for task in tasks)
            )

    def _slots(self, loop: asyncio.AbstractEventLoop) -> asyncio.Semaphore:
        """Return the semaphore bound to ``loop``."""
        with self._lock:
            slots = self._slots_by_loop.get(loop)
            if slots is None:
                slots = asyncio.Semaphore(self._max_inflight)
                self._slots_by_loop[loop] = slots
            return slots

    def _quarantined(self, loop: asyncio.AbstractEventLoop, adapter_id: str) -> bool:
        """Return whether ``adapter_id`` has a live abandoned inspect on ``loop``."""
        with self._lock:
            mapping = self._detached_by_loop.get(loop)
            if mapping is None:
                return False
            tasks = mapping.get(adapter_id)
            return tasks is not None and any(not task.done() for task in tasks)

    def _finish_detached(
        self,
        loop: asyncio.AbstractEventLoop,
        adapter_id: str,
        task: asyncio.Future[object],
    ) -> None:
        """Absorb one abandoned inspect and lift quarantine when none remain."""
        _absorb_abandoned(task)
        with self._lock:
            mapping = self._detached_by_loop.get(loop)
            if mapping is None:
                return
            tasks = mapping.get(adapter_id)
            if tasks is None:
                return
            tasks.discard(task)
            if not tasks:
                del mapping[adapter_id]
            if not mapping:
                self._detached_by_loop.pop(loop, None)

    def _abandon(
        self,
        loop: asyncio.AbstractEventLoop,
        adapter_id: str,
        task: asyncio.Future[object],
    ) -> None:
        """Cancel ``task`` without waiting and quarantine it if it stays live."""
        if task.done():
            _absorb_abandoned(task)
            return
        task.add_done_callback(lambda done: self._finish_detached(loop, adapter_id, done))
        task.cancel()
        if task.done():
            self._finish_detached(loop, adapter_id, task)
            return
        with self._lock:
            mapping = self._detached_by_loop.setdefault(loop, {})
            mapping.setdefault(adapter_id, set()).add(task)
        _logger.info("guardrail adapter quarantined adapter_id=%s", adapter_id)

    async def run[T](
        self,
        fn: Callable[[], Awaitable[T]],
        timeout: float,
        *,
        adapter_id: str,
    ) -> T:
        """Await ``fn`` and abandon it when ``timeout`` elapses.

        Args:
            fn: Zero-argument coroutine factory for one inspect.
            timeout: Positive seconds budget, including slot wait.
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
        loop = asyncio.get_running_loop()
        if self._quarantined(loop, adapter_id):
            raise ClassifierTimeoutError("adapter is quarantined after ignoring cancellation")
        deadline = loop.time() + timeout
        slots = self._slots(loop)
        acquired = False
        inspect: asyncio.Future[object] | None = None
        try:
            try:
                async with asyncio.timeout(max(0.0, deadline - loop.time())):
                    await slots.acquire()
            except TimeoutError as exc:
                raise ClassifierTimeoutError("classifier exceeded its per-check timeout") from exc
            acquired = True
            if self._quarantined(loop, adapter_id):
                raise ClassifierTimeoutError("adapter is quarantined after ignoring cancellation")
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise ClassifierTimeoutError("classifier exceeded its per-check timeout")
            inspect = asyncio.ensure_future(fn())
            try:
                done, _pending = await asyncio.wait({inspect}, timeout=remaining)
            except asyncio.CancelledError:
                self._abandon(loop, adapter_id, inspect)
                raise
            if inspect in done:
                return cast(T, inspect.result())
            self._abandon(loop, adapter_id, inspect)
            raise ClassifierTimeoutError("classifier exceeded its per-check timeout")
        finally:
            if acquired:
                slots.release()


class _NativeCallbackRunner:
    """One lazily started daemon loop shared by native guardrail callbacks."""

    def __init__(self) -> None:
        """Create an unstarted runner."""
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._loop: asyncio.AbstractEventLoop | None = None

    def loop(self) -> asyncio.AbstractEventLoop:
        """Return the running daemon loop, starting it on first use."""
        with self._lock:
            current = self._loop
            if current is not None and current.is_running():
                return current
            self._ready.clear()
            thread = threading.Thread(
                target=self._run_forever,
                name="exp-guardrail-native",
                daemon=True,
            )
            thread.start()
        if not self._ready.wait(timeout=5.0):
            raise RuntimeError("native guardrail callback loop failed to start")
        started = self._loop
        if started is None or not started.is_running():
            raise RuntimeError("native guardrail callback loop failed to start")
        return started

    def _run_forever(self) -> None:
        """Own one event loop for the life of the process."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        self._ready.set()
        loop.run_forever()

    def submit[T](self, coro: Coroutine[object, object, T]) -> T:
        """Run ``coro`` on the daemon loop and return when it finishes.

        Detached quarantined inspects stay on the loop after ``coro``
        returns, so the Rust worker is not blocked on cancellation.

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
    still ignoring cancellation.

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
