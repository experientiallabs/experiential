"""Cancellable async classifier execution under a per-loop inflight cap."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Coroutine
from weakref import WeakKeyDictionary

MAX_INFLIGHT_ASYNC_CLASSIFIER_CALLS = 32


class ClassifierTimeoutError(TimeoutError):
    """A classifier exceeded its per-check timeout or inflight wait."""


class BoundedInspect:
    """Run inspect coroutines under ``asyncio.timeout`` and a per-loop cap.

    Each event loop has its own semaphore. Cancellation exits the slot
    immediately, so a cancelled inspect cannot retain capacity for later
    healthy adapters. Native callbacks use a private loop and therefore cannot
    consume the Python gateway loop's slots.
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
        self._slots_by_loop: WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore] = (
            WeakKeyDictionary()
        )

    def _slots(self) -> asyncio.Semaphore:
        """Return the semaphore bound to the running event loop."""
        loop = asyncio.get_running_loop()
        slots = self._slots_by_loop.get(loop)
        if slots is None:
            slots = asyncio.Semaphore(self._max_inflight)
            self._slots_by_loop[loop] = slots
        return slots

    async def run[T](self, fn: Callable[[], Awaitable[T]], timeout: float) -> T:
        """Await ``fn`` and cancel it when ``timeout`` elapses.

        Args:
            fn: Zero-argument coroutine factory for one inspect.
            timeout: Positive seconds budget, including slot wait.

        Returns:
            The inspect result.

        Raises:
            ClassifierTimeoutError: The budget elapsed before ``fn`` returned.
            Exception: Whatever ``fn`` raised.
        """
        if timeout <= 0:
            raise ClassifierTimeoutError("classifier timeout is not positive")
        try:
            async with asyncio.timeout(timeout):
                async with self._slots():
                    return await fn()
        except TimeoutError as exc:
            raise ClassifierTimeoutError("classifier exceeded its per-check timeout") from exc


def run_on_private_loop[T](coro: Coroutine[object, object, T]) -> T:
    """Run one coroutine on a private event loop for native callbacks.

    Rust invokes control-plane methods on worker threads that do not own the
    Python gateway loop. ``asyncio.run`` creates a private loop on that
    thread so the callback never attaches to, or blocks, the gateway loop.

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
        return asyncio.run(coro)
    raise RuntimeError(
        "native guardrail callbacks cannot run on a thread that already owns an event loop"
    )
