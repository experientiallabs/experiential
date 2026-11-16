"""One request-owned budget for optional parallel inspection at blocking boundaries."""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Callable, Coroutine
from concurrent.futures import Future


class InputWait:
    """Share the first response or predispatch boundary across every inspected subject.

    Attributes:
        _boundary: Thread-safe first blocking instant; pending during safe generation.
        _lock: Serialize competing boundary announcements without renewing their budget.
    """

    def __init__(self) -> None:
        """Start with no blocking wait and no retained request content."""
        self._boundary: Future[float] = Future()
        self._lock = threading.Lock()

    def begin(self, *, elapsed_seconds: float = 0.0) -> None:
        """Start once before work that must wait for moderation."""
        with self._lock:
            if not self._boundary.done():
                self._boundary.set_result(time.monotonic() - elapsed_seconds)

    async def run[T](
        self,
        inspect: Callable[[], Coroutine[object, object, T]],
        *,
        wait_ms: int,
        deadline: float,
        native_elapsed: Callable[[], float | None] | None = None,
    ) -> T | None:
        """Return a ready verdict or expire only this optional policy's wait.

        Args:
            inspect: Isolated inspection coroutine factory with its own capacity bound.
            wait_ms: Shared additional waiting allowance after the first blocking boundary.
            deadline: Existing request deadline, which never extends the grace.
            native_elapsed: Native boundary clock for distinct optional policy budgets.

        Returns:
            The inspection result, or None when this policy's remaining wait is spent.
        """
        if self._boundary.done() and self._boundary.result() + wait_ms / 1000 <= time.monotonic():
            return None
        if native_elapsed is not None and (elapsed := native_elapsed()) is not None:
            self.begin(elapsed_seconds=elapsed)
        task = asyncio.create_task(inspect())
        boundary = asyncio.wrap_future(self._boundary)
        try:
            # A distinct, smaller policy budget still expires before the native
            # request-wide ceiling. Only this uncommon chain needs clock polling;
            # the native ceiling never depends on this event loop being scheduled.
            while native_elapsed is not None and not self._boundary.done() and not task.done():
                if (elapsed := native_elapsed()) is not None:
                    self.begin(elapsed_seconds=elapsed)
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                await asyncio.wait({task}, timeout=min(0.005, remaining))
            if not self._boundary.done():
                await asyncio.wait(
                    {task, boundary},
                    timeout=max(0, deadline - time.monotonic()),
                    return_when=asyncio.FIRST_COMPLETED,
                )
            if task.done():
                return task.result()
            if not self._boundary.done():
                return None
            remaining = min(deadline, self._boundary.result() + wait_ms / 1000) - time.monotonic()
            if remaining > 0:
                await asyncio.wait({task}, timeout=remaining)
            if task.done():
                return task.result()
            return None
        finally:
            if not task.done():
                task.cancel()
                # BoundedInspect retains the isolated worker until cancellation actually
                # completes. Waiting for that worker here would renew the response grace.
                task.add_done_callback(_consume)


def _consume(task: asyncio.Task[object]) -> None:
    """Retrieve a cancelled inspection's result without delaying delivery."""
    if not task.cancelled():
        task.exception()
