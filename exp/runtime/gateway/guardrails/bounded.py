"""Bounded classifier execution that cannot stall the request deadline."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

MAX_INFLIGHT_CLASSIFIER_CALLS = 32
_INFLIGHT = threading.BoundedSemaphore(MAX_INFLIGHT_CLASSIFIER_CALLS)


class ClassifierTimeoutError(TimeoutError):
    """A classifier exceeded its per-check timeout without returning."""


def run_bounded[T](
    fn: Callable[[], T],
    timeout: float,
    *,
    slots: threading.BoundedSemaphore | None = None,
) -> T:
    """Run ``fn`` on a worker thread and raise if it exceeds ``timeout``.

    The caller waits at most ``timeout`` seconds. ``fn`` never runs on the
    caller's thread, so a synchronous inspect cannot block an asyncio event
    loop that invoked this helper through ``asyncio.to_thread``. Timed-out
    workers keep their slot until they finish, so at most
    ``MAX_INFLIGHT_CLASSIFIER_CALLS`` inspects can be live. Further calls
    fail closed at the same timeout instead of starting another thread. The
    worker is a daemon so a hung adapter cannot stall process shutdown.

    Args:
        fn: Synchronous inspect call.
        timeout: Positive seconds budget.
        slots: Optional inflight limiter. Tests inject a smaller semaphore.
            Production uses the process-wide classifier cap.

    Returns:
        The inspect result.

    Raises:
        ClassifierTimeoutError: The wait expired, or no worker slot was free
            before the timeout.
        Exception: Whatever ``fn`` raised.
    """
    if timeout <= 0:
        raise ClassifierTimeoutError("classifier timeout is not positive")
    limiter = _INFLIGHT if slots is None else slots
    started = time.monotonic()
    if not limiter.acquire(timeout=timeout):
        raise ClassifierTimeoutError("classifier worker capacity is exhausted")
    remaining = timeout - (time.monotonic() - started)
    if remaining <= 0:
        limiter.release()
        raise ClassifierTimeoutError("classifier exceeded its per-check timeout")
    box: list[T] = []
    errors: list[Exception] = []
    done = threading.Event()

    def worker() -> None:
        """Execute the inspect call, then release the inflight slot."""
        try:
            box.append(fn())
        except Exception as exc:  # noqa: BLE001 - inspect failures are returned to the chain
            errors.append(exc)
        finally:
            done.set()
            limiter.release()

    thread = threading.Thread(target=worker, name="exp-guardrail-inspect", daemon=True)
    try:
        thread.start()
    except RuntimeError:
        limiter.release()
        raise
    if not done.wait(remaining):
        raise ClassifierTimeoutError("classifier exceeded its per-check timeout")
    if errors:
        raise errors[0]
    return box[0]
