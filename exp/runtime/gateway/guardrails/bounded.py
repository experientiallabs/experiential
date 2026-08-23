"""Bounded classifier execution that cannot stall the request deadline."""

from __future__ import annotations

import threading
from collections.abc import Callable


class ClassifierTimeoutError(TimeoutError):
    """A classifier exceeded its per-check timeout without returning."""


def run_bounded[T](fn: Callable[[], T], timeout: float) -> T:
    """Run ``fn`` on a daemon thread and raise if it exceeds ``timeout``.

    The caller waits at most ``timeout`` seconds. ``fn`` never runs on the
    caller's thread, so a synchronous inspect cannot block an asyncio event
    loop that invoked this helper through ``asyncio.to_thread``. The worker
    is a daemon so a hung adapter cannot stall process shutdown.

    Args:
        fn: Synchronous inspect call.
        timeout: Positive seconds budget.

    Returns:
        The inspect result.

    Raises:
        ClassifierTimeoutError: The wait expired before ``fn`` returned.
        Exception: Whatever ``fn`` raised.
    """
    if timeout <= 0:
        raise ClassifierTimeoutError("classifier timeout is not positive")
    box: list[T] = []
    errors: list[Exception] = []
    done = threading.Event()

    def worker() -> None:
        """Execute the inspect call and publish its outcome."""
        try:
            box.append(fn())
        except Exception as exc:  # noqa: BLE001 - inspect failures are returned to the chain
            errors.append(exc)
        finally:
            done.set()

    thread = threading.Thread(target=worker, name="exp-guardrail-inspect", daemon=True)
    thread.start()
    if not done.wait(timeout):
        raise ClassifierTimeoutError("classifier exceeded its per-check timeout")
    if errors:
        raise errors[0]
    return box[0]
