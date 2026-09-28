"""Retain the learner launcher's process-wide stop authority across Ray lifecycle calls."""

import signal
from collections.abc import Iterator
from contextlib import contextmanager
from threading import current_thread, main_thread


@contextmanager
def preserve_stop_handlers() -> Iterator[None]:
    """Restore caller stop handlers after an owned synchronous Ray operation, even on error.

    Ray installs a process-exiting SIGTERM handler during initialization. Reinstalling
    the caller's handlers preserves asyncio's existing wakeup descriptor and callbacks;
    this boundary never removes or re-registers event-loop signal handlers.

    Yields:
        Control to initialization or shutdown of the caller-owned Ray runtime.

    Raises:
        RuntimeError: The caller cannot restore process handlers from a worker thread.
    """
    if current_thread() is not main_thread():
        raise RuntimeError("resident Ray lifecycle requires the main thread for signal ownership")
    stop_signals = [signal.SIGINT, signal.SIGTERM]
    if hasattr(signal, "SIGBREAK"):
        stop_signals.append(signal.SIGBREAK)
    previous = {number: signal.getsignal(number) for number in stop_signals}
    try:
        yield
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)
