"""Run independent calls at once, at most a bound in flight, stopping at the first failure."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from contextvars import copy_context


def run_at_once[T](calls: Sequence[Callable[[], T]], concurrency: int) -> tuple[T, ...]:
    """Return every call's result in input order, with at most ``concurrency`` in flight.

    With a bound of one the calls run in order on this thread, exactly like a loop. Otherwise
    each call runs in a copy of the caller's context (spend ceilings and call evidence travel
    with it); the first failure, including a budget stop, cancels every call not yet started
    and is raised once the running ones finish, so no new work starts after a stop.

    Raises:
        ValueError: The bound is not positive.
    """
    if concurrency < 1:
        raise ValueError("concurrency must be positive")
    if concurrency == 1 or len(calls) <= 1:
        return tuple(call() for call in calls)
    with ThreadPoolExecutor(max_workers=min(concurrency, len(calls))) as pool:
        futures = [pool.submit(copy_context().run, call) for call in calls]
        done, _ = wait(futures, return_when=FIRST_EXCEPTION)
        if any(future.exception() is not None for future in done):
            for future in futures:
                future.cancel()
        return tuple(future.result() for future in futures)
