"""Bounded independent council draws with inherited accounting and failure cancellation."""

from collections.abc import Callable, Sequence
from concurrent.futures import CancelledError
from functools import partial
from threading import Event, Lock

from exp.optimize.evaluation.rules.agent_loop import run_bounded


def council_draws[T](calls: Sequence[Callable[[], T]], concurrency: int) -> tuple[T, ...]:
    """Run independent draws in input order, stopping queued work after a terminal failure.

    Context propagation is owned by run_bounded. Each callback still owns its real native
    request ledger, result identity and cleanup; this helper never redraws or erases an attempt.

    Args:
        calls: Independent native owner invocations, each with a distinct saved identity.
        concurrency: Maximum concurrent invocations.

    Returns:
        Results in the original order, regardless of completion order.

    Raises:
        BaseException: The first observed real failure, after already-running owners finish.
        ValueError: The concurrency is invalid.
    """
    stopped, lock = Event(), Lock()
    failures: list[BaseException] = []

    def checked(call: Callable[[], T]) -> T:
        """Leave pending callbacks undispatched after another callback fails."""
        if stopped.is_set():
            raise CancelledError("council work cancelled after a terminal failure")
        try:
            return call()
        except BaseException as error:
            with lock:
                if not failures:
                    failures.append(error)
                stopped.set()
            raise

    try:
        return run_bounded(tuple(partial(checked, call) for call in calls), concurrency)
    except BaseException:
        if failures:
            raise failures[0] from None
        raise
