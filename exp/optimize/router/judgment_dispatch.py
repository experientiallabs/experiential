"""Bounded concurrent judgment execution with ordered evidence and durable cleanup."""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_COMPLETED, CancelledError, Future, ThreadPoolExecutor, wait

from exp.common.evaluations import EvaluationCellEvidence


def dispatch_judgments[ItemT](
    items: Sequence[ItemT],
    execute: Callable[[ItemT, Callable[[], bool]], EvaluationCellEvidence],
    *,
    maximum_concurrency: int,
    on_completed: Callable[[int], None] | None = None,
) -> tuple[EvaluationCellEvidence, ...]:
    """Run one bounded window, preserving input order and draining admitted work on failure.

    Args:
        items: Canonically ordered, independently bound judgment inputs.
        execute: Executes and persists one cell; checks cancellation before provider admission.
        maximum_concurrency: Positive total concurrency across the complete judgment phase.
        on_completed: Optional completion count callback, invoked only on the calling thread.

    Returns:
        Completed evidence in the original input order, regardless of completion order.

    Raises:
        ValueError: The concurrency allowance is not positive.
        BaseException: An execution failed or the caller interrupted; running work is drained
            so completed responses can be persisted before the original failure is raised.
    """
    if maximum_concurrency <= 0:
        raise ValueError("judgment concurrency must be positive")
    if maximum_concurrency == 1:
        serial_results = []
        for item in items:
            serial_results.append(execute(item, lambda: False))
            if on_completed is not None:
                on_completed(len(serial_results))
        return tuple(serial_results)
    if not items:
        return ()
    cancelled = threading.Event()
    failure_lock = threading.Lock()
    failures: list[BaseException] = []

    def invoke(item: ItemT) -> EvaluationCellEvidence:
        """Stop unstarted work and retain the original failure before waking the coordinator."""
        if cancelled.is_set():
            raise CancelledError
        try:
            return execute(item, cancelled.is_set)
        except BaseException as exc:
            with failure_lock:
                if not failures and not isinstance(exc, CancelledError):
                    failures.append(exc)
            cancelled.set()
            raise

    concurrency = min(maximum_concurrency, len(items))
    executor = ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="exp-judge")
    pending: dict[Future[EvaluationCellEvidence], int] = {}
    results: dict[int, EvaluationCellEvidence] = {}
    next_index = 0

    def fill_window() -> None:
        """Admit only enough work to use the remaining phase-wide allowance."""
        nonlocal next_index
        while len(pending) < concurrency and next_index < len(items) and not cancelled.is_set():
            pending[executor.submit(invoke, items[next_index])] = next_index
            next_index += 1

    try:
        fill_window()
        while pending:
            completed, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in sorted(completed, key=pending.__getitem__):
                results[pending.pop(future)] = future.result()
                if on_completed is not None:
                    on_completed(len(results))
            fill_window()
    except BaseException as exc:
        cancelled.set()
        for future in pending:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
        if isinstance(exc, CancelledError) and failures:
            raise failures[0] from exc
        raise
    else:
        executor.shutdown(wait=True)
    return tuple(results[index] for index in range(len(items)))
