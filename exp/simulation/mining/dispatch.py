"""Bounded parallel dispatch of the paid calls one tagging pass makes.

Tagging a corpus is latency bound, not compute bound. One call answers a whole question bank
about one rendered episode and spends seconds waiting on the endpoint, so a corpus of thousands
of episodes takes hours of almost pure waiting when the calls are made one after another. Running
a bounded number of them at once is the whole of the speedup, and the bound is what keeps a local
run from becoming a load test against someone else's endpoint.

Three properties make that safe on work that costs money per item:

- **Order is an input property.** Results come back in the order the items were given, whatever
  order they completed in, so what the caller assembles from them cannot depend on scheduling.
- **A failure is the same failure every run.** Items are admitted in order and no further item is
  admitted once one has failed, which makes the lowest-indexed failure the one that is always
  observed. That exact failure is raised, so a corpus-wide run reports the same first problem on
  every attempt instead of whichever worker happened to lose the race.
- **Nothing in flight is abandoned.** Work already admitted has already been paid for, so a
  failure drains the pool rather than cancelling it: every answer that was bought reaches its
  caller's commit path before the failure surfaces.

The dispatcher is generic over the item and the result and knows nothing about decisions, banks or
caches. It lives beside the mining code that uses it because the dependency direction forbids
simulation from importing the equivalent judgment dispatcher that optimize owns.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait

DEFAULT_TAGGING_CONCURRENCY = 8
"""Calls in flight at once when a caller does not choose.

Eight is the allowance the rest of this project uses for simultaneous provider work. It turns a
multi-hour serial pass over a large corpus into well under an hour while staying a load a single
local run can defend: the ceiling is per run, so what the endpoint sees stays bounded and
predictable.
"""


def dispatch_tagging[ItemT, ResultT](
    items: Sequence[ItemT],
    answer: Callable[[ItemT], ResultT],
    *,
    maximum_concurrency: int,
) -> tuple[ResultT, ...]:
    """Answer every item with at most ``maximum_concurrency`` in flight, in input order.

    Args:
        items: Work items in their canonical order, each answerable independently of the others.
        answer: Answers one item. It runs on a worker thread, so everything it reaches has to be
            safe to use from several threads at once.
        maximum_concurrency: Positive ceiling on items in flight. One runs every item inline on
            the calling thread, with no pool and no worker thread, so a caller that wants a
            strictly serial pass gets exactly that.

    Returns:
        One result per item, in the order the items were supplied.

    Raises:
        ValueError: The concurrency ceiling is not positive.
        BaseException: The lowest-indexed item's failure, re-raised unchanged after every
            already-admitted item has finished. Failures are never swallowed and never merged.
    """
    if maximum_concurrency < 1:
        raise ValueError("tagging concurrency must be at least one call in flight")
    if maximum_concurrency == 1 or len(items) <= 1:
        return tuple(answer(item) for item in items)
    workers = min(maximum_concurrency, len(items))
    failed = threading.Event()
    results: dict[int, ResultT] = {}
    failures: dict[int, BaseException] = {}
    pending: dict[Future[ResultT], int] = {}
    admitted = 0
    pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="exp-tag")

    def admit() -> None:
        """Fill the window in item order, and stop admitting once anything has failed."""
        nonlocal admitted
        while len(pending) < workers and admitted < len(items) and not failed.is_set():
            pending[pool.submit(answer, items[admitted])] = admitted
            admitted += 1

    try:
        admit()
        while pending:
            completed, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in sorted(completed, key=pending.__getitem__):
                index = pending.pop(future)
                try:
                    results[index] = future.result()
                except BaseException as exc:  # noqa: BLE001 - re-raised below, by index
                    failures[index] = exc
                    failed.set()
            admit()
    finally:
        failed.set()
        pool.shutdown(wait=True, cancel_futures=True)
    if failures:
        raise failures[min(failures)]
    return tuple(results[index] for index in range(len(items)))
