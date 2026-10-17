"""Run a registered replay panel: the interrupted-run policy and concurrent execution.

The held-out audit and the build's miss retest both run a frozen panel of registered replays
(:mod:`exp.optimize.evaluation.trace_replay`). This module is their one execution path: it
refuses interrupted runs unless resumption was asked for, then runs every trace through
:func:`execute_fidelity_evaluation`. Consent and terminal presentation stay with the commands.
"""

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime

from exp.common.progress import ProgressHook, qualified, report
from exp.common.project import ProjectStore
from exp.optimize.evaluation.fidelity import PreparedFidelity, execute_fidelity_evaluation
from exp.optimize.evaluation.trace_replay import DEADLINE, interrupted

PANEL_CONCURRENCY = 4  # traces of a frozen panel replayed at once
PANEL_STAGE = "Replaying traces"  # counts finished panel traces


def require_resumable(
    store: ProjectStore, registrations: tuple[PreparedFidelity, ...], *, resume: bool
) -> int:
    """Count interrupted traces, refusing them before any consent or call without ``resume``.

    Raises:
        ValueError: A trace was interrupted and ``resume`` was not given.
    """
    stopped = sum(interrupted(store, prepared) for prepared in registrations)
    if stopped and not resume:
        raise ValueError(
            f"{stopped} trace(s) were interrupted before saving a result; run the same "
            "command with --retry-incomplete to resume them in place"
        )
    return stopped


def execute_panel(
    store: ProjectStore,
    registrations: tuple[PreparedFidelity, ...],
    *,
    progress: ProgressHook | None = None,
    guard: Callable[[], None] | None = None,
    resume: bool = False,
) -> None:
    """Run registered panel traces concurrently through the fidelity owner.

    Panel traces read frozen memory and never publish, so they are independent; each keeps its
    own ledger and session. Finished traces replay untouched; with ``resume``, each interrupted
    trace continues in place from its first uncommitted step.

    Args:
        store: Project owning the runs.
        registrations: Saved registrations, in panel order.
        progress: Observer; each trace's events carry its ``trace i/n`` qualifier, and
            ``Replaying traces k/n`` counts finished traces.
        guard: Checked before each provider slot claim, for example an audit's panel lease.
        resume: Continue interrupted traces in place in a fresh resumption window.

    Raises:
        ValueError: A trace was interrupted and ``resume`` was not given; nothing is sent.
    """
    require_resumable(store, registrations, resume=resume)
    total = len(registrations)
    window = datetime.now(UTC) + DEADLINE if resume else None
    with ThreadPoolExecutor(PANEL_CONCURRENCY) as pool:
        report(progress, PANEL_STAGE, completed=0, total=total)
        futures = [
            pool.submit(
                execute_fidelity_evaluation,
                store,
                prepared,
                progress=qualified(progress, f"trace {index}/{total}"),
                dispatch_guard=guard,
                resume_until=window,
            )
            for index, prepared in enumerate(registrations, start=1)
        ]
        for finished, future in enumerate(as_completed(futures), start=1):
            future.result()
            report(progress, PANEL_STAGE, completed=finished, total=total)
