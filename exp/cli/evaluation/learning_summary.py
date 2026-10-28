"""The short learning and retest result lines ``exp build`` and ``exp eval --audit`` print.

Both commands save their full learning output beside the saved run (:func:`save_details`) and
show only these lines on the terminal. The module stays light so that ``exp --help`` does not
load the learning lifecycle.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from exp.common.core.files import write_text_atomic
from exp.common.project import ProjectStore
from exp.common.project.paths import validate_local_id
from exp.optimize.evaluation.fidelity.recovery import QUALIFICATION

if TYPE_CHECKING:
    from exp.cli.evaluation.miss_retest import RetestOutcome
    from exp.optimize.evaluation.learning.traces import LearningCurve, LearningOutcome


@dataclass(frozen=True)
class LearningSummary:
    """What a command's learning stage did, for its summary lines.

    Attributes:
        recall: Memory-recall alias, or None when the catalog has no decision model.
        outcome: The learning pass, or None when it did not run or the spend was declined.
        curve: Learning progress over the learned memory.
        retest: The miss retest, or None when no learned trace had a miss.
    """

    recall: str | None = None
    outcome: LearningOutcome | None = None
    curve: LearningCurve | None = None
    retest: RetestOutcome | None = None


def learning_lines(learning: LearningSummary, noun: str = "traces") -> tuple[str, ...]:
    """The learning and retest result lines, warnings included, in terminal markup."""
    outcome = learning.outcome
    if outcome is None:
        return ()
    calls = (
        "no new model calls (saved results)"
        if outcome.attempts_used == 0
        else f"{outcome.attempts_used:,} model attempts"
    )
    learned = (
        f"[green]✓[/green] learned {outcome.learned}/{outcome.traces} {noun} · "
        f"memory v{outcome.head_version} · {calls}"
    )
    curve = learning.curve
    if curve is not None and curve.points:
        exact = sum(point.exact for point in curve.points)
        steps = sum(point.steps for point in curve.points)
        learned += f"\n  replies exact while learning {_percent(exact, steps)} ({exact}/{steps})"
        if curve.quarters:
            last = curve.quarters[-1]
            learned += f", last {last.last - last.first + 1} traces {last.share}"
        if curve.qualification is not None:
            learned += f" · {QUALIFICATION}"
    lines = [learned]
    if outcome.incomplete:
        lines.append(
            f"[yellow]warning[/yellow] {outcome.incomplete} trace(s) stopped on every allowed "
            "attempt and were not learned"
        )
    if outcome.retired:
        lines.append(
            f"[yellow]warning[/yellow] {outcome.retired} trace(s) retired after deterministic "
            "refusals or by an operator: not learned and not scored (named in the report)"
        )
    if outcome.stopped is not None:
        lines.append(f"[yellow]warning[/yellow] {outcome.stopped}")
    retest = learning.retest
    if retest is not None:
        total, changes = retest.total, retest.changes
        lines.append(
            f"[green]✓[/green] retested {retest.planned} "
            f"{'trace' if retest.planned == 1 else 'traces'} that had misses · "
            f"{_percent(total.exact, total.steps)} replies exact ({total.exact}/{total.steps}) · "
            f"fixed {changes.fixed} of {changes.missed} missed turns · "
            f"{changes.regressed} regressed"
            + ("" if retest.qualification is None else f" · {QUALIFICATION}")
        )
    return tuple(lines)


def _percent(part: int, whole: int) -> str:
    """A whole-number percentage, or a dash without a denominator."""
    return f"{part / whole:.0%}" if whole else "-"


def save_details(store: ProjectStore, kind: str, identity: str, text: str) -> Path:
    """Write a command's full output beside its saved run state, like an evaluation report.

    Args:
        store: Project owning the run.
        kind: Runtime family, such as ``learning`` or ``audits``.
        identity: Exact identity of the saved state the output describes.
        text: The rendered full output.

    Returns:
        ``<project>/runtime/<kind>/<identity>/report.txt``, written atomically.
    """
    validate_local_id(identity, label=f"{kind} identity")
    path = store.paths.runtime_directory / kind / identity / "report.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomic(path, text)
    return path
