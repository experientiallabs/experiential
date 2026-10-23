"""Miss retest in the existing evaluation terminal conventions."""

from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from rich.console import Console
from rich.text import Text

from exp.cli.evaluation.held_out_fidelity import (
    aggregate_line,
    execute_runs,
    render_traces,
    share,
)
from exp.cli.evaluation.view import heading
from exp.cli.shared.consent import can_prompt, require_spend_consent
from exp.cli.shared.options import usage_error
from exp.cli.shared.theme import EXP_THEME
from exp.common.progress import ProgressHook
from exp.common.project import ProjectStore
from exp.common.release_revision import installed_release_revision
from exp.optimize.evaluation.fidelity import execute_fidelity_evaluation, saved_fidelity_report
from exp.optimize.evaluation.miss_retest import (
    TurnChanges,
    register_miss_retest,
    saved_retest_results,
    turn_changes,
)
from exp.optimize.evaluation.trace_learning import learning_curve
from exp.optimize.evaluation.trace_replay import ReplayAggregate, aggregate


@dataclass(frozen=True)
class RetestOutcome:
    """The completed miss retest, for a calling command's summary.

    Attributes:
        total: Sums over the retested traces.
        planned: Traces in the retest panel.
        changes: Turns fixed and regressed against their first measurement.
    """

    total: ReplayAggregate
    planned: int
    changes: TurnChanges


def run_miss_retest(
    project: str,
    count: int,
    *,
    root: Path,
    yes: bool,
    non_interactive: bool,
    dry_run: bool,
    verbose: bool = False,
    retry_incomplete: bool = False,
    details: Console | None = None,
    terminal: Console | None = None,
    progress: ProgressHook | None = None,
    pause: Callable[[], AbstractContextManager[None]] | None = None,
    command: str | None = None,
) -> RetestOutcome | None:
    """Replay learned traces the simulator got wrong on its final memory, or replay the result.

    Args:
        project: Existing project with a completed build and learning pass.
        count: Maximum missed traces, earliest in consumption order first.
        root: Existing project and artifact root.
        yes: Explicit authorization after the undefined-cost warning.
        non_interactive: Disable terminal questions.
        dry_run: Register the runs and stop before credentials or provider calls.
        verbose: Also print every trace, every turn, and every criterion.
        retry_incomplete: Replace interrupted runs, buying their calls again.
        details: Where the full retest output is written; defaults to the terminal.
        terminal: The user's terminal, for the spend question and errors.
        progress: The calling command's single progress line; without it, the retest owns one.
        pause: Releases that progress line while the spend question is asked.
        command: Exact command a refusal names; defaults to the bare command.

    Returns:
        The completed retest, or None for a dry run or a declined spend.
    """
    screen = terminal or Console(theme=EXP_THEME)
    console = details or screen
    with usage_error(OSError, ValueError):
        store = ProjectStore(root, project)
        panel = register_miss_retest(
            store,
            count,
            code_revision=installed_release_revision(),
            clock=lambda: datetime.now(UTC),
            retry_incomplete=retry_incomplete,
        )
        heading(console, project, "Miss retest")
        judge = panel.prepared[0].spec.judge_alias
        steps = sum(len(case.steps) for case in panel.cases)
        console.print(
            Text(
                f"{len(panel.cases)} of {panel.learned} learned traces had a miss · {steps} turns "
                f"· hybrid simulator {panel.world_model.artifact_id} · "
                f"memory v{panel.memory_version} · judge {judge}",
                style="dim",
            )
        )
        if panel.retired:
            console.print(
                Text(
                    f"{len(panel.retired)} earlier attempts could not complete; "
                    "registered fresh attempts",
                    style="dim",
                )
            )
        if dry_run:
            for prepared in panel.prepared:
                console.print(Text(f"  {prepared.spec.run_id}"))
            console.print(
                "Prepared without provider calls. Run the same command without --dry-run."
            )
            return None
        pending = [p for p in panel.prepared if saved_fidelity_report(store, p) is None]
        if not pending:
            console.print(Text("Saved results. No provider calls.", style="dim"))
            for prepared in panel.prepared:
                execute_fidelity_evaluation(store, prepared)
        else:
            interactive = not non_interactive and can_prompt(screen)
            with pause() if pause is not None else nullcontext():
                consented = require_spend_consent(
                    screen,
                    root=root,
                    yes=yes,
                    estimated_cost_usd=None,
                    cost_is_upper_bound=False,
                    command=command or f"exp build {project}",
                    non_interactive=not interactive,
                )
            if not consented:
                return None
            execute_runs(screen, store, panel.prepared, progress)
        results = saved_retest_results(store, panel)
        if verbose:
            render_traces(console, store, panel.prepared, results, "Retested trace")
        console.print()
        total, changes = aggregate(results), turn_changes(panel, results)
        console.print(Text(aggregate_line(total, len(panel.cases), "Miss retest")))
        console.print(Text(changes_line(changes)))
        curve = learning_curve(store, "all")
        if curve is not None:
            console.print(Text(curve.headline()))
        console.print(
            Text(
                "The retest reads what memory learned from these traces, so it checks that "
                "lessons stuck; test-then-train fidelity measures tasks memory has not learned.",
                style="dim",
            )
        )
    return RetestOutcome(total=total, planned=len(panel.cases), changes=changes)


def changes_line(changes: TurnChanges) -> str:
    """One line: exact turns before and after learning, then fixed and regressed turns.

    Args:
        changes: Turn comparison over completed retests.

    Returns:
        Plain summary text with explicit denominators.
    """
    steps = changes.steps
    before = f"{changes.before_exact}/{steps} ({share(changes.before_exact, steps)})"
    after = f"{changes.after_exact}/{steps} ({share(changes.after_exact, steps)})"
    return (
        f"  exact before learning {before} → after {after} · fixed {changes.fixed} of "
        f"{changes.missed} missed turns · {changes.regressed} regressed"
    )
