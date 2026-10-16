"""Miss retest in the existing evaluation terminal conventions."""

from datetime import UTC, datetime
from pathlib import Path

from rich.console import Console
from rich.text import Text

from exp.cli.evaluation.held_out_fidelity import (
    aggregate_line,
    execute_runs,
    render_traces,
    require_resumable,
    share,
)
from exp.cli.evaluation.view import heading
from exp.cli.shared.consent import can_prompt, require_spend_consent
from exp.cli.shared.options import usage_error
from exp.cli.shared.theme import EXP_THEME
from exp.common.project import ProjectStore
from exp.common.release_revision import installed_release_revision
from exp.optimize.evaluation.fidelity import execute_fidelity_evaluation, saved_fidelity_report
from exp.optimize.evaluation.fidelity_recovery import QUALIFICATION, qualify
from exp.optimize.evaluation.miss_retest import (
    TurnChanges,
    register_miss_retest,
    saved_retest_results,
    turn_changes,
)
from exp.optimize.evaluation.trace_learning import learning_curve
from exp.optimize.evaluation.trace_replay import aggregate


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
) -> None:
    """Replay learned traces the simulator got wrong on its final memory, or replay the result.

    Args:
        project: Existing project with a completed build and learning pass.
        count: Maximum missed traces, earliest in consumption order first.
        root: Existing project and artifact root.
        yes: Explicit authorization after the undefined-cost warning.
        non_interactive: Disable terminal questions.
        dry_run: Register the runs and stop before credentials or provider calls.
        verbose: Also print every trace, every turn, and every criterion.
        retry_incomplete: Resume interrupted runs in place: saved work replays, and a request
            lost in flight is sent again and reported as an unknown-outcome retry.
    """
    console = Console(theme=EXP_THEME)
    with usage_error(OSError, ValueError):
        store = ProjectStore(root, project)
        panel = register_miss_retest(
            store,
            count,
            code_revision=installed_release_revision(),
            clock=lambda: datetime.now(UTC),
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
            return
        pending = [p for p in panel.prepared if saved_fidelity_report(store, p) is None]
        if not pending:
            console.print(Text("Saved results. No provider calls.", style="dim"))
            for prepared in panel.prepared:
                execute_fidelity_evaluation(store, prepared)
        else:
            require_resumable(store, panel.prepared, resume=retry_incomplete)
            interactive = not non_interactive and can_prompt(console)
            if not require_spend_consent(
                console,
                root=root,
                yes=yes,
                estimated_cost_usd=None,
                cost_is_upper_bound=False,
                command=f"exp eval {project} --retest-misses {count}",
                non_interactive=not interactive,
            ):
                return
            execute_runs(console, store, panel.prepared, resume=retry_incomplete)
        results = saved_retest_results(store, panel)
        if verbose:
            render_traces(console, store, panel.prepared, results, "Retested trace")
        console.print()
        qualification = qualify([(row.unknown_outcomes, row.exact, row.steps) for row in results])
        label = "Miss retest" if qualification is None else f"Miss retest ({QUALIFICATION})"
        console.print(Text(aggregate_line(aggregate(results), len(panel.cases), label)))
        for line in qualification.lines() if qualification is not None else ():
            console.print(Text(line))
        console.print(Text(changes_line(turn_changes(panel, results))))
        curve = learning_curve(store)
        if curve is not None:
            console.print(Text(curve.headline()))
        console.print(
            Text(
                "The retest reads what memory learned from these traces, so it checks that "
                "lessons stuck; test-then-train fidelity measures tasks memory has not learned.",
                style="dim",
            )
        )


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
