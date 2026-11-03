"""Public manifest-driven development factory using the existing paid stage owners."""

from pathlib import Path

import typer
from rich.console import Console
from rich.markup import escape
from rich.text import Text

from exp.cli.evaluation.spend import review_cost
from exp.cli.evaluation.spend_ceiling import ceiling_line, enforced_ceiling, inventory_terms
from exp.cli.shared.consent import require_spend_consent
from exp.cli.shared.options import usage_error
from exp.cli.shared.progress import progress_display
from exp.cli.shared.theme import EXP_THEME
from exp.common.core.artifacts import stable_id
from exp.common.project import ProjectStore
from exp.common.release_revision import installed_execution_revision
from exp.optimize.evaluation.executable.costs import evaluation_costs
from exp.optimize.evaluation.factory.runner import (
    TaskFactorySpec,
    factory_attempt_bound,
    factory_population,
    factory_requires_execution,
    preflight_task_factory,
    run_task_factory,
    saved_task_factory,
)
from exp.runtime.models.spend_ceiling import SpendCeilingReached, UnpricedDispatchRefused
from exp.simulation.task_distribution import select_distribution


def run_pipeline(
    project: str, path: Path, *, root: Path, yes: bool, non_interactive: bool, dry_run: bool
) -> None:
    """Validate a frozen mixture, qualify its families and verify delivered package bytes.

    Args:
        project: Existing source project containing seed worlds and independent witnesses.
        path: Typed TaskFactorySpec JSON manifest, with no credentials or generated code.
        root: Existing project store root.
        yes: Explicit authorization of the displayed ceiling, including the attempts of any
            unpriced roles.
        non_interactive: Fail rather than prompt when paid work lacks authorization.
        dry_run: Check available source contracts without dispatch or publication.
    """
    console = Console(theme=EXP_THEME)
    try:
        _run_pipeline(
            console,
            project,
            path,
            root=root,
            yes=yes,
            non_interactive=non_interactive,
            dry_run=dry_run,
        )
    except (SpendCeilingReached, UnpricedDispatchRefused) as stop:
        console.print(f"[red]error[/red] {escape(str(stop))}")
        console.print(
            "Completed work is saved; the operation's spend receipt is under the project's "
            "runtime/spend directory. Run the same command again to resume."
        )
        raise typer.Exit(code=1) from None


def _run_pipeline(
    console: Console,
    project: str,
    path: Path,
    *,
    root: Path,
    yes: bool,
    non_interactive: bool,
    dry_run: bool,
) -> None:
    """Run the factory under its consented ceiling; see :func:`run_pipeline`."""
    with usage_error(OSError, ValueError, RuntimeError):
        spec = TaskFactorySpec.model_validate_json(path.read_bytes())
        store = ProjectStore(root, project)
        revision = installed_execution_revision()
        completed = saved_task_factory(store, spec, code_revision=revision)
        if completed is None:
            preflight_task_factory(store, spec, code_revision=revision)
        planned = select_distribution(spec.distribution, factory_population(store, spec))
        console.print(
            f"Task factory: {len(spec.families)} seeds, {len(planned.components)} distinct "
            f"source/workflow components, {len(spec.actors)} actors; at most "
            f"{factory_attempt_bound(store, spec)} author/actor/proxy attempts until "
            f"{spec.deadline.isoformat()}. Runtime inference has separate allowances."
        )
        console.print_json(data={"planned_coverage": planned.model_dump(mode="json")})
        if dry_run:
            console.print(
                "Sources and bounds checked. No new calls, runtime execution or packages."
            )
            return
        command = f"exp eval {project} --pipeline {path}"
        operation = stable_id("pipeline", spec)
        # One ceiling over every author and actor role: priced roles in dollars, unpriced ones
        # within their displayed attempts, which only an explicit confirmation authorizes.
        authorized: float | None = None
        unpriced = 0
        if factory_requires_execution(store, spec, code_revision=revision):
            known, unpriced = inventory_terms(evaluation_costs(store, spec))
            review_cost(console, store, spec)
            console.print(Text(ceiling_line(known, unpriced), style="dim"))
            if not require_spend_consent(
                console,
                root=root,
                yes=yes,
                estimated_cost_usd=None if unpriced else known,
                cost_is_upper_bound=not unpriced,
                command=command,
                non_interactive=non_interactive,
            ):
                return
            authorized = known
        with (
            enforced_ceiling(
                store,
                operation,
                stage=operation,
                ceiling_usd=authorized,
                unpriced_attempts=unpriced,
                command=command,
            ),
            progress_display(console) as progress,
        ):
            result = completed or run_task_factory(
                store, spec, code_revision=revision, progress=progress
            )
        console.print_json(data=result.model_dump(mode="json"))
