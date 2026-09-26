"""Project-first evaluation setup, durable execution, and report inspection."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from pathlib import Path

import typer
from rich.console import Console
from rich.prompt import FloatPrompt
from rich.table import Table
from rich.text import Text

from exp.cli.evaluation.setup import configure_evaluation
from exp.cli.evaluation.view import heading, inspect_report, render_report
from exp.cli.shared.consent import can_prompt, require_spend_consent
from exp.cli.shared.options import ROOT_OPTION, usage_error
from exp.cli.shared.picker import PickerOption, choose_one
from exp.cli.shared.progress import progress_display
from exp.cli.shared.theme import EXP_THEME
from exp.common.models import load_model_catalog
from exp.common.progress import ProgressEvent, ProgressHook
from exp.common.project import ProjectStore
from exp.common.release_revision import installed_release_revision
from exp.optimize.evaluation.export import export_report
from exp.optimize.evaluation.judging_resume import prepare_judging_revision
from exp.optimize.evaluation.runs import (
    EvaluationRun,
    evaluation_tasks,
    execute_run,
    list_runs,
    load_defaults,
    load_run,
    prepare_run,
    save_defaults,
    save_run,
)
from exp.runtime.models import RuntimeModelCatalog
from exp.runtime.models.budget import SpendLimitReached
from exp.simulation.engines.text.errors import SimulationContentionError

_console = Console(theme=EXP_THEME)


def run_evaluation(
    project: str | None = typer.Argument(None, metavar="PROJECT"),
    models: str | None = typer.Option(
        None, "--models", help="Comma-separated configured model aliases."
    ),
    root: Path = ROOT_OPTION,
    repeats: int | None = typer.Option(
        None, "--repeats", min=1, help="Valid repeats per distinct scenario."
    ),
    concurrency: int | None = typer.Option(None, "--concurrency", min=1),
    maximum_steps: int | None = typer.Option(None, "--max-steps", min=1),
    maximum_output_tokens: int | None = typer.Option(
        None, "--max-output-tokens", min=1, help="Cumulative generated tokens per rollout."
    ),
    resume: str | None = typer.Option(None, "--resume", help="Resume an exact saved run ID."),
    report: str | None = typer.Option(
        None, "--report", help="Inspect a completed run without provider calls."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Prepare and show the matrix and estimate."
    ),
    yes: bool = typer.Option(False, "--yes", "-y"),
    non_interactive: bool = typer.Option(False, "--non-interactive"),
) -> None:
    """Evaluate models against a project's simulated scenarios and configured judge.

    Args:
        project: Named grounded project, or an interactive project picker.
        models: Explicit comma-separated worker aliases.
        root: Local project and artifact root.
        repeats: Valid repeats per scenario; infrastructure retries are separate.
        concurrency: Maximum simultaneously active rollouts.
        maximum_steps: Candidate-turn budget.
        maximum_output_tokens: Cumulative rollout generation budget.
        resume: Exact saved run ID; rejects configuration overrides.
        report: Completed run ID to inspect without dispatch.
        dry_run: Stop before credentials and provider calls.
        yes: Confirm the quote, including any budget warning.
        non_interactive: Disable terminal questions.
    """
    interactive = not non_interactive and can_prompt(_console)
    with usage_error(OSError, ValueError):
        if project is None:
            if not interactive:
                raise ValueError("provide PROJECT or run exp eval in an interactive terminal")
            options = tuple(
                PickerOption(path.name, path.name)
                for path in sorted((root / "projects").glob("*/project.toml"))
                for path in (path.parent,)
            )
            if not options:
                raise ValueError(
                    "no local projects; run exp build NAME --traces PATH --source chat-json"
                )
            picked = choose_one(_console, title="Evaluation project", options=options)
            if not picked.values:
                return
            project = picked.values[0]
        store = ProjectStore(root, project)
        overrides = (models, repeats, concurrency, maximum_steps, maximum_output_tokens)
        if report is not None:
            if resume is not None or any(value is not None for value in overrides):
                raise ValueError("--report cannot be combined with execution options")
            completed = load_run(store, report)
            _results(store, completed, interactive=interactive)
            return
        if resume is not None and any(value is not None for value in overrides):
            raise ValueError("--resume uses frozen settings; start a new evaluation to change them")
        evaluation_tasks(store)
        defaults = load_defaults(store)
        if resume is None and interactive and models is None:
            resume = _project_screen(store)
            if resume == "exit":
                return
            if resume is not None:
                selected = load_run(store, resume)
                if selected.status == "completed":
                    _results(store, selected, interactive=interactive)
                    return
        catalog = load_model_catalog(store.model_catalog_path)
        if resume is not None:
            with progress_display(_console, single_line=True) as progress:
                progress(ProgressEvent(stage="Loading saved evaluation"))
                run = load_run(store, resume)
                if run.status == "failed" and run.stage in {"judging", "judgments"}:
                    pointer = prepare_judging_revision(
                        store,
                        run.prepared,
                        catalog,
                        previous=run.judging_revision,
                        created_at=datetime.now(UTC),
                        code_revision=installed_release_revision(),
                    )
                    run = run.model_copy(update={"judging_revision": pointer})
        else:
            aliases = (
                tuple(value.strip() for value in models.split(",") if value.strip())
                if models is not None
                else defaults.models
            )
            options = defaults.options.model_copy(
                update={
                    **({"repeats": repeats} if repeats is not None else {}),
                    **({"maximum_concurrency": concurrency} if concurrency is not None else {}),
                    **({"maximum_steps": maximum_steps} if maximum_steps is not None else {}),
                    **(
                        {"maximum_rollout_output_tokens": maximum_output_tokens}
                        if maximum_output_tokens is not None
                        else {}
                    ),
                }
            )
            selected_defaults = defaults.model_copy(update={"models": aliases, "options": options})
            if interactive and models is None:
                selected_defaults = configure_evaluation(
                    _console, store, catalog, selected_defaults
                )
                if selected_defaults is None:
                    return
            with progress_display(_console, single_line=True) as progress:
                run = prepare_run(
                    store,
                    catalog,
                    selected_defaults,
                    code_revision=installed_release_revision(),
                    progress=progress,
                )
                save_defaults(store, selected_defaults)
        _preflight(store, run)
        if dry_run:
            _console.print("Prepared without provider calls. Resume with:")
            _console.print(f"exp eval {project} --root {root} --resume {run.run_id}", markup=False)
            return
        while True:
            if interactive and not yes:
                selected_run = _review(store, run)
                if selected_run is None:
                    return
                run = selected_run
            if not require_spend_consent(
                _console,
                root=root,
                yes=yes,
                estimated_cost_usd=run.spending_limit_usd,
                command=f"exp eval {project} --resume {run.run_id}",
                non_interactive=not interactive,
            ):
                return
            save_run(store, run)
            heading(_console, project, "Running evaluation · Ctrl-C to pause")
            try:
                with progress_display(_console, single_line=True) as progress:
                    progress(ProgressEvent(stage="Preparing evaluation"))
                    execute_run(
                        store,
                        run,
                        RuntimeModelCatalog(catalog),
                        provider_spend_consented=True,
                        progress=_compact_progress(progress),
                    )
                break
            except SpendLimitReached as exc:
                run = load_run(store, run.run_id)
                _console.print(
                    f"Paused at the ${exc.limit_usd:,.2f} spending limit. Completed calls saved."
                )
                if not interactive:
                    _console.print(
                        Text(f"Resume: exp eval {project} --root {root} --resume {run.run_id}")
                    )
                    return
                yes = False
                _console.print("Increase the spending limit to continue this evaluation.")
            except SimulationContentionError:
                _console.print("Paused: rollout state is busy. Completed work saved.")
                _console.print(
                    Text(f"Resume: exp eval {project} --root {root} --resume {run.run_id}")
                )
                return
            except KeyboardInterrupt:
                _console.print(
                    f"Saved. Resume: exp eval {project} --root {root} --resume {run.run_id}",
                    markup=False,
                )
                raise typer.Exit(130) from None
        finished = load_run(store, run.run_id)
        _results(store, finished, interactive=interactive)


def _results(project: ProjectStore, run: EvaluationRun, *, interactive: bool) -> None:
    """Choose the compact interactive report or a script-readable export receipt.

    Args:
        project: Local project containing the run's immutable evidence.
        run: Saved evaluation whose results should be displayed and exported.
        interactive: Whether to offer report browsing instead of an export receipt.

    Raises:
        OSError: Local report evidence cannot be read or exports cannot be written.
        ValueError: Saved report evidence is missing or inconsistent.
    """
    if interactive:
        inspect_report(_console, project, run)
    else:
        render_report(_console, project, run)
        _, html = export_report(project, run)
        _console.print(Text(f"\nReport: {html}"))


def _project_screen(project: ProjectStore) -> str | None:
    """Keep starting an evaluation separate from browsing saved results.

    Args:
        project: Built project supplying scenarios and saved evaluation runs.

    Returns:
        A saved run ID, ``None`` for a new evaluation, or ``"exit"`` on cancellation.

    Raises:
        OSError: Saved project or run metadata cannot be read.
        ValueError: The build or stored evaluation evidence is invalid.
    """
    tasks = evaluation_tasks(project)
    heading(_console, project.paths.project_id, f"{len(tasks)} scenarios")
    while True:
        choices = [PickerOption("new", "New evaluation")]
        runs = list_runs(project)
        if runs:
            choices.append(PickerOption("saved", "Saved evaluations"))
        choices.append(PickerOption("exit", "Back"))
        choice = choose_one(_console, title="Evaluations", options=choices)
        if not choice.values or choice.values[0] == "exit":
            return "exit"
        if choice.values[0] == "new":
            return None
        selected = choose_one(
            _console,
            title="Saved evaluations",
            options=tuple(
                PickerOption(
                    run.run_id,
                    f"{run.created_at.astimezone():%b %d, %H:%M:%S} · {run.status.title()}",
                    f"{len(run.prepared.setup.candidates)} models",
                )
                for run in runs
            ),
        )
        if selected.values:
            return selected.values[0]


def _preflight(project: ProjectStore, run: EvaluationRun) -> None:
    """Show the model matrix and costs in one short launch review.

    Args:
        project: Project supplying the display name.
        run: Prepared evaluation supplying frozen models, repeats, and cost estimates.
    """
    cost = run.prepared.cost
    setup = run.prepared.setup
    heading(_console, project.paths.project_id, "Review evaluation")
    _console.print(
        f"{cost.scenario_count} scenarios × {cost.worker_count} models × "
        f"{setup.repeats} {'run' if setup.repeats == 1 else 'runs'} = "
        f"{cost.judgment_count} rollouts\n"
    )
    _console.print(Text("Models: " + ", ".join(candidate.alias for candidate in setup.candidates)))
    _console.print(Text(f"World model: {setup.world_model_settings.world_model_alias}"))
    _console.print(
        Text(f"Judge: {run.prepared.judge_request.model.model_id} ({setup.judgment_status})")
    )
    if run.judging_revision is not None:
        _console.print("Retry judging with full model context. Saved rollouts are reused.")
    _console.print(
        f"\nEstimated ${cost.estimated_cost_usd:,.2f} · "
        f"Spending limit ${run.spending_limit_usd:,.2f}"
    )


def _review(project: ProjectStore, run: EvaluationRun) -> EvaluationRun | None:
    """Require an explicit launch action before requesting spend consent.

    Args:
        project: Project supplying the display name for review screens.
        run: Prepared evaluation supplying the frozen per-stage cost breakdown.

    Returns:
        Reviewed run only after Start evaluation; None after Back or cancellation.
        Cost details and spending-limit edits never authorize a provider call.
    """
    while True:
        choice = choose_one(
            _console,
            title="Ready",
            options=(
                PickerOption(
                    "start", "Resume evaluation" if run.status != "prepared" else "Start evaluation"
                ),
                PickerOption("limit", "Spending limit"),
                PickerOption("cost", "Cost details"),
                PickerOption("back", "Back"),
            ),
        )
        if not choice.values or choice.values[0] == "back":
            return None
        if choice.values[0] == "start":
            if (
                run.required_spending_limit_usd
                and run.spending_limit_usd < run.required_spending_limit_usd
            ):
                _console.print("Increase the spending limit before resuming.")
                continue
            return run
        if choice.values[0] == "limit":
            suggested = max(run.spending_limit_usd, run.required_spending_limit_usd or 0)
            limit = FloatPrompt.ask(
                "Total spending limit ($)",
                console=_console,
                default=math.ceil(suggested * 100) / 100,
            )
            if not math.isfinite(limit) or limit <= 0:
                _console.print("Enter a positive dollar amount.")
                continue
            run = run.model_copy(update={"spending_limit_usd": limit})
            _preflight(project, run)
            continue
        heading(_console, project.paths.project_id, "Cost details")
        table = Table("Stage", "Estimate", box=None)
        for label, component in (
            ("Assistant", run.prepared.cost.workers),
            ("World model", run.prepared.cost.simulation),
            ("Retrieval", run.prepared.cost.retrieval),
            ("Judge", run.prepared.cost.judge),
        ):
            table.add_row(
                label,
                f"${component.estimated_cost_usd:,.4f}",
            )
        _console.print(table)
        _console.print(Text(run.prepared.cost.estimate_basis), style="dim")
        _console.print(
            f"Captured turns with measured tokens: {run.prepared.cost.measured_turns:g} / "
            f"{run.prepared.cost.captured_turns:g}. Pauses before exceeding the spending limit.",
            style="dim",
        )


def _compact_progress(progress: ProgressHook) -> ProgressHook:
    """Keep durable detailed events intact while rendering only the stage and counts.

    Args:
        progress: Terminal progress sink receiving the compact event projection.

    Returns:
        Observer that forwards stage and counts without changing source events.
    """

    def observe(event: ProgressEvent) -> None:
        """Forward a concise view of the engine's observed progress."""
        stage = {
            "preflight": "Checking evaluation",
            "simulation": "Starting rollouts",
            "evaluation cells": "Rollouts",
            "judging": "Starting judging",
            "judgments": "Judging",
            "report": "Building report",
            "completed": "Complete",
        }.get(event.stage, event.stage)
        progress(ProgressEvent(stage=stage, completed=event.completed, total=event.total))

    return observe
