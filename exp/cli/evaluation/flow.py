"""Project-first evaluation setup, durable execution, and report inspection."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.prompt import Confirm, IntPrompt
from rich.table import Table

from exp.cli.evaluation.view import inspect_report, render_report
from exp.cli.shared.consent import can_prompt, require_spend_consent
from exp.cli.shared.options import ROOT_OPTION, usage_error
from exp.cli.shared.picker import PickerOption, choose_many, choose_one
from exp.cli.shared.progress import progress_display
from exp.cli.shared.theme import EXP_THEME
from exp.common.models import ModelCatalog, load_model_catalog
from exp.common.project import ProjectStore
from exp.common.project.config_sqlite import list_projects
from exp.common.release_revision import installed_release_revision
from exp.optimize.evaluation.prepare import ModelEvaluationOptions
from exp.optimize.evaluation.runs import (
    EvaluationDefaults,
    EvaluationRun,
    evaluation_tasks,
    execute_run,
    list_runs,
    load_defaults,
    load_run,
    prepare_run,
    save_defaults,
)
from exp.runtime.models import RuntimeModelCatalog

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
        yes: Confirm an in-budget quote.
        non_interactive: Disable terminal questions.
    """
    interactive = not non_interactive and can_prompt(_console)
    with usage_error(OSError, ValueError):
        if project is None:
            if not interactive:
                raise ValueError("provide PROJECT or run exp eval in an interactive terminal")
            options = tuple(
                PickerOption(project_id, project_id) for project_id in list_projects(root)
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
            render_report(_console, store, completed)
            if interactive:
                inspect_report(_console, store, completed)
            return
        if resume is not None and any(value is not None for value in overrides):
            raise ValueError("--resume uses frozen settings; start a new evaluation to change them")
        defaults = load_defaults(store)
        catalog = load_model_catalog(store.model_catalog_path)
        if resume is None and interactive and models is None:
            resume = _project_screen(store)
            if resume == "exit":
                return
            if resume is not None:
                selected = load_run(store, resume)
                if selected.status == "completed":
                    render_report(_console, store, selected)
                    inspect_report(_console, store, selected)
                    return
        if resume is not None:
            run = load_run(store, resume)
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
                selected_defaults = _configure(store, catalog, selected_defaults)
                if selected_defaults is None:
                    return
            run = prepare_run(
                store, catalog, selected_defaults, code_revision=installed_release_revision()
            )
            save_defaults(store, selected_defaults)
        _preflight(store, run)
        if dry_run:
            _console.print("Prepared without provider calls. Resume with:")
            _console.print(f"exp eval {project} --root {root} --resume {run.run_id}", markup=False)
            return
        if not require_spend_consent(
            _console,
            root=root,
            yes=yes,
            estimated_cost_usd=run.prepared.cost.maximum_cost_usd,
            command=f"exp eval {project} --resume {run.run_id}",
            non_interactive=not interactive,
        ):
            return
        _console.print(
            "Ctrl-C stops queued work and drains active rollouts; finished work stays saved."
        )
        try:
            with progress_display(_console) as progress:
                execute_run(
                    store,
                    run,
                    RuntimeModelCatalog(catalog),
                    provider_spend_consented=True,
                    progress=progress,
                )
        except KeyboardInterrupt:
            _console.print(
                f"Saved. Resume: exp eval {project} --root {root} --resume {run.run_id}",
                markup=False,
            )
            raise typer.Exit(130) from None
        finished = load_run(store, run.run_id)
        render_report(_console, store, finished)
        if interactive:
            inspect_report(_console, store, finished)


def _project_screen(project: ProjectStore) -> str | None:
    """Show project coverage and let the user resume or inspect saved work."""
    tasks = evaluation_tasks(project)
    config = project.load_project()
    _console.print(f"\n[bold]{project.paths.project_id}[/bold] · {len(tasks)} distinct scenarios")
    if config.models:
        _console.print(
            f"Environment: {config.models.world_model} · Judge: {config.models.judge}", markup=False
        )
    runs = list_runs(project)
    choices = [PickerOption("new", "New evaluation", "choose models and review settings")]
    choices.extend(
        PickerOption(run.run_id, f"{run.status.title()} · {run.run_id}", run.stage) for run in runs
    )
    choices.append(PickerOption("exit", "Back"))
    choice = choose_one(_console, title="Model evaluations", options=choices)
    if not choice.values:
        return "exit"
    return None if choice.values[0] == "new" else choice.values[0]


def _configure(
    project: ProjectStore, catalog: ModelCatalog, defaults: EvaluationDefaults
) -> EvaluationDefaults | None:
    """Collect model choices and explicit rollout budgets through terminal controls."""
    del project
    chosen = choose_many(
        _console,
        title="Models to evaluate",
        minimum=2,
        preselected=defaults.models,
        options=tuple(
            PickerOption(
                alias,
                alias,
                f"{alias} · reasoning {model.capabilities.reasoning_effort or 'provider default'}",
            )
            for alias, model in sorted(catalog.models.items())
            if model.capabilities is not None
            and model.capabilities.supports_completions is not False
        ),
    )
    if not chosen.values:
        return None
    options = defaults.options
    repeats = IntPrompt.ask("Valid runs per scenario", default=options.repeats, console=_console)
    concurrency = IntPrompt.ask(
        "Parallel rollouts", default=options.maximum_concurrency, console=_console
    )
    steps = IntPrompt.ask(
        "Maximum steps per rollout", default=options.maximum_steps, console=_console
    )
    tokens = IntPrompt.ask(
        "Maximum generated tokens per rollout",
        default=options.maximum_rollout_output_tokens,
        console=_console,
    )
    parsed = ModelEvaluationOptions.model_validate(
        {
            **options.model_dump(),
            "repeats": repeats,
            "maximum_concurrency": concurrency,
            "maximum_steps": steps,
            "maximum_rollout_output_tokens": tokens,
        }
    )
    if not Confirm.ask("Save these project defaults?", default=True, console=_console):
        return None
    return defaults.model_copy(update={"models": chosen.values, "options": parsed})


def _preflight(project: ProjectStore, run: EvaluationRun) -> None:
    """Display the exact matrix, judge, capacities, and separately priced execution stages."""
    cost = run.prepared.cost
    setup = run.prepared.setup
    _console.print(f"\n[bold]Review evaluation · {project.paths.project_id}[/bold]")
    _console.print(
        f"{cost.scenario_count} distinct scenarios × {cost.worker_count} models × "
        f"{setup.repeats} repeats = {cost.judgment_count} planned rollouts"
    )
    _console.print(
        f"{setup.maximum_steps} steps · "
        f"{setup.maximum_rollout_output_tokens:,} output tokens/rollout · "
        f"{setup.maximum_concurrency} parallel workers"
    )
    _console.print(
        f"World model: {setup.world_model_settings.world_model_alias} · "
        f"Judge: {run.prepared.judge_request.model.model_id}",
        markup=False,
    )
    _console.print(
        f"Judge status: {setup.judgment_status}. "
        "Invalid attempts are excluded from quality and assistant cost."
    )
    table = Table("Stage", "Estimate", "Reserved maximum", box=None)
    for label, component in (
        ("Assistant", cost.workers),
        ("World model", cost.simulation),
        ("Retrieval", cost.retrieval),
        ("Judge", cost.judge),
    ):
        table.add_row(
            label, f"${component.estimated_cost_usd:.4f}", f"${component.maximum_cost_usd:.4f}"
        )
    _console.print(table)
    _console.print(f"Run: {run.run_id}", markup=False)
