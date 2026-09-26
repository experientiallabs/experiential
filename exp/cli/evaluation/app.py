"""Light CLI command declaration for the evaluation application boundary."""

from pathlib import Path

import typer

from exp.cli.shared.options import ROOT_OPTION


def evaluate(
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
    # Defer the application graph at command dispatch, keeping config/help independent.
    from exp.cli.evaluation.flow import run_evaluation

    run_evaluation(
        project=project,
        models=models,
        root=root,
        repeats=repeats,
        concurrency=concurrency,
        maximum_steps=maximum_steps,
        maximum_output_tokens=maximum_output_tokens,
        resume=resume,
        report=report,
        dry_run=dry_run,
        yes=yes,
        non_interactive=non_interactive,
    )
