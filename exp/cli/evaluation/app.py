"""Light CLI command declaration for the evaluation application boundary."""

from pathlib import Path

import typer

from exp.cli.shared.options import ROOT_OPTION

FIDELITY_OPTION = typer.Option(
    None, "--fidelity", help="Evaluate a pinned simulation-fidelity workload."
)

RETRY_INCOMPLETE_OPTION = typer.Option(
    False,
    "--retry-incomplete",
    help="With --audit, buy again the calls of runs that stopped incomplete.",
)

AUDIT_OPTION = typer.Option(
    None,
    "--audit",
    min=1,
    metavar="N",
    help="Independently audit the simulator on N held-out traces it never learned.",
)

VERBOSE_OPTION = typer.Option(
    False,
    "--verbose",
    help="With --fidelity or --audit, also print every turn and the configuration.",
)


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
    fidelity: Path | None = FIDELITY_OPTION,
    audit: int | None = AUDIT_OPTION,
    retry_incomplete: bool = RETRY_INCOMPLETE_OPTION,
    verbose: bool = VERBOSE_OPTION,
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
        fidelity: Explicit reconstruction workload, separate from worker-model evaluation.
        audit: Number of held-out traces replayed on a separately learned fit-only memory.
        retry_incomplete: Replace interrupted audit runs, buying their calls again.
        verbose: Plain per-turn fidelity detail, for agents and logs.
    """
    if retry_incomplete and audit is None:
        raise typer.BadParameter("--retry-incomplete applies only with --audit")
    if verbose and fidelity is None and audit is None:
        raise typer.BadParameter("--verbose applies only with --fidelity or --audit")
    if audit is not None:
        if (
            project is None
            or fidelity is not None
            or resume is not None
            or report is not None
            or any(
                value is not None
                for value in (models, repeats, concurrency, maximum_steps, maximum_output_tokens)
            )
        ):
            raise typer.BadParameter(
                "--audit requires PROJECT and cannot combine with other evaluation modes"
            )
        from exp.cli.evaluation.held_out_fidelity import run_audit

        run_audit(
            project,
            audit,
            root=root,
            yes=yes,
            non_interactive=non_interactive,
            dry_run=dry_run,
            verbose=verbose,
            retry_incomplete=retry_incomplete,
        )
        return
    if fidelity is not None:
        if (
            project is None
            or resume is not None
            or report is not None
            or any(
                value is not None
                for value in (models, repeats, concurrency, maximum_steps, maximum_output_tokens)
            )
        ):
            raise typer.BadParameter(
                "--fidelity requires PROJECT and cannot combine with worker-evaluation options"
            )
        from exp.cli.evaluation.fidelity import run_fidelity

        run_fidelity(
            project,
            fidelity,
            root=root,
            yes=yes,
            non_interactive=non_interactive,
            dry_run=dry_run,
            verbose=verbose,
        )
        return
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
