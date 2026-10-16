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
    help="With --held-out-fidelity or --retest-misses, resume interrupted runs in place: saved "
    "steps replay, and a request lost in flight is sent again and reported.",
)

RETEST_MISSES_OPTION = typer.Option(
    None,
    "--retest-misses",
    min=1,
    metavar="N",
    help="Replay up to N learned traces the simulator got wrong, on its final memory.",
)

HELD_OUT_FIDELITY_OPTION = typer.Option(
    None,
    "--held-out-fidelity",
    min=1,
    metavar="N",
    help="Measure the simulator learned with --hold-out on N held-out traces it never saw.",
)

INCLUDE_INSPECTED_OPTION = typer.Option(
    False,
    "--include-inspected",
    help="With --held-out-fidelity, also re-run held-out lineages an earlier --verbose audit "
    "showed; reported as a development re-audit, never independent.",
)

VERBOSE_OPTION = typer.Option(
    False,
    "--verbose",
    help="With --fidelity, --held-out-fidelity or --retest-misses, also print every turn and "
    "the configuration.",
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
    held_out_fidelity: int | None = HELD_OUT_FIDELITY_OPTION,
    retest_misses: int | None = RETEST_MISSES_OPTION,
    retry_incomplete: bool = RETRY_INCOMPLETE_OPTION,
    include_inspected: bool = INCLUDE_INSPECTED_OPTION,
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
        held_out_fidelity: Number of held-out build traces replayed on the fit-only simulator.
        retest_misses: Number of missed learned traces replayed on the final memory.
        retry_incomplete: Resume interrupted held-out or retest runs in place.
        include_inspected: Re-run held-out lineages an earlier audit showed in detail.
        verbose: Plain per-turn fidelity detail, for agents and logs.
    """
    replay = held_out_fidelity if held_out_fidelity is not None else retest_misses
    if retry_incomplete and replay is None:
        raise typer.BadParameter(
            "--retry-incomplete applies only with --held-out-fidelity or --retest-misses"
        )
    if include_inspected and held_out_fidelity is None:
        raise typer.BadParameter("--include-inspected applies only with --held-out-fidelity")
    if verbose and fidelity is None and replay is None:
        raise typer.BadParameter(
            "--verbose applies only with --fidelity, --held-out-fidelity or --retest-misses"
        )
    if retest_misses is not None:
        if (
            project is None
            or fidelity is not None
            or held_out_fidelity is not None
            or resume is not None
            or report is not None
            or any(
                value is not None
                for value in (models, repeats, concurrency, maximum_steps, maximum_output_tokens)
            )
        ):
            raise typer.BadParameter(
                "--retest-misses requires PROJECT and cannot combine with other evaluation modes"
            )
        from exp.cli.evaluation.miss_retest import run_miss_retest

        run_miss_retest(
            project,
            retest_misses,
            root=root,
            yes=yes,
            non_interactive=non_interactive,
            dry_run=dry_run,
            verbose=verbose,
            retry_incomplete=retry_incomplete,
        )
        return
    if held_out_fidelity is not None:
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
                "--held-out-fidelity requires PROJECT and cannot combine with other evaluation "
                "modes"
            )
        from exp.cli.evaluation.held_out_fidelity import run_held_out_fidelity

        run_held_out_fidelity(
            project,
            held_out_fidelity,
            root=root,
            yes=yes,
            non_interactive=non_interactive,
            dry_run=dry_run,
            verbose=verbose,
            retry_incomplete=retry_incomplete,
            include_inspected=include_inspected,
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
