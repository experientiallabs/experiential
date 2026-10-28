"""Light CLI command declaration for the evaluation application boundary."""

from pathlib import Path

import typer

from exp.cli.shared.options import ROOT_OPTION

TRACE_ANALYSIS_OPTION = typer.Option(
    None,
    "--trace-analysis",
    help="Council spec path, or auto for the five-model council; --models overrides actors.",
)

REFINE_TASK_OPTION = typer.Option(
    None, "--refine-task", help="Refine a simulator task against protected verifier challenges."
)

CALIBRATION_OPTION = typer.Option(
    None, "--calibrate", help="Calibrate a frozen task from independently labeled saved evidence."
)

FIDELITY_OPTION = typer.Option(
    None, "--fidelity", help="Evaluate a pinned simulation-fidelity workload."
)

RETRY_INCOMPLETE_OPTION = typer.Option(
    False,
    "--retry-incomplete",
    help="With --audit, resume interrupted runs in place: saved steps replay, and a request "
    "lost in flight is sent again and reported.",
)

AUDIT_OPTION = typer.Option(
    None,
    "--audit",
    min=1,
    metavar="N",
    help="Independently audit the simulator on N held-out traces it never learned.",
)

INCLUDE_INSPECTED_OPTION = typer.Option(
    False,
    "--include-inspected",
    help="With --audit, also re-run held-out lineages an earlier --verbose audit showed; "
    "reported as a development re-audit, never independent.",
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
    include_inspected: bool = INCLUDE_INSPECTED_OPTION,
    verbose: bool = VERBOSE_OPTION,
    trace_analysis: Path | None = TRACE_ANALYSIS_OPTION,
    refine_task: Path | None = REFINE_TASK_OPTION,
    calibration: Path | None = CALIBRATION_OPTION,
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
        retry_incomplete: Resume interrupted audit runs in place.
        include_inspected: Re-run held-out lineages an earlier audit showed in detail.
        verbose: Plain per-turn fidelity detail, for agents and logs.
        trace_analysis: Simulator-only actor council and read-only evaluation refinement.
        refine_task: Bounded task refinement with independent qualification checks.
        calibration: Saved-evidence verifier and rubric calibration without new inference.
    """
    if calibration is not None:
        if (
            project is None
            or any(
                value is not None
                for value in (
                    fidelity,
                    audit,
                    trace_analysis,
                    refine_task,
                    resume,
                    report,
                    models,
                    repeats,
                    concurrency,
                    maximum_steps,
                    maximum_output_tokens,
                )
            )
            or verbose
            or retry_incomplete
            or include_inspected
        ):
            raise typer.BadParameter(
                "--calibrate requires PROJECT and cannot combine with other modes "
                "or rollout overrides"
            )
        from exp.cli.evaluation.calibration import run_calibration

        run_calibration(project, calibration, root=root, dry_run=dry_run)
        return
    if refine_task is not None:
        if (
            project is None
            or fidelity is not None
            or audit is not None
            or trace_analysis is not None
            or resume is not None
            or report is not None
            or verbose
            or any(
                value is not None
                for value in (models, repeats, concurrency, maximum_steps, maximum_output_tokens)
            )
        ):
            raise typer.BadParameter(
                "--refine-task requires PROJECT and cannot combine with other evaluation modes "
                "or model and rollout overrides"
            )
        from exp.cli.evaluation.task_refinement import run_task_refinement

        run_task_refinement(
            project,
            refine_task,
            root=root,
            yes=yes,
            non_interactive=non_interactive,
            dry_run=dry_run,
        )
        return
    if trace_analysis == Path("auto"):
        if (
            project is None
            or fidelity is not None
            or audit is not None
            or resume is not None
            or report is not None
            or verbose
            or any(value is not None for value in (repeats, maximum_steps, maximum_output_tokens))
        ):
            raise typer.BadParameter(
                "--trace-analysis auto requires PROJECT; "
                "other evaluation overrides are not supported"
            )
        from exp.cli.evaluation.built_council import run_built_council

        run_built_council(
            project,
            tuple(alias.strip() for alias in models.split(",") if alias.strip())
            if models is not None
            else None,
            root=root,
            maximum_concurrency=concurrency or 1,
            yes=yes,
            non_interactive=non_interactive,
            dry_run=dry_run,
        )
        return
    if trace_analysis is not None:
        if (
            project is None
            or fidelity is not None
            or audit is not None
            or resume is not None
            or report is not None
            or any(
                value is not None
                for value in (models, repeats, concurrency, maximum_steps, maximum_output_tokens)
            )
        ):
            raise typer.BadParameter(
                "--trace-analysis requires PROJECT and cannot combine with other evaluation modes"
            )
        from exp.cli.evaluation.trace_analysis import run_trace_analysis

        run_trace_analysis(
            project,
            trace_analysis,
            root=root,
            yes=yes,
            non_interactive=non_interactive,
            dry_run=dry_run,
        )
        return
    if retry_incomplete and audit is None:
        raise typer.BadParameter("--retry-incomplete applies only with --audit")
    if include_inspected and audit is None:
        raise typer.BadParameter("--include-inspected applies only with --audit")
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
