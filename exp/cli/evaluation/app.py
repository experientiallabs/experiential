"""Light CLI command declaration for the evaluation application boundary."""

from pathlib import Path
from typing import Literal

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

MATERIALIZE_OPTION = typer.Option(
    None, "--materialize", help="Materialize declared native edits from a frozen checkpoint."
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

AuditMemory = Literal["learned", "none", "lessons", "bare", "guidance"]
AUDIT_MEMORIES: tuple[AuditMemory, ...] = ("learned", "none", "lessons", "bare", "guidance")
AUDIT_MEMORY_OPTION = typer.Option(
    None,
    "--audit-memory",
    metavar="learned|none|lessons|bare|guidance",
    help="With --audit, the memory the held-out replays read: all learned memory (default), "
    "none (recorded originals only), only the checkpoint's screened lessons, bare (no "
    "originals and no memory), or guidance (originals plus temporary session-scoped notes "
    "that are never persisted). Every other spec field is identical.",
)
AUDIT_CHECKPOINT_OPTION = typer.Option(
    None,
    "--audit-checkpoint",
    min=1,
    metavar="K",
    help="With --audit, read the frozen memory version that had consumed exactly K fit "
    "trajectories instead of the final one (for sample-count grids).",
)

VERBOSE_OPTION = typer.Option(
    False,
    "--verbose",
    help="With --fidelity or --audit, also print every turn and the configuration.",
)


CONTINUATIONS_OPTION = typer.Option(
    None,
    "--continuations",
    help="Run one ContinuationCommand JSON spec: continuation studies, witnessed SFT export, "
    "trajectory trace export, saved-evidence analysis, environment comparison, task-factory "
    "authoring, or executable "
    "tasks (action harbor: propose-worlds, construct-assets, register, qualify, "
    "prepare-training, export or serve).",
)
PIPELINE_OPTION = typer.Option(
    None, "--pipeline", help="Qualify and deliver a frozen multi-family task-factory manifest."
)


def _exclusive_mode(
    project: str | None,
    *,
    conflicts: tuple[str | int | Path | None, ...],
    flags: tuple[bool, ...] = (),
    message: str,
) -> str:
    """Require a project and refuse every option the selected evaluation mode cannot combine with.

    Args:
        project: Named project from the command line, which every exclusive mode requires.
        conflicts: Option values that must stay unset for this mode.
        flags: Boolean options that must stay off for this mode.
        message: Usage error naming the mode and what it cannot combine with.

    Returns:
        The required project name.

    Raises:
        typer.BadParameter: The project is missing or a conflicting option was supplied.
    """
    if project is None or any(value is not None for value in conflicts) or any(flags):
        raise typer.BadParameter(message)
    return project


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
    audit_memory: str | None = AUDIT_MEMORY_OPTION,
    audit_checkpoint: int | None = AUDIT_CHECKPOINT_OPTION,
    verbose: bool = VERBOSE_OPTION,
    trace_analysis: Path | None = TRACE_ANALYSIS_OPTION,
    refine_task: Path | None = REFINE_TASK_OPTION,
    calibration: Path | None = CALIBRATION_OPTION,
    materialize: Path | None = MATERIALIZE_OPTION,
    continuations: Path | None = CONTINUATIONS_OPTION,
    pipeline: Path | None = PIPELINE_OPTION,
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
        audit_memory: Memory profile the audit's held-out replays read.
        audit_checkpoint: Earlier frozen memory version the audit reads, by consumed traces.
        verbose: Plain per-turn fidelity detail, for agents and logs.
        trace_analysis: Simulator-only actor council and read-only evaluation refinement.
        refine_task: Bounded task refinement with independent qualification checks.
        calibration: Saved-evidence verifier and rubric calibration without new inference.
        materialize: Frozen native artifact handoff, without provider calls.
        continuations: Explicit runtime study or saved-evidence training export specification.
        pipeline: Bounded multi-family development qualification and verified package delivery.

    Returns:
        None. The selected mode prints its own result.

    Raises:
        typer.BadParameter: PROJECT is missing or options conflict with the selected mode.
    """
    if audit_memory is not None and (audit is None or audit_memory not in AUDIT_MEMORIES):
        raise typer.BadParameter(
            "--audit-memory applies only with --audit and is one of learned, none, lessons, "
            "bare or guidance"
        )
    if audit_checkpoint is not None and audit is None:
        raise typer.BadParameter("--audit-checkpoint applies only with --audit")
    rollout = (models, repeats, concurrency, maximum_steps, maximum_output_tokens)
    audit_flags = (verbose, retry_incomplete, include_inspected)
    if pipeline is not None:
        selected = _exclusive_mode(
            project,
            conflicts=(
                continuations,
                materialize,
                calibration,
                fidelity,
                audit,
                trace_analysis,
                refine_task,
                resume,
                report,
                *rollout,
            ),
            flags=audit_flags,
            message="--pipeline requires PROJECT and cannot combine with other evaluation modes "
            "or rollout overrides",
        )
        from exp.cli.evaluation.pipeline import run_pipeline

        run_pipeline(
            selected,
            pipeline,
            root=root,
            yes=yes,
            non_interactive=non_interactive,
            dry_run=dry_run,
        )
        return
    if continuations is not None:
        selected = _exclusive_mode(
            project,
            conflicts=(
                materialize,
                calibration,
                fidelity,
                audit,
                trace_analysis,
                refine_task,
                resume,
                report,
                *rollout,
            ),
            flags=audit_flags,
            message="--continuations requires PROJECT and cannot combine with other evaluation "
            "modes",
        )
        from exp.cli.evaluation.continuations import run_continuations

        run_continuations(
            selected,
            continuations,
            root=root,
            yes=yes,
            non_interactive=non_interactive,
            dry_run=dry_run,
        )
        return
    if materialize is not None:
        selected = _exclusive_mode(
            project,
            conflicts=(
                calibration,
                fidelity,
                audit,
                trace_analysis,
                refine_task,
                resume,
                report,
                *rollout,
            ),
            flags=audit_flags,
            message="--materialize requires PROJECT and cannot combine with other modes "
            "or rollout overrides",
        )
        from exp.cli.evaluation.native_artifacts import run_materialization

        run_materialization(selected, materialize, root=root, dry_run=dry_run)
        return
    if calibration is not None:
        selected = _exclusive_mode(
            project,
            conflicts=(fidelity, audit, trace_analysis, refine_task, resume, report, *rollout),
            flags=audit_flags,
            message="--calibrate requires PROJECT and cannot combine with other modes "
            "or rollout overrides",
        )
        from exp.cli.evaluation.calibration import run_calibration

        run_calibration(selected, calibration, root=root, dry_run=dry_run)
        return
    if refine_task is not None:
        selected = _exclusive_mode(
            project,
            conflicts=(fidelity, audit, trace_analysis, resume, report, *rollout),
            flags=(verbose,),
            message="--refine-task requires PROJECT and cannot combine with other evaluation "
            "modes or model and rollout overrides",
        )
        from exp.cli.evaluation.task_refinement import run_task_refinement

        run_task_refinement(
            selected,
            refine_task,
            root=root,
            yes=yes,
            non_interactive=non_interactive,
            dry_run=dry_run,
        )
        return
    if trace_analysis == Path("auto"):
        selected = _exclusive_mode(
            project,
            conflicts=(
                fidelity,
                audit,
                resume,
                report,
                repeats,
                maximum_steps,
                maximum_output_tokens,
            ),
            flags=(verbose,),
            message="--trace-analysis auto requires PROJECT; "
            "other evaluation overrides are not supported",
        )
        from exp.cli.evaluation.built_council import run_built_council

        run_built_council(
            selected,
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
        selected = _exclusive_mode(
            project,
            conflicts=(fidelity, audit, resume, report, *rollout),
            message="--trace-analysis requires PROJECT and cannot combine with other "
            "evaluation modes",
        )
        from exp.cli.evaluation.trace_analysis import run_trace_analysis

        run_trace_analysis(
            selected,
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
        selected = _exclusive_mode(
            project,
            conflicts=(fidelity, resume, report, *rollout),
            message="--audit requires PROJECT and cannot combine with other evaluation modes",
        )
        from exp.cli.evaluation.held_out_fidelity import run_audit

        run_audit(
            selected,
            audit,
            root=root,
            yes=yes,
            non_interactive=non_interactive,
            dry_run=dry_run,
            verbose=verbose,
            retry_incomplete=retry_incomplete,
            include_inspected=include_inspected,
            memory_profile=next(m for m in AUDIT_MEMORIES if m == (audit_memory or "learned")),
            checkpoint=audit_checkpoint,
        )
        return
    if fidelity is not None:
        selected = _exclusive_mode(
            project,
            conflicts=(resume, report, *rollout),
            message="--fidelity requires PROJECT and cannot combine with worker-evaluation options",
        )
        from exp.cli.evaluation.fidelity import run_fidelity

        run_fidelity(
            selected,
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
