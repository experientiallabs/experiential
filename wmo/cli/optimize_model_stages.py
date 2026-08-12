"""Stage execution for the one-command model optimizer."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import typer
from rich.console import Console
from rich.markup import escape

from wmo.cli.optimize_model_plan import (
    _compact_fingerprint,
    _fit_fingerprint,
    _policy_fit_identity,
    _RunPaths,
    _scenario_identity,
)

if TYPE_CHECKING:
    from wmo.optimize.routing.compression import CompressionConfig
    from wmo.optimize.routing.outcomes import OutcomeMatrix
    from wmo.optimize.routing.pipeline import (
        RunManifest,
        SpendLedger,
        StageDecision,
        StageRecord,
        SweepSpendProjection,
    )
    from wmo.optimize.routing.policy import EmbedderSpec, RoutingPolicy
    from wmo.optimize.routing.report import ImprovementReport
    from wmo.optimize.routing.sweep import SweepPlan


def _run_stages(
    decisions: list[StageDecision],
    *,
    console: Console,
    manifest: RunManifest,
    ledger: SpendLedger,
    paths: _RunPaths,
    plan: SweepPlan,
    projection: SweepSpendProjection,
    model_dir: Path,
    pool_file: Path,
    embedder: EmbedderSpec,
    compression: CompressionConfig | None,
    fallback: str | None,
    baseline: str | None,
    cost_quality: float,
    allow_uneven_coverage: bool,
    already_measured: int = 0,
) -> RunManifest:
    """Walk the plan, running what it said would run and recording each stage as it completes.

    The manifest is saved after EVERY stage, not once at the end: a run that dies on the fit has
    still paid for its sweep, and the next run must know that.

    """
    from wmo.optimize.routing.pipeline import Stage

    # The loop saves after every stage, which is what makes a rejected sweep survive: the
    # coverage gate lives in the FIT iteration, so the SWEEP iteration's save has already run by
    # the time the gate can stop the run.
    for decision in decisions:
        if not decision.will_run:
            console.print(
                f"\n[bold]{decision.stage.value}[/bold] [dim]SKIP: {escape(decision.reason)}[/dim]"
            )
            continue
        console.print(
            f"\n[bold]{decision.stage.value}[/bold] [dim]({escape(decision.reason)})[/dim]"
        )
        match decision.stage:
            case Stage.SWEEP:
                ledger.check(Stage.SWEEP, projection.total_usd, basis=projection.basis)
                record = _stage_sweep(
                    console,
                    plan,
                    model_dir=model_dir,
                    pool_file=pool_file,
                    already_measured=already_measured,
                )
                ledger.record(record.total_spend_usd)
            case Stage.COMPACT:
                record = _stage_compact(console, compression)
            case Stage.FIT:
                _enforce_coverage(console, paths.matrix, allow_uneven=allow_uneven_coverage)
                record = _stage_fit(
                    console,
                    paths,
                    embedder=embedder,
                    compression=compression,
                    fallback=fallback,
                    allow_uneven=allow_uneven_coverage,
                )
            case Stage.TUNE:
                record = _stage_tune(console, paths, cost_quality=cost_quality)
            case _:
                record = _stage_report(console, paths, model_dir=model_dir, baseline=baseline)
        manifest = manifest.with_record(record)
        manifest.save(paths.manifest)
    return manifest


def _now() -> str:
    """This moment as an ISO-8601 UTC stamp, for the manifest's completion times."""
    return datetime.now(tz=UTC).isoformat()


def _stage_sweep(
    console: Console,
    plan: SweepPlan,
    *,
    model_dir: Path,
    pool_file: Path,
    already_measured: int = 0,
) -> StageRecord:
    """Measure every candidate closed-loop and record what it cost.

    Deliberately does NOT judge the evidence it produced. The coverage contract that withholds a
    fit lives in `_enforce_coverage`, which runs against whatever matrix the fit is about to
    consume, so it binds a matrix this stage just bought AND one an earlier run bought that this
    run skipped past. Enforcing it here instead would let a recorded-then-skipped sweep carry a
    biased matrix into the fit with no gate at all.

    The matrix is written and this stage is recorded whatever the evidence looks like: those cells
    were paid for, their `error` fields are the diagnosis, and re-running after fixing the cause
    must not buy them a second time.
    """
    from wmo.cli.route_sweep_cmd import _compressor_note, cell_progress, print_world_model_spend
    from wmo.optimize.routing.compression import compression_signature
    from wmo.optimize.routing.pipeline import Stage, StageRecord, file_sha256
    from wmo.optimize.routing.sweep import execute_sweep
    from wmo.simulation import WorldModelEnv
    from wmo.simulation.model import load_world_model

    world_model, _serve_provider = load_world_model(model_dir)
    run = execute_sweep(
        plan,
        world_model=world_model,
        env_factory=lambda: WorldModelEnv(world_model, score_on_close=True),
        on_outcome=cell_progress(console, plan.cells - already_measured),
    )
    matrix = run.matrix
    scored = sum(1 for outcome in matrix.outcomes if outcome.scored)
    console.print(
        f"  [green]✓[/green] {len(matrix.outcomes)} cell(s), {scored} scored -> "
        f"{escape(str(plan.out_path))}\n"
        f"  measured candidate spend ${run.candidate_usd:.4f}{_compressor_note(run)} (the world "
        "model's own serve/judge cost is metered separately)",
        soft_wrap=True,
    )
    print_world_model_spend(console, run)
    record = StageRecord(
        stage=Stage.SWEEP,
        fingerprint={
            "pool": file_sha256(pool_file),
            "scenarios": _scenario_identity(plan),
            "episodes": str(plan.episodes),
            "max_steps": str(plan.max_steps),
            "compression": compression_signature(plan.compression),
        },
        artifact_path=str(plan.out_path),
        artifact_identity=file_sha256(plan.out_path),
        completed_at=_now(),
        spend_usd=run.candidate_usd,
        compressor_spend_usd=run.compressor_usd,
        world_model_spend_usd=run.world_model_usd,
    )
    return record


def _stage_compact(console: Console, compression: CompressionConfig | None) -> StageRecord:
    """Record the D-COMPRESS arm this run configured into the sweep and the fit.

    The compaction slot as it actually turned out. The design reserved it as a step between sweep
    and fit, and the seam that landed made it a configuration instead: one arm, applied by the sweep
    to every candidate call it measures and by the fit to every text it embeds into the bank. So
    this stage runs nothing and buys nothing, and it writes no artifact of its own. What it does own
    is the record that the arm was applied, which is what lets a later run say the compact row is
    current instead of silently assuming it.

    The compressor's own per-call bill is real and is NOT lost by having no line here: it is
    metered inside the sweep and reported as the compressor's share of that stage's candidate
    spend (`StageRecord.compressor_spend_usd`).
    """
    from wmo.optimize.routing.compression import compression_signature
    from wmo.optimize.routing.pipeline import Stage, StageRecord

    signature = compression_signature(compression)
    console.print(
        f"  [green]✓[/green] configured {escape(signature)}\n"
        "  no separate step and no separate bill: this arm is what the sweep measures every "
        "candidate through and what the fit embeds its bank through, so the endpoint routes on "
        "the representation it will serve",
        soft_wrap=True,
    )
    return StageRecord(
        stage=Stage.COMPACT,
        fingerprint=_compact_fingerprint(compression),
        completed_at=_now(),
    )


def _enforce_coverage(console: Console, matrix_path: Path, *, allow_uneven: bool) -> None:
    """Show what the fit would be weighed on, and withhold it when that is not a comparison.

    `route sweep`'s contract, applied at the point it actually protects something: the fit. Running
    it here rather than at the end of the sweep is what lets a rejected sweep be RECORDED, because
    a later run that skips the sweep still passes through this gate on its way to the fit. So a
    biased matrix cannot reach a fitter by having been bought on an earlier invocation, and the
    cells that were paid for are never bought twice.

    Both refusals are free and repeatable: nothing has spent anything by the time this runs on a
    resumed run. Zero scored cells has no opt-out, exactly as in `route sweep`, since there is
    nothing to fit and `fit` would fail anyway.

    Raises:
        typer.Exit: The matrix is not fit-ready (exit code 1).
    """
    from wmo.cli.route_sweep_cmd import (
        BIAS_ACCEPTED_NOTE,
        NO_EVIDENCE_WARNING,
        print_coverage,
        uneven_warning,
    )
    from wmo.optimize.routing.outcomes import OutcomeMatrix
    from wmo.optimize.routing.sweep import coverage

    matrix = OutcomeMatrix.load(matrix_path)
    rows = coverage(matrix)
    print_coverage(console, rows)
    if not any(outcome.scored for outcome in matrix.outcomes):
        console.print(NO_EVIDENCE_WARNING)
        raise typer.Exit(1)
    warning = uneven_warning(rows)
    if warning is None:
        return
    console.print(warning)
    if not allow_uneven:
        console.print(
            "  fix the lost cells and re-run, drop the candidate that lost them, or re-run "
            "with [bold]--allow-uneven-coverage[/bold] to fit on this matrix anyway (the "
            "matrix is on disk and recorded, so re-running will not buy these cells again)"
        )
        raise typer.Exit(1)
    console.print(BIAS_ACCEPTED_NOTE)


def _stage_fit(
    console: Console,
    paths: _RunPaths,
    *,
    embedder: EmbedderSpec,
    compression: CompressionConfig | None,
    fallback: str | None,
    allow_uneven: bool,
) -> StageRecord:
    """Fit the guarded kNN policy on the swept matrix, into the path `wmo serve` reads.

    `compression` is the compaction stage's arm, and passing it here is the fit half of
    representation consistency: `fit_knn_artifact` embeds the bank through the compressor and stamps
    both `compression` and `fit_compression` on the policy, which is what the mount gate re-checks.
    The arm cannot disagree with the one the matrix was measured under, because the same flag
    configured the sweep and the sweep re-measures whenever it moves (the arm is in its
    fingerprint).
    """
    from wmo.optimize.routing.knn import fit_knn_artifact, fit_provenance
    from wmo.optimize.routing.outcomes import load_matrix_with_digest, split_router_scenarios
    from wmo.optimize.routing.pipeline import Stage, StageRecord

    matrix, source = load_matrix_with_digest(paths.matrix)
    try:
        router_split = split_router_scenarios(matrix.scenario_ids())
    except ValueError as exc:
        console.print(f"[red]fit failed[/red] {escape(str(exc))}")
        raise typer.Exit(1) from exc
    try:
        fitted = fit_knn_artifact(
            matrix,
            out_path=paths.policy,
            matrix_source=source,
            embedder=embedder,
            fit_ids=list(router_split.fit_ids),
            fallback=fallback,
            compression=compression,
        )
    except ValueError as exc:
        # Nothing is wrong with the flags: the matrix this run measured cannot be fitted. Exit 1
        # like the coverage gate, not 2 with a usage banner (`route_app.student` precedent).
        console.print(f"[red]fit failed[/red] {escape(str(exc))}")
        raise typer.Exit(1) from exc
    _rebaseline_dial_snapshot(console, paths.policy, fitted.policy)
    console.print(
        f"  [green]✓[/green] knn policy over {fitted.scenarios} scenario(s) -> "
        f"{escape(str(paths.policy))}\n"
        f"  bank {escape(str(fitted.bank_path))}, fallback {escape(fitted.policy.default_model)}\n"
        f"  routed away from the fallback {fitted.routed_share:.1%} of the time on the fit set "
        f"(IN-SAMPLE: every request retrieves its own row; the report measures held out)",
        soft_wrap=True,
    )
    return StageRecord(
        stage=Stage.FIT,
        fingerprint=_fit_fingerprint(
            matrix=paths.matrix,
            embedder=embedder,
            compression=compression,
            fallback=fallback,
            allow_uneven=allow_uneven,
        ),
        artifact_path=str(paths.policy),
        artifact_identity=fit_provenance(fitted.policy),
        completed_at=_now(),
    )


def _rebaseline_dial_snapshot(console: Console, policy_path: Path, fitted: RoutingPolicy) -> None:
    """Retire an as-fitted snapshot that the fit just now superseded.

    `route tune` refuses to dial a policy whose `policy.base.json` describes a DIFFERENT fit,
    because for a human running the commands separately that snapshot is the only sign that a
    refit happened, and dialing it would silently replace the new fit with a slid copy of the old
    one. In an orchestrated run the refit is not a surprise: this command performed it one step
    ago and is about to dial the result, so the snapshot is stale by construction and the tune
    stage would otherwise refuse a chain it asked for itself. The remedy is the one that error
    message prescribes, taken deliberately and said out loud.

    Only ever removes a snapshot of a SUPERSEDED fit. A snapshot that still matches (a redo that
    reproduced the same fit) is left exactly where it is, so the dial keeps re-applying from it.
    """
    from wmo.optimize.routing.knn import fit_provenance
    from wmo.optimize.routing.policy import RoutingPolicy

    base_path = policy_path.with_name(f"{policy_path.stem}.base{policy_path.suffix}")
    if not base_path.is_file():
        return
    try:
        stale = fit_provenance(RoutingPolicy.load(base_path)) != fit_provenance(fitted)
    except (OSError, ValueError):
        stale = True  # an unreadable snapshot is not a baseline for anything
    if not stale:
        return
    base_path.unlink()
    console.print(
        f"  re-baselined the dial: {escape(base_path.name)} was the as-fitted snapshot of the "
        "fit this stage just replaced, so the next dial applies to the new fit",
        soft_wrap=True,
    )


def _stage_tune(console: Console, paths: _RunPaths, *, cost_quality: float) -> StageRecord:
    """Set the endpoint's dial, preserving `route tune`'s as-fitted snapshot semantics."""
    from wmo.optimize.routing.knn import tune_policy_dial
    from wmo.optimize.routing.pipeline import Stage, StageRecord, file_sha256

    fit_identity = _policy_fit_identity(paths.policy)
    try:
        dialed = tune_policy_dial(paths.policy, cost_quality)
    except ValueError as exc:
        console.print(f"[red]tune failed[/red] {escape(str(exc))}")
        raise typer.Exit(1) from exc
    knobs = dialed.knobs
    console.print(
        f"  [green]✓[/green] cost_quality={dialed.cost_quality:g} ({dialed.named_point})\n"
        f"  knobs: floor_q={knobs.floor_q:g}, cost knob lam={knobs.pick_lam:g}, "
        f"guard={knobs.guard_mode}, z={knobs.knn_z:g}\n"
        f"  as fitted: {escape(str(dialed.base_path))}",
        soft_wrap=True,
    )
    return StageRecord(
        stage=Stage.TUNE,
        fingerprint={"policy_fit": fit_identity, "cost_quality": f"{cost_quality:g}"},
        artifact_path=str(paths.policy),
        artifact_identity=file_sha256(paths.policy),
        completed_at=_now(),
    )


def _stage_report(
    console: Console, paths: _RunPaths, *, model_dir: Path, baseline: str | None
) -> StageRecord:
    """Score the tuned policy against its anchor on the same held-out scenarios."""
    from wmo.optimize.routing.outcomes import OutcomeMatrix
    from wmo.optimize.routing.pareto import PARETO_FILENAME, held_out_curve
    from wmo.optimize.routing.pipeline import Stage, StageRecord, file_sha256
    from wmo.optimize.routing.policy import RoutingPolicy

    matrix = OutcomeMatrix.load(paths.matrix)
    policy = RoutingPolicy.load(paths.policy)
    anchor = baseline or policy.default_model
    try:
        report = build_endpoint_scorecard(matrix, policy, baseline=anchor, endpoint=model_dir.name)
    except (KeyError, ValueError) as exc:
        console.print(
            f"[red]report failed[/red] cannot report against {escape(anchor)}: {escape(str(exc))}\n"
            "  name a pool model the sweep scored with --baseline, or re-run the sweep so the "
            "anchor has scored episodes"
        )
        raise typer.Exit(1) from exc
    paths.report.parent.mkdir(parents=True, exist_ok=True)
    paths.report.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    console.print(
        f"  [green]✓[/green] report over {report.headline.scenarios_compared} commonly-scored "
        f"scenario(s) -> {escape(str(paths.report))}",
        soft_wrap=True,
    )
    # The measured cost/quality curve. It goes in MODEL_DIR (a serving artifact, mounted by
    # GET /config beside policy.json), not the disposable optimize manifest dir, so the
    # platform's Pareto graph renders this workload's real frontier (D-PARETO). Additive: a
    # curve failure warns rather than failing a report that already succeeded.
    try:
        curve = held_out_curve(matrix, policy, judge="world-model verifier")
        pareto_path = model_dir / PARETO_FILENAME
        pareto_path.write_text(curve.model_dump_json(indent=2), encoding="utf-8")
        console.print(
            f"  [green]✓[/green] pareto curve ({sum(1 for p in curve.points if p.on_frontier)} "
            f"frontier point(s), recommended {curve.recommended}) -> "
            f"{escape(str(pareto_path))}",
            soft_wrap=True,
        )
    except (ValueError, FileNotFoundError) as exc:
        console.print(
            f"  [yellow]![/yellow] pareto curve skipped: {escape(str(exc))}", soft_wrap=True
        )
    return StageRecord(
        stage=Stage.REPORT,
        fingerprint={
            "matrix": file_sha256(paths.matrix),
            "policy": file_sha256(paths.policy),
            "anchor": anchor,
        },
        artifact_path=str(paths.report),
        artifact_identity=file_sha256(paths.report),
        completed_at=_now(),
    )


def build_endpoint_scorecard(
    matrix: OutcomeMatrix, policy: RoutingPolicy, *, baseline: str, endpoint: str
) -> ImprovementReport:
    """THE SCORECARD SEAM: the one call the report stage makes to produce its numbers.

    Today this is `wmo.optimize.routing.report.build_report`, the paired held-out comparison the
    endpoint already cites. The richer three-objective scorecard has since landed as
    `wmo.optimize.routing.scorecard` (effective cost per COMPLETED task, the cache-aware
    accounting rule, the ablation ladder); wiring it in is a deliberate follow-up rather than
    part of this command's first release, because its `Arm`/`ConditionLabel` inputs describe a
    grid this stage does not yet build. When that happens only the body of this function changes:
    the stage, the manifest fingerprints, and the ending that renders the result all stay as they
    are.

    Args:
        matrix: Held-out candidate outcome matrix.
        policy: Tuned routing policy to score.
        baseline: Pool candidate used as the report's named anchor.
        endpoint: World-model endpoint name attached to the report.

    Returns:
        The paired held-out improvement report for the endpoint.
    """
    from wmo.optimize.routing.report import build_report

    return build_report(matrix, policy, baseline=baseline, endpoint=endpoint, generated_at=_now())


# ----------------------------------------------------------------------------------- the payoff


def _print_payoff(console: Console, name: str, *, paths: _RunPaths, cost_quality: float) -> None:
    """Close on what the endpoint is now, what it bought, and how to serve it.

    Three objectives, each against the same named anchor over the same scenarios, each carrying
    where its number came from. A number that cannot honestly be computed prints its reason.
    """
    from wmo.optimize.routing.knn import cost_quality_named_point
    from wmo.optimize.routing.policy import RoutingPolicy
    from wmo.optimize.routing.report import ImprovementReport

    report = ImprovementReport.model_validate_json(paths.report.read_text(encoding="utf-8"))
    policy = RoutingPolicy.load(paths.policy)
    head = report.headline
    anchor = report.baseline.model_id
    console.print(
        f"\n  policy: {policy.kind} (guarded, fallback {escape(policy.default_model)})   "
        f"dial: {cost_quality:g} {cost_quality_named_point(cost_quality).lower()}"
    )
    quality = (head.accuracy - head.baseline_accuracy) * 100
    console.print(
        f"  quality  {quality:+.1f}pt vs {escape(anchor)}   [dim](world-model simulated, "
        f"{head.scenarios_compared} held-out scenario(s) scored on both sides)[/dim]"
    )
    console.print(f"  cost     {_cost_line(head.cost_per_run_usd, head.baseline_cost_per_run_usd)}")
    console.print(f"  latency  {_latency_line(head.latency_p50_ms, head.baseline_latency_p50_ms)}")
    if head.scenarios_excluded:
        console.print(
            f"  [dim]{head.scenarios_excluded} scenario(s) left out: one side had no scored "
            "episode, and a comparison over different scenarios is not a comparison[/dim]"
        )
    console.print(
        f"\n  serve it:   [bold]wmo serve --name {escape(name)}[/bold]\n"
        f'  endpoint:   POST /v1/chat/completions  (model="{escape(name)}")',
        soft_wrap=True,
    )


def _cost_line(routed: float, anchor: float) -> str:
    """The cost delta, or the reason there is not one to state."""
    if anchor <= 0.0:
        return (
            "[dim]no delta: the anchor's episodes measured $0, so a percentage would divide by "
            "zero (an unpriced pool entry, or a matrix from a run that recorded no cost)[/dim]"
        )
    percent = (routed - anchor) / anchor * 100
    return (
        f"{percent:+.0f}% per episode   [dim](measured candidate-side at list prices, "
        "single-shot; cache effects not modeled)[/dim]"
    )


def _latency_line(routed_ms: float, anchor_ms: float) -> str:
    """The p50 delta, or the reason there is not one to state."""
    if anchor_ms <= 0.0:
        return (
            "[dim]no delta: the anchor's episodes recorded no per-call timings, so there is "
            "nothing to compare[/dim]"
        )
    return (
        f"p50 {(routed_ms - anchor_ms) / 1000:+.2f}s   [dim](wall time per policy call during "
        "the sweep, env time excluded)[/dim]"
    )
