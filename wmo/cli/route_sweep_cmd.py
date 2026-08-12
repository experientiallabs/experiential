"""Route sweep command and its evidence-presentation helpers."""

from __future__ import annotations

import itertools
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from wmo.cli.consent import require_spend_consent
from wmo.common.config import ARTIFACT_DIR, WorldModelStore
from wmo.common.project import ArtifactStoreError, ProjectStore
from wmo.common.tasks import resolve_task_set

if TYPE_CHECKING:
    # Type-only: real imports are local to the commands and helpers that construct or inspect
    # these values, so importing this module never pulls the optimize/engine/env/distill/pool
    # bodies behind it.
    from wmo.optimize.routing.outcomes import ScenarioOutcome
    from wmo.optimize.routing.sweep import CandidateCoverage, DeferredRisk, SweepPlan, SweepRun

from wmo.cli.route_constants import (
    _DEFAULT_HISTORY_CHARS,
    _DEFAULT_POOL_PATH,
    COMPRESSOR_IDS_HELP,
    DEFAULT_MATRIX_FILENAME,
)

_console = Console()


def sweep(
    model: str = typer.Argument(
        None, help="World model to measure against (default: the only one built under --root)."
    ),
    pool_file: str = typer.Option(
        _DEFAULT_POOL_PATH,
        "--pool",
        # The doubled brackets are escaped: typer renders help through rich markup, which
        # otherwise swallows them and prints an empty pair.
        help="Candidate pool TOML: one \\[\\[model]] table per candidate.",
    ),
    project: str = typer.Option(
        "default",
        "--project",
        help="Project that owns the immutable task set built from canonical trace evidence.",
    ),
    task_set: str | None = typer.Option(
        None,
        "--task-set",
        help="Immutable task-set ID. Omit only when the project has exactly one task set.",
    ),
    scenarios: int = typer.Option(
        20,
        "--scenarios",
        min=1,
        help="Cap on held-out immutable tasks measured (a deterministic task-ID prefix).",
    ),
    episodes: int = typer.Option(
        1, "--episodes", min=1, help="Episodes per (candidate, task) cell."
    ),
    max_steps: int = typer.Option(
        20, "--max-steps", min=1, help="Step budget per episode (also the cost estimate's cap)."
    ),
    concurrency: int = typer.Option(
        1,
        "--concurrency",
        min=1,
        help="Cells measured at once (1 = one at a time). Changes only how long the sweep takes, "
        "never what it measures, and a sweep interrupted at one value resumes at another. Your "
        "PROVIDER LIMITS are the real ceiling, not this number: every candidate call and every "
        "world-model serve and judge call is a request, and the world model's own calls all come "
        "out of ONE account's bucket, so raising this past what that bucket allows turns cells "
        "into throttling errors instead of results.",
    ),
    history_chars: int = typer.Option(
        _DEFAULT_HISTORY_CHARS,
        "--history-chars",
        min=1,
        help="Characters of each observation the agent sees on later turns. Raise it for an "
        "environment whose tool payloads are large: too small and the agent cannot see what it "
        "just fetched, so it re-fetches. Changes what candidates are measured on, so matrices "
        "swept at different values are not comparable.",
    ),
    assume_input_tokens: int = typer.Option(
        2000,
        "--assume-input-tokens",
        min=0,
        help="ASSUMED input tokens per policy call, for the cost estimate only.",
    ),
    assume_output_tokens: int = typer.Option(
        250,
        "--assume-output-tokens",
        min=0,
        help="ASSUMED output tokens per policy call, for the cost estimate only.",
    ),
    out: str = typer.Option(
        DEFAULT_MATRIX_FILENAME, "--out", help="Where to write the OutcomeMatrix JSON."
    ),
    root: str = typer.Option(ARTIFACT_DIR, "--root", help="Project dir."),
    yes: bool = typer.Option(
        False,
        "--yes",
        help="Consent to the projected spend up front. Required in a non-interactive "
        "session (CI, cron, piped output, redirected input), where the run otherwise "
        "refuses to start.",
    ),
    allow_uneven_coverage: bool = typer.Option(
        False,
        "--allow-uneven-coverage",
        help="Hand the matrix to `fit` even when the candidates were not scored on the same "
        "evidence: different scenarios, or different numbers of surviving episodes on the same "
        "scenarios. The fit is then biased (both fitters skip unscored rows and weigh the rest per "
        "episode); the coverage table prints either way.",
    ),
    compressor: str = typer.Option(
        None,
        "--compressor",
        help="D-COMPRESS: measure every candidate call through this compressor "
        f"({COMPRESSOR_IDS_HELP}), so the matrix is the compressed ARM of the grid. Default: "
        "uncompressed. `fit` requires the matrix arm to match the policy it stamps.",
    ),
    aggressiveness: float = typer.Option(
        0.0,
        "--aggressiveness",
        min=0.0,
        max=1.0,
        help="Compressor-defined dial in [0, 1] for --compressor: 0.0 is a no-op and higher "
        "never removes less, but it is not an exact removal fraction (the achieved ratio is "
        "recorded per episode).",
    ),
) -> None:
    """Measure every pool candidate closed-loop and write the outcome matrix `fit` consumes.

    This is step one of the routing workflow: nothing else produces an `OutcomeMatrix`.

        wmo optimize route sweep support --project support --pool .wmo/pool.toml
        wmo optimize route fit matrix.json --kind knn

    Every (candidate, scenario, episode) cell runs one full episode against the world model,
    which scores it (`WorldModelEnv(..., score_on_close=True)`): a matrix without verified
    rewards is not evidence. Tasks come only from the project's immutable held-out TaskSet
    partition. The candidate tool surface is summarized from fit tasks only, so a candidate is
    not scored on guessing tools visible only in held-out evidence. The whole sweep runs the
    world model frozen, so no cell's predictions become another cell's retrieved demos and the
    result does not depend on sweep order.

    Nothing measured is lost, and nothing measured is bought twice. Every cell lands in
    `<out>.partial.jsonl` the moment it completes, so a sweep killed at hour five keeps the cells
    it paid for; re-running the same command measures only what is missing and then writes the
    matrix and removes the sidecar. Changing what the sweep measures (the pool, the scenario cut,
    episodes, the step budget, the observation window, the compressor) makes those rows a
    different arm, and the command says so and stops rather than merging two arms into one matrix.
    `--concurrency N` runs N cells at once, which is the difference between a six-hour grid and a
    one-hour grid; it is not part of what the sweep measures, so a run interrupted at one value
    resumes at another.

    Spend is confirmed before the first episode runs, and consent is said, never inferred: at a
    terminal the projected cost is a question, and with nobody to ask (CI, cron, piped output,
    `| tee`, or a redirected stdin, which is not a person even when stdout is a terminal) the run
    REFUSES with exit code 2 unless `--yes` was passed, naming what it would have spent.
    What that estimate multiplies is ASSUMED tokens per policy call by the real
    cell and call counts, so it is a projection, never a measurement; the measured candidate spend
    is printed when the sweep finishes. Before that question is asked, every candidate's backend
    is resolved as far as it goes without a request: its kind's static
    requirements from the entry alone, then its lazy SDK client forced to BUILD, which imports the
    SDK and resolves credentials locally. So a candidate that could never be called is a usage
    error at the boundary, not a mid-sweep abort with earlier candidates already paid for. Two
    things stay first-cell failures because seeing them needs a request (bedrock AWS credentials,
    tinker service reachability); the pre-flight names them per entry when the pool has one.

    Fit-readiness is a coverage contract, not a nonzero count. A cell goes unscored when its
    episode errored (provider throttle, agent crash, judge failure), both fitters SKIP unscored
    rows, and what they do with the rest is episode-weighted, so the contract is that every
    candidate has the same number of scored episodes on the same scenarios. Two ways to break it,
    both blocked: a matrix where candidate A was scored on 20 scenarios and B on 11 ranks them on
    DIFFERENT task sets, and a matrix where both cover all 20 but A kept 3 episodes on a scenario
    where B kept 1 weighs that scenario three times as heavily for A, because `--kind rank`
    averages every surviving episode into its cluster mean and both kinds pick their
    default/fallback model off episode-weighted means (`routing._overall_best`,
    `knn.best_single_on_fit`). A knn BANK cell is that pair's own mean, so the bank is milder, but
    milder is not unbiased and it is the same matrix either way. Either break leaves the policy
    decided by whichever cells each candidate happened to lose. So per-candidate scored counts
    ALWAYS print, and when the evidence differs the command still writes the matrix (those cells
    were paid for, and their `error` fields are the diagnosis) but WITHHOLDS the `fit` handoff and
    exits non-zero, naming each candidate, the scenarios it has no scored episode for, and the
    scenarios where it kept fewer episodes than the best-covered candidate.
    `--allow-uneven-coverage` is the opt-out for an operator who knows the bias and wants the
    partial data anyway (one candidate's backend down for the whole sweep, say): it prints the same
    coverage table and stops treating the difference as fatal. Losing the SAME cells for every
    candidate is not uneven, since the comparison stays like-for-like on less data and the counts
    show the loss.

    Exit code 1 when the matrix is not fit-ready: no cell scored at all, or unequal scored evidence
    without `--allow-uneven-coverage`. `sweep && fit` in a script then stops instead of fitting on
    it, and the matrix is written either way.

    Args:
        model: Built world model to simulate while measuring candidates.
        pool_file: Candidate pool TOML to measure.
        project: Project that owns the immutable canonical task-set artifact.
        task_set: Optional exact task-set artifact ID when the project has several.
        scenarios: Maximum number of held-out tasks to include.
        episodes: Episode attempts per candidate and task.
        max_steps: Per-episode agent step limit.
        concurrency: Concurrent cells, which changes duration but not measured evidence.
        history_chars: Observation history retained for each agent step.
        assume_input_tokens: Input-token estimate used only for spend consent.
        assume_output_tokens: Output-token estimate used only for spend consent.
        out: Outcome matrix destination.
        root: Project artifact directory containing the world model.
        yes: Whether to consent to the projected spend without prompting.
        allow_uneven_coverage: Whether to retain a matrix with unmatched scored evidence.
        compressor: Optional compressor evaluated as part of the measured arm.
        aggressiveness: Compressor-specific dial in the inclusive range from zero to one.

    Raises:
        typer.BadParameter: Sweep inputs, selected candidates, or coverage settings are invalid.
    """
    from wmo.optimize.routing.compression import resolve_compression
    from wmo.optimize.routing.sweep import (
        SweepError,
        coverage,
        execute_sweep,
        plan_sweep,
        preflight_pool,
        resumable_cells,
    )
    from wmo.simulation import WorldModelEnv
    from wmo.simulation.model import load_world_model

    out_path = Path(out)
    if compressor is None and aggressiveness > 0.0:
        raise typer.BadParameter("--aggressiveness needs --compressor to apply it")
    sweep_compression = None
    if compressor is not None:
        try:
            # Checked before a single episode is paid for, and against the SERVING rule: there
            # is no point measuring an arm whose compressor could never be mounted.
            sweep_compression = resolve_compression(compressor, aggressiveness)
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
    store = WorldModelStore(root)
    try:
        model_dir = store.resolve(model)
    except FileNotFoundError as exc:
        raise typer.BadParameter(str(exc)) from exc
    except ValueError as exc:
        # `resolve` says "pass --name", the option `wmo serve`/`play`/`demo` carry. Here the
        # model is a POSITIONAL, so say what a user of this command actually types.
        names = store.list_names()
        raise typer.BadParameter(
            f"multiple world models built ({', '.join(names)}); name one as the MODEL argument, "
            f"e.g. `wmo optimize route sweep {names[0]}`"
        ) from exc
    # Everything knowable without spending is settled BEFORE the cost question: a candidate whose
    # backend cannot even be constructed, or an --out that cannot be written, would otherwise
    # surface after the sweep had already paid for cells it then throws away.
    try:
        tasks = resolve_task_set(ProjectStore(Path(root), project).artifacts, task_set)
        preflight = preflight_pool(Path(pool_file))
    except (ArtifactStoreError, SweepError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    print_deferred_risks(_console, preflight.deferred)
    try:
        plan = plan_sweep(
            model_dir=model_dir,
            pool=preflight.pool,
            out_path=out_path,
            task_set=tasks,
            scenarios=scenarios,
            episodes=episodes,
            max_steps=max_steps,
            assume_input_tokens=assume_input_tokens,
            assume_output_tokens=assume_output_tokens,
            history_chars=history_chars,
            compression=sweep_compression,
            max_concurrency=concurrency,
        )
    except (SweepError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    print_underfilled_task_note(_console, plan)
    try:
        # Before the money question, not after it: a sidecar left by a run of a DIFFERENT plan is
        # refused here, while refusing still costs nothing.
        already_measured = resumable_cells(plan)
    except SweepError as exc:
        raise typer.BadParameter(str(exc)) from exc
    world_model, _serve_provider = load_world_model(model_dir)

    print_cost_estimate(_console, plan, already_measured=already_measured)
    _confirm_cost(plan, yes=yes)

    _console.print(
        f"sweeping {len(plan.pool.models)} candidate(s) over {len(plan.tasks)} held-out "
        f"task(s) from [bold]{escape(plan.task_set_id)}[/bold], {episodes} episode(s) each…"
    )
    run = execute_sweep(
        plan,
        world_model=world_model,
        env_factory=lambda: WorldModelEnv(world_model, score_on_close=True),
        on_outcome=cell_progress(_console, plan.cells - already_measured),
    )
    matrix = run.matrix
    scored = sum(1 for outcome in matrix.outcomes if outcome.scored)
    # `escape(out)`: a bracketed path segment would otherwise be read as markup and dropped, so
    # the line would print a path that does not exist (and this one is meant to be copied).
    _console.print(
        f"[green]✓[/green] {len(matrix.outcomes)} cell(s), {scored} scored -> {escape(out)}\n"
        f"  measured candidate spend ${run.candidate_usd:.4f}{_compressor_note(run)} (the world "
        "model's own serve/judge cost is metered separately)",
        soft_wrap=True,  # a path a user copies must not be wrapped
    )
    print_world_model_spend(_console, run)
    rows = coverage(matrix)
    print_coverage(_console, rows)
    if scored == 0:
        # Exit non-zero: a matrix with no verified reward is not evidence, so `sweep && fit` in a
        # script must stop here rather than fit on it. The rows are on disk for their `error`s.
        # No --allow-uneven-coverage escape: there is nothing to fit, so `fit` would fail anyway.
        _console.print(NO_EVIDENCE_WARNING)
        raise typer.Exit(1)
    warning = uneven_warning(rows)
    if warning is not None:
        _console.print(warning)
        if not allow_uneven_coverage:
            _console.print(
                "  fix the lost cells and sweep again, drop the candidate that lost them, or "
                "re-run with [bold]--allow-uneven-coverage[/bold] to fit on this matrix anyway"
            )
            raise typer.Exit(1)
        _console.print(BIAS_ACCEPTED_NOTE)
    _console.print(
        f"  next: [bold]wmo optimize route fit {escape(out)} --kind knn[/bold]", soft_wrap=True
    )


# ------------------------------------------------------------------ shared sweep presentation
# Rendering for the sweep's typed results, shared by `route sweep` and `optimize model`'s sweep
# stage so the coverage contract reads the same whichever command a user reached it through.
# Every one takes its console explicitly: the two commands own different ones.

NO_EVIDENCE_WARNING = (
    "[yellow]warning[/yellow] no cell was scored, so this matrix is not evidence and "
    "fitting will fail; read the `error` field of a row to see what broke"
)

BIAS_ACCEPTED_NOTE = (
    "  --allow-uneven-coverage was passed: fitting on it anyway, with that bias accepted"
)

# Scenario ids shown per candidate before the column summarizes the rest: enough to see the pattern
# in a table an operator reads, without a 20-id line per row.
_LOST_SHOWN = 5


def print_deferred_risks(console: Console, deferred: tuple[DeferredRisk, ...]) -> None:
    """Name what the request-free pre-flight could not close, per candidate that carries it.

    Args:
        console: Destination for the operator-facing preflight note.
        deferred: Candidate risks that preflight could not verify without a request.
    """
    if not deferred:
        return
    console.print(
        "[yellow]note[/yellow] the pre-flight makes no request, so one thing per candidate "
        "below can still fail at its first cell (the matrix records it as that cell's `error`):"
    )
    for risk in deferred:
        console.print(f"  {escape(risk.candidate)} (kind={risk.kind.value}): {risk.risk}")


def print_underfilled_task_note(console: Console, plan: SweepPlan) -> None:
    """Say when an immutable task set has fewer held-out tasks than the requested sweep cap.

    Args:
        console: Destination for the operator-facing note.
        plan: Sweep plan whose held-out task selection is being described.
    """
    if not plan.underfilled:
        return
    console.print(
        f"[yellow]note[/yellow] task set {plan.task_set_id} has only {len(plan.tasks)} held-out "
        "task(s) for this sweep cap. Coverage remains immutable; inspect coverage.json before "
        "treating the fitted policy as broad traffic evidence."
    )


def _compressor_note(run: SweepRun) -> str:
    """Name the compressor's share of candidate spend, but only when one actually billed.

    The D-COMPRESS rule folds the compressor's inference cost into the candidate figure, so on a
    compressed arm that number is not just the models. Saying so on an UNCOMPRESSED sweep would
    be noise about a stage that did not run, which is why this is conditional on a nonzero bill
    rather than on the flag.
    """
    if run.compressor_usd <= 0.0:
        return ""
    return f" (incl. ${run.compressor_usd:.4f} compressor)"


def print_world_model_spend(console: Console, run: SweepRun) -> None:
    """The OTHER half of a sweep's bill: what the simulator charged to run the evaluation.

    Printed as its own line, never folded into the candidate figure above it. The candidate side
    is the serving cost a customer would pay and the policy is fitted to trade off; this side is
    eval infrastructure that exists only because the measurement happened. One number covering
    both would misprice both.

    Args:
        console: Destination for the metering summary.
        run: Completed sweep whose evaluation-side usage is reported.
    """
    gap = run.metering_gap
    if run.episodes_metered == 0:
        console.print(f"  world-model spend {gap}")
        return
    usage = run.world_model_usage
    phases = ", ".join(
        f"{phase.value} ${bucket.cost_usd:.4f}" for phase, bucket in sorted(usage.by_phase.items())
    )
    detail = f" ({phases})" if phases else ""
    console.print(
        f"  measured world-model spend ${run.world_model_usd:.4f} over {run.episodes_metered} "
        f"session(s){detail}: eval infrastructure, not serving cost"
        + (f"\n  [yellow]note[/yellow] {gap}" if gap is not None else ""),
        soft_wrap=True,
    )
    if run.usage_path is not None:
        console.print(
            f'  recorded as kind="sweep" -> {escape(str(run.usage_path))}', soft_wrap=True
        )


def cell_progress(console: Console, cells: int) -> Callable[[ScenarioOutcome], None]:
    """A per-cell progress line: which cell, what it scored, what it cost.

    Args:
        console: Destination for each completed cell's progress line.
        cells: Total number of cells expected in the current invocation.

    Returns:
        A callback that reports each completed scenario outcome.
    """
    done = itertools.count(1)

    def _on_outcome(outcome: ScenarioOutcome) -> None:
        reward = "unscored" if outcome.reward is None else f"{outcome.reward:.2f}"
        console.print(
            f"  [{next(done)}/{cells}] {escape(outcome.model)} {escape(outcome.scenario_id)} "
            f"ep{outcome.episode}: reward={reward} ${outcome.cost_usd:.5f} "
            f"steps={outcome.steps}"
        )

    return _on_outcome


def uneven_warning(rows: list[CandidateCoverage]) -> str | None:
    """The warning for coverage that is not a comparison, or None when it is one.

    Two different failures, so two different messages: candidates ranked on different scenario
    SETS, and candidates ranked on the same scenarios with different numbers of surviving EPISODES.
    Both bias a fit; naming which one happened is what makes the message actionable.

    Args:
        rows: Per-candidate scored-coverage measurements.

    Returns:
        A warning explaining the bias, or None when coverage is comparable.
    """
    from wmo.optimize.routing.sweep import Unevenness, unevenness

    counts = ", ".join(f"{escape(row.candidate)} {row.scored}" for row in rows)
    match unevenness(rows):
        case Unevenness.EVEN:
            return None
        case Unevenness.SCENARIOS:
            return (
                "[yellow]warning[/yellow] candidates were scored on DIFFERENT scenarios (scored "
                f"cells: {counts}), so `fit` would rank them on different task sets: it skips "
                "unscored rows, and the policy that comes out is biased by which scenarios each "
                "candidate lost. The paid cells are on disk and their `error` field says what "
                "broke."
            )
        case Unevenness.EPISODES:
            return (
                "[yellow]warning[/yellow] candidates cover the same scenarios but kept DIFFERENT "
                f"numbers of scored episodes on them (scored cells: {counts}; the table above "
                "says which scenarios were thinned, as kept/most). Both fitters weigh EPISODES: "
                "--kind rank averages every surviving episode into its cluster mean, so a "
                "scenario one candidate kept 1 of 3 episodes on counts a third as much for it, "
                "and both kinds pick their default/fallback model off the same episode-weighted "
                "means. What comes out then turns on which episodes happened to fail."
            )


def print_coverage(console: Console, rows: list[CandidateCoverage]) -> None:
    """Show what each candidate would be weighed on: its scored cells, and what it lost.

    The last column names every scenario where this candidate holds less evidence than the
    best-covered candidate does: a bare id is a scenario with no scored episode at all, and
    `id 1/3` is a scenario where it kept 1 of the 3 episodes another candidate kept. Both change
    what a fitter weighs, so both are per candidate here rather than summed into Unscored.

    Args:
        console: Destination for the coverage table and diagnostics.
        rows: Per-candidate scored-coverage measurements to render.
    """
    most: Counter[str] = Counter()
    for row in rows:
        for sid, count in row.scored_episodes:
            most[sid] = max(most[sid], count)
    table = Table(title="Scored coverage per candidate (`fit` SKIPS unscored cells)")
    table.add_column("Candidate", no_wrap=True)
    table.add_column("Scored", justify="right")
    table.add_column("Unscored", justify="right")
    table.add_column("Scenarios lost, or thinned (kept/most)")
    for row in rows:
        gaps = [
            sid if count == 0 else f"{sid} {count}/{most[sid]}"
            for sid, count in row.scored_episodes
            if count == 0 or count < most[sid]
        ]
        lost = ", ".join(gaps[:_LOST_SHOWN])
        if len(gaps) > _LOST_SHOWN:
            lost += f" (+{len(gaps) - _LOST_SHOWN} more)"
        # Scenario ids are corpus data (trace ids) and candidate names are operator strings: both
        # reach a rich console, where `[a]` is markup that would silently drop from the table.
        table.add_row(
            escape(row.candidate),
            f"{row.scored:,}",
            f"{row.unscored:,}",
            escape(lost) if lost else "-",
        )
    console.print(table)
    for row in rows:
        if row.scored == 0 and row.first_error is not None:
            # A candidate that was never scored at all is the one failure a coverage table cannot
            # explain, and its cause is already on disk. Surfacing the first one names the entry
            # and points at the pool file, which is where the fix is.
            console.print(
                f"  [yellow]{escape(row.candidate)}[/yellow] was never scored; its first cell "
                f"failed with: {escape(row.first_error)}\n"
                "    fix that entry in the pool file, drop it, or retry once the cause has cleared"
            )


def print_cost_estimate(console: Console, plan: SweepPlan, *, already_measured: int = 0) -> None:
    """Render the projected spend, stating exactly which parts are assumed.

    Honest by construction: the CELL and CALL counts are real (the step budget is the per-episode
    cap, so calls are an upper bound), the tokens per call are an assumption the flags name, and
    the per-candidate $/Mtok is the pool entry's own price row. The world model's serve and judge
    calls are a separate meter (the D12 cost split) and are deliberately absent.

    `already_measured` is the count a resume will reuse instead of buying. The table above it is
    still the whole grid, because that is what the per-candidate arithmetic describes; the line
    under it says how much of that grid this run is actually paying for.

    Args:
        console: Destination for the spend table and assumptions.
        plan: Sweep plan containing the priced candidate grid.
        already_measured: Cells a resumed sweep will reuse without buying again.
    """
    table = Table(title="Route sweep cost estimate (ASSUMED tokens, not a measurement)")
    table.add_column("Candidate", no_wrap=True)
    table.add_column("Episodes", justify="right")
    table.add_column("Calls (max)", justify="right")
    table.add_column("$/Mtok in", justify="right")
    table.add_column("$/Mtok out", justify="right")
    table.add_column("USD (est)", justify="right")
    for line in plan.cost_lines:
        table.add_row(
            # A rich cell renders markup: an operator-chosen name like `gpt[a]` would print as
            # `gpt`, making two candidates indistinguishable in the table they confirm spend from
            # (and a name containing a closing tag would abort the command outright).
            escape(line.candidate),
            f"{line.episodes:,}",
            f"{line.calls:,}",
            f"{line.input_per_mtok:.3f}",
            f"{line.output_per_mtok:.3f}",
            f"{line.usd:.2f}",
        )
    console.print(table)
    console.print(
        f"{plan.cells} cell(s) = {len(plan.cost_lines)} candidate(s) x "
        f"{len(plan.tasks)} held-out task(s) x {plan.episodes} episode(s); estimated "
        f"total ${plan.total_usd:.2f}"
    )
    if already_measured:
        console.print(
            f"  RESUMING: {already_measured} of those cell(s) are already measured beside the "
            f"matrix and are NOT bought again, so this run measures {plan.cells - already_measured}"
            " and spends proportionally less than the total above."
        )
    if plan.max_concurrency > 1:
        console.print(
            f"  {plan.max_concurrency} cell(s) run at once, so the sweep finishes sooner for the "
            "same money; it does not change what is measured."
        )
    console.print(
        f"  ASSUMPTION: {plan.assume_input_tokens:,} input + {plan.assume_output_tokens:,} output "
        f"token(s) per policy call, and every episode running its full {plan.max_steps}-call "
        "budget. Calls are capped, tokens per call are NOT measured: set "
        "--assume-input-tokens/--assume-output-tokens from your own numbers, or read the measured "
        "spend this command prints when it finishes."
    )
    console.print(
        "  Candidate side only: the world model's own serve and judge calls are metered "
        "separately and are NOT in this figure."
    )


def _confirm_cost(plan: SweepPlan, *, yes: bool) -> None:
    """Confirm the projected spend before any episode runs.

    Consent is said, never inferred: a non-interactive session cannot answer a prompt, so a
    spending run REFUSES unless `--yes` was passed. This command shipped proceed-and-note for
    its first day, and the equivalent branch in `wmo optimize model` spent a scripted caller's
    real money it never agreed to; every spend surface now shares one refusal
    (`wmo.cli.consent.require_spend_consent`).

    Raises:
        typer.Exit: The user declined (exit code 0), or a non-interactive session was not told
            `--yes` (exit code 2).
    """
    if not require_spend_consent(
        _console,
        yes=yes,
        spend=f"~${plan.total_usd:.2f} across {plan.cells} cell(s)",
        command="wmo optimize route sweep",
    ):
        raise typer.Exit(0)


def register(app: typer.Typer) -> None:
    """Register the route sweep command on its parent Typer app.

    Args:
        app: Parent Typer application that owns the route command group.
    """
    app.command("sweep", help="Measure routing candidates closed-loop into an outcome matrix.")(
        sweep
    )
