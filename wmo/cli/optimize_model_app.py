"""`wmo optimize model`: the one-command optimizer, staged and resumable.

After `wmo build`, this turns a world model into an optimized, servable endpoint in one command:
preflight -> sweep -> fit -> tune -> report, one cost forecast up front, one question, and a
three-objective headline at the end. Every stage calls the same library function the matching
manual command calls (`wmo.optimize.sweep`, `knn.fit_knn_artifact`, `knn.tune_policy_dial`,
`report.build_report`), so consent, metering, and artifacts stay single-sourced and a user can
drop to any manual command mid-flow. The only artifact format this command adds is its manifest.

`--compressor` activates the compaction slot between sweep and fit. It is not a paid step of its
own: it configures the sweep (which then measures that D-COMPRESS arm) and the fit (which embeds
its bank through the compressor, the representation-consistency rule), so the plan table shows it
as a row that spends nothing while the compressor's own per-call bill lands folded into the sweep's
candidate spend.

Not in this build: the `--distill` stage (train a student, gate it, add it to the pool, re-sweep).
It is named in `wmo.optimize.pipeline.STAGE_ORDER` so its arrival is additive; passing `--distill`
today is a usage error that names the manual command instead. Its PREFLIGHT already exists,
though (`wmo.optimize.teacher`), and that refusal carries the teacher-search verdict on whatever
matrix this model has: usually there is no teacher gap, and the cheapest run is the one the
evidence says to skip.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

import typer
from pydantic import BaseModel, ConfigDict, ValidationError
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from wmo.cli.consent import require_spend_consent
from wmo.cli.route_app import (
    BIAS_ACCEPTED_NOTE,
    NO_EVIDENCE_WARNING,
    _compressor_note,
    cell_progress,
    print_coverage,
    print_deferred_risks,
    print_tiny_corpus_note,
    print_world_model_spend,
    uneven_warning,
)
from wmo.config import ARTIFACT_DIR, WorldModelStore
from wmo.engine import load_world_model
from wmo.env import WorldModelEnv
from wmo.env.closed_loop import scenario_id
from wmo.optimize.compression import (
    CompressionConfig,
    compression_signature,
    resolve_compression,
)
from wmo.optimize.knn import (
    COST_QUALITY_BALANCED,
    DEFAULT_KNN_MIN_PAIRS,
    DEFAULT_RAG_NUM,
    DEFAULT_RAG_THRES,
    cost_quality_named_point,
    fit_knn_artifact,
    fit_provenance,
    tune_policy_dial,
)
from wmo.optimize.outcomes import OutcomeMatrix, load_matrix_with_digest
from wmo.optimize.pipeline import (
    BUILT_STAGES,
    CONFIGURED_STAGES,
    MANIFEST_DIRNAME,
    MANIFEST_FILENAME,
    MATRIX_FILENAME,
    REPORT_FILENAME,
    RESERVED_STAGES,
    BudgetExceeded,
    RunManifest,
    SpendLedger,
    Stage,
    StageDecision,
    StageRecord,
    StageStatus,
    SweepSpendProjection,
    decide_stage,
    file_sha256,
    forced_stages,
    load_manifest,
    planned_stages,
    project_sweep_spend,
)
from wmo.optimize.policy import (
    AZURE_EMBEDDER_DEPLOYMENT,
    AZURE_EMBEDDER_ENV,
    DEFAULT_KNN_Z,
    POLICY_FILENAME,
    EmbedderSpec,
    RoutingPolicy,
    embedder_provenance,
    probe_embedder,
    resolve_embedder,
)
from wmo.optimize.report import ImprovementReport, build_report
from wmo.optimize.sweep import (
    SweepError,
    SweepPlan,
    coverage,
    execute_sweep,
    plan_sweep,
    resolve_config,
    resumable_cells,
)
from wmo.optimize.sweep import preflight_pool as run_preflight
from wmo.optimize.teacher import TeacherSearchVerdict, select_teacher
from wmo.providers.pool import DEFAULT_POOL_PATH

_console = Console()

DEFAULT_SCENARIOS = 20
DEFAULT_EPISODES = 1
DEFAULT_MAX_STEPS = 20
# The sweep's cost projection needs a per-call token assumption. These are `route sweep`'s own
# defaults, not flags here: the orchestrator's surface stays the decisions a user owns, and the
# plan table names the assumption in words so the number is never read as a measurement.
ASSUMED_INPUT_TOKENS = 2000
ASSUMED_OUTPUT_TOKENS = 250

_KNN_KNOBS = (
    f"z={DEFAULT_KNN_Z:g} k={DEFAULT_RAG_NUM} thres={DEFAULT_RAG_THRES:g} "
    f"pairs={DEFAULT_KNN_MIN_PAIRS} se_floor=True q=0.05"
)
"""The knn fit this command performs. Fixed: the validated champion, dialed after the fact."""


def optimize_model(  # noqa: PLR0913 - each flag is one decision a user owns (see the help text)
    world_model: str = typer.Argument(
        None, help="Built world model to optimize (default: the only one under --root)."
    ),
    pool_file: str = typer.Option(
        str(DEFAULT_POOL_PATH),
        "--pool",
        # The doubled brackets are escaped: typer renders help through rich markup, which
        # otherwise swallows them and prints an empty pair.
        help="Candidate pool TOML the router chooses between: one \\[\\[model]] table per "
        "candidate, as `wmo providers set` writes it.",
    ),
    traces_file: str = typer.Option(
        None,
        "--traces",
        help="Trace corpus the held-out scenarios come from (default: the model's own "
        "traces.otel.jsonl). A build keeps no copy of the corpus it read, so pass the file here.",
    ),
    scenarios: int = typer.Option(
        DEFAULT_SCENARIOS,
        "--scenarios",
        min=1,
        help="Cap on held-out scenarios measured. More scenarios is better evidence and more "
        "spend, linearly.",
    ),
    episodes: int = typer.Option(
        DEFAULT_EPISODES,
        "--episodes",
        min=1,
        help="Episodes per (candidate, scenario) cell. Raise it when your rewards are noisy.",
    ),
    max_steps: int = typer.Option(
        DEFAULT_MAX_STEPS, "--max-steps", min=1, help="Step budget per episode."
    ),
    concurrency: int = typer.Option(
        1,
        "--concurrency",
        min=1,
        help="Cells the sweep measures at once (1 = one at a time). Changes only how long the "
        "sweep takes, never what it measures, so it is not part of what decides whether the "
        "sweep stage can be skipped. Your PROVIDER LIMITS are the real ceiling: every candidate "
        "call and every world-model serve and judge call is a request, and the world model's own "
        "calls all come out of ONE account's bucket.",
    ),
    cost_quality: float = typer.Option(
        COST_QUALITY_BALANCED,
        "--cost-quality",
        min=0.0,
        max=1.0,
        help="The endpoint's one dial, set at the end: 0.0 = max quality, 1.0 = max savings. "
        "0.25 is the shipped default.",
    ),
    fallback: str = typer.Option(
        None,
        "--fallback",
        help="Pool model every request uses unless the evidence says otherwise, and the anchor "
        "the closing numbers are quoted against. Default: the best single model on the sweep.",
    ),
    baseline: str = typer.Option(
        None,
        "--baseline",
        help="Compare the final numbers against this pool model instead of the fallback.",
    ),
    compressor: str = typer.Option(
        None,
        "--compressor",
        help="Compress every request through this compressor before routing it (identity | "
        "truncate | llmlingua2-endpoint). The sweep then measures that arm and the fit embeds "
        "its bank through the same compressor, so the endpoint serves what was measured. "
        "Default: no compression.",
    ),
    aggressiveness: float = typer.Option(
        0.0,
        "--aggressiveness",
        min=0.0,
        max=1.0,
        help="Compressor-defined dial in [0, 1] for --compressor: 0.0 is a no-op and higher never "
        "removes less, but it is not an exact removal fraction (the achieved ratio is measured per "
        "episode).",
    ),
    embedder: str = typer.Option(
        "auto",
        "--embedder",
        help="What the fitted policy routes on: auto | hashing | azure. `auto` uses "
        "text-embedding-3-large when AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT are set "
        "(semantic features, billed to that resource) and hashing-512 otherwise; the resolution "
        "is always printed.",
    ),
    distill: str = typer.Option(
        None,
        "--distill",
        help="NOT IN THIS BUILD. Reserved for the distillation stage (train a student, gate it, "
        "add it to the pool, re-sweep it). Use `wmo optimize distill run` for now, and "
        "`wmo optimize distill probe <matrix.json>` to find out whether you should at all: "
        "passing this flag prints that verdict for this model's own matrix.",
    ),
    force_from: str = typer.Option(
        None,
        "--force-from",
        help="Redo this stage and everything after it, even when its inputs are unchanged: "
        "sweep | fit | tune | report.",
    ),
    max_usd: float = typer.Option(
        None,
        "--max-usd",
        min=0.0,
        help="Stop before any paid stage whose projection would carry this run past this many "
        "USD, counting what earlier runs already spent. Candidate spend and the world model's "
        "own eval spend both count against it. The run stays resumable and prints how to "
        "continue it.",
    ),
    allow_uneven_coverage: bool = typer.Option(
        False,
        "--allow-uneven-coverage",
        help="Fit even when the candidates were not scored on the same evidence. The fit is then "
        "biased; the coverage table prints either way.",
    ),
    root: str = typer.Option(ARTIFACT_DIR, "--root", help="Project dir holding the built models."),
    yes: bool = typer.Option(
        False,
        "--yes",
        help="Consent to the projected spend up front. Required in a non-interactive "
        "session (CI, cron, piped output), where a spending run otherwise refuses to start.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print the plan table (what would run, what it is projected to cost) and exit "
        "without spending anything or touching any artifact.",
    ),
) -> None:
    """Measure, fit, tune, and report a routing policy for a world model, in one command.

    The whole routing workflow with one question in it:

        wmo optimize model support

    Stage by stage that is `route sweep` (the only paid step: every candidate runs the model's own
    held-out scenarios closed-loop), `route fit --kind knn`, `route tune`, and `route report`. One
    plan table prints before anything spends, showing what each stage will do and what it is
    projected to cost, and one confirmation covers the run.

    Re-running is cheap and safe. A stage is skipped when its inputs are unchanged, and the reason
    is printed either way, so a run that stopped halfway resumes at the stage that stopped it:

        wmo optimize model support                 # resumes; unchanged stages say why they skipped
        wmo optimize model support --force-from sweep   # buy fresh cells anyway
        wmo optimize model support --yes --max-usd 25   # scripted, with a hard spend cap
        wmo optimize model support --concurrency 6      # six cells at once, same evidence

    Resume is per CELL inside the sweep, not just per stage: a sweep that died at hour five keeps
    every cell it paid for (in `<matrix>.partial.jsonl`) and the next run measures only what is
    missing, so an interrupted grid never gets bought twice.

    `--compressor` measures and fits a compressed arm end to end, which is the same thing a
    `route sweep --compressor` plus `route fit --compressor` pair does by hand:

        wmo optimize model support --compressor llmlingua2-endpoint --aggressiveness 0.4

    Artifacts land exactly where the manual commands put them, so you can drop to any of them
    mid-flow and this command resumes around it: `policy.json` (plus its evidence bank) in the
    model's own directory where `wmo serve` reads it, and the outcome matrix, report, and run
    manifest under `<model>/optimize/`. Deleting that directory resets resume and breaks nothing.
    """
    if distill is not None:
        raise typer.BadParameter(_distill_reserved_message(world_model=world_model, root=root))
    if compressor is None and aggressiveness > 0.0:
        raise typer.BadParameter("--aggressiveness needs --compressor to apply it")
    compression: CompressionConfig | None = None
    if compressor is not None:
        try:
            # Resolved at PLAN time, before the table: an unknown id, or an implementation that
            # could never be mounted, is a usage error. Discovering it after the sweep this arm
            # configured has been paid for would be discovering it too late to matter.
            compression = resolve_compression(compressor, aggressiveness)
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
    try:
        embedder_spec, resolution = _resolve_embedder_choice(embedder)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    redo = _parse_force_from(force_from, compacting=compression is not None)
    stages = planned_stages(compacting=compression is not None)
    store = WorldModelStore(root)
    try:
        model_dir = store.resolve(world_model)
    except FileNotFoundError as exc:
        raise typer.BadParameter(str(exc)) from exc
    except ValueError as exc:
        names = store.list_names()
        raise typer.BadParameter(
            f"multiple world models built ({', '.join(names)}); name one as the WORLD-MODEL "
            f"argument, e.g. `wmo optimize model {names[0]}`"
        ) from exc

    run_dir = model_dir / MANIFEST_DIRNAME
    paths = _RunPaths(
        manifest=run_dir / MANIFEST_FILENAME,
        matrix=run_dir / MATRIX_FILENAME,
        policy=model_dir / POLICY_FILENAME,
        report=run_dir / REPORT_FILENAME,
    )
    read = load_manifest(paths.manifest, world_model=model_dir.name)
    if read.warning is not None:
        _console.print(f"[yellow]note[/yellow] {escape(read.warning)}")
    manifest = read.manifest

    # Preflight runs before the plan table by necessity: it is what proves the candidates are
    # usable and what prices them, and both have to be true before an operator is asked to
    # authorize anything. It spends nothing, so running it unconditionally costs only time.
    try:
        config = resolve_config(model_dir)
        preflight = run_preflight(Path(pool_file))
        print_deferred_risks(_console, preflight.deferred)
        plan = plan_sweep(
            model_dir=model_dir,
            config=config,
            pool=preflight.pool,
            out_path=paths.matrix,
            traces_file=Path(traces_file) if traces_file is not None else None,
            scenarios=scenarios,
            episodes=episodes,
            max_steps=max_steps,
            assume_input_tokens=ASSUMED_INPUT_TOKENS,
            assume_output_tokens=ASSUMED_OUTPUT_TOKENS,
            compression=compression,
            max_concurrency=concurrency,
        )
        # Read before the plan table so the sweep row can say how much of the grid a previous
        # attempt already bought, and so a sidecar from a different plan is refused for free.
        already_measured = resumable_cells(plan)
    except SweepError as exc:
        raise typer.BadParameter(str(exc)) from exc
    print_tiny_corpus_note(_console, plan)
    # Both flags name a pool candidate, and the pool is loaded by the pre-flight above, so a typo
    # is knowable here for free: a boundary error rather than a surprise after the sweep has been
    # paid for and the fit written. --fallback used to survive this far and then be printed in the
    # plan table as if it were a real model, only failing inside the fit stage.
    known = [entry.name for entry in preflight.pool.models]
    for flag, value, why in (
        ("--fallback", fallback, "the fit can only guard a candidate the sweep measures"),
        ("--baseline", baseline, "the report can only anchor on a candidate the sweep measures"),
    ):
        if value is not None and value not in known:
            raise typer.BadParameter(
                f"{flag} '{value}' is not a model in {pool_file}; {why}. "
                f"Available: {', '.join(known)}"
            )

    # Printed before the plan table, the way `route fit` prints it before the fit: the embedder
    # decides what the policy can route on, and an operator who meant to fit on semantic vectors
    # should see that it resolved to hashed features before authorizing any spend.
    _console.print(resolution)
    # The world-model side of a sweep is not projectable from arithmetic, but once this model has
    # been swept once its OWN measured ratio is, and it is far too big to leave out of a cap
    # (7.0x the candidate side on a real tau corpus).
    projection = project_sweep_spend(plan.total_usd, manifest.record_for(Stage.SWEEP))
    decisions = _plan_stages(
        stages,
        manifest=manifest,
        paths=paths,
        plan=plan,
        pool_file=Path(pool_file),
        embedder=embedder_spec,
        compression=compression,
        fallback=fallback,
        baseline=baseline,
        cost_quality=cost_quality,
        allow_uneven=allow_uneven_coverage,
        redo=redo,
    )
    _print_plan(
        _console,
        model_dir.name,
        pool_file=Path(pool_file),
        pool_size=len(plan.pool.models),
        plan=plan,
        decisions=decisions,
        cost_quality=cost_quality,
        fallback=fallback,
        anchor=_report_anchor(paths.policy, baseline=baseline, fallback=fallback),
        embedder=embedder_spec,
        compression=compression,
        projection=projection,
        paths=paths,
        already_measured=already_measured,
    )
    # After the plan, before any consent or spend question: the whole point of a dry run is
    # reading the table above without committing to anything, so it exits here even when the
    # budget check below would have refused the real run (the table already shows the numbers).
    if dry_run:
        _console.print("\ndry run: nothing was run and nothing was spent")
        raise typer.Exit(0)

    # One throwaway embedding before anything is bought, and only when a fit will actually happen:
    # `--embedder auto` turns the mere presence of AZURE_OPENAI_* into a network dependency, and
    # those variables routinely point at a resource that serves chat but hosts no embedding
    # deployment. Without this the failure lands after the sweep has been paid for. After the dry
    # run exits, because a dry run promises to reach nothing.
    if any(decision.stage is Stage.FIT and decision.will_run for decision in decisions):
        try:
            probe_embedder(embedder_spec)
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc

    # Seeded from every dollar this model's optimization has already spent, both sides:
    # --max-usd bounds the optimization, not one invocation of it (see `SpendLedger`).
    ledger = SpendLedger(max_usd=max_usd, spent_usd=manifest.lifetime_spend_usd)
    try:
        # Before the question, not after it: being asked to approve a run and then told it cannot
        # start is a worse experience than being told first, and both numbers are known here.
        if _will_sweep(decisions):
            ledger.check(Stage.SWEEP, projection.total_usd, basis=projection.basis)
    except BudgetExceeded as exc:
        _print_budget_stop(model_dir.name, exc)
        raise typer.Exit(1) from exc
    if not _confirm(decisions, plan, yes=yes):
        _console.print("nothing was run and nothing was spent")
        raise typer.Exit(0)

    try:
        manifest = _run_stages(
            decisions,
            manifest=manifest,
            ledger=ledger,
            paths=paths,
            plan=plan,
            projection=projection,
            model_dir=model_dir,
            pool_file=Path(pool_file),
            embedder=embedder_spec,
            compression=compression,
            fallback=fallback,
            baseline=baseline,
            cost_quality=cost_quality,
            allow_uneven_coverage=allow_uneven_coverage,
            already_measured=already_measured,
        )
    except BudgetExceeded as exc:
        _print_budget_stop(model_dir.name, exc)
        raise typer.Exit(1) from exc
    # No save here: `_run_stages` persists after every stage it runs, which is what keeps a run
    # that dies mid-flight resumable.
    _print_payoff(_console, model_dir.name, paths=paths, cost_quality=cost_quality)


class _RunPaths(BaseModel):
    """Where this run's artifacts live. Serving paths stay where serving already looks."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    manifest: Path
    matrix: Path
    policy: Path
    report: Path


def _distill_reserved_message(*, world_model: str | None, root: str) -> str:
    """Why `--distill` is refused, plus this model's teacher-search verdict when one is readable.

    The stage is not wired, but its PREFLIGHT is (`wmo.optimize.teacher`), and the preflight is
    the half that decides whether the stage should run at all. Printing the verdict here means
    the product surface already answers the real question ("should I distill?") in the same place
    an operator asked for it, and usually answers no, which is the cheapest possible outcome.

    Everything about the lookup is best effort: an unresolvable model, an absent matrix, or a
    matrix too small to compare adds nothing to the message rather than replacing a usage error
    with an unrelated one.
    """
    base = (
        "--distill is reserved and not implemented in this build: the distillation stage "
        "(train a student, gate it, add it to the pool, re-sweep it, merge the matrix) is "
        "still a separate workflow. Run `wmo optimize distill run --config <toml>`, then "
        "`wmo optimize route student <run-dir> --input-per-mtok ... --output-per-mtok ...` "
        "to put the student in the pool, then re-run this command without --distill."
    )
    found = _teacher_verdict(world_model=world_model, root=root)
    if found is None:
        return base
    matrix_path, verdict = found
    return (
        f"{base} Worth knowing before you spend anything, the teacher-search verdict on this "
        f"model's current matrix: {verdict.reason} "
        f"(`wmo optimize distill probe {matrix_path}` prints the full table.)"
    )


def _teacher_verdict(
    *, world_model: str | None, root: str
) -> tuple[Path, TeacherSearchVerdict] | None:
    """This model's teacher search over the matrix already on disk, or None when unavailable."""
    try:
        model_dir = WorldModelStore(root).resolve(world_model)
    except (FileNotFoundError, ValueError):
        return None
    matrix_path = model_dir / MANIFEST_DIRNAME / MATRIX_FILENAME
    if not matrix_path.is_file():
        return None
    try:
        return matrix_path, select_teacher(OutcomeMatrix.load(matrix_path))
    except (ValidationError, ValueError, OSError):
        return None


def _resolve_embedder_choice(choice: str) -> tuple[EmbedderSpec, str]:
    """`--embedder` as the spec to fit with, plus the one line explaining the choice.

    The resolution itself is `wmo.optimize.policy.resolve_embedder`, shared with
    `wmo optimize route fit` so the two commands cannot disagree about what `auto` means. What is
    decided here is the narrower surface: this command takes no `--deployment`, `--endpoint`, or
    `--dim`, because its whole promise is one command with no routing knobs in it. So `azure` reads
    the same standard environment pair `auto` looks for rather than flags that do not exist, and
    the operator who needs to name a specific deployment is pointed at the manual fit that has
    those flags.

    Raises:
        ValueError: An unknown choice, or `azure` with nothing in the environment to point it at.
    """
    endpoint = os.environ.get(AZURE_EMBEDDER_ENV[1])
    if choice == "azure" and not endpoint:
        raise ValueError(
            f"--embedder azure needs {' and '.join(AZURE_EMBEDDER_ENV)} in the environment: this "
            "command resolves the azure deployment from that standard pair rather than taking "
            "--deployment/--endpoint flags. Export them, or fit by hand with `wmo optimize route "
            "fit <matrix.json> --kind knn --embedder azure --deployment <name> --endpoint <url>`."
        )
    explicit_azure = choice == "azure"
    return resolve_embedder(
        choice,
        dim=None,
        deployment=AZURE_EMBEDDER_DEPLOYMENT if explicit_azure else None,
        endpoint=endpoint if explicit_azure else None,
        api_key_env=AZURE_EMBEDDER_ENV[0] if explicit_azure else None,
    )


def _parse_force_from(force_from: str | None, *, compacting: bool) -> frozenset[Stage]:
    """`--force-from` as the set of stages it invalidates, or a usage error naming the choices."""
    if force_from is None:
        return frozenset()
    redoable = [stage for stage in BUILT_STAGES if stage is not Stage.PREFLIGHT]
    try:
        stage = Stage(force_from)
    except ValueError as exc:
        raise typer.BadParameter(
            f"unknown stage '{force_from}' for --force-from; use "
            f"{' | '.join(item.value for item in redoable)}"
        ) from exc
    if stage in CONFIGURED_STAGES:
        # Forcing a configuration is a category error, whether or not it is active on this run:
        # the compaction stage does no work to redo. What changes its outcome is the arm itself,
        # and changing that already re-measures the sweep (the arm is in its fingerprint).
        instead = (
            "Change --compressor/--aggressiveness to measure a different arm"
            if compacting
            else "This run named no compressor for it to configure anything with"
        )
        raise typer.BadParameter(
            f"stage '{stage.value}' configures the sweep and the fit rather than running on its "
            f"own, so there is nothing to force. {instead}, or force one of "
            f"{' | '.join(item.value for item in redoable)}"
        )
    if stage in RESERVED_STAGES:
        raise typer.BadParameter(
            f"stage '{stage.value}' is a reserved slot that this build does not run, so there is "
            f"nothing to force; use {' | '.join(item.value for item in redoable)}"
        )
    if stage is Stage.PREFLIGHT:
        raise typer.BadParameter(
            "preflight always runs (it spends nothing), so --force-from preflight would change "
            f"nothing; use {' | '.join(item.value for item in redoable)}"
        )
    return forced_stages(stage)


def _plan_stages(
    stages: tuple[Stage, ...],
    *,
    manifest: RunManifest,
    paths: _RunPaths,
    plan: SweepPlan,
    pool_file: Path,
    embedder: EmbedderSpec,
    compression: CompressionConfig | None,
    fallback: str | None,
    baseline: str | None,
    cost_quality: float,
    allow_uneven: bool,
    redo: frozenset[Stage],
) -> list[StageDecision]:
    """Decide every stage against what is on disk, upstream first.

    A stage downstream of one that will run is planned as running too, without consulting its
    fingerprints: the inputs it would be compared against are the artifacts the upstream stage is
    about to replace, so any verdict taken now would describe a state that no longer exists by the
    time the stage is reached. Saying "sweep reruns first" is both true and what the plan table
    then honors.

    COMPACT is the exception, decided on its own fingerprint however dirty the run above it is. Its
    only input is the arm the operator named, which no other stage writes, so borrowing the "runs
    after sweep, which will change its input" line for it would print something false. It still
    dirties what follows: the fit embeds its bank through the compressor, so a changed arm is a
    changed fit.
    """
    decisions: list[StageDecision] = []
    running: Stage | None = None
    for stage in stages:
        if stage is Stage.PREFLIGHT:
            continue
        if running is not None and stage not in CONFIGURED_STAGES:
            decisions.append(
                StageDecision(
                    stage=stage,
                    status=StageStatus.RUN,
                    reason=f"runs after {running.value}, which will change its input",
                )
            )
            continue
        live = _live_inputs(
            stage,
            paths=paths,
            plan=plan,
            pool_file=pool_file,
            embedder=embedder,
            compression=compression,
            fallback=fallback,
            baseline=baseline,
            cost_quality=cost_quality,
            allow_uneven=allow_uneven,
        )
        decision = decide_stage(
            stage,
            manifest=manifest,
            forced=stage in redo,
            fingerprint=live.fingerprint,
            artifact=live.artifact,
            artifact_identity=live.artifact_identity,
            skip_summary=live.skip_summary,
        )
        decisions.append(decision)
        if decision.will_run:
            running = stage
    return decisions


class _StageInputs(BaseModel):
    """What one stage consumes and produces right now: everything `decide_stage` compares.

    `artifact` is None for a stage that writes no file (compaction configures the two stages
    around it), which leaves its fingerprint the whole of its verdict.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    fingerprint: dict[str, str]
    artifact: Path | None = None
    artifact_identity: str | None = None
    skip_summary: str


def _compact_fingerprint(compression: CompressionConfig | None) -> dict[str, str]:
    """The arm the compaction stage configured, rendered once for both the plan and the record.

    Deliberately the same string `compression_signature` puts in the SWEEP fingerprint, and
    deliberately recorded twice: the sweep's copy is the one that forces cells to be re-measured
    when the arm moves, since that is what makes its rewards mean something different. This copy
    only decides whether the compact ROW reads as run or skipped, and sharing the rendering is what
    keeps the two from ever describing different arms.
    """
    return {"compression": compression_signature(compression)}


def _fit_fingerprint(
    *,
    matrix: Path,
    embedder: EmbedderSpec,
    compression: CompressionConfig | None,
    fallback: str | None,
    allow_uneven: bool,
) -> dict[str, str]:
    """Everything that decides what the fit produces, for both the plan and the record.

    One function so the planned and recorded fingerprints cannot drift, which matters most for the
    conditional key: `compression` is present only when a compressor is named. Adding it
    unconditionally would refit every model whose manifest predates the flag in order to record a
    value meaning "no compression", which is exactly what the key's absence already says. Both
    directions still rerun the fit, since an added key and a removed one are each a difference.
    """
    fingerprint = {
        "matrix": file_sha256(matrix),
        "kind": "knn",
        "fallback": fallback or "auto",
        "knobs": _KNN_KNOBS,
        "embedder": embedder_provenance(embedder),
        # Whether the operator accepted biased evidence is part of what produced this fit. Without
        # it here, consent given once would stick silently: a later run WITHOUT the flag would skip
        # the fit, never reach the coverage gate, and leave a policy fitted on knowingly-uneven
        # evidence with nothing saying so.
        "allow_uneven": str(allow_uneven),
    }
    if compression is not None:
        # The fit embeds its bank through the compressor and stamps the arm on the policy, so the
        # arm is an input to the fit in its own right, not only through the matrix it reads.
        fingerprint["compression"] = compression_signature(compression)
    return fingerprint


def _live_inputs(
    stage: Stage,
    *,
    paths: _RunPaths,
    plan: SweepPlan,
    pool_file: Path,
    embedder: EmbedderSpec,
    compression: CompressionConfig | None,
    fallback: str | None,
    baseline: str | None,
    cost_quality: float,
    allow_uneven: bool,
) -> _StageInputs:
    """What `stage` would consume and produce right now, read off the filesystem."""
    match stage:
        case Stage.SWEEP:
            return _StageInputs(
                fingerprint={
                    "pool": file_sha256(pool_file),
                    "scenarios": _scenario_identity(plan),
                    "episodes": str(plan.episodes),
                    "max_steps": str(plan.max_steps),
                    # A matrix's rewards belong to ONE D-COMPRESS arm, so a changed compressor
                    # means different evidence and the cells have to be bought again. This is the
                    # fingerprint that makes `--compressor` re-measure rather than reuse cells
                    # that ran under a different arm.
                    "compression": compression_signature(plan.compression),
                },
                artifact=paths.matrix,
                artifact_identity=file_sha256(paths.matrix),
                skip_summary="same pool, same scenarios, same episodes",
            )
        case Stage.COMPACT:
            return _StageInputs(
                fingerprint=_compact_fingerprint(compression),
                skip_summary="same compressor, same aggressiveness",
            )
        case Stage.FIT:
            return _StageInputs(
                fingerprint=_fit_fingerprint(
                    matrix=paths.matrix,
                    embedder=embedder,
                    compression=compression,
                    fallback=fallback,
                    allow_uneven=allow_uneven,
                ),
                artifact=paths.policy,
                artifact_identity=_policy_fit_identity(paths.policy),
                skip_summary="same matrix, same knn knobs",
            )
        case Stage.TUNE:
            return _StageInputs(
                fingerprint={
                    "policy_fit": _policy_fit_identity(paths.policy),
                    "cost_quality": f"{cost_quality:g}",
                },
                artifact=paths.policy,
                artifact_identity=file_sha256(paths.policy),
                skip_summary=f"same fit, dial already at {cost_quality:g}",
            )
        case _:
            return _StageInputs(
                fingerprint={
                    "matrix": file_sha256(paths.matrix),
                    "policy": file_sha256(paths.policy),
                    "anchor": _report_anchor(paths.policy, baseline=baseline, fallback=fallback),
                },
                artifact=paths.report,
                artifact_identity=file_sha256(paths.report),
                skip_summary="same matrix, same policy, same anchor",
            )


def _report_anchor(policy_path: Path, *, baseline: str | None, fallback: str | None) -> str:
    """The model the closing numbers are quoted against, resolved the same way at plan and run.

    Defaulting to the fitted policy's own fallback is what makes the headline answer the question
    an operator actually has ("is routing better than just using the model I would have used?").
    Resolving it identically here and in the report stage matters for a duller reason: a planned
    anchor that did not match the recorded one would make the report look changed on every
    resume, and re-run forever.
    """
    if baseline is not None:
        return baseline
    if policy_path.is_file():
        try:
            return RoutingPolicy.load(policy_path).default_model
        except (OSError, ValueError):
            pass
    # No policy yet, so the fit is about to pick one and the report is running regardless.
    return fallback or "the fitted fallback"


def _scenario_identity(plan: SweepPlan) -> str:
    """The scenario SET the sweep will measure, as an id list a change is visible in."""
    return ",".join(scenario_id(scenario) for scenario in plan.scenarios)


def _policy_fit_identity(policy_path: Path) -> str:
    """Which FIT the policy on disk came from, surviving the dial that `tune` applies to it.

    A byte hash would call every tuned policy a stranger to the fit that produced it, so `fit`
    would rerun on every resume. `fit_provenance` strips the dial suffix, so this changes when
    someone refits (by hand or otherwise) and not when the dial moves.
    """
    if not policy_path.is_file():
        return "missing"
    try:
        return fit_provenance(RoutingPolicy.load(policy_path))
    except (OSError, ValueError):
        # An unreadable policy is a rerun, not a crash: the fit stage overwrites it anyway.
        return "unreadable"


# --------------------------------------------------------------------------------- the plan table


def _stage_plan_text(
    stage: Stage,
    *,
    plan: SweepPlan,
    compression: CompressionConfig | None,
    cost_quality: float,
    fallback: str | None,
    anchor: str,
    already_measured: int = 0,
) -> str:
    """One line saying what this stage will actually do, in the operator's terms."""
    match stage:
        case Stage.PREFLIGHT:
            return f"resolve {len(plan.pool.models)} backend(s), check prices"
        case Stage.SWEEP:
            grid = (
                f"{len(plan.pool.models)} candidate(s) x {len(plan.scenarios)} scenario(s) "
                f"x {plan.episodes} episode(s)"
            )
            pace = f", {plan.max_concurrency} at a time" if plan.max_concurrency > 1 else ""
            # A resumed sweep is not the grid it prints: saying so here is what stops the row
            # from reading as a bill for cells the last attempt already paid for.
            resumed = (
                f"; {already_measured} already measured, {plan.cells - already_measured} to buy"
                if already_measured
                else ""
            )
            return f"{grid}{pace}{resumed}"
        case Stage.COMPACT:
            # Says what it IS rather than implying a step: the arm plus the two stages it sets up.
            return f"{escape(compression_signature(compression))}, configures sweep and fit"
        case Stage.FIT:
            return f"knn (guarded, fallback {escape(fallback or 'best single on the sweep')})"
        case Stage.TUNE:
            return f"cost_quality {cost_quality:g} ({cost_quality_named_point(cost_quality)})"
        case _:
            return f"3-objective headline vs {escape(anchor)}"


def _report_estimate(policy_path: Path, fitting_with: EmbedderSpec) -> str:
    """What the report stage costs, or why that cannot honestly be a number.

    Priced off the embedder the report will ACTUALLY use, which `build_report` takes from the
    policy, not off the one this command fits with. They differ exactly when the fit is skipped
    and the policy on disk came from a manual `route fit --embedder azure`: quoting this run's
    hashing spec would then print "free" for a stage about to embed every scenario for money.
    """
    spec = fitting_with
    if policy_path.is_file():
        try:
            spec = RoutingPolicy.load(policy_path).embedder
        except (OSError, ValueError):
            spec = fitting_with  # unreadable: the fit stage is about to replace it anyway
    if spec.kind == "hashing":
        return "free"
    return "unpriced (embeds every scenario; the pool prices completions, not embeddings)"


def _print_plan(
    console: Console,
    name: str,
    *,
    pool_file: Path,
    pool_size: int,
    plan: SweepPlan,
    decisions: list[StageDecision],
    cost_quality: float,
    fallback: str | None,
    anchor: str,
    embedder: EmbedderSpec,
    compression: CompressionConfig | None,
    projection: SweepSpendProjection,
    paths: _RunPaths,
    already_measured: int = 0,
) -> None:
    """The whole run in one table, printed before anything spends.

    Every estimate names itself a projection. The preflight row reads `ok` rather than `will run`
    because it already has: it is what priced the sweep row above it.
    """
    console.print(
        f"\n[bold]optimize model: {escape(name)}[/bold]    "
        f"pool: {pool_size} candidate(s) ({escape(str(pool_file))})\n"
    )
    table = Table(show_header=True, box=None, pad_edge=False, padding=(0, 2))
    table.add_column("stage", no_wrap=True)
    table.add_column("plan")
    table.add_column("est. cost", justify="right")
    table.add_column("status")
    table.add_row(
        Stage.PREFLIGHT.value,
        _stage_plan_text(
            Stage.PREFLIGHT,
            plan=plan,
            compression=compression,
            cost_quality=cost_quality,
            fallback=fallback,
            anchor=anchor,
        ),
        "free",
        "[green]ok[/green]",
    )
    for decision in decisions:
        table.add_row(
            decision.stage.value,
            _stage_plan_text(
                decision.stage,
                plan=plan,
                compression=compression,
                cost_quality=cost_quality,
                fallback=fallback,
                anchor=anchor,
                already_measured=already_measured,
            ),
            _estimate_text(decision.stage, plan=plan, embedder=embedder, paths=paths),
            _status_text(decision),
        )
    console.print(table)
    running = [decision for decision in decisions if decision.will_run]
    total = _projected_total(decisions, plan)
    if not running:
        console.print(
            "\n  every stage is current, so this run has nothing to do and will spend nothing "
            "(pass --force-from <stage> to redo one anyway)"
        )
        return
    if not _will_sweep(decisions):
        # Only free stages left to do. A "~$0.00" line with a paragraph of token assumptions
        # under it is noise about spend that cannot happen.
        console.print(
            "\n  nothing here spends: the sweep is current, and fit, tune, and report are free"
        )
        return
    console.print(
        f"\n  estimated candidate spend ~${total:.2f} (a projection: {ASSUMED_INPUT_TOKENS:,} "
        f"input + {ASSUMED_OUTPUT_TOKENS:,} assumed output token(s) per call, times the real cell "
        f"and call counts, at each candidate's own pool price)"
    )
    console.print(_world_model_forecast(projection, compressed=plan.compression is not None))


def _print_budget_stop(name: str, exc: BudgetExceeded) -> None:
    """Report a cap stop as a pause, not a failure: everything finished is on disk and recorded."""
    _console.print(
        f"\n[yellow]stopped at the spend cap[/yellow] {escape(str(exc))}\n"
        "  every finished stage is on disk and recorded, so nothing is lost. Resume with a "
        f"higher cap: [bold]wmo optimize model {escape(name)} --max-usd <more>[/bold]"
    )


def _will_sweep(decisions: list[StageDecision]) -> bool:
    """Whether this run will buy cells, which is the only thing that costs candidate money."""
    return any(decision.stage is Stage.SWEEP and decision.will_run for decision in decisions)


def _world_model_forecast(projection: SweepSpendProjection, *, compressed: bool) -> str:
    """What to say about the OTHER side of the bill before the operator authorizes the run.

    Two honest answers, never a zero. With a prior sweep of this model there is a measured ratio
    to forecast from, and the line quotes both the number and the single observation it rests on.
    Without one the side is simply not projectable, and saying so has to include how large it can
    be, or "not in this figure" reads as "not much" and the operator plans against the wrong
    number.
    """
    # A compressed arm makes the candidate projection wrong in BOTH directions at once, so say
    # so rather than let the reader assume the usual over-estimate is the whole story.
    arm = (
        "\n  on a compressed arm that candidate figure is an OVER-estimate (it assumes "
        "uncompressed tokens) while the compressor's OWN per-call cost is not in it at all, "
        "because nothing predicts that in advance. Both are measured when the sweep finishes."
        if compressed
        else ""
    )
    if projection.projected:
        return (
            f"  plus a projected ~${projection.world_model_usd:.2f} world-model side "
            f"({projection.basis}), so ~${projection.total_usd:.2f} total is what --max-usd is "
            "checked against. That half is a forecast from one prior sweep, not arithmetic." + arm
        )
    return (
        "  the world model's own serve and judge calls are NOT in that figure and are not "
        "projectable before this model's first sweep: nothing predicts the simulator's and the "
        "judge's token use per episode in advance. It is not a rounding error either, measuring "
        "7.0x the candidate side on one real tau corpus, so treat the number above as a lower "
        "bound. Both sides are measured and reported when the sweep finishes, and the next run "
        "forecasts this line from them." + arm
    )


def _estimate_text(
    stage: Stage, *, plan: SweepPlan, embedder: EmbedderSpec, paths: _RunPaths
) -> str:
    """One stage's `est. cost` cell: a projection, `free`, or why neither can be honest."""
    match stage:
        case Stage.SWEEP:
            return f"~${plan.total_usd:.2f}"
        case Stage.COMPACT:
            # Not free and not its own line: the compressor bills per call, and every one of those
            # calls happens inside the sweep, so its cost arrives folded into the sweep's measured
            # candidate spend (the D-COMPRESS accounting rule). Saying "free" here would promise
            # the operator a compressor that costs nothing.
            return "included in sweep"
        case Stage.REPORT:
            return _report_estimate(paths.policy, embedder)
        case _:
            return "free"


def _projected_total(decisions: list[StageDecision], plan: SweepPlan) -> float:
    """What the running stages are projected to spend. Only the sweep has a priced projection."""
    return sum(
        plan.total_usd
        for decision in decisions
        if decision.will_run and decision.stage is Stage.SWEEP
    )


def _status_text(decision: StageDecision) -> str:
    """One stage's `status` cell, carrying the reason on both the skip and the run path."""
    if decision.status is StageStatus.SKIP:
        return f"[dim]SKIP ({escape(decision.reason)})[/dim]"
    return f"will run [dim]({escape(decision.reason)})[/dim]"


def _confirm(decisions: list[StageDecision], plan: SweepPlan, *, yes: bool) -> bool:
    """The run's single spend confirmation. One question, before the first paid call.

    Asked whenever the SWEEP will run, rather than whenever the candidate projection is nonzero.
    A pool priced at zero still spends on the world-model side, and that is exactly the case
    where the simulator's cost is the whole bill, so keying the question on a candidate-side
    number would skip it precisely when it matters most. Fit, tune, and report cost nothing, so a
    run of only those does not need permission to happen.

    A non-interactive session cannot answer, so a spending run REFUSES rather than proceeding:
    consent must be said (`--yes`), never inferred from the absence of a terminal. Every spend
    surface (`route sweep`, `optimize distill`, both harness environments) shares the one
    implementation in `wmo.cli.consent`; all of them originally shipped
    proceed-silently-or-note, and the proceed branch here cost a scripted caller real money it
    never agreed to.

    Raises:
        typer.Exit: code 2 when a spending run cannot ask and was not told `--yes`.
    """
    if not any(decision.will_run for decision in decisions):
        return False
    if yes or not _will_sweep(decisions):
        return True
    return require_spend_consent(
        _console,
        yes=yes,
        spend=f"~${_projected_total(decisions, plan):.2f} across {plan.cells} sweep cell(s)",
        command="wmo optimize model",
        alternative="--dry-run to see the plan without spending",
    )


# ------------------------------------------------------------------------------ stage execution


def _run_stages(
    decisions: list[StageDecision],
    *,
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
    # The loop saves after every stage, which is what makes a rejected sweep survive: the
    # coverage gate lives in the FIT iteration, so the SWEEP iteration's save has already run by
    # the time the gate can stop the run.
    for decision in decisions:
        if not decision.will_run:
            _console.print(
                f"\n[bold]{decision.stage.value}[/bold] [dim]SKIP: {escape(decision.reason)}[/dim]"
            )
            continue
        _console.print(
            f"\n[bold]{decision.stage.value}[/bold] [dim]({escape(decision.reason)})[/dim]"
        )
        match decision.stage:
            case Stage.SWEEP:
                ledger.check(Stage.SWEEP, projection.total_usd, basis=projection.basis)
                record = _stage_sweep(
                    plan,
                    model_dir=model_dir,
                    pool_file=pool_file,
                    already_measured=already_measured,
                )
                ledger.record(record.total_spend_usd)
            case Stage.COMPACT:
                record = _stage_compact(compression)
            case Stage.FIT:
                _enforce_coverage(paths.matrix, allow_uneven=allow_uneven_coverage)
                record = _stage_fit(
                    paths,
                    embedder=embedder,
                    compression=compression,
                    fallback=fallback,
                    allow_uneven=allow_uneven_coverage,
                )
            case Stage.TUNE:
                record = _stage_tune(paths, cost_quality=cost_quality)
            case _:
                record = _stage_report(paths, model_dir=model_dir, baseline=baseline)
        manifest = manifest.with_record(record)
        manifest.save(paths.manifest)
    return manifest


def _now() -> str:
    """This moment as an ISO-8601 UTC stamp, for the manifest's completion times."""
    return datetime.now(tz=UTC).isoformat()


def _stage_sweep(
    plan: SweepPlan, *, model_dir: Path, pool_file: Path, already_measured: int = 0
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
    world_model, _serve_provider = load_world_model(model_dir)
    run = execute_sweep(
        plan,
        world_model=world_model,
        env_factory=lambda: WorldModelEnv(world_model, score_on_close=True),
        on_outcome=cell_progress(_console, plan.cells - already_measured),
    )
    matrix = run.matrix
    scored = sum(1 for outcome in matrix.outcomes if outcome.scored)
    _console.print(
        f"  [green]✓[/green] {len(matrix.outcomes)} cell(s), {scored} scored -> "
        f"{escape(str(plan.out_path))}\n"
        f"  measured candidate spend ${run.candidate_usd:.4f}{_compressor_note(run)} (the world "
        "model's own serve/judge cost is metered separately)",
        soft_wrap=True,
    )
    print_world_model_spend(_console, run)
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


def _stage_compact(compression: CompressionConfig | None) -> StageRecord:
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
    signature = compression_signature(compression)
    _console.print(
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


def _enforce_coverage(matrix_path: Path, *, allow_uneven: bool) -> None:
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
    matrix = OutcomeMatrix.load(matrix_path)
    rows = coverage(matrix)
    print_coverage(_console, rows)
    if not any(outcome.scored for outcome in matrix.outcomes):
        _console.print(NO_EVIDENCE_WARNING)
        raise typer.Exit(1)
    warning = uneven_warning(rows)
    if warning is None:
        return
    _console.print(warning)
    if not allow_uneven:
        _console.print(
            "  fix the lost cells and re-run, drop the candidate that lost them, or re-run "
            "with [bold]--allow-uneven-coverage[/bold] to fit on this matrix anyway (the "
            "matrix is on disk and recorded, so re-running will not buy these cells again)"
        )
        raise typer.Exit(1)
    _console.print(BIAS_ACCEPTED_NOTE)


def _stage_fit(
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
    matrix, source = load_matrix_with_digest(paths.matrix)
    try:
        fitted = fit_knn_artifact(
            matrix,
            out_path=paths.policy,
            matrix_source=source,
            embedder=embedder,
            fallback=fallback,
            compression=compression,
        )
    except ValueError as exc:
        # Nothing is wrong with the flags: the matrix this run measured cannot be fitted. Exit 1
        # like the coverage gate, not 2 with a usage banner (`route_app.student` precedent).
        _console.print(f"[red]fit failed[/red] {escape(str(exc))}")
        raise typer.Exit(1) from exc
    _rebaseline_dial_snapshot(paths.policy, fitted.policy)
    _console.print(
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


def _rebaseline_dial_snapshot(policy_path: Path, fitted: RoutingPolicy) -> None:
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
    _console.print(
        f"  re-baselined the dial: {escape(base_path.name)} was the as-fitted snapshot of the "
        "fit this stage just replaced, so the next dial applies to the new fit",
        soft_wrap=True,
    )


def _stage_tune(paths: _RunPaths, *, cost_quality: float) -> StageRecord:
    """Set the endpoint's dial, preserving `route tune`'s as-fitted snapshot semantics."""
    fit_identity = _policy_fit_identity(paths.policy)
    try:
        dialed = tune_policy_dial(paths.policy, cost_quality)
    except ValueError as exc:
        _console.print(f"[red]tune failed[/red] {escape(str(exc))}")
        raise typer.Exit(1) from exc
    knobs = dialed.knobs
    _console.print(
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


def _stage_report(paths: _RunPaths, *, model_dir: Path, baseline: str | None) -> StageRecord:
    """Score the tuned policy against its anchor on the same held-out scenarios."""
    matrix = OutcomeMatrix.load(paths.matrix)
    policy = RoutingPolicy.load(paths.policy)
    anchor = baseline or policy.default_model
    try:
        report = build_endpoint_scorecard(matrix, policy, baseline=anchor, endpoint=model_dir.name)
    except (KeyError, ValueError) as exc:
        _console.print(
            f"[red]report failed[/red] cannot report against {escape(anchor)}: {escape(str(exc))}\n"
            "  name a pool model the sweep scored with --baseline, or re-run the sweep so the "
            "anchor has scored episodes"
        )
        raise typer.Exit(1) from exc
    paths.report.parent.mkdir(parents=True, exist_ok=True)
    paths.report.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    _console.print(
        f"  [green]✓[/green] report over {report.headline.scenarios_compared} commonly-scored "
        f"scenario(s) -> {escape(str(paths.report))}",
        soft_wrap=True,
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

    Today this is `wmo.optimize.report.build_report`, the paired held-out comparison the endpoint
    already cites. The richer three-objective scorecard has since landed as
    `wmo.optimize.scorecard` (effective cost per COMPLETED task, the cache-aware accounting rule,
    the ablation ladder); wiring it in is a deliberate follow-up rather than part of this command's
    first release, because its `Arm`/`ConditionLabel` inputs describe a grid this stage does not
    yet build. When that happens only the body of this function changes: the stage, the manifest
    fingerprints, and the ending that renders the result all stay as they are.
    """
    return build_report(matrix, policy, baseline=baseline, endpoint=endpoint, generated_at=_now())


# ----------------------------------------------------------------------------------- the payoff


def _print_payoff(console: Console, name: str, *, paths: _RunPaths, cost_quality: float) -> None:
    """Close on what the endpoint is now, what it bought, and how to serve it.

    Three objectives, each against the same named anchor over the same scenarios, each carrying
    where its number came from. A number that cannot honestly be computed prints its reason.
    """
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
