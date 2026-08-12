"""Planning, fingerprints, and presentation for staged model optimization."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import typer
from pydantic import BaseModel, ConfigDict, ValidationError
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from wmo.cli import optimize_model_app as _app
from wmo.cli.consent import require_spend_consent
from wmo.common.config import WorldModelStore

if TYPE_CHECKING:
    from wmo.optimize.routing.compression import CompressionConfig
    from wmo.optimize.routing.pipeline import (
        BudgetExceeded,
        RunManifest,
        Stage,
        StageDecision,
        SweepSpendProjection,
    )
    from wmo.optimize.routing.policy import EmbedderSpec
    from wmo.optimize.routing.sweep import SweepPlan
    from wmo.optimize.routing.teacher import TeacherSearchVerdict

DEFAULT_SCENARIOS = 20
DEFAULT_EPISODES = 1
DEFAULT_MAX_STEPS = 20
ASSUMED_INPUT_TOKENS = 2000
ASSUMED_OUTPUT_TOKENS = 250
_DEFAULT_POOL_PATH = ".wmo/pool.toml"
_COST_QUALITY_BALANCED = 0.25
_DEFAULT_KNN_Z = 0.5
_DEFAULT_RAG_NUM = 50
_DEFAULT_RAG_THRES = 0.95
_DEFAULT_KNN_MIN_PAIRS = 8
_ROUTER_SPLIT_VERSION = "scenario-hash-70-30-v1"
_KNN_KNOBS = (
    f"z={_DEFAULT_KNN_Z:g} k={_DEFAULT_RAG_NUM} thres={_DEFAULT_RAG_THRES:g} "
    f"pairs={_DEFAULT_KNN_MIN_PAIRS} se_floor=True q=0.05 split={_ROUTER_SPLIT_VERSION}"
)


class _RunPaths(BaseModel):
    """Where this run's artifacts live. Serving paths stay where serving already looks."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    manifest: Path
    matrix: Path
    policy: Path
    report: Path


def _is_disabled_in(pool_file: Path, name: str) -> bool:
    """Whether `name` is a roster entry that exists but is turned off (enabled = false).

    Best-effort for an error message: the pool already loaded once through preflight, so a
    second read here cannot fail in a new way, and any surprise reads as plain "not a model".
    """
    from wmo.common.providers.pool import load_pool

    try:
        return any(
            entry.name == name and not entry.enabled for entry in load_pool(pool_file).models
        )
    except (FileNotFoundError, ValueError):
        return False


def _distill_reserved_message(*, world_model: str | None, root: str) -> str:
    """Why `--distill` is refused, plus this model's teacher-search verdict when one is readable.

    The stage is not wired, but its PREFLIGHT is (`wmo.optimize.routing.teacher`), and the
    preflight is the half that decides whether the stage should run at all. Printing the verdict
    here means the product surface already answers the real question ("should I distill?") in the
    same place an operator asked for it, and usually answers no, which is the cheapest possible
    outcome.

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
    from wmo.optimize.routing.outcomes import OutcomeMatrix
    from wmo.optimize.routing.pipeline import MANIFEST_DIRNAME, MATRIX_FILENAME
    from wmo.optimize.routing.teacher import select_teacher

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

    The resolution itself is `wmo.optimize.routing.policy.resolve_embedder`, shared with
    `wmo optimize route fit` so the two commands cannot disagree about what `auto` means. What is
    decided here is the narrower surface: this command takes no `--deployment`, `--endpoint`, or
    `--dim`, because its whole promise is one command with no routing knobs in it. So `azure` reads
    the same standard environment pair `auto` looks for rather than flags that do not exist, and
    the operator who needs to name a specific deployment is pointed at the manual fit that has
    those flags.

    Raises:
        ValueError: An unknown choice, or `azure` with nothing in the environment to point it at.
    """
    from wmo.optimize.routing.policy import (
        AZURE_EMBEDDER_DEPLOYMENT,
        AZURE_EMBEDDER_ENV,
        resolve_embedder,
    )

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
    from wmo.optimize.routing.pipeline import (
        BUILT_STAGES,
        CONFIGURED_STAGES,
        RESERVED_STAGES,
        Stage,
        forced_stages,
    )

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
    from wmo.optimize.routing.pipeline import (
        CONFIGURED_STAGES,
        Stage,
        StageDecision,
        StageStatus,
        decide_stage,
    )

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
    from wmo.optimize.routing.compression import compression_signature

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
    from wmo.optimize.routing.compression import compression_signature
    from wmo.optimize.routing.pipeline import file_sha256
    from wmo.optimize.routing.policy import embedder_provenance

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
    from wmo.optimize.routing.compression import compression_signature
    from wmo.optimize.routing.pipeline import Stage, file_sha256

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
    from wmo.optimize.routing.policy import RoutingPolicy

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
    from wmo.optimize.routing.evaluation import scenario_id

    return ",".join(scenario_id(scenario) for scenario in plan.scenarios)


def _policy_fit_identity(policy_path: Path) -> str:
    """Which FIT the policy on disk came from, surviving the dial that `tune` applies to it.

    A byte hash would call every tuned policy a stranger to the fit that produced it, so `fit`
    would rerun on every resume. `fit_provenance` strips the dial suffix, so this changes when
    someone refits (by hand or otherwise) and not when the dial moves.
    """
    from wmo.optimize.routing.knn import fit_provenance
    from wmo.optimize.routing.policy import RoutingPolicy

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
    from wmo.optimize.routing.compression import compression_signature
    from wmo.optimize.routing.evaluation import scenario_id
    from wmo.optimize.routing.knn import cost_quality_named_point
    from wmo.optimize.routing.outcomes import split_router_scenarios
    from wmo.optimize.routing.pipeline import Stage

    router_split = split_router_scenarios([scenario_id(scenario) for scenario in plan.scenarios])
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
            return (
                f"knn over {len(router_split.fit_ids)} fit scenario(s) "
                f"(guarded, fallback {escape(fallback or 'best single on the fit split')})"
            )
        case Stage.TUNE:
            return f"cost_quality {cost_quality:g} ({cost_quality_named_point(cost_quality)})"
        case _:
            return (
                f"3-objective headline vs {escape(anchor)} over "
                f"{len(router_split.report_ids)} router-held-out scenario(s)"
            )


def _report_estimate(policy_path: Path, fitting_with: EmbedderSpec) -> str:
    """What the report stage costs, or why that cannot honestly be a number.

    Priced off the embedder the report will ACTUALLY use, which `build_report` takes from the
    policy, not off the one this command fits with. They differ exactly when the fit is skipped
    and the policy on disk came from a manual `route fit --embedder azure`: quoting this run's
    hashing spec would then print "free" for a stage about to embed every scenario for money.
    """
    from wmo.optimize.routing.policy import RoutingPolicy

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
    from wmo.optimize.routing.pipeline import Stage

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
    _app._console.print(
        f"\n[yellow]stopped at the spend cap[/yellow] {escape(str(exc))}\n"
        "  every finished stage is on disk and recorded, so nothing is lost. Resume with a "
        f"higher cap: [bold]wmo optimize model {escape(name)} --max-usd <more>[/bold]"
    )


def _will_sweep(decisions: list[StageDecision]) -> bool:
    """Whether this run will buy cells, which is the only thing that costs candidate money."""
    from wmo.optimize.routing.pipeline import Stage

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
    from wmo.optimize.routing.pipeline import Stage

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
    from wmo.optimize.routing.pipeline import Stage

    return sum(
        plan.total_usd
        for decision in decisions
        if decision.will_run and decision.stage is Stage.SWEEP
    )


def _status_text(decision: StageDecision) -> str:
    """One stage's `status` cell, carrying the reason on both the skip and the run path."""
    from wmo.optimize.routing.pipeline import StageStatus

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
    consent must be said (`--yes`), never inferred from the absence of an interactive session on
    BOTH streams (a redirected stdin is not a person, even under a terminal stdout). Every spend
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
        _app._console,
        yes=yes,
        spend=f"~${_projected_total(decisions, plan):.2f} across {plan.cells} sweep cell(s)",
        command="wmo optimize model",
        alternative="--dry-run to see the plan without spending",
    )
