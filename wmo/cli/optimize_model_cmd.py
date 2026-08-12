"""One-command staged routing optimization CLI command."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import typer
from rich.markup import escape

from wmo.cli import optimize_model_app as _app
from wmo.cli.optimize_model_plan import (
    _COST_QUALITY_BALANCED,
    _DEFAULT_POOL_PATH,
    ASSUMED_INPUT_TOKENS,
    ASSUMED_OUTPUT_TOKENS,
    DEFAULT_EPISODES,
    DEFAULT_MAX_STEPS,
    DEFAULT_SCENARIOS,
    _confirm,
    _distill_reserved_message,
    _is_disabled_in,
    _parse_force_from,
    _plan_stages,
    _print_budget_stop,
    _print_plan,
    _report_anchor,
    _resolve_embedder_choice,
    _RunPaths,
    _will_sweep,
)
from wmo.cli.optimize_model_stages import _print_payoff, _run_stages
from wmo.common.config import ARTIFACT_DIR, WorldModelStore

if TYPE_CHECKING:
    from wmo.optimize.routing.compression import CompressionConfig


def optimize_model(  # noqa: PLR0913 - each flag is one decision a user owns (see the help text)
    world_model: str = typer.Argument(
        None, help="Built world model to optimize (default: the only one under --root)."
    ),
    pool_file: str = typer.Option(
        _DEFAULT_POOL_PATH,
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
        _COST_QUALITY_BALANCED,
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
        "truncate). The sweep then measures that arm and the fit embeds "
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
        "session (CI, cron, piped output, redirected input), where a spending run "
        "otherwise refuses to start.",
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

        wmo optimize model support --compressor truncate --aggressiveness 0.4

    Artifacts land exactly where the manual commands put them, so you can drop to any of them
    mid-flow and this command resumes around it: `policy.json` (plus its evidence bank) in the
    model's own directory where `wmo serve` reads it, and the outcome matrix, report, and run
    manifest under `<model>/optimize/`. Deleting that directory resets resume and breaks nothing.

    Args:
        options: Inputs accepted by this callable.
    Raises:
        ValueError: If the requested operation cannot be completed.
    """
    from wmo.cli.route_sweep_cmd import print_deferred_risks, print_tiny_corpus_note
    from wmo.optimize.routing.compression import resolve_compression
    from wmo.optimize.routing.evaluation import scenario_id
    from wmo.optimize.routing.outcomes import split_router_scenarios
    from wmo.optimize.routing.pipeline import (
        MANIFEST_DIRNAME,
        MANIFEST_FILENAME,
        MATRIX_FILENAME,
        REPORT_FILENAME,
        BudgetExceeded,
        SpendLedger,
        Stage,
        load_manifest,
        planned_stages,
        project_sweep_spend,
    )
    from wmo.optimize.routing.policy import POLICY_FILENAME, probe_embedder
    from wmo.optimize.routing.sweep import SweepError, plan_sweep, resolve_config, resumable_cells
    from wmo.optimize.routing.sweep import preflight_pool as run_preflight

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
        _app._console.print(f"[yellow]note[/yellow] {escape(read.warning)}")
    manifest = read.manifest

    # Preflight runs before the plan table by necessity: it is what proves the candidates are
    # usable and what prices them, and both have to be true before an operator is asked to
    # authorize anything. It spends nothing, so running it unconditionally costs only time.
    try:
        config = resolve_config(model_dir)
        preflight = run_preflight(Path(pool_file))
        print_deferred_risks(_app._console, preflight.deferred)
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
    try:
        split_router_scenarios([scenario_id(scenario) for scenario in plan.scenarios])
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    print_tiny_corpus_note(_app._console, plan)
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
            # Say which of the two different repairs applies: an entry that IS in the file but
            # turned off needs its flag flipped, not a re-registration.
            if _is_disabled_in(Path(pool_file), value):
                raise typer.BadParameter(
                    f"{flag} '{value}' is disabled (enabled = false) in {pool_file}; {why}. "
                    "Flip it back on to use it here"
                )
            raise typer.BadParameter(
                f"{flag} '{value}' is not a model in {pool_file}; {why}. "
                f"Available: {', '.join(known)}"
            )

    # Printed before the plan table, the way `route fit` prints it before the fit: the embedder
    # decides what the policy can route on, and an operator who meant to fit on semantic vectors
    # should see that it resolved to hashed features before authorizing any spend.
    _app._console.print(resolution)
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
        _app._console,
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
        _app._console.print("\ndry run: nothing was run and nothing was spent")
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
        _app._console.print("nothing was run and nothing was spent")
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
        # The cap is a clean stop, not a failure: every stage that completed is recorded and the
        # next run resumes at the one that did not start.
        _print_budget_stop(model_dir.name, exc)
        raise typer.Exit(1) from exc
    except typer.Exit:
        # A stage refusal follows the command's ordinary exit path.
        raise
    except Exception:
        raise
    # No save here: `_run_stages` persists after every stage it runs, which is what keeps a run
    # that dies mid-flight resumable.
    _print_payoff(_app._console, model_dir.name, paths=paths, cost_quality=cost_quality)
