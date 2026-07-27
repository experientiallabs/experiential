"""`wmo optimize route`: sweep, fit, tune, and report learned inference policies.

The routing optimizer's CLI face, sitting beside `wmo optimize harness` in the optimizer
family. The workflow chains in one direction:

    student -> pool -> sweep -> OutcomeMatrix -> fit -> policy.json -> tune / report

`student` puts a freshly distilled adapter into the candidate pool, which is what makes a trained
model routable at all. `sweep` is the producer: it measures every candidate on the world model's
own held-out scenarios and writes the `OutcomeMatrix` everything downstream consumes (a research
adapter such as RouterBench can write the same artifact instead). `fit` emits the policy artifact
serving loads, `report` the improvement report the endpoint cites, and `tune` is the one post-fit
control: it moves a fitted policy's cost/quality dial without refitting.

`pin` sits outside that chain: it installs a `kind="static"` policy for one pool model, so a
single candidate is serveable before any measurement exists, which is the honest zero-evidence
starting point a fit is compared against. Vocabulary note: "route" is developer-facing CLI only;
customer copy never says router.
"""

from __future__ import annotations

import itertools
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import typer
from llm_waterfall import ChatMaxTokensField
from pydantic import ValidationError
from rich.console import Console
from rich.markup import escape
from rich.prompt import Confirm
from rich.table import Table

from wmo.config import ARTIFACT_DIR, WorldModelStore
from wmo.distill.store import MODEL_CARD_FILE, DistillModelCard, student_pool_entry
from wmo.engine import load_world_model
from wmo.env import WorldModelEnv
from wmo.optimize.knn import (
    COST_QUALITY_ANCHORS,
    DialResult,
    KnnFitOutcome,
    fit_knn_artifact,
    tune_policy_dial,
)
from wmo.optimize.outcomes import OutcomeMatrix, ScenarioOutcome, load_matrix_with_digest
from wmo.optimize.policy import (
    POLICY_FILENAME,
    EmbedderSpec,
    RoutingPolicy,
    embedder_provenance,
)
from wmo.optimize.report import build_report
from wmo.optimize.routing import evaluate_policy, fit_rank_policy, rerank_policy
from wmo.optimize.sweep import (
    CandidateCoverage,
    DeferredRisk,
    SweepError,
    SweepPlan,
    SweepRun,
    Unevenness,
    coverage,
    execute_sweep,
    plan_sweep,
    preflight_pool,
    resolve_config,
    unevenness,
)
from wmo.providers.pool import (
    DEFAULT_POOL_PATH,
    PoolEntry,
    PoolLockTimeout,
    load_pool,
    upsert_pool_entry,
)

# The two output-budget parameter names any OpenAI-compatible backend accepts.
_MAX_TOKENS_FIELDS: tuple[ChatMaxTokensField, ...] = ("max_tokens", "max_completion_tokens")

route_app = typer.Typer(
    help="Make models routable, measure them closed-loop, then fit, tune, and report policies.",
    no_args_is_help=True,
)

_console = Console()

DEFAULT_MATRIX_FILENAME = "matrix.json"
"""Default `sweep --out`: the outcome matrix `fit` takes as its argument."""


@route_app.command("sweep")
def sweep(
    model: str = typer.Argument(
        None, help="World model to measure against (default: the only one built under --root)."
    ),
    pool_file: str = typer.Option(
        str(DEFAULT_POOL_PATH),
        "--pool",
        # The doubled brackets are escaped: typer renders help through rich markup, which
        # otherwise swallows them and prints an empty pair.
        help="Candidate pool TOML: one \\[\\[model]] table per candidate.",
    ),
    traces_file: str = typer.Option(
        None,
        "--traces",
        help="Trace corpus the scenarios come from (default: the model's own "
        "traces.otel.jsonl, as `wmo demo --traces` resolves it). A build does not keep a copy "
        "of the corpus it read, so pass the file here.",
    ),
    scenarios: int = typer.Option(
        20,
        "--scenarios",
        min=1,
        help="Cap on held-out scenarios measured (a deterministic prefix by trace id).",
    ),
    episodes: int = typer.Option(
        1, "--episodes", min=1, help="Episodes per (candidate, scenario) cell."
    ),
    max_steps: int = typer.Option(
        20, "--max-steps", min=1, help="Step budget per episode (also the cost estimate's cap)."
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
    yes: bool = typer.Option(False, "--yes", help="Skip the cost confirmation."),
    allow_uneven_coverage: bool = typer.Option(
        False,
        "--allow-uneven-coverage",
        help="Hand the matrix to `fit` even when the candidates were not scored on the same "
        "evidence: different scenarios, or different numbers of surviving episodes on the same "
        "scenarios. The fit is then biased (both fitters skip unscored rows and weigh the rest per "
        "episode); the coverage table prints either way.",
    ),
) -> None:
    """Measure every pool candidate closed-loop and write the outcome matrix `fit` consumes.

    This is step one of the routing workflow: nothing else produces an `OutcomeMatrix`.

        wmo optimize route sweep support --traces traces.otel.jsonl --pool .wmo/pool.toml
        wmo optimize route fit matrix.json --kind knn

    Every (candidate, scenario, episode) cell runs one full episode against the world model,
    which scores it (`WorldModelEnv(..., score_on_close=True)`): a matrix without verified
    rewards is not evidence. Scenarios are the task prompts of the corpus's TEST band, the third
    band of the build's deterministic 3-way split, which prompt optimization and knowledge
    extraction never saw; the candidates' tool surface is summarized from the TRAIN band only
    (the same discipline), so a candidate is not scored on guessing what tools exist. What that
    buys is a policy fitted on prompts no GEPA candidate was SELECTED on. It is not isolation
    from the environment: a build indexes the full corpus for serving, so the world model can
    still retrieve a held-out trace's own recorded steps as demos when it simulates that
    scenario. Bands are also cut per trace id, not per task text, so a task repeated across
    traces can appear on both sides. The whole sweep runs the world model frozen, so no cell's
    predictions become another cell's retrieved demos and the result does not depend on sweep
    order.

    Spend is confirmed before the first episode runs (`--yes` skips, as in
    `wmo optimize harness --mode distill`). What that estimate multiplies is ASSUMED tokens per
    policy call by the real cell and call counts, so it is a projection, never a measurement;
    the measured candidate spend is printed when the sweep finishes. Before that question is asked,
    every candidate's backend is resolved as far as it goes without a request: its kind's static
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
    """
    out_path = Path(out)
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
        config = resolve_config(model_dir)
        preflight = preflight_pool(Path(pool_file))
    except SweepError as exc:
        raise typer.BadParameter(str(exc)) from exc
    print_deferred_risks(_console, preflight.deferred)
    try:
        plan = plan_sweep(
            model_dir=model_dir,
            config=config,
            pool=preflight.pool,
            out_path=out_path,
            traces_file=Path(traces_file) if traces_file is not None else None,
            scenarios=scenarios,
            episodes=episodes,
            max_steps=max_steps,
            assume_input_tokens=assume_input_tokens,
            assume_output_tokens=assume_output_tokens,
        )
    except SweepError as exc:
        raise typer.BadParameter(str(exc)) from exc
    print_tiny_corpus_note(_console, plan)
    world_model, _serve_provider = load_world_model(model_dir)

    print_cost_estimate(_console, plan)
    _confirm_cost(yes=yes)

    _console.print(
        f"sweeping {len(plan.pool.models)} candidate(s) over {len(plan.scenarios)} held-out "
        f"scenario(s) of [bold]{escape(model_dir.name)}[/bold], {episodes} episode(s) each…"
    )
    run = execute_sweep(
        plan,
        world_model=world_model,
        env_factory=lambda: WorldModelEnv(world_model, score_on_close=True),
        on_outcome=cell_progress(_console, plan.cells),
    )
    matrix = run.matrix
    scored = sum(1 for outcome in matrix.outcomes if outcome.scored)
    # `escape(out)`: a bracketed path segment would otherwise be read as markup and dropped, so
    # the line would print a path that does not exist (and this one is meant to be copied).
    _console.print(
        f"[green]✓[/green] {len(matrix.outcomes)} cell(s), {scored} scored -> {escape(out)}\n"
        f"  measured candidate spend ${run.candidate_usd:.4f} (the world model's own serve/judge "
        "cost is metered separately)",
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
    """Name what the request-free pre-flight could not close, per candidate that carries it."""
    if not deferred:
        return
    console.print(
        "[yellow]note[/yellow] the pre-flight makes no request, so one thing per candidate "
        "below can still fail at its first cell (the matrix records it as that cell's `error`):"
    )
    for risk in deferred:
        console.print(f"  {escape(risk.candidate)} (kind={risk.kind.value}): {risk.risk}")


def print_tiny_corpus_note(console: Console, plan: SweepPlan) -> None:
    """Say when the corpus was too small to leave a held-out band to measure on."""
    if not plan.tiny_corpus:
        return
    console.print(
        f"[yellow]note[/yellow] {plan.trace_count} trace(s) is too few for a held-out band, so "
        "these scenarios come from the FULL corpus: they are not leak-free, and a policy "
        "fitted on them is a smoke test, not evidence"
    )


def print_world_model_spend(console: Console, run: SweepRun) -> None:
    """The OTHER half of a sweep's bill: what the simulator charged to run the evaluation.

    Printed as its own line, never folded into the candidate figure above it. The candidate side
    is the serving cost a customer would pay and the policy is fitted to trade off; this side is
    eval infrastructure that exists only because the measurement happened. One number covering
    both would misprice both.
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
    """A per-cell progress line: which cell, what it scored, what it cost."""
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
    """
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


def print_cost_estimate(console: Console, plan: SweepPlan) -> None:
    """Render the projected spend, stating exactly which parts are assumed.

    Honest by construction: the CELL and CALL counts are real (the step budget is the per-episode
    cap, so calls are an upper bound), the tokens per call are an assumption the flags name, and
    the per-candidate $/Mtok is the pool entry's own price row. The world model's serve and judge
    calls are a separate meter (the D12 cost split) and are deliberately absent.
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
        f"{len(plan.scenarios)} held-out scenario(s) x {plan.episodes} episode(s); estimated "
        f"total ${plan.total_usd:.2f}"
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


def _confirm_cost(*, yes: bool) -> None:
    """Confirm the projected spend before any episode runs.

    Every pool entry is priced (`load_pool` refuses an unpriced candidate), so the spend is
    accountable and `--yes` always applies, the rule `wmo optimize harness --mode distill`
    uses. A non-interactive session cannot answer a prompt, so it proceeds and says so instead
    of hanging.

    Raises:
        typer.Exit: The user declined (exit code 0).
    """
    if yes:
        return
    if not _console.is_terminal:
        _console.print(
            "non-interactive session: proceeding without confirmation (pass --yes to say so "
            "explicitly)"
        )
        return
    if not Confirm.ask("Proceed?", default=True):
        raise typer.Exit(0)


@route_app.command("student")
def student(
    card_dir: str = typer.Argument(
        ...,
        help="The distillation run dir, or an adapter version dir: whichever holds "
        "model_card.json.",
    ),
    input_per_mtok: float = typer.Option(
        ...,
        "--input-per-mtok",
        min=0.0,
        help="Prompt-token price at the serving endpoint, USD per 1M tokens. Required: an "
        "unpriced candidate reports $0 and a cost-aware policy would route everything to it.",
    ),
    output_per_mtok: float = typer.Option(
        ...,
        "--output-per-mtok",
        min=0.0,
        help="Completion-token price at the serving endpoint, USD per 1M tokens.",
    ),
    name: str = typer.Option(
        "student", "--name", help="Pool handle: what policy artifacts and request logs call it."
    ),
    pool: str = typer.Option(
        str(DEFAULT_POOL_PATH), "--pool", help="Candidate pool TOML to add the entry to."
    ),
    endpoint: str = typer.Option(
        None,
        "--endpoint",
        help="OpenAI-compatible base URL. Default: Tinker's serving endpoint.",
    ),
    api_key_env: str = typer.Option(
        None,
        "--api-key-env",
        help="Env var holding the endpoint's API key. Default: TINKER_API_KEY on Tinker's own "
        "endpoint; on any other --endpoint the provider's WMO_ENDPOINT_API_KEY fallback is used, "
        "so a Tinker key is never sent to a host you named.",
    ),
    chat_max_tokens_field: str = typer.Option(
        None,
        "--chat-max-tokens-field",
        help="Output-budget parameter the endpoint accepts: max_tokens | max_completion_tokens. "
        "Default: max_tokens on Tinker's endpoint, max_completion_tokens on any other.",
    ),
    yes: bool = typer.Option(
        False, "--yes", help="Skip the confirmation when an entry of this name already exists."
    ),
) -> None:
    """Add a distilled student to the candidate pool, so the router can select it.

    The keystone step between training and serving: a run produces a `tinker://` adapter, and this
    turns it into a `[[model]]` entry the sweep measures, the fitter routes to, and the endpoint
    calls, with no hand-edited TOML in between:

        wmo optimize route student .wmo/distill/support --input-per-mtok 0.1 --output-per-mtok 0.4

    On Tinker's own endpoint the entry reads its credential from `TINKER_API_KEY`, so export that
    before serving. Point `--endpoint` somewhere else and the Tinker defaults do NOT follow: the
    entry falls back to `WMO_ENDPOINT_API_KEY` and to `max_completion_tokens`, so a Tinker key is
    never sent to a host you named. `--api-key-env` and `--chat-max-tokens-field` set either
    explicitly.

    To serve the student on its own with no measurement at all, follow this with
    `wmo optimize route pin <world-model> --model student`; to have the router CHOOSE between the
    student and the rest of the roster, run `wmo optimize route fit` on a matrix that covers both.
    """
    card_path = Path(card_dir) / MODEL_CARD_FILE
    if not card_path.is_file():
        raise typer.BadParameter(
            f"no {MODEL_CARD_FILE} at {card_path}; pass a distillation run directory (the one "
            "holding config.toml and metrics.jsonl) or an adapter version directory "
            "(.wmo/adapters/<name>/vN)"
        )
    if endpoint is not None and not endpoint.strip():
        # `--endpoint "$UNSET_VAR"` is the way this happens. Falling back to Tinker's endpoint
        # would silently serve a different host than the script meant to name.
        raise typer.BadParameter(
            "--endpoint is empty; give the OpenAI-compatible base URL, or drop the flag to use "
            "Tinker's serving endpoint"
        )
    if api_key_env is not None and not api_key_env.strip():
        # Same accident as an empty --endpoint. An empty string reaches `pool_provider` as a
        # falsy api_key_env, which it reads as "no explicit credential" and skips its own
        # unset-variable check, so the misconfiguration would only surface as a 401 at request
        # time with no hint.
        raise typer.BadParameter(
            "--api-key-env is empty; name the environment variable holding the endpoint's key, "
            "or drop the flag to use the provider's default credentials"
        )
    if chat_max_tokens_field is not None and chat_max_tokens_field not in _MAX_TOKENS_FIELDS:
        raise typer.BadParameter(
            f"unknown --chat-max-tokens-field {chat_max_tokens_field!r}; use "
            f"{' or '.join(_MAX_TOKENS_FIELDS)}"
        )
    try:
        card = DistillModelCard.model_validate_json(card_path.read_text(encoding="utf-8"))
    except ValidationError as exc:
        raise typer.BadParameter(f"cannot read the model card at {card_path}: {exc}") from exc
    try:
        entry = student_pool_entry(
            card,
            name=name,
            input_per_mtok=input_per_mtok,
            output_per_mtok=output_per_mtok,
            endpoint=endpoint,
            api_key_env=api_key_env,
            chat_max_tokens_field=cast("ChatMaxTokensField | None", chat_max_tokens_field),
        )
    except ValidationError as exc:
        raise typer.BadParameter(f"cannot build a pool entry for '{name}': {exc}") from exc
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    pool_path = Path(pool)
    if _pool_has(pool_path, name) and not yes and not _confirm_replace(pool_path, name):
        _console.print(
            f"left {pool_path} unchanged; pass --yes to replace '{name}' (comments in the file "
            "are not preserved by a replacement), or --name <other> to keep both"
        )
        raise typer.Exit(0)
    try:
        replaced = upsert_pool_entry(entry, pool_path)
    except PoolLockTimeout as exc:
        # Nothing is wrong with the flags, so this is not a BadParameter: another writer is in the
        # way. Exit non-zero (and say to retry) so a script does not read it as a registration.
        _console.print(f"[red]pool busy[/red] {escape(str(exc))}")
        raise typer.Exit(1) from exc
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    verb = "replaced" if replaced else "added"
    _console.print(
        f"[green]✓[/green] {verb} pool candidate [bold]{name}[/bold] -> {pool_path}\n"
        f"  {card.base_model} adapter at {entry.model}\n"
        f"  ${input_per_mtok:g}/${output_per_mtok:g} per 1M in/out tokens, "
        f"{_credential_note(entry)}\n"
        f"  serve it directly: wmo optimize route pin <world-model> --model {name}",
        soft_wrap=True,
    )


def _credential_note(entry: PoolEntry) -> str:
    """How this entry authenticates, so the summary never names a key it will not send."""
    if entry.api_key_env is not None:
        return f"credential from {entry.api_key_env}"
    return "credential from WMO_ENDPOINT_API_KEY (the custom-endpoint fallback)"


def _pool_has(path: Path, name: str) -> bool:
    """Whether `path` already carries an entry called `name` (False when there is no pool yet)."""
    if not path.is_file():
        return False
    try:
        return any(entry.name == name for entry in load_pool(path).models)
    except (ValueError, FileNotFoundError):
        # An unreadable pool is upsert_pool_entry's error to raise, with its own message; do not
        # pre-empt it here with a confirmation prompt about an entry we cannot see.
        return False


def _confirm_replace(path: Path, name: str) -> bool:
    """Confirm repointing an existing pool handle; a non-interactive run declines."""
    try:
        return Confirm.ask(f"Replace the existing '{name}' entry in {path}?", default=False)
    except EOFError:
        return False


@route_app.command("fit")
def fit(
    matrix_file: str = typer.Argument(..., help="OutcomeMatrix JSON (closed-loop eval output)."),
    kind: str = typer.Option(
        "rank",
        "--kind",
        help="knn (guarded nearest-neighbor evidence, the validated champion) | rank "
        "(Avengers cluster ranks).",
    ),
    out: str = typer.Option(
        POLICY_FILENAME, "--out", help="Where to write the fitted policy JSON."
    ),
    fallback: str = typer.Option(
        None,
        "--fallback",
        help="(knn) Baseline model every request uses unless the evidence says otherwise. "
        "Default: the best single model on the fit set.",
    ),
    z: float = typer.Option(
        0.5,
        "--z",
        min=0.0,
        help="(knn) Confidence knob: standard errors of paired evidence a pick must clear to "
        "leave the fallback (doubled when it is also pricier). Higher = stricter = more "
        "requests stay on the fallback; 0 routes on any positive difference.",
    ),
    rag_num: int = typer.Option(50, "--rag-num", min=1, help="(knn) Neighbor budget."),
    rag_thres: float = typer.Option(
        0.95,
        "--rag-thres",
        min=0.0,
        max=1.0,
        help="(knn) Keep neighbors above this fraction of the rag-num-th best similarity.",
    ),
    min_pairs: int = typer.Option(
        8, "--min-pairs", min=0, help="(knn) Neighbors scored on both sides before routing away."
    ),
    floor_q: float = typer.Option(
        0.05,
        "--floor-q",
        min=0.0,
        max=1.0,
        help="Novelty floor quantile: abstain to the fallback when a query's best bank "
        "similarity is below this quantile of the bank's own nearest-neighbor sims "
        "(coverage/robustness knob for task drift; 0 = off, the exact validated champion).",
    ),
    se_floor: bool = typer.Option(
        True,
        "--se-floor/--no-se-floor",
        help="(knn) Floor the guard's standard error on thin neighborhoods (small-bank safety).",
    ),
    clusters: int = typer.Option(64, "--clusters", min=1, help="k-means cluster count."),
    seed: int = typer.Option(42, "--seed", help="Clustering seed."),
    top_k_clusters: int = typer.Option(2, "--top-k-clusters", min=1),
    beta: float = typer.Option(6.0, "--beta", help="Cluster softmax sharpness."),
    cost_weight: float = typer.Option(
        0.0,
        "--cost-weight",
        min=0.0,
        help="Quality/cost knob: reward points paid per average-call-cost unit (0 = pure "
        "accuracy ranking, the Avengers reference behavior).",
    ),
    embedder: str = typer.Option("hashing", "--embedder", help="hashing | azure"),
    dim: int = typer.Option(512, "--dim", help="Embedding dimension."),
    deployment: str = typer.Option(None, "--deployment", help="(azure) embedding deployment."),
    endpoint: str = typer.Option(None, "--endpoint", help="(azure) resource endpoint."),
    api_key_env: str = typer.Option(
        None, "--api-key-env", help="(azure) env var holding the account key."
    ),
) -> None:
    """Fit a routing policy on an outcome matrix (kNN evidence or Avengers cluster ranks)."""
    if kind not in ("rank", "knn"):
        raise typer.BadParameter(f"unknown kind '{kind}'; use knn or rank")
    matrix, source = load_matrix_with_digest(Path(matrix_file))
    if embedder not in ("hashing", "azure"):
        raise typer.BadParameter(f"unknown embedder '{embedder}'; use hashing or azure")
    spec = (
        EmbedderSpec(dim=dim)
        if embedder == "hashing"
        else EmbedderSpec(
            kind="azure",
            dim=dim,
            deployment=deployment,
            endpoint=endpoint,
            api_key_env=api_key_env,
        )
    )
    out_path = Path(out)
    if rag_thres <= 0.0:
        # typer's min is inclusive but the artifact field requires > 0; fail before the fit
        # writes a sidecar it will then abandon.
        raise typer.BadParameter("--rag-thres must be greater than 0")
    if kind == "knn":
        if cost_weight > 0.0:
            raise typer.BadParameter(
                "--cost-weight re-ranks cluster evidence and applies to --kind rank only; a knn "
                "policy trades cost through its dial instead: fit it, then "
                "`wmo optimize route tune <policy.json> --cost-quality <0..1>`"
            )
        fitted = fit_knn_artifact(
            matrix,
            out_path=out_path,
            matrix_source=source,
            embedder=spec,
            fallback=fallback,
            z=z,
            rag_num=rag_num,
            rag_thres=rag_thres,
            min_pairs=min_pairs,
            se_floor=se_floor,
            floor_q=floor_q,
        )
        print_knn_fit(_console, fitted, out=out, z=z)
        return
    built = spec.build()  # ONE embedder for fit and evaluation; azure would otherwise embed twice
    policy = fit_rank_policy(
        matrix,
        embedder=spec,
        n_clusters=clusters,
        seed=seed,
        top_k_clusters=top_k_clusters,
        beta=beta,
        fitted_from=(
            f"{source} rank seed={seed} k={clusters} topk={top_k_clusters} beta={beta:g} "
            f"cost_weight={cost_weight:g} {embedder_provenance(spec)}"
        ),
    )
    if cost_weight > 0.0:
        policy = rerank_policy(policy, cost_weight=cost_weight)
    policy.save(out_path)
    result = evaluate_policy(policy, matrix, matrix.scenario_ids(), embedder=built)
    _console.print(
        f"[green]✓[/green] fitted {len(policy.clusters)} clusters over "
        f"{result.scenarios} scenarios -> {out}\n"
        f"  fit-set accuracy {result.accuracy:.4f}, cost/scenario ${result.cost_per_scenario:.5f}"
    )


def print_knn_fit(console: Console, fitted: KnnFitOutcome, *, out: str, z: float) -> None:
    """Report a written knn policy: where its evidence is, and what it scored in-sample."""
    console.print(
        f"[green]✓[/green] fitted knn policy over {fitted.scenarios} scenarios -> {out}\n"
        f"  bank {fitted.bank_path}, fallback {fitted.policy.default_model}, z={z}\n"
        f"  routed away from the fallback {fitted.routed_share:.1%} of the time; cost/scenario "
        f"${fitted.cost_per_scenario:.5f}\n"
        f"  fit-set accuracy {fitted.fit_accuracy:.4f} is IN-SAMPLE (every request retrieves its "
        "own row); measure on held-out scenarios with `wmo optimize route report`"
    )


@route_app.command("pin")
def pin(
    world_model: str = typer.Argument(
        None, help="Built world model whose endpoint serves this policy. Default: the only one."
    ),
    model: str = typer.Option(
        ...,
        "--model",
        help="Pool entry every request goes to (a `wmo optimize route student` name).",
    ),
    pool: str = typer.Option(
        str(DEFAULT_POOL_PATH), "--pool", help="Candidate pool TOML to snapshot into the policy."
    ),
    root: str = typer.Option(ARTIFACT_DIR, "--root", help="Project dir holding the built models."),
    out: str = typer.Option(
        None, "--out", help="Override where the policy JSON lands (default: the model's own dir)."
    ),
    yes: bool = typer.Option(
        False, "--yes", help="Skip the confirmation when a policy is already installed."
    ),
) -> None:
    """Serve one pool model as an endpoint, with no matrix and no fit.

    A `kind="static"` policy sends every request to `--model`, which is all a single distilled
    student needs to be reachable through the OpenAI-compatible endpoint:

        wmo optimize route student .wmo/distill/support --input-per-mtok 0.1 --output-per-mtok 0.4
        wmo optimize route pin support --model student
        wmo serve --name support

    The policy is written to the world model's artifact dir, because that is where `wmo serve`
    looks for one. This is the honest "before" state the routing story is told against: a static
    endpoint has learned nothing and saves nothing, and `GET /v1/endpoints/<name>/savings` will
    say so. Replace it with `wmo optimize route fit` on a real outcome matrix to let the router
    choose per request.
    """
    try:
        model_dir = WorldModelStore(root).resolve(world_model)
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    pool_path = Path(pool)
    try:
        roster = load_pool(pool_path)
    except FileNotFoundError as exc:
        raise typer.BadParameter(str(exc)) from exc
    except ValueError as exc:
        raise typer.BadParameter(f"cannot read the pool at {pool_path}: {exc}") from exc
    if all(entry.name != model for entry in roster.models):
        available = ", ".join(entry.name for entry in roster.models)
        raise typer.BadParameter(
            f"no pool model named '{model}' in {pool_path}; available: {available}"
        )
    out_path = Path(out) if out else model_dir / POLICY_FILENAME
    if out_path.is_file() and not yes and not _confirm_overwrite(out_path):
        _console.print(f"left {out_path} in place")
        raise typer.Exit(0)
    policy = RoutingPolicy(
        kind="static",
        default_model=model,
        pool=roster.models,
        fitted_from=f"pinned to {model} from {pool_path} (no outcome matrix)",
    )
    policy.save(out_path)
    _console.print(
        f"[green]✓[/green] pinned endpoint [bold]{model_dir.name}[/bold] to "
        f"[bold]{model}[/bold] -> {out_path}\n"
        f"  every request goes to {model}; nothing is measured and nothing is saved yet\n"
        f"  serve it: wmo serve --name {model_dir.name}\n"
        "  to let the router choose per request instead, replace this with "
        "`wmo optimize route fit <matrix.json>`",
        soft_wrap=True,
    )


def _confirm_overwrite(path: Path) -> bool:
    """Confirm replacing an installed policy; a non-interactive run declines.

    Worth asking about: the file being replaced may be a fitted knn policy, whose evidence bank
    sidecar this static policy will not use and does not remove.
    """
    try:
        return Confirm.ask(f"Replace the policy already at {path}?", default=False)
    except EOFError:
        return False


@route_app.command("tune")
def tune(
    policy_file: str = typer.Argument(POLICY_FILENAME, help="Fitted knn policy JSON to re-tune."),
    cost_quality: float = typer.Option(
        ...,
        "--cost-quality",
        min=0.0,
        max=1.0,
        help="The endpoint's one dial: 0.0 = max quality, 1.0 = max savings. 0.25 is the "
        "shipped default. See the anchor table this command prints for what each end measured.",
    ),
) -> None:
    """Set a fitted policy's cost/quality dial in place, without refitting anything.

    The dial maps to the policy's knobs along the measured frontier (see
    `wmo.optimize.knn.apply_cost_quality`). The first successful run copies the un-tuned artifact
    to `policy.base.json` and every later run re-reads THAT, so the dial is always applied to the
    policy as fitted and sliding twice never compounds:

        wmo optimize route tune models/support/policy.json --cost-quality 0.6

    That snapshot is only a valid baseline for the fit it came from, so this command refuses to
    run when the two disagree (refit the policy and the stale snapshot must be deleted, not
    silently dialed back over the new fit). A tune that is rejected writes nothing at all, and
    every write it does make is atomic.

    The evidence bank is untouched, so this is instant. A served endpoint can be dialed without
    touching files at all: `PUT /v1/endpoints/{name}/config`.
    """
    try:
        dialed = tune_policy_dial(Path(policy_file), cost_quality)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    print_dial(_console, dialed)


def print_dial(console: Console, dialed: DialResult) -> None:
    """Report an applied dial position against the frontier that was actually measured."""
    knobs = dialed.knobs
    console.print(
        f"[green]✓[/green] cost_quality={dialed.cost_quality:g} "
        f"({dialed.named_point}) -> {dialed.policy_path}\n"
        f"  knobs: floor_q={knobs.floor_q:g}, cost knob lam={knobs.pick_lam:g}, "
        f"guard={knobs.guard_mode}, z={knobs.knn_z:g}\n"
        f"  as fitted: {dialed.base_path}\n"
        f"  measured on routerbench-ours9 (5 held-out splits, vs the best single model):"
    )
    for anchor in COST_QUALITY_ANCHORS:
        marker = "->" if anchor.cost_quality == dialed.cost_quality else "  "
        console.print(
            f"  {marker} {anchor.cost_quality:<5g} {anchor.quality_delta_points:+.2f}pt "
            f"@ {anchor.cost_delta_percent:+.1f}% cost"
            + (f"  [dim]{anchor.named_point}[/dim]" if anchor.named_point != "Custom" else "")
        )


@route_app.command("report")
def report(
    matrix_file: str = typer.Argument(..., help="OutcomeMatrix JSON with held-out scenarios."),
    policy_file: str = typer.Argument(..., help="Fitted policy JSON."),
    baseline: str = typer.Option(
        ..., "--baseline", help="Frontier pool model the report compares against."
    ),
    endpoint: str = typer.Option("endpoint", "--endpoint", help="Endpoint id for the report."),
    out: str = typer.Option("report.json", "--out", help="Where to write the report JSON."),
) -> None:
    """Build the improvement report for a fitted policy over a matrix."""
    matrix = OutcomeMatrix.load(Path(matrix_file))
    policy = RoutingPolicy.load(Path(policy_file))
    improvement = build_report(
        matrix,
        policy,
        baseline=baseline,
        endpoint=endpoint,
        generated_at=datetime.now(tz=UTC).isoformat(),
    )
    Path(out).write_text(improvement.model_dump_json(indent=2), encoding="utf-8")
    headline = improvement.headline
    _console.print(
        f"[green]✓[/green] report -> {out}\n"
        f"  routed acc {headline.accuracy:.4f} @ ${headline.cost_per_run_usd:.5f}/run vs "
        f"{baseline} {headline.baseline_accuracy:.4f} @ ${headline.baseline_cost_per_run_usd:.5f}"
    )
