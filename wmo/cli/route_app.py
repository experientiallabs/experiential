"""`wmo optimize route`: sweep, fit, tune, and report learned inference policies.

The routing optimizer's CLI face, sitting beside `wmo optimize harness` in the optimizer
family. The workflow chains in one direction:

    sweep -> OutcomeMatrix -> fit -> policy.json -> tune (the dial) / report (the evidence)

`sweep` is the producer: it measures every candidate in the pool on the world model's own
held-out scenarios and writes the `OutcomeMatrix` everything downstream consumes (a research
adapter such as RouterBench can write the same artifact instead). `fit` emits the policy
artifact serving loads, `report` the improvement report the endpoint cites, and `tune` is the
one post-fit control: it moves a fitted policy's cost/quality dial without refitting.
Vocabulary note: "route" is developer-facing CLI only; customer copy never says router.
"""

from __future__ import annotations

import itertools
import os
from datetime import UTC, datetime
from pathlib import Path

import typer
from pydantic import BaseModel, ConfigDict
from rich.console import Console
from rich.markup import escape
from rich.prompt import Confirm
from rich.table import Table

from wmo.config import ARTIFACT_DIR, WorldModelStore, load_config
from wmo.core.types import Trace
from wmo.engine import load_world_model, split_holdout
from wmo.env import WorldModelEnv
from wmo.env.closed_loop import evaluate_pool
from wmo.env.scenarios import scenarios_from_traces, tools_hint_from_traces
from wmo.ingest import get_adapter
from wmo.optimize.knn import (
    COST_QUALITY_ANCHORS,
    apply_cost_quality,
    cost_quality_knobs,
    cost_quality_named_point,
    fit_knn_policy,
)
from wmo.optimize.outcomes import OutcomeMatrix, ScenarioOutcome
from wmo.optimize.policy import (
    KNN_BANK_FILENAME,
    POLICY_FILENAME,
    EmbedderSpec,
    RoutingPolicy,
)
from wmo.optimize.report import build_report
from wmo.optimize.routing import evaluate_policy, fit_rank_policy, rerank_policy
from wmo.providers.base import TokenUsage
from wmo.providers.pool import DEFAULT_POOL_PATH, ModelPool, load_pool, pool_provider
from wmo.serving.traces_source import TRACES_FILENAME, local_traces_path

route_app = typer.Typer(
    help="Measure a candidate pool closed-loop, then fit and report the inference policy.",
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
        help="Hand the matrix to `fit` even when candidates were scored on DIFFERENT scenarios. "
        "The ranking is then biased (both fitters skip unscored rows); the coverage table prints "
        "either way.",
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
    the measured candidate spend is printed when the sweep finishes. Every candidate's provider is
    CONSTRUCTED before that question is asked, so a backend that cannot be built at all is a usage
    error rather than a mid-sweep abort with earlier candidates already paid for.

    Fit-readiness is a coverage contract, not a nonzero count. A cell goes unscored when its
    episode errored (provider throttle, agent crash, judge failure) and both fitters SKIP unscored
    rows, so a matrix where candidate A was scored on 20 scenarios and B on 11 ranks the two on
    DIFFERENT task sets. That is not a comparison: the policy it fits is biased by whichever
    scenarios each candidate happened to lose. So per-candidate scored counts ALWAYS print, and
    when candidates lost different scenarios the command still writes the matrix (those cells were
    paid for, and their `error` fields are the diagnosis) but WITHHOLDS the `fit` handoff and exits
    non-zero, naming each candidate and the scenarios it has no scored episode for.
    `--allow-uneven-coverage` is the opt-out for an operator who knows the bias and wants the
    partial data anyway (one candidate's backend down for the whole sweep, say): it prints the same
    coverage table and stops treating the difference as fatal. Losing the SAME scenarios for every
    candidate is not uneven, since the comparison stays like-for-like on fewer scenarios and the
    counts show the loss. Differing episode counts within a commonly scored scenario change that
    cell's noise, not which scenarios a candidate was ranked on, so they show in the Unscored
    column without blocking.

    Exit code 1 when the matrix is not fit-ready: no cell scored at all, or unequal scored coverage
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
    try:
        config = load_config(model_dir)
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    try:
        pool = load_pool(Path(pool_file))
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    # Everything knowable without spending is checked BEFORE the cost question: a candidate whose
    # backend cannot even be constructed, or an --out that cannot be written, would otherwise
    # surface after the sweep had already paid for cells it then throws away.
    _check_pool_backends(pool)
    _check_out_writable(out_path)

    traces = _corpus_traces(model_dir, config.trace_adapter, traces_file)
    train, holdout, tiny_corpus = split_holdout(
        traces, config.train_split, (1.0 - config.train_split) / 2
    )
    # Sorted by trace id first, so `--scenarios` always cuts the same prefix of the same corpus.
    ordered = sorted(holdout, key=lambda trace: trace.trace_id)
    sweep_scenarios = scenarios_from_traces(ordered)[:scenarios]
    if not sweep_scenarios:
        raise typer.BadParameter(
            f"the {len(holdout)} held-out trace(s) of world model '{model_dir.name}' carry no "
            "task prompt, so there is nothing to measure; rebuild from a corpus whose traces "
            "record the instruction they were given"
        )
    if tiny_corpus:
        _console.print(
            f"[yellow]note[/yellow] {len(traces)} trace(s) is too few for a held-out band, so "
            "these scenarios come from the FULL corpus: they are not leak-free, and a policy "
            "fitted on them is a smoke test, not evidence"
        )
    world_model, _serve_provider = load_world_model(model_dir)

    lines = _estimate_cost(
        pool,
        episodes_per_candidate=len(sweep_scenarios) * episodes,
        calls_per_episode=max_steps,
        input_tokens=assume_input_tokens,
        output_tokens=assume_output_tokens,
    )
    _print_cost_estimate(
        lines,
        scenario_count=len(sweep_scenarios),
        episodes=episodes,
        max_steps=max_steps,
        input_tokens=assume_input_tokens,
        output_tokens=assume_output_tokens,
    )
    _confirm_cost(yes=yes)

    cells = len(pool.models) * len(sweep_scenarios) * episodes
    _console.print(
        f"sweeping {len(pool.models)} candidate(s) over {len(sweep_scenarios)} held-out "
        f"scenario(s) of [bold]{escape(model_dir.name)}[/bold], {episodes} episode(s) each…"
    )
    done = itertools.count(1)

    def _on_outcome(outcome: ScenarioOutcome) -> None:
        reward = "unscored" if outcome.reward is None else f"{outcome.reward:.2f}"
        _console.print(
            f"  [{next(done)}/{cells}] {escape(outcome.model)} {escape(outcome.scenario_id)} "
            f"ep{outcome.episode}: reward={reward} ${outcome.cost_usd:.5f} "
            f"steps={outcome.steps}"
        )

    # Frozen for the whole sweep (the `wmo.evals.closed_loop` precedent): without it a
    # candidate's PREDICTED steps enter the shared retrieval buffer and become demos for the next
    # candidate, so the comparison this matrix exists to make would depend on sweep order.
    with world_model.frozen():
        matrix = evaluate_pool(
            lambda: WorldModelEnv(world_model, score_on_close=True),
            pool,
            sweep_scenarios,
            episodes_per_scenario=episodes,
            max_steps=max_steps,
            tools_hint=tools_hint_from_traces(train) or None,
            on_outcome=_on_outcome,
        )
    matrix.save(out_path)
    scored = sum(1 for outcome in matrix.outcomes if outcome.scored)
    spent = sum(outcome.cost_usd for outcome in matrix.outcomes)
    # `escape(out)`: a bracketed path segment would otherwise be read as markup and dropped, so
    # the line would print a path that does not exist (and this one is meant to be copied).
    _console.print(
        f"[green]✓[/green] {len(matrix.outcomes)} cell(s), {scored} scored -> {escape(out)}\n"
        f"  measured candidate spend ${spent:.4f} (the world model's own serve/judge cost is "
        "metered separately)",
        soft_wrap=True,  # a path a user copies must not be wrapped
    )
    coverage = _coverage(matrix)
    _print_coverage(coverage)
    if scored == 0:
        # Exit non-zero: a matrix with no verified reward is not evidence, so `sweep && fit` in a
        # script must stop here rather than fit on it. The rows are on disk for their `error`s.
        # No --allow-uneven-coverage escape: there is nothing to fit, so `fit` would fail anyway.
        _console.print(
            "[yellow]warning[/yellow] no cell was scored, so this matrix is not evidence and "
            "fitting will fail; read the `error` field of a row to see what broke"
        )
        raise typer.Exit(1)
    if _uneven_coverage(coverage):
        counts = ", ".join(f"{escape(row.candidate)} {row.scored}" for row in coverage)
        _console.print(
            "[yellow]warning[/yellow] candidates were scored on DIFFERENT scenarios (scored cells: "
            f"{counts}), so `fit` would rank them on different task sets: it skips unscored rows, "
            "and the policy that comes out is biased by which scenarios each candidate lost. The "
            "paid cells are on disk and their `error` field says what broke."
        )
        if not allow_uneven_coverage:
            _console.print(
                "  fix the lost cells and sweep again, drop the candidate that lost them, or "
                "re-run with [bold]--allow-uneven-coverage[/bold] to fit on this matrix anyway"
            )
            raise typer.Exit(1)
        _console.print(
            "  --allow-uneven-coverage was passed: fitting on it anyway, with that bias accepted"
        )
    _console.print(
        f"  next: [bold]wmo optimize route fit {escape(out)} --kind knn[/bold]", soft_wrap=True
    )


def _check_pool_backends(pool: ModelPool) -> None:
    """Construct every candidate's provider BEFORE the sweep spends anything.

    `evaluate_pool` builds a candidate's provider lazily, at that candidate's FIRST CELL, so any
    reason a backend cannot be built (an unset `api_key_env`, a kind that refuses the explicit key
    the entry names, a config its backend rejects) used to abort the run as a raw traceback after
    every earlier candidate had been fully paid for, with no matrix written. Building all of them
    here turns that into a usage error at the boundary, before the cost question.

    Construction only, never a request. Every backend in `wmo.providers.registry` only stores its
    config in `__init__` and defers the SDK import, the credential read, and the client itself to
    `_get_client()`, so building the whole pool costs nothing and touches no network. Verifying a
    candidate over the wire is deliberately NOT done here: `wmo providers verify` bills a real call
    per model, which would spend money inside a pre-flight whose whole job is to run before any
    spend is authorized, and would make the cost estimate printed next understate what the command
    had already spent. The residual gap is worth stating plainly: whatever a backend resolves
    lazily (a Bedrock region and AWS credentials, an optional SDK extra, an Azure api-version)
    still surfaces at that candidate's first cell, because seeing it earlier needs either a request
    or a copy of each backend's lazy resolution that would drift from it.

    Every constructed provider is discarded: `evaluate_pool` still builds its own per cell, so
    per-cell provider state (the tinker provider's per-episode prompt history) is unchanged.

    Reports EVERY unusable candidate, not just the first: a pool is edited as a file, so an
    operator fixing one entry at a time pays a full round trip per typo.

    Raises:
        typer.BadParameter: One or more candidates cannot be used, each named with its kind.
    """
    problems: list[str] = []
    for entry in pool.models:
        try:
            pool_provider(entry)
        except Exception as exc:  # noqa: BLE001 - anything here is a usage error, never a spend
            # `pool_provider` already prefixes its own failures with the entry name and kind; a
            # surprise from deeper down gets the same identification so the file is editable.
            detail = str(exc)
            problems.append(
                detail
                if detail.startswith(f"pool model '{entry.name}'")
                else f"pool model '{entry.name}' (kind={entry.kind.value}): {detail}"
            )
    if problems:
        raise typer.BadParameter(
            "; ".join(problems)
            + ". Fix or remove those entries in the pool file, then re-run (checked all "
            + f"{len(pool.models)} candidate(s) before spending anything)"
        )


class _CandidateCoverage(BaseModel):
    """One candidate's scored coverage: what a fitter would actually rank it on."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate: str
    scored: int  # cells with a verified reward
    unscored: int  # cells whose episode or scoring failed (skipped by both fitters)
    lost_scenarios: tuple[str, ...]  # scenarios with NO scored episode for this candidate


# Lost scenario ids shown per candidate before the column summarizes the rest: enough to see the
# pattern in a table an operator reads, without a 20-id line per row.
_LOST_SHOWN = 5


def _coverage(matrix: OutcomeMatrix) -> list[_CandidateCoverage]:
    """Per-candidate scored coverage over the swept scenarios, in pool order."""
    swept = matrix.scenario_ids()
    rows: list[_CandidateCoverage] = []
    for name in matrix.model_names():
        cells = [outcome for outcome in matrix.outcomes if outcome.model == name]
        has_reward = {cell.scenario_id for cell in cells if cell.scored}
        rows.append(
            _CandidateCoverage(
                candidate=name,
                scored=sum(1 for cell in cells if cell.scored),
                unscored=sum(1 for cell in cells if not cell.scored),
                lost_scenarios=tuple(sid for sid in swept if sid not in has_reward),
            )
        )
    return rows


def _uneven_coverage(coverage: list[_CandidateCoverage]) -> bool:
    """Whether the candidates were scored on DIFFERENT scenarios, which is not a comparison.

    Compared as SETS of lost scenarios, not counts: two candidates can lose the same number of
    scenarios and still have been ranked on disjoint task subsets. A scenario every candidate lost
    is even (the comparison is like-for-like on what is left), and an episode-count difference
    inside a commonly scored scenario is noise on that cell rather than a different task set, so
    neither blocks (see `sweep`).
    """
    return len({row.lost_scenarios for row in coverage}) > 1


def _print_coverage(coverage: list[_CandidateCoverage]) -> None:
    """Show what each candidate would be ranked on: its scored cells and the scenarios it lost."""
    table = Table(title="Scored coverage per candidate (`fit` SKIPS unscored cells)")
    table.add_column("Candidate", no_wrap=True)
    table.add_column("Scored", justify="right")
    table.add_column("Unscored", justify="right")
    table.add_column("Scenarios with no scored episode")
    for row in coverage:
        lost = ", ".join(row.lost_scenarios[:_LOST_SHOWN])
        if len(row.lost_scenarios) > _LOST_SHOWN:
            lost += f" (+{len(row.lost_scenarios) - _LOST_SHOWN} more)"
        # Scenario ids are corpus data (trace ids) and candidate names are operator strings: both
        # reach a rich console, where `[a]` is markup that would silently drop from the table.
        table.add_row(
            escape(row.candidate),
            f"{row.scored:,}",
            f"{row.unscored:,}",
            escape(lost) if lost else "-",
        )
    _console.print(table)


def _check_out_writable(out_path: Path) -> None:
    """Prove `--out` can be written BEFORE the sweep spends anything.

    `OutcomeMatrix.save` creates the parent directory and writes, but only once every episode has
    run: a destination that cannot be created (a parent component that is a regular file, an
    unwritable directory, a path that is itself a directory) would throw the whole paid sweep
    away with an OS error and no message. Checked without creating anything, so declining the
    cost confirmation still leaves the filesystem untouched.

    Raises:
        typer.BadParameter: The matrix could not be written there.
    """
    resolved = out_path if out_path.is_absolute() else Path.cwd() / out_path
    existing = resolved.parent
    while not existing.exists() and existing != existing.parent:
        existing = existing.parent
    problem: str | None = None
    if not existing.is_dir():
        problem = f"{existing} is not a directory"
    elif not os.access(existing, os.W_OK):
        problem = f"{existing} is not writable"
    elif resolved.is_dir():
        problem = f"{resolved} is a directory"
    elif resolved.exists() and not os.access(resolved, os.W_OK):
        problem = f"{resolved} is not writable"
    if problem is not None:
        raise typer.BadParameter(
            f"cannot write the outcome matrix to {out_path}: {problem}. --out must name a "
            "writable JSON file path"
        )


def _corpus_traces(model_dir: Path, adapter_name: str, explicit: str | None) -> list[Trace]:
    """Ingest the corpus the sweep takes its scenarios from: `--traces`, else the model's own.

    A build does NOT persist the corpus it read (it keeps prompts, metrics and the retrieval
    index), so `local_traces_path` finds a file only for a Hub-downloaded model or a shipped
    example. `--traces` is the same escape hatch `wmo demo` carries for exactly this reason, and
    the failure names it.
    """
    if explicit is not None:
        path = Path(explicit)
        if not path.is_file():
            raise typer.BadParameter(f"no trace file at {path} (--traces)")
    else:
        found = local_traces_path(model_dir)
        if found is None:
            raise typer.BadParameter(
                f"no trace corpus for world model '{model_dir.name}': looked for "
                f"{model_dir / TRACES_FILENAME} and {model_dir.parent.parent / TRACES_FILENAME}. "
                "Sweep scenarios are the model's OWN held-out task prompts, and a build keeps no "
                "copy of the corpus it read, so pass `--traces <the file wmo build --file read>` "
                f"or put that file at {model_dir / TRACES_FILENAME}"
            )
        path = found
    try:
        traces = get_adapter(adapter_name).from_file(str(path))
    except (OSError, ValueError) as exc:  # unknown adapter, unreadable or malformed corpus
        raise typer.BadParameter(f"cannot ingest {path}: {exc}") from exc
    if not traces:
        raise typer.BadParameter(
            f"{path} ingested no traces with the '{adapter_name}' adapter, so there are no "
            "scenarios to sweep"
        )
    return traces


class _CostLine(BaseModel):
    """One candidate's projected sweep spend under the stated per-call token assumption."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate: str
    episodes: int
    calls: int
    input_per_mtok: float
    output_per_mtok: float
    usd: float


def _estimate_cost(
    pool: ModelPool,
    *,
    episodes_per_candidate: int,
    calls_per_episode: int,
    input_tokens: int,
    output_tokens: int,
) -> list[_CostLine]:
    """Project each candidate's spend, priced by its OWN pool entry (overrides included)."""
    per_call = TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens)
    calls = episodes_per_candidate * calls_per_episode
    lines: list[_CostLine] = []
    for entry in pool.models:
        price = entry.price()
        lines.append(
            _CostLine(
                candidate=entry.name,
                episodes=episodes_per_candidate,
                calls=calls,
                input_per_mtok=price.input_per_mtok,
                output_per_mtok=price.output_per_mtok,
                usd=calls * entry.cost_usd(per_call),
            )
        )
    return lines


def _print_cost_estimate(
    lines: list[_CostLine],
    *,
    scenario_count: int,
    episodes: int,
    max_steps: int,
    input_tokens: int,
    output_tokens: int,
) -> None:
    """Render the projected spend, stating exactly which parts are assumed.

    Honest by construction: the CELL and CALL counts are real (`--max-steps` is the per-episode
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
    for line in lines:
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
    _console.print(table)
    _console.print(
        f"{len(lines) * scenario_count * episodes} cell(s) = {len(lines)} candidate(s) x "
        f"{scenario_count} held-out scenario(s) x {episodes} episode(s); estimated total "
        f"${sum(line.usd for line in lines):.2f}"
    )
    _console.print(
        f"  ASSUMPTION: {input_tokens:,} input + {output_tokens:,} output token(s) per policy "
        f"call, and every episode running its full {max_steps}-call budget. Calls are capped, "
        "tokens per call are NOT measured: set --assume-input-tokens/--assume-output-tokens from "
        "your own numbers, or read the measured spend this command prints when it finishes."
    )
    _console.print(
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
    matrix = OutcomeMatrix.load(Path(matrix_file))
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
    built = spec.build()  # ONE embedder for fit and evaluation; azure would otherwise embed twice
    if kind == "knn":
        if cost_weight > 0.0:
            raise typer.BadParameter(
                "--cost-weight re-ranks cluster evidence and applies to --kind rank only; a knn "
                "policy trades cost through its dial instead: fit it, then "
                "`wmo optimize route tune <policy.json> --cost-quality <0..1>`"
            )
        # The sidecar goes beside the policy file: that is where serving resolves it from.
        policy = fit_knn_policy(
            matrix,
            bank_path=out_path.parent / KNN_BANK_FILENAME,
            embedder=spec,
            embed_with=built,
            guard_model=fallback,
            rag_num=rag_num,
            rag_thres=rag_thres,
            z=z,
            min_pairs=min_pairs,
            se_floor=se_floor,
            floor_q=floor_q,
            fitted_from=f"{matrix_file} knn z={z} k={rag_num} q={floor_q} {embedder}-{dim}",
        )
    else:
        policy = fit_rank_policy(
            matrix,
            embedder=spec,
            n_clusters=clusters,
            seed=seed,
            top_k_clusters=top_k_clusters,
            beta=beta,
            fitted_from=f"{matrix_file} seed={seed} k={clusters} {embedder}-{dim}",
        )
        if cost_weight > 0.0:
            policy = rerank_policy(policy, cost_weight=cost_weight)
    policy.save(out_path)
    result = evaluate_policy(policy, matrix, matrix.scenario_ids(), embedder=built)
    if kind == "knn":
        routed = 1.0 - result.model_mix.get(policy.default_model, 0.0)
        _console.print(
            f"[green]✓[/green] fitted knn policy over {result.scenarios} scenarios -> {out}\n"
            f"  bank {out_path.parent / KNN_BANK_FILENAME}, fallback {policy.default_model}, "
            f"z={z}\n"
            f"  routed away from the fallback {routed:.1%} of the time; cost/scenario "
            f"${result.cost_per_scenario:.5f}\n"
            f"  fit-set accuracy {result.accuracy:.4f} is IN-SAMPLE (every request retrieves its "
            "own row); measure on held-out scenarios with `wmo optimize route report`"
        )
        return
    _console.print(
        f"[green]✓[/green] fitted {len(policy.clusters)} clusters over "
        f"{result.scenarios} scenarios -> {out}\n"
        f"  fit-set accuracy {result.accuracy:.4f}, cost/scenario ${result.cost_per_scenario:.5f}"
    )


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
    `wmo.optimize.knn.apply_cost_quality`). The first run copies the un-tuned artifact to
    `policy.base.json` and every later run re-reads THAT, so the dial is always applied to the
    policy as fitted and sliding twice never compounds:

        wmo optimize route tune models/support/policy.json --cost-quality 0.6

    The evidence bank is untouched, so this is instant. A served endpoint can be dialed without
    touching files at all: `PUT /v1/endpoints/{name}/config`.
    """
    path = Path(policy_file)
    if not path.is_file():
        raise typer.BadParameter(f"no policy file at {path}")
    base_path = path.with_name(f"{path.stem}.base{path.suffix}")
    if not base_path.is_file():
        # Preserve the artifact as fitted the first time, so `tune` is always re-appliable from
        # the fit and never from an already-slid copy of itself.
        base_path.write_bytes(path.read_bytes())
    base = RoutingPolicy.load(base_path)
    try:
        tuned = apply_cost_quality(base, cost_quality)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    tuned.save(path)
    knobs = cost_quality_knobs(cost_quality)
    _console.print(
        f"[green]✓[/green] cost_quality={cost_quality:g} "
        f"({cost_quality_named_point(cost_quality)}) -> {path}\n"
        f"  knobs: floor_q={knobs.floor_q:g}, cost knob lam={knobs.pick_lam:g}, "
        f"guard={knobs.guard_mode}, z={knobs.knn_z:g}\n"
        f"  as fitted: {base_path}\n"
        f"  measured on routerbench-ours9 (5 held-out splits, vs the best single model):"
    )
    for anchor in COST_QUALITY_ANCHORS:
        marker = "->" if anchor.cost_quality == cost_quality else "  "
        _console.print(
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
