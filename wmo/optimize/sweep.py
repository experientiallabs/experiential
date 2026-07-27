"""The routing sweep's library core: plan a closed-loop measurement, then run it.

`wmo optimize route sweep` and the sweep stage of `wmo optimize model` are two faces of THIS
module, so the two cannot drift. What lives here once: the scenario cut (the corpus's held-out
band, sorted by trace id, capped by `scenarios`), the backend pre-flight that resolves every
candidate before anything is spent, the projected cost table's arithmetic, and the coverage
contract that decides whether the resulting matrix is fit-ready.

Rendering deliberately does NOT live here. Every function returns typed data or raises
`SweepError` carrying a message its caller prints, so each CLI face owns its own console, its own
spend confirmation, and its own progress display while the measurement itself is single-sourced.

The two seams a caller supplies are the world model and the env factory. They are parameters
rather than imports so the caller decides how the model is loaded and how the env is
constructed, and so tests can stub both at the CLI module they already patch.
"""

from __future__ import annotations

import os
from collections import Counter
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from wmo.config import HarnessConfig, load_config
from wmo.engine import split_holdout
from wmo.env.closed_loop import evaluate_pool
from wmo.env.scenarios import Scenario, scenarios_from_traces, tools_hint_from_traces
from wmo.ingest import get_adapter
from wmo.optimize.outcomes import OutcomeMatrix, ScenarioOutcome
from wmo.providers.base import ProviderKind, TokenUsage
from wmo.providers.pool import ModelPool, load_pool, prepare_pool_provider
from wmo.serving.traces_source import TRACES_FILENAME, local_traces_path

if TYPE_CHECKING:
    from wmo.core.types import Trace
    from wmo.engine.world_model import WorldModel
    from wmo.env.base import Env


class SweepError(ValueError):
    """A sweep cannot proceed, for a reason the operator can fix.

    Every message names what went wrong and what to do about it, because both CLI faces render
    it verbatim as a usage error (AGENTS rule 9: error messages are part of the interface).
    """


# What each kind still cannot know before its first cell, and why. Both are measured properties
# of the backend (see `BedrockProvider.prepare` and `TinkerChatProvider.prepare`), not caution: a
# pre-flight that resolved them would have to make a request, which is the one thing it may not do.
DEFERRED_RISK: dict[ProviderKind, str] = {
    ProviderKind.BEDROCK: (
        "AWS credentials (boto3 resolves them by walking a chain that can reach the "
        "instance-metadata endpoint over the network, and it builds a client with no credentials "
        "at all)"
    ),
    ProviderKind.TINKER: (
        "the Tinker service being reachable and serving this model (constructing the client "
        "connects and pins a server-side session for the whole process)"
    ),
}


class DeferredRisk(BaseModel):
    """One candidate's residual first-cell risk that a request-free pre-flight cannot close."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate: str
    kind: ProviderKind
    risk: str


class PoolPreflight(BaseModel):
    """The candidate roster, proven constructible, plus what could still fail at a first cell."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    pool: ModelPool
    deferred: tuple[DeferredRisk, ...]


class CostLine(BaseModel):
    """One candidate's projected sweep spend under the stated per-call token assumption."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate: str
    episodes: int
    calls: int
    input_per_mtok: float
    output_per_mtok: float
    usd: float


class SweepPlan(BaseModel):
    """Everything a sweep will do, computed before a single cell is paid for.

    Held as data so the same plan can be printed as a cost table by `route sweep`, folded into
    `optimize model`'s one plan table, and then executed, without any of the three recomputing
    what the others decided.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_dir: Path
    out_path: Path
    pool: ModelPool
    scenarios: tuple[Scenario, ...]
    episodes: int
    max_steps: int
    tools_hint: str | None
    trace_count: int  # traces the corpus ingested, which is what decides `tiny_corpus`
    tiny_corpus: bool  # too small for a held-out band, so the scenarios are not leak-free
    assume_input_tokens: int
    assume_output_tokens: int
    cost_lines: tuple[CostLine, ...]

    @property
    def cells(self) -> int:
        """Cells the sweep will run: candidates x scenarios x episodes."""
        return len(self.pool.models) * len(self.scenarios) * self.episodes

    @property
    def total_usd(self) -> float:
        """The projected candidate-side spend (a projection, never a measurement)."""
        return sum(line.usd for line in self.cost_lines)


def resolve_config(model_dir: Path) -> HarnessConfig:
    """The built model's config, or a `SweepError` naming why it could not be read."""
    try:
        return load_config(model_dir)
    except (FileNotFoundError, ValueError) as exc:
        raise SweepError(str(exc)) from exc


def preflight_pool(pool_file: Path) -> PoolPreflight:
    """Load the candidate pool and resolve every backend locally, BEFORE anything is spent.

    `evaluate_pool` builds a candidate's provider lazily at that candidate's FIRST CELL, and every
    backend then builds its SDK client lazily inside that first call, so any reason a candidate
    cannot be used (an unset `api_key_env`, a kind that refuses the explicit key the entry names,
    a missing SDK extra, an Azure config with no api_version, a Bedrock entry whose region
    resolves nowhere) used to abort the run as a raw traceback after every earlier candidate had
    been fully paid for, with no matrix written. `prepare_pool_provider` closes that: it checks the
    kind's static requirements from the entry alone, then forces the lazy client to be BUILT, which
    imports the SDK and resolves credentials from the environment and local credential files.

    No request, ever. Every `prepare` is documented network-free, and verifying a candidate over
    the wire is deliberately NOT done here: `wmo providers verify` bills a real call per model,
    which would spend money inside a pre-flight whose whole job is to run before any spend is
    authorized, and would make the cost estimate printed next understate what the command had
    already spent. Some backends therefore keep a residual gap (`DEFERRED_RISK`), returned per
    entry rather than left for the operator to discover mid-sweep.

    Every prepared provider is discarded: `evaluate_pool` still builds its own per cell, so
    per-cell provider state (the tinker provider's per-episode prompt history) is unchanged.

    Reports EVERY unusable candidate, not just the first: a pool is edited as a file, so an
    operator fixing one entry at a time pays a full round trip per typo.

    Raises:
        SweepError: The pool could not be read, or one or more candidates cannot be used.
    """
    try:
        pool = load_pool(pool_file)
    except (FileNotFoundError, ValueError) as exc:
        raise SweepError(str(exc)) from exc
    problems: list[str] = []
    for entry in pool.models:
        try:
            prepare_pool_provider(entry)
        except Exception as exc:  # noqa: BLE001 - anything here is a usage error, never a spend
            # `prepare_pool_provider` already prefixes its own failures with the entry name and
            # kind; a surprise from deeper down gets the same identification so the file is
            # editable from the message.
            detail = str(exc)
            problems.append(
                detail
                if detail.startswith(f"pool model '{entry.name}'")
                else f"pool model '{entry.name}' (kind={entry.kind.value}): {detail}"
            )
    if problems:
        raise SweepError(
            "; ".join(problems)
            + ". Fix or remove those entries in the pool file, then re-run (checked all "
            + f"{len(pool.models)} candidate(s) before spending anything)"
        )
    return PoolPreflight(
        pool=pool,
        deferred=tuple(
            DeferredRisk(candidate=entry.name, kind=entry.kind, risk=DEFERRED_RISK[entry.kind])
            for entry in pool.models
            if entry.kind in DEFERRED_RISK
        ),
    )


def plan_sweep(
    *,
    model_dir: Path,
    config: HarnessConfig,
    pool: ModelPool,
    out_path: Path,
    traces_file: Path | None,
    scenarios: int,
    episodes: int,
    max_steps: int,
    assume_input_tokens: int,
    assume_output_tokens: int,
) -> SweepPlan:
    """Cut the held-out scenario set and project the spend, without touching the filesystem.

    Everything knowable without spending is settled here: an `out_path` that cannot be written
    and a corpus that carries no task prompt are both boundary errors, raised before a caller
    asks its operator to authorize anything.

    Raises:
        SweepError: The destination is unwritable, the corpus is missing or unreadable, or the
            held-out band carries no task prompt to measure.
    """
    _check_out_writable(out_path)
    traces = _corpus_traces(model_dir, config.trace_adapter, traces_file)
    train, holdout, tiny_corpus = split_holdout(
        traces, config.train_split, (1.0 - config.train_split) / 2
    )
    # Sorted by trace id first, so `scenarios` always cuts the same prefix of the same corpus.
    ordered = sorted(holdout, key=lambda trace: trace.trace_id)
    cut = scenarios_from_traces(ordered)[:scenarios]
    if not cut:
        raise SweepError(
            f"the {len(holdout)} held-out trace(s) of world model '{model_dir.name}' carry no "
            "task prompt, so there is nothing to measure; rebuild from a corpus whose traces "
            "record the instruction they were given"
        )
    return SweepPlan(
        model_dir=model_dir,
        out_path=out_path,
        pool=pool,
        scenarios=tuple(cut),
        episodes=episodes,
        max_steps=max_steps,
        tools_hint=tools_hint_from_traces(train) or None,
        trace_count=len(traces),
        tiny_corpus=tiny_corpus,
        assume_input_tokens=assume_input_tokens,
        assume_output_tokens=assume_output_tokens,
        cost_lines=tuple(
            _estimate_cost(
                pool,
                episodes_per_candidate=len(cut) * episodes,
                calls_per_episode=max_steps,
                input_tokens=assume_input_tokens,
                output_tokens=assume_output_tokens,
            )
        ),
    )


def execute_sweep(
    plan: SweepPlan,
    *,
    world_model: WorldModel,
    env_factory: Callable[[], Env],
    on_outcome: Callable[[ScenarioOutcome], None] | None = None,
) -> OutcomeMatrix:
    """Run every cell of `plan` against a FROZEN world model and write the matrix.

    Frozen for the whole sweep (the `wmo.evals.closed_loop` precedent): without it a candidate's
    PREDICTED steps enter the shared retrieval buffer and become demos for the next candidate, so
    the comparison this matrix exists to make would depend on sweep order.

    `env_factory` must build an env that scores on close (`score_on_close=True`): a matrix
    without verified rewards is not evidence, and `evaluate_pool` refuses one that does not score.
    """
    with world_model.frozen():
        matrix = evaluate_pool(
            env_factory,
            plan.pool,
            list(plan.scenarios),
            episodes_per_scenario=plan.episodes,
            max_steps=plan.max_steps,
            tools_hint=plan.tools_hint,
            on_outcome=on_outcome,
        )
    matrix.save(plan.out_path)
    return matrix


class CandidateCoverage(BaseModel):
    """One candidate's scored coverage: what a fitter would actually weigh it on."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate: str
    scored: int  # cells with a verified reward
    unscored: int  # cells whose episode or scoring failed (skipped by both fitters)
    # Scored EPISODE COUNT per swept scenario, in sweep order. Counts, not presence: both fitters
    # weigh episodes, not scenarios (see `unevenness`), so `X: 3` and `X: 1` are different
    # evidence even though both "cover" X.
    scored_episodes: tuple[tuple[str, int], ...]
    first_error: str | None  # first error text among this candidate's unscored cells, if any

    @property
    def lost_scenarios(self) -> tuple[str, ...]:
        """Swept scenarios this candidate has NO scored episode for."""
        return tuple(sid for sid, count in self.scored_episodes if count == 0)


class Unevenness(StrEnum):
    """How the candidates' scored evidence differs, when it does."""

    EVEN = "even"
    SCENARIOS = "scenarios"  # candidates were scored on different scenario SETS
    EPISODES = "episodes"  # same scenarios, different numbers of scored episodes


def coverage(matrix: OutcomeMatrix) -> list[CandidateCoverage]:
    """Per-candidate scored coverage over the swept scenarios, in pool order."""
    swept = matrix.scenario_ids()
    rows: list[CandidateCoverage] = []
    for name in matrix.model_names():
        cells = [outcome for outcome in matrix.outcomes if outcome.model == name]
        per_scenario: Counter[str] = Counter(cell.scenario_id for cell in cells if cell.scored)
        errors = [cell.error for cell in cells if not cell.scored and cell.error]
        rows.append(
            CandidateCoverage(
                candidate=name,
                scored=sum(1 for cell in cells if cell.scored),
                unscored=sum(1 for cell in cells if not cell.scored),
                scored_episodes=tuple((sid, per_scenario[sid]) for sid in swept),
                first_error=errors[0] if errors else None,
            )
        )
    return rows


def unevenness(rows: list[CandidateCoverage]) -> Unevenness:
    """Whether the candidates were scored on the same evidence, and if not, how it differs.

    Compared as per-(candidate, scenario) scored EPISODE COUNTS, because that is what the fitters
    weigh. Presence is not enough: `fit_rank_policy` averages every surviving EPISODE
    independently, so a candidate that kept 3 episodes on a scenario and one that kept 1 carry
    different effective scenario weights into the same cluster mean, and the models chosen as
    `default_model` (`routing.py:_overall_best`) and as the knn fallback/guard
    (`knn.py:best_single_on_fit`) are picked off those same episode-weighted means. The knn BANK is
    milder, since its cells are per-scenario means, but "milder" is not "unbiased", and it is the
    same matrix either way.

    Losing the same episodes of the same scenarios for EVERY candidate is even: the comparison
    stays like-for-like on less data, and the counts show the loss.
    """
    if len({row.scored_episodes for row in rows}) <= 1:
        return Unevenness.EVEN
    if len({row.lost_scenarios for row in rows}) > 1:
        return Unevenness.SCENARIOS
    return Unevenness.EPISODES


def _check_out_writable(out_path: Path) -> None:
    """Prove the matrix destination can be written BEFORE the sweep spends anything.

    `OutcomeMatrix.save` creates the parent directory and writes, but only once every episode has
    run: a destination that cannot be created (a parent component that is a regular file, an
    unwritable directory, a path that is itself a directory) would throw the whole paid sweep
    away with an OS error and no message. Checked without creating anything, so declining the
    cost confirmation still leaves the filesystem untouched.

    Raises:
        SweepError: The matrix could not be written there.
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
        raise SweepError(
            f"cannot write the outcome matrix to {out_path}: {problem}. --out must name a "
            "writable JSON file path"
        )


def _corpus_traces(model_dir: Path, adapter_name: str, explicit: Path | None) -> list[Trace]:
    """Ingest the corpus the sweep takes its scenarios from: the explicit file, else the model's.

    A build does NOT persist the corpus it read (it keeps prompts, metrics and the retrieval
    index), so `local_traces_path` finds a file only for a Hub-downloaded model or a shipped
    example. `--traces` is the same escape hatch `wmo demo` carries for exactly this reason, and
    the failure names it.
    """
    if explicit is not None:
        if not explicit.is_file():
            raise SweepError(f"no trace file at {explicit} (--traces)")
        path = explicit
    else:
        found = local_traces_path(model_dir)
        if found is None:
            raise SweepError(
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
        raise SweepError(f"cannot ingest {path}: {exc}") from exc
    if not traces:
        raise SweepError(
            f"{path} ingested no traces with the '{adapter_name}' adapter, so there are no "
            "scenarios to sweep"
        )
    return traces


def _estimate_cost(
    pool: ModelPool,
    *,
    episodes_per_candidate: int,
    calls_per_episode: int,
    input_tokens: int,
    output_tokens: int,
) -> list[CostLine]:
    """Project each candidate's spend, priced by its OWN pool entry (overrides included)."""
    per_call = TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens)
    calls = episodes_per_candidate * calls_per_episode
    lines: list[CostLine] = []
    for entry in pool.models:
        price = entry.price()
        lines.append(
            CostLine(
                candidate=entry.name,
                episodes=episodes_per_candidate,
                calls=calls,
                input_per_mtok=price.input_per_mtok,
                output_per_mtok=price.output_per_mtok,
                usd=calls * entry.cost_usd(per_call),
            )
        )
    return lines
