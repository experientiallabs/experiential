"""DeepSWE v1.1 -> OutcomeMatrix: published mini-swe-agent trials as routing evidence.

The research adapter the `OutcomeMatrix` docstring invites: DeepSWE v1.1 publishes a dense
per-trial table (50 model-and-effort configs x 113 long-horizon SWE tasks, one scaffold,
~22.5k trials with graded scores and measured USD costs), and this module converts it into the
routing optimizer's native artifact so `wmo optimize route fit` and the research protocols run
on it unchanged. Provenance: https://deepswe.datacurve.ai, artifacts v1.1 (trials, tasks,
live leaderboard), plus each task's `instruction.md` from the datacurve-ai/deep-swe repository
(Apache-2.0).

What converts, and the honesty rules:

- INTEGRITY GATE: before anything is written, every config's trial-mean pass rate must
  reproduce the published leaderboard's `pass_at_1` exactly (the publisher's own crosscheck);
  a single mismatch aborts the conversion.
- POOL: the configs whose model our price table covers (the OpenAI and Anthropic families;
  41 of the 50). The other 9 (gemini/glm/grok/kimi/muse) are vendors the table does not price,
  and an unpriced or invented price would poison every cost number downstream. Entry names are
  `<model>@<effort>` (e.g. `claude-opus-5@high`); models served through Vertex AI keep their
  Anthropic model identity. `reasoning_effort` stays None on the entries: the pool's typed
  effort dial has no `xhigh`, and these entries are measurement snapshots, not serveable rungs.
- OUTCOMES: one `ScenarioOutcome` per scored trial: `reward` is the graded fail-to-pass
  fraction (`f2p`, the objective the lab validated; binary pass/fail overstates the model-tier
  gap ~6x on this data), `success` the published binary verdict, `cost_usd` the published
  per-trial measured cost, `usage` the published token counts (`n_cache_tokens` verified to be
  a subset of `n_input_tokens`, matching `TokenUsage`'s contract). The 21 scored trials that
  publish no `cost_usd` become UNSCORED rows (`reward=None`, `error` says why): pricing them
  $0 would corrupt every cost mean, and the lab's own fits dropped that task entirely.
- EMBEDDINGS: the recorded Qwen3-Embedding-0.6B vectors (the local embedder's default model)
  are re-shaped into the row-aligned `.npy` that `CachedTaskEmbedder` serves, so fits and
  reproductions are offline and bit-exact.
- GROUPS: `scenario_groups.json` maps each task to its repository, the grouping key
  `split_router_scenarios_grouped` needs (38 of the 113 tasks share a repository with another;
  an ungrouped split leaks same-repo near-duplicates into the fit bank).

Everything this module writes is a build output for the published artifact bundle
(HF: matrix.json + task_embeddings.npy + scenario_groups.json); nothing here is committed to
git. `wmo optimize route convert-deepswe` is the CLI face; `wmo reproduce run` consumes the
published bundle through the `deepswe-coding` manifest.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from statistics import fmean

import numpy as np
from pydantic import BaseModel, ConfigDict

from wmo.optimize.outcomes import OutcomeMatrix, ScenarioOutcome
from wmo.providers.base import ProviderKind, TokenUsage
from wmo.providers.pool import PoolEntry

logger = logging.getLogger(__name__)

MATRIX_FILENAME = "matrix.json"
EMBEDDINGS_FILENAME = "task_embeddings.npy"
GROUPS_FILENAME = "scenario_groups.json"

# Every instruction.md ends with this harness boilerplate. It is identical on all 113 tasks and
# carries no task signal, so it is stripped before the text reaches an embedder (and the
# recorded vectors were computed on the stripped text).
PROMPT_BOILERPLATE = (
    "\nIMPORTANT: Please work on this in a new branch from main and "
    "commit everything when you are done.\n"
)

# USD per 1M tokens: (input, cached input read, output, cache write). Fetched live 2026-07-28
# from the OpenAI and Anthropic pricing pages, Anthropic rows re-verified 2026-07-31 (the same
# table the coding-router product ships as `router.router_core.STANDARD`; opus-5 prices match
# opus-4-8, sonnet-4-6 matches post-intro sonnet-5). This table is also the pool filter: a
# DeepSWE config converts only if its model has a row here.
MODEL_PRICES_USD_PER_MTOK: dict[str, tuple[float, float, float, float]] = {
    "gpt-5.4-nano": (0.20, 0.02, 1.25, 0.20),
    "gpt-5.4-mini": (0.75, 0.075, 4.50, 0.75),
    "gpt-5.4": (2.50, 0.25, 15.00, 2.50),
    "gpt-5.5": (5.00, 0.50, 30.00, 5.00),
    "gpt-5.3-codex": (1.75, 0.175, 14.00, 1.75),
    # The GPT-5.6 family is the first to charge for cache writes (1.25x input).
    "gpt-5.6-luna": (1.00, 0.10, 6.00, 1.25),
    "gpt-5.6-terra": (2.50, 0.25, 15.00, 3.125),
    "gpt-5.6-sol": (5.00, 0.50, 30.00, 6.25),
    "claude-haiku-4-5": (1.00, 0.10, 5.00, 1.25),
    "claude-sonnet-4-6": (3.00, 0.30, 15.00, 3.75),
    "claude-sonnet-5": (3.00, 0.30, 15.00, 3.75),
    "claude-opus-4-8": (5.00, 0.50, 25.00, 6.25),
    "claude-opus-5": (5.00, 0.50, 25.00, 6.25),
    "claude-fable-5": (10.00, 1.00, 50.00, 12.50),
}


def price_table_span() -> tuple[float, float]:
    """The (cheapest, priciest) blended $/1M price at a 1:1 in:out mix, over the whole table.

    The pre-split lab's `arms` CLI printed this as its price-table health check (0.72 -> 30.00,
    41x); keeping the same number computable here is what proves the table ported intact.
    """
    blended = [(row[0] + row[2]) / 2 for row in MODEL_PRICES_USD_PER_MTOK.values()]
    return min(blended), max(blended)


class _Trial(BaseModel):
    """One published trial row (the fields this conversion reads; the rest pass through)."""

    model_config = ConfigDict(extra="ignore")

    config: str
    task_name: str
    trial_name: str
    included_in_score: bool
    model: str
    reasoning_effort: str | None = None
    f2p: float | None = None
    passed: bool = False
    cost_usd: float | None = None
    outcome: str | None = None
    n_agent_steps: int | None = None
    n_input_tokens: int | None = None
    n_output_tokens: int | None = None
    n_cache_tokens: int | None = None


class _Task(BaseModel):
    """One published task row (id + the repository that groups it)."""

    model_config = ConfigDict(extra="ignore")

    id: str
    repository: str


class DeepsweConversion(BaseModel):
    """What one conversion produced, for the CLI to print and tests to pin."""

    model_config = ConfigDict(frozen=True)

    matrix_path: Path
    embeddings_path: Path
    groups_path: Path
    models: int
    scenarios: int
    scored_outcomes: int
    unscored_outcomes: int
    crosscheck: str  # the integrity gate's verdict over ALL published configs
    dropped_configs: list[str]  # configs whose model the price table does not cover


class DeepsweTopArm(BaseModel):
    """The strongest converted arm's headline numbers (the conversion's hard gate)."""

    model_config = ConfigDict(frozen=True)

    name: str
    graded: float  # task-mean of per-cell mean f2p
    pass_at_1: float  # task-mean of per-cell mean binary pass
    cost_per_task: float  # task-mean of per-cell mean measured USD
    tasks: int


def _dotted(model: str) -> str:
    """The provider's real model id: `gpt-5-6-terra` -> `gpt-5.6-terra` (claude ids as-is)."""
    return re.sub(r"^gpt-(\d)-(\d)", r"gpt-\1.\2", model)


def _pool_entry(model: str, effort: str | None) -> PoolEntry:
    """One converted arm as a pool entry, priced from the table (which admitted it)."""
    prices = MODEL_PRICES_USD_PER_MTOK[model]
    anthropic = model.startswith("claude")
    return PoolEntry(
        name=f"{model}@{effort or 'default'}",
        kind=ProviderKind.ANTHROPIC if anthropic else ProviderKind.OPENAI_RESPONSES,
        model=model,
        input_per_mtok=prices[0],
        cached_input_per_mtok=prices[1],
        output_per_mtok=prices[2],
        cache_write_per_mtok=prices[3],
    )


def _load_rows(source: Path) -> tuple[list[_Trial], list[_Task], dict[str, float]]:
    """Read and validate the three published artifacts under `source`.

    Raises:
        ValueError: An artifact's row count disagrees with its own declared header (the
            publisher's self-description; a partial download must not convert quietly).
    """
    trials_raw = json.loads((source / "trials.json").read_text(encoding="utf-8"))
    tasks_raw = json.loads((source / "tasks.json").read_text(encoding="utf-8"))
    leaderboard = json.loads((source / "leaderboard-live.json").read_text(encoding="utf-8"))
    if len(trials_raw["rows"]) != trials_raw["n_trials"]:
        raise ValueError("trials.json row count disagrees with its own header; refusing to load")
    if len(tasks_raw["rows"]) != tasks_raw["n_tasks"]:
        raise ValueError("tasks.json row count disagrees with its own header; refusing to load")
    trials = [_Trial.model_validate(row) for row in trials_raw["rows"]]
    tasks = [_Task.model_validate(row) for row in tasks_raw["rows"]]
    published = {row["config"]: float(row["pass_at_1"]) for row in leaderboard["rows"]}
    return trials, tasks, published


def _crosscheck(scored: list[_Trial], published: dict[str, float]) -> str:
    """Reproduce every config's published pass@1 from the raw trials, or refuse to convert.

    Raises:
        ValueError: Any config's recomputed trial-mean pass rate differs from the published
            leaderboard by more than 1e-9 (or is missing from it).
    """
    by_config: dict[str, list[float]] = {}
    for trial in scored:
        by_config.setdefault(trial.config, []).append(float(trial.passed))
    off = sorted(
        config
        for config, values in by_config.items()
        if config not in published or abs(fmean(values) - published[config]) > 1e-9
    )
    if off:
        raise ValueError(
            f"{len(off)} of {len(by_config)} configs do not reproduce the published pass@1 "
            f"(first: {off[:3]}); the trials and leaderboard artifacts disagree, refusing to "
            "convert"
        )
    return f"{len(by_config)}/{len(by_config)} configs reproduce published pass@1, 0 off"


def _task_text(source: Path, task_id: str) -> str:
    """One task's instruction.md with the shared harness boilerplate stripped."""
    path = source / "deep-swe-main" / "tasks" / task_id / "instruction.md"
    if not path.is_file():
        raise FileNotFoundError(
            f"no instruction.md for task '{task_id}' under {source / 'deep-swe-main' / 'tasks'}; "
            "extract the deep-swe repository tarball into the source directory first"
        )
    raw = path.read_text(encoding="utf-8")
    return raw[: -len(PROMPT_BOILERPLATE)] if raw.endswith(PROMPT_BOILERPLATE) else raw


def convert_deepswe(source: Path, *, embedding_cache: Path, out: Path) -> DeepsweConversion:
    """Convert the published DeepSWE artifacts under `source` into an OutcomeMatrix bundle.

    Args:
        source: Directory holding `trials.json`, `tasks.json`, `leaderboard-live.json`, and
            the extracted `deep-swe-main/tasks/<id>/instruction.md` texts.
        embedding_cache: JSON of task id -> recorded embedding vector (the lab's
            Qwen3-Embedding-0.6B cache); must cover every task at one width.
        out: Where `matrix.json`, `task_embeddings.npy`, and `scenario_groups.json` land.

    Returns:
        The conversion's counts, verdicts, and artifact paths.

    Raises:
        ValueError: The integrity gate failed, the embedding cache does not cover the tasks,
            or two tasks share instruction text (a text-keyed vector cache cannot tell them
            apart, per `CachedTaskEmbedder`).
    """
    trials, tasks, published = _load_rows(source)
    scored = [trial for trial in trials if trial.included_in_score]
    crosscheck = _crosscheck(scored, published)

    arm_of: dict[str, str] = {}  # config -> pool entry name
    effort_of: dict[str, tuple[str, str | None]] = {}
    dropped: set[str] = set()
    for trial in scored:
        model = _dotted(trial.model)
        if model not in MODEL_PRICES_USD_PER_MTOK:
            dropped.add(trial.config)
            continue
        arm_of[trial.config] = f"{model}@{trial.reasoning_effort or 'default'}"
        effort_of[trial.config] = (model, trial.reasoning_effort)
    pool = [_pool_entry(*effort_of[config]) for config in sorted(arm_of, key=lambda c: arm_of[c])]

    texts = {task.id: _task_text(source, task.id) for task in tasks}
    if len(set(texts.values())) != len(texts):
        raise ValueError(
            "two tasks share identical instruction text; the text-keyed vector cache cannot "
            "tell them apart (see CachedTaskEmbedder), refusing to convert"
        )
    groups = {task.id: task.repository for task in tasks}

    kept = sorted(
        (trial for trial in scored if trial.config in arm_of),
        key=lambda trial: (trial.task_name, arm_of[trial.config], trial.trial_name),
    )
    outcomes: list[ScenarioOutcome] = []
    episode_counter: dict[tuple[str, str], int] = {}
    unscored = 0
    for trial in kept:
        name = arm_of[trial.config]
        episode = episode_counter.get((trial.task_name, name), 0)
        episode_counter[(trial.task_name, name)] = episode + 1
        priced = trial.cost_usd is not None
        if not priced:
            unscored += 1
        outcomes.append(
            ScenarioOutcome(
                scenario_id=trial.task_name,
                task=texts[trial.task_name],
                model=name,
                episode=episode,
                # Reward carries the graded objective; an unpriced trial is unscored evidence
                # rather than a $0 lie (module docstring).
                reward=trial.f2p if priced else None,
                success=trial.passed,
                steps=trial.n_agent_steps or 0,
                stop_reason=trial.outcome or "",
                usage=TokenUsage(
                    input_tokens=trial.n_input_tokens or 0,
                    output_tokens=trial.n_output_tokens or 0,
                    cached_input_tokens=trial.n_cache_tokens or 0,
                ),
                cost_usd=trial.cost_usd or 0.0,
                error=None if priced else "published trial carries no cost_usd; unscored so "
                "cost means stay honest",
            )
        )
    matrix = OutcomeMatrix(pool=pool, outcomes=outcomes)

    cache: dict[str, list[float]] = json.loads(embedding_cache.read_text(encoding="utf-8"))
    order: list[str] = []
    seen: set[str] = set()
    for outcome in matrix.outcomes:  # first-appearance order, the CachedTaskEmbedder contract
        if outcome.scenario_id not in seen:
            seen.add(outcome.scenario_id)
            order.append(outcome.scenario_id)
    missing = [task_id for task_id in order if task_id not in cache]
    if missing:
        raise ValueError(
            f"embedding cache {embedding_cache} misses {len(missing)} of {len(order)} tasks "
            f"(first: {missing[:3]}); re-embed with the local model before converting"
        )
    vectors = np.asarray([cache[task_id] for task_id in order], dtype=np.float32)

    out.mkdir(parents=True, exist_ok=True)
    matrix.save(out / MATRIX_FILENAME)
    np.save(out / EMBEDDINGS_FILENAME, vectors)
    (out / GROUPS_FILENAME).write_text(
        json.dumps({task_id: groups[task_id] for task_id in order}, indent=1), encoding="utf-8"
    )
    logger.info(
        "converted DeepSWE: %d arms x %d tasks, %d scored trials (%d unscored) -> %s",
        len(pool),
        len(order),
        len(outcomes) - unscored,
        unscored,
        out,
    )
    return DeepsweConversion(
        matrix_path=out / MATRIX_FILENAME,
        embeddings_path=out / EMBEDDINGS_FILENAME,
        groups_path=out / GROUPS_FILENAME,
        models=len(pool),
        scenarios=len(order),
        scored_outcomes=len(outcomes) - unscored,
        unscored_outcomes=unscored,
        crosscheck=crosscheck,
        dropped_configs=sorted(dropped),
    )


def top_arm(matrix: OutcomeMatrix) -> DeepsweTopArm:
    """The converted matrix's strongest arm, aggregated the way the lab reported it.

    Per-cell means over a (task, arm)'s scored trials, then a mean over tasks: the aggregation
    that produced the pre-split lab's `deepswe` golden row, so its numbers are directly
    comparable (and pinned in the conversion tests).
    """
    graded: dict[str, dict[str, list[float]]] = {}
    binary: dict[str, dict[str, list[float]]] = {}
    cost: dict[str, dict[str, list[float]]] = {}
    for outcome in matrix.outcomes:
        if outcome.reward is None:
            continue
        graded.setdefault(outcome.model, {}).setdefault(outcome.scenario_id, []).append(
            outcome.reward
        )
        binary.setdefault(outcome.model, {}).setdefault(outcome.scenario_id, []).append(
            float(outcome.success)
        )
        cost.setdefault(outcome.model, {}).setdefault(outcome.scenario_id, []).append(
            outcome.cost_usd
        )

    def task_mean(cells: dict[str, list[float]]) -> float:
        return fmean(fmean(values) for values in cells.values())

    name = max(graded, key=lambda model: task_mean(graded[model]))
    return DeepsweTopArm(
        name=name,
        graded=task_mean(graded[name]),
        pass_at_1=task_mean(binary[name]),
        cost_per_task=task_mean(cost[name]),
        tasks=len(graded[name]),
    )
