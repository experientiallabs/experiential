"""Run REAL Terminal-Bench-2 episodes for every pool candidate; emit an OutcomeMatrix.

The bench-defaults product path needs a `real_episode` outcome matrix for TB2, and none
existed: the only real-episode matrix producer in the repo is tau's
`packages/environment-capture/tau-bench/rl/real_episodes.py`, and every TB2 runner here
(`wmo optimize harness`, `wmo optimize distill run`) produces harness verdicts or adapters
rather than routing training data. This script is the missing converter, built on the
product's own seam: `wmo.evals.harbor.HarborScorer` drives harbor's `terminus_2` agent
against the pinned TB2 registry dataset, and every graded trial becomes one
`ScenarioOutcome`.

Two measurement decisions are load-bearing and deliberate:

* The agent is harbor's OWN terminus-2, never the wmo pi bridge. The distill lane measured
  our pi scaffold needing 2-3x terminus-2's turns on these exact tasks and losing 39-59% of
  the reward to scaffold overhead, so a grid that swapped scaffolds in would be measuring
  the scaffold, not the candidate. The scaffold is held fixed; the model is the variable.
* Rows are priced by THIS pool entry's `cost_usd`, never by harbor's own `cost_usd` field.
  litellm prices only models it recognizes and silently reports $0 for the rest, which would
  read as free rather than as missing.

Per-cell durability comes from harbor: each trial owns a directory under the model's job
dir, and `HarborScorer.score` prunes only ungradeable trials before letting harbor resume,
so an interrupted grid re-pays the boundary trials rather than the matrix. `--resume` reads
the rows file back and skips models whose cells are already all harvested.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

from harbor.models.job.config import JobConfig
from harbor.models.trial.paths import TrialPaths
from harbor.models.trial.result import TrialResult

from wmo.core.types import JsonObject
from wmo.evals.harbor.scorer import HarborScorer, TaskEnvironment
from wmo.evals.harbor.scorer import _trial_reward as _scorer_trial_reward
from wmo.harness.doc import HarnessDoc
from wmo.optimize.outcomes import OutcomeMatrix, ScenarioOutcome
from wmo.providers.base import TokenUsage
from wmo.providers.pool import PoolEntry, load_pool

# `_scorer_trial_reward` is deliberately the scorer's OWN reward reader rather than a copy: it
# encodes the bool/finite/in-range rejections that decide scored-vs-excluded, and a second
# implementation here would drift from the product's definition of a graded trial.
logger = logging.getLogger(__name__)

TASK_LABEL = "terminal-bench-2"
JUDGE_LABEL = "tb2-verifier (harbor terminus-2, pytest/ctrf)"
HARBOR_TERMINUS_2 = "harbor.agents.terminus_2.terminus_2:Terminus2"

# Terminus-2 knobs held identical across every candidate: the scaffold is not the variable.
# max_turns matches the distill lane's TB2 pin. enable_summarize keeps a context overflow
# from silently removing a task from the denominator (it compacts and continues instead).
COMMON_AGENT_KWARGS: JsonObject = {
    "max_turns": 100,
    "parser_name": "json",
    "enable_summarize": True,
    "suppress_max_turns_warning": True,
    "collect_rollout_details": False,
    "record_terminal_session": False,
    "store_all_messages": False,
}

# A candidate litellm cannot price or size gets an explicit row, else `get_model_context_limit`
# falls back to 1M tokens and terminus-2 never compacts. Context numbers are conservative
# published values; they bound compaction, they do not enter any reported cost.
DEFAULT_MAX_INPUT_TOKENS = 200_000
DEFAULT_MAX_OUTPUT_TOKENS = 16_384

# Retry only the transport faults this corpus actually produced: the tb2-cost corner hit a
# Bedrock/provider ServiceUnavailable window under sustained load where single calls were
# fine, and E2B sandbox creation flakes the same way. A retry re-enters harbor's resume, so
# graded trials are never re-paid.
RETRY_DELAYS_S = (30.0, 120.0, 300.0, 600.0)


def _agent_wiring(entry: PoolEntry) -> tuple[str, JsonObject]:
    """The litellm route string and terminus-2 kwargs for one pool entry.

    Credentials travel in `llm_kwargs` (terminus-2 forwards it to the LiteLLM constructor,
    which spreads it into every `acompletion` call) because the scorer forces the harbor
    agent's `env` to empty and a subprocess would not inherit anything set here.
    """
    kwargs: JsonObject = dict(COMMON_AGENT_KWARGS)
    llm_kwargs: JsonObject = {}

    if entry.kind.value == "anthropic":
        route = f"anthropic/{entry.model}"
        llm_kwargs["api_key"] = _require_env("ANTHROPIC_API_KEY", entry)
    elif entry.kind.value == "azure":
        if not entry.deployment or not entry.endpoint:
            raise ValueError(f"pool entry {entry.name!r}: azure needs endpoint and deployment")
        route = f"azure/{entry.deployment}"
        kwargs["api_base"] = entry.endpoint
        llm_kwargs["api_key"] = _require_env(entry.api_key_env or "AZURE_API_KEY", entry)
        if entry.api_version:
            llm_kwargs["api_version"] = entry.api_version
    elif entry.kind.value == "openrouter":
        route = f"openrouter/{entry.model}"
        llm_kwargs["api_key"] = _require_env("OPENROUTER_API_KEY", entry)
    elif entry.kind.value == "openai":
        route = f"openai/{entry.model}"
        if entry.endpoint:
            kwargs["api_base"] = entry.endpoint
        llm_kwargs["api_key"] = _require_env(entry.api_key_env or "OPENAI_API_KEY", entry)
    else:
        raise ValueError(f"pool entry {entry.name!r}: unsupported kind {entry.kind.value!r}")

    # Route strings litellm does not carry in its own table need a size hint so terminus-2
    # compacts at the right point. Prices go in too for harbor's own bookkeeping; our rows
    # are priced from the pool entry regardless.
    if entry.kind.value in {"openai", "openrouter"}:
        price = entry.price()
        kwargs["model_info"] = {
            "max_input_tokens": DEFAULT_MAX_INPUT_TOKENS,
            "max_output_tokens": DEFAULT_MAX_OUTPUT_TOKENS,
            "input_cost_per_token": price.input_per_mtok / 1e6,
            "output_cost_per_token": price.output_per_mtok / 1e6,
        }

    kwargs["llm_kwargs"] = llm_kwargs
    return route, kwargs


def _require_env(name: str, entry: PoolEntry) -> str:
    value = os.environ.get(name)
    if not value:
        raise ValueError(
            f"pool entry {entry.name!r} needs ${name}; source the project .env files first "
            "(set -a; source .../.env; set +a)"
        )
    return value


def _job_template(
    *, jobs_dir: Path, task_environment: TaskEnvironment, concurrency: int
) -> JobConfig:
    """A bare harbor template for the pinned TB2 registry dataset.

    The agent entry stays empty: the scorer owns agent identity, model, and kwargs, and
    validates that the template did not set them.
    """
    return JobConfig.model_validate(
        {
            "job_name": "placeholder-overwritten-by-scorer",
            "jobs_dir": str(jobs_dir),
            "n_concurrent_trials": concurrency,
            "environment": {"type": task_environment},
            "datasets": [{"name": "terminal-bench", "version": "2.0"}],
            "agents": [{}],
        }
    )


def _harvest(job_dir: Path, entry: PoolEntry) -> list[ScenarioOutcome]:
    """Read every trial directory under `job_dir` into priced ScenarioOutcome rows.

    Scans the directory rather than reading a ScoreReport so a crashed or cancelled model
    still yields its finished cells. Episode indices come from sorting each task's trial
    names, matching how the scorer projects attempts.
    """
    if not job_dir.is_dir():
        return []
    by_task: dict[str, list[Path]] = defaultdict(list)
    for child in sorted(job_dir.iterdir()):
        if not child.is_dir():
            continue
        result_path = TrialPaths(child).result_path
        if not result_path.is_file():
            continue
        by_task[child.name.split("__")[0]].append(child)

    rows: list[ScenarioOutcome] = []
    for trial_dirs in by_task.values():
        for episode, trial_dir in enumerate(sorted(trial_dirs, key=lambda p: p.name)):
            row = _row_from_trial(trial_dir, entry=entry, episode=episode)
            if row is not None:
                rows.append(row)
    return rows


def _row_from_trial(trial_dir: Path, *, entry: PoolEntry, episode: int) -> ScenarioOutcome | None:
    result_path = TrialPaths(trial_dir).result_path
    try:
        trial = TrialResult.model_validate_json(result_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("unreadable trial result %s: %s", result_path, exc)
        return None

    n_in, n_cache, n_out, _litellm_cost = trial.compute_token_cost_totals()
    usage = TokenUsage(
        input_tokens=int(n_in or 0),
        cached_input_tokens=int(n_cache or 0),
        output_tokens=int(n_out or 0),
    )
    agent_result = trial.agent_result
    metadata = dict(agent_result.metadata or {}) if agent_result is not None else {}
    call_times_ms = metadata.get("api_request_times_msec") or []
    call_seconds = [float(ms) / 1000.0 for ms in call_times_ms if isinstance(ms, (int, float))]

    reward = _scorer_trial_reward(trial, reward_key="reward")
    exception_type = trial.exception_info.exception_type if trial.exception_info else None
    # An agent timeout that still produced a verifier reward is a real benchmark outcome, not
    # an infrastructure hole: TB2's own wall budget expiring is one of the ways a model fails.
    unscored = reward is None
    error = None
    if unscored:
        error = f"unscored: {exception_type or 'no verifier reward'}"

    return ScenarioOutcome(
        scenario_id=trial.task_name or trial_dir.name.split("__")[0],
        task=TASK_LABEL,
        model=entry.name,
        episode=episode,
        reward=reward,
        success=bool(reward is not None and reward >= 1.0),
        steps=int(metadata.get("n_episodes") or 0),
        stop_reason=str(exception_type or ""),
        usage=usage,
        cost_usd=entry.cost_usd(usage),
        call_seconds=call_seconds,
        error=error,
    )


async def _score_model(
    entry: PoolEntry,
    *,
    task_ids: Sequence[str],
    attempts: int,
    jobs_root: Path,
    task_environment: TaskEnvironment,
    concurrency: int,
    doc: HarnessDoc,
) -> None:
    """Drive one candidate over every pinned task. Raises on unrecovered transport faults."""
    jobs_dir = jobs_root / entry.name
    jobs_dir.mkdir(parents=True, exist_ok=True)
    route, agent_kwargs = _agent_wiring(entry)
    logger.info(
        "scoring %s as %s (%d tasks x %d attempts)", entry.name, route, len(task_ids), attempts
    )

    scorer = await HarborScorer.create(
        _job_template(
            jobs_dir=jobs_dir, task_environment=task_environment, concurrency=concurrency
        ),
        list(task_ids),
        provider_config=entry.provider_config(),
        reward_key="reward",
        reward_mode="raw",
        attempts=attempts,
        task_environment=task_environment,
        # terminus-2 ignores harness_backend (it runs its own scaffold); "e2b" is what lifts
        # the scorer's local-execution guard that would otherwise pin concurrency to 1.
        harness_backend="e2b",
        agent_concurrency=concurrency,
        harbor_retries=0,
        agent_import_path=HARBOR_TERMINUS_2,
        agent_model_name=route,
        extra_agent_kwargs=agent_kwargs,
        # A single ungradeable trial must not abort the grid; it becomes an excluded row.
        missing_reward="zero",
    )
    await asyncio.to_thread(scorer.score, doc)


def _write_rows(rows_path: Path, rows: Sequence[ScenarioOutcome]) -> None:
    """Rewrite the rows file from the harvested set (harvest is idempotent per trial dir)."""
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(row.model_dump_json() for row in rows)
    rows_path.write_text(body + ("\n" if body else ""), encoding="utf-8")


def _read_rows(rows_path: Path) -> list[ScenarioOutcome]:
    if not rows_path.is_file():
        return []
    rows: list[ScenarioOutcome] = []
    for line in rows_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(ScenarioOutcome.model_validate_json(line))
    return rows


def _ledger_append(ledger_path: Path, record: JsonObject) -> None:
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with ledger_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True, help="pool TOML with the candidates")
    parser.add_argument("--task-ids", type=Path, required=True, help="JSON array of TB2 task ids")
    parser.add_argument("--out-dir", type=Path, required=True, help="cohort artifact directory")
    parser.add_argument("--episodes", type=int, default=2, help="attempts per task (cohort pin)")
    parser.add_argument("--concurrency", type=int, default=6, help="concurrent trials per model")
    parser.add_argument(
        "--task-environment", default="e2b", choices=("e2b", "docker"), help="harbor task env"
    )
    parser.add_argument("--only", nargs="*", default=None, help="restrict to these pool names")
    parser.add_argument("--limit-tasks", type=int, default=None, help="first N tasks (smoke only)")
    parser.add_argument(
        "--retries", type=int, default=len(RETRY_DELAYS_S), help="transport-fault retries per model"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    pool = load_pool(args.pool)
    entries = list(pool.models)
    if args.only:
        wanted = set(args.only)
        unknown = wanted - {entry.name for entry in entries}
        if unknown:
            raise SystemExit(f"--only names models absent from the pool: {sorted(unknown)}")
        entries = [entry for entry in entries if entry.name in wanted]

    task_environment: TaskEnvironment = "e2b" if args.task_environment == "e2b" else "docker"
    task_ids = json.loads(args.task_ids.read_text(encoding="utf-8"))
    if args.limit_tasks is not None:
        task_ids = task_ids[: args.limit_tasks]

    out_dir = args.out_dir
    jobs_root = out_dir / "harbor"
    rows_path = out_dir / "rows.jsonl"
    matrix_path = out_dir / "matrix.json"
    ledger_path = out_dir / "ledger.jsonl"
    out_dir.mkdir(parents=True, exist_ok=True)

    doc = HarnessDoc.baseline()
    expected_cells = len(task_ids) * args.episodes

    for entry in entries:
        job_dir = jobs_root / entry.name / f"wmo-{doc.doc_hash[:12]}"
        existing = _harvest(job_dir, entry)
        if len(existing) >= expected_cells:
            logger.info(
                "%s already has %d/%d cells, skipping", entry.name, len(existing), expected_cells
            )
            continue

        started = time.time()
        attempt = 0
        while True:
            try:
                asyncio.run(
                    _score_model(
                        entry,
                        task_ids=task_ids,
                        attempts=args.episodes,
                        jobs_root=jobs_root,
                        task_environment=task_environment,
                        concurrency=args.concurrency,
                        doc=doc,
                    )
                )
                break
            except KeyboardInterrupt:
                raise
            except Exception as exc:  # noqa: BLE001 - transport faults are the expected case
                harvested = _harvest(job_dir, entry)
                logger.warning(
                    "%s failed on attempt %d with %s: %s (%d cells banked)",
                    entry.name,
                    attempt + 1,
                    type(exc).__name__,
                    exc,
                    len(harvested),
                )
                _ledger_append(
                    ledger_path,
                    {
                        "model": entry.name,
                        "event": "attempt_failed",
                        "attempt": attempt + 1,
                        "error_type": type(exc).__name__,
                        "error": str(exc)[:500],
                        "cells_banked": len(harvested),
                    },
                )
                if attempt >= args.retries:
                    logger.error("%s exhausted %d retries, moving on", entry.name, args.retries)
                    break
                time.sleep(RETRY_DELAYS_S[min(attempt, len(RETRY_DELAYS_S) - 1)])
                attempt += 1

        rows = _harvest(job_dir, entry)
        scored = [row for row in rows if row.scored]
        spend = sum(row.cost_usd for row in rows)
        _ledger_append(
            ledger_path,
            {
                "model": entry.name,
                "event": "model_complete",
                "cells": len(rows),
                "cells_expected": expected_cells,
                "scored": len(scored),
                "solved": sum(1 for row in scored if row.success),
                "cost_usd": round(spend, 4),
                "wall_seconds": round(time.time() - started, 1),
            },
        )
        logger.info(
            "%s: %d/%d cells, %d scored, %d solved, $%.2f, %.0fs",
            entry.name,
            len(rows),
            expected_cells,
            len(scored),
            sum(1 for row in scored if row.success),
            spend,
            time.time() - started,
        )

        # Rewrite the full matrix after every model so an interrupted grid still has one.
        all_rows: list[ScenarioOutcome] = []
        for candidate in pool.models:
            candidate_dir = jobs_root / candidate.name / f"wmo-{doc.doc_hash[:12]}"
            all_rows.extend(_harvest(candidate_dir, candidate))
        _write_rows(rows_path, all_rows)
        matrix = OutcomeMatrix(pool=list(pool.models), outcomes=all_rows)
        matrix_path.write_text(matrix.model_dump_json(indent=2), encoding="utf-8")

    final = _read_rows(rows_path)
    logger.info(
        "grid done: %d rows, %d scored, $%.2f total",
        len(final),
        sum(1 for row in final if row.scored),
        sum(row.cost_usd for row in final),
    )


if __name__ == "__main__":
    main()
