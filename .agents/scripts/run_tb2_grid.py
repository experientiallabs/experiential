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
import subprocess
import time
from collections import defaultdict
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import litellm
from harbor.models.job.config import DatasetConfig, JobConfig
from harbor.models.trial.paths import TrialPaths
from harbor.models.trial.result import TrialResult

from wmo.core.types import JsonObject
from wmo.evals.harbor.scorer import HarborScorer, TaskEnvironment

# The scorer's OWN reward reader rather than a copy: it encodes the bool/finite/in-range
# rejections that decide scored-vs-excluded, and a second implementation here would drift from
# the product's definition of a graded trial.
from wmo.evals.harbor.scorer import _trial_reward as _scorer_trial_reward
from wmo.evals.harbor.tasks import resolve_harbor_tasks
from wmo.harness.doc import HarnessDoc
from wmo.optimize.outcomes import OutcomeMatrix, ScenarioOutcome
from wmo.providers.base import TokenUsage
from wmo.providers.pool import ModelPool, PoolEntry, load_pool
from wmo.runs.ledger import LedgerLine

logger = logging.getLogger(__name__)

TASK_LABEL = "terminal-bench-2"
# One arm: no compaction lever is under test here, so the grid holds the identity arm only.
ARM_NAME = "identity"
JUDGE_LABEL = "tb2-verifier (harbor terminus-2, pytest/ctrf)"
HARBOR_TERMINUS_2 = "harbor.agents.terminus_2.terminus_2:Terminus2"

# Terminus-2 knobs held identical across every candidate: the scaffold is not the variable.
# max_turns matches the distill lane's TB2 pin. enable_summarize keeps a context overflow
# from silently removing a task from the denominator (it compacts and continues instead).
MAX_TURNS = 100
COMMON_AGENT_KWARGS: JsonObject = {
    "max_turns": MAX_TURNS,
    "parser_name": "json",
    "enable_summarize": True,
    "suppress_max_turns_warning": True,
    "collect_rollout_details": False,
    "record_terminal_session": False,
    "store_all_messages": False,
}

# A route litellm has never heard of gets an explicit `model_info`, else its context limit
# falls back to 1M, terminus-2 never compacts, and the provider rejects the oversized prompt
# instead. Measured on this pool: litellm knows the gpt-5.x and claude routes but NOT
# azure/DeepSeek-V4-Pro, azure/Kimi-K2.6, azure/FW-GLM-5.2, the Fireworks kimi-k3 route, or
# either OpenRouter qwen. The override is applied ONLY to routes litellm cannot resolve, so a
# model whose real limit is known keeps it (clamping gpt-5.6 to 200k would have made it compact
# for no reason). 128k is at or below every unknown candidate's published limit, so it can only
# compact EARLIER than necessary, never overflow; it is applied identically to all of them and
# enters no reported cost.
UNKNOWN_ROUTE_MAX_INPUT_TOKENS = 128_000
UNKNOWN_ROUTE_MAX_OUTPUT_TOKENS = 16_384

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

    if not _litellm_knows(route):
        price = entry.price()
        kwargs["model_info"] = {
            "max_input_tokens": UNKNOWN_ROUTE_MAX_INPUT_TOKENS,
            "max_output_tokens": UNKNOWN_ROUTE_MAX_OUTPUT_TOKENS,
            "input_cost_per_token": price.input_per_mtok / 1e6,
            "output_cost_per_token": price.output_per_mtok / 1e6,
        }

    kwargs["llm_kwargs"] = llm_kwargs
    return route, kwargs


def _litellm_knows(route: str) -> bool:
    """Whether litellm can resolve this route's context limits on its own."""
    try:
        litellm.get_model_info(route)
    except Exception:  # noqa: BLE001 - litellm raises several unrelated types for "unmapped"
        return False
    return True


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


def _ledger_line(
    *,
    event: str,
    tip_sha: str,
    episodes: int,
    chunk: int | None = None,
    cells: int = 0,
    scored: int = 0,
    candidate_usd: float = 0.0,
    wall_s: float = 0.0,
    cumulative_usd: float = 0.0,
    note: str = "",
) -> LedgerLine:
    """One conforming ledger line.

    `LedgerLine` is `extra="forbid"` on purpose: the runner and `wmo runs backfill` both
    validate against it and SKIP what fails, so a line carrying an unrecognized key is dropped
    by both and its chunk's cells never reach the runs tables. The first version of this runner
    wrote `model`, `solved` and `cost_usd` as top-level keys and every line was silently
    discarded. Anything this schema has no field for goes in `note`.
    """
    return LedgerLine(
        event=event,
        arm=ARM_NAME,
        chunk=chunk,
        cells=cells,
        scored=scored,
        candidate_usd=candidate_usd,
        wall_s=wall_s,
        ts=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        cumulative_usd=cumulative_usd,
        tip_sha=tip_sha,
        max_steps=MAX_TURNS,
        episodes=episodes,
        note=note,
    )


def _ledger_append(ledger_path: Path, line: LedgerLine) -> None:
    """Append one conforming ledger line; append-only so a SIGKILL costs one line, not the file."""
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with ledger_path.open("a", encoding="utf-8") as handle:
        handle.write(line.model_dump_json() + "\n")


def _tip_sha() -> str:
    """The repo tip this cohort was measured at; every ledger line carries it."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() or "unknown"


def _repair_ledger(out_dir: Path, *, pool: ModelPool, episodes: int, tip_sha: str) -> None:
    """Rewrite `ledger.jsonl` from the chunk files so every line conforms to `LedgerLine`.

    Needed because the ledger is written by several concurrent model processes and any line a
    reader cannot validate is DROPPED by both the runner and `wmo runs backfill`, taking its
    chunk's cells out of the runs tables with it. Regenerating from the chunk files (the actual
    evidence) makes conformance a property of this pass rather than of whichever process wrote
    the line. The original is preserved as `ledger.raw.jsonl` so nothing is lost, including the
    retry notes that have no chunk of their own.
    """
    arm_dir = out_dir / ARM_NAME
    ledger_path = out_dir / "ledger.jsonl"
    chunks = sorted(arm_dir.glob("chunk-*.json"), key=lambda p: int(p.stem.removeprefix("chunk-")))
    if not chunks:
        return
    if ledger_path.is_file() and not (out_dir / "ledger.raw.jsonl").is_file():
        ledger_path.rename(out_dir / "ledger.raw.jsonl")

    by_index = {index: entry.name for index, entry in enumerate(pool.models)}
    cumulative = 0.0
    lines: list[LedgerLine] = []
    for chunk_file in chunks:
        chunk = int(chunk_file.stem.removeprefix("chunk-"))
        payload = json.loads(chunk_file.read_text(encoding="utf-8"))
        rows = [ScenarioOutcome.model_validate(raw) for raw in payload.get("outcomes", [])]
        scored = [row for row in rows if row.scored]
        spend = sum(row.cost_usd for row in rows)
        cumulative += spend
        lines.append(
            _ledger_line(
                event="chunk",
                tip_sha=tip_sha,
                episodes=episodes,
                chunk=chunk,
                cells=len(rows),
                scored=len(scored),
                candidate_usd=round(spend, 6),
                cumulative_usd=round(cumulative, 6),
                note=(
                    f"{by_index.get(chunk, f'chunk-{chunk}')}: {len(scored)} scored, "
                    f"{sum(1 for row in scored if row.success)} solved"
                ),
            )
        )
    ledger_path.write_text("\n".join(line.model_dump_json() for line in lines) + "\n", "utf-8")
    logger.info("regenerated %s with %d conforming chunk lines", ledger_path.name, len(lines))


async def _prewarm_tasks(task_ids: Sequence[str]) -> None:
    """Download every pinned TB2 task into harbor's shared cache once, serially.

    Harbor caches registry tasks under a single `~/.cache/harbor/tasks/` tree, and it resolves
    them lazily on first use. Running several models as parallel processes therefore has them
    racing to populate the same directories, and a reader can observe a half-written task: this
    run hit `FileNotFoundError` on a tests/ file that appeared on disk moments later. Warming
    the cache serially before any parallel work makes every later read read-only.
    """
    dataset = DatasetConfig.model_validate({"name": "terminal-bench", "version": "2.0"})
    resolved = await resolve_harbor_tasks(dataset, list(task_ids))
    logger.info("prewarmed %d harbor tasks into the shared cache", len(resolved))


def _consolidate(
    arm_dir: Path,
    *,
    pool: ModelPool,
    rows_path: Path,
    matrix_path: Path,
    episodes: int,
    tip_sha: str,
) -> list[ScenarioOutcome]:
    """Assemble rows.jsonl and matrix.json from every chunk file in the arm directory.

    Idempotent and safe to run while other models are still going: it reads only completed
    chunk files, so a partial grid consolidates into a partial (but valid) matrix.
    """
    rows: list[ScenarioOutcome] = []
    for chunk_file in sorted(arm_dir.glob("chunk-*.json")):
        payload = json.loads(chunk_file.read_text(encoding="utf-8"))
        for raw in payload.get("outcomes", []):
            rows.append(ScenarioOutcome.model_validate(raw))
    _write_rows(rows_path, rows)
    matrix = OutcomeMatrix(pool=list(pool.models), outcomes=rows)
    matrix_path.write_text(matrix.model_dump_json(indent=2), encoding="utf-8")
    _repair_ledger(arm_dir.parent, pool=pool, episodes=episodes, tip_sha=tip_sha)
    scored = [row for row in rows if row.scored]
    logger.info(
        "consolidated %d rows (%d scored, %d solved) from %d chunks, $%.2f",
        len(rows),
        len(scored),
        sum(1 for row in scored if row.success),
        len(list(arm_dir.glob("chunk-*.json"))),
        sum(row.cost_usd for row in rows),
    )
    return rows


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
    parser.add_argument(
        "--prewarm-tasks",
        action="store_true",
        help="download the pinned tasks into harbor's shared cache and exit (run before batches)",
    )
    parser.add_argument(
        "--consolidate-only",
        action="store_true",
        help="rebuild rows.jsonl and matrix.json from existing chunks, running no episodes",
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

    # Grid-directory layout, because that is what `wmo runs backfill` replays into the platform
    # runs tables: cohort.json beside an arm subdirectory holding chunk-N.json files, with a
    # ledger naming the arm. One chunk per candidate is the natural unit here (a model is what
    # this runner completes atomically), so a resumed grid re-emits only the models it re-ran.
    out_dir = args.out_dir
    arm_dir = out_dir / ARM_NAME
    jobs_root = out_dir / "harbor"
    rows_path = out_dir / "rows.jsonl"
    matrix_path = arm_dir / "matrix.json"
    ledger_path = out_dir / "ledger.jsonl"
    arm_dir.mkdir(parents=True, exist_ok=True)
    tip_sha = _tip_sha()

    if args.prewarm_tasks:
        asyncio.run(_prewarm_tasks(task_ids))
        return

    if args.consolidate_only:
        _consolidate(
            arm_dir,
            pool=pool,
            rows_path=rows_path,
            matrix_path=matrix_path,
            episodes=args.episodes,
            tip_sha=tip_sha,
        )
        return

    doc = HarnessDoc.baseline()
    expected_cells = len(task_ids) * args.episodes
    # Chunk index is the candidate's position in the FULL pool, not in the filtered --only set,
    # so a chunk file keeps naming the same model across partial reruns.
    pool_order = {entry.name: index for index, entry in enumerate(pool.models)}

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
                    _ledger_line(
                        event="retry",
                        tip_sha=tip_sha,
                        episodes=args.episodes,
                        chunk=pool_order[entry.name],
                        cells=len(harvested),
                        scored=sum(1 for row in harvested if row.scored),
                        note=(
                            f"{entry.name}: attempt {attempt + 1} failed with "
                            f"{type(exc).__name__}: {str(exc)[:200]}"
                        ),
                    ),
                )
                if attempt >= args.retries:
                    logger.error("%s exhausted %d retries, moving on", entry.name, args.retries)
                    break
                time.sleep(RETRY_DELAYS_S[min(attempt, len(RETRY_DELAYS_S) - 1)])
                attempt += 1

        rows = _harvest(job_dir, entry)
        scored = [row for row in rows if row.scored]
        spend = sum(row.cost_usd for row in rows)
        chunk = pool_order[entry.name]
        (arm_dir / f"chunk-{chunk}.json").write_text(
            json.dumps({"outcomes": [row.model_dump(mode="json") for row in rows]}, indent=2),
            encoding="utf-8",
        )
        _ledger_append(
            ledger_path,
            _ledger_line(
                event="chunk",
                tip_sha=tip_sha,
                episodes=args.episodes,
                chunk=chunk,
                cells=len(rows),
                scored=len(scored),
                candidate_usd=round(spend, 6),
                wall_s=round(time.time() - started, 1),
                cumulative_usd=round(spend, 6),
                note=(
                    f"{entry.name}: {len(scored)} scored, "
                    f"{sum(1 for row in scored if row.success)} solved, "
                    f"{expected_cells} expected"
                ),
            ),
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

    # The matrix is assembled from chunk files by --consolidate rather than after every model,
    # because several models run as separate processes (one per provider, to spread rate limits)
    # and a per-model rewrite of one shared file would race. Each process owns exactly its own
    # chunk file; consolidation is a separate, idempotent, read-only-of-chunks pass.
    _consolidate(
        arm_dir,
        pool=pool,
        rows_path=rows_path,
        matrix_path=matrix_path,
        episodes=args.episodes,
        tip_sha=tip_sha,
    )


if __name__ == "__main__":
    main()
