"""Run the frozen coding-router outcome matrix with resumable, budgeted cells."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shutil
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from coding_model_router_usage import exact_cost_usd, usage_from_trace
from e2b import Sandbox
from harbor.models.job.config import JobConfig
from harbor.models.trial.result import TrialResult

from wmo.agents.default import default_agent
from wmo.core.files import write_text_atomic
from wmo.evals.harbor.scorer import HarborScorer
from wmo.harness.scoring import ScoreCell
from wmo.optimize.outcomes import OutcomeMatrix, ScenarioOutcome
from wmo.providers.pool import ModelPool, PoolEntry, load_pool

logger = logging.getLogger("coding-model-router-matrix")

EXPERIMENT_ID = "coding-router-20260728"
BENCHMARKS = ("terminal-bench-2", "swe-bench-verified")
MAX_LOGICAL_ATTEMPTS = 3
RETRY_DELAYS_S = (15, 60)
# One in-flight cell can use 20 calls with up to 4,096 output tokens each. Reserving $500 is
# deliberately conservative for a one-million-token context on the most expensive frozen arm,
# including cache writes, and prevents concurrency from overshooting the operator's cap before
# realized usage is persisted.
CELL_SPEND_RESERVATION_USD = 500.0
E2B_ACCOUNT_CAP = 100
SCAFFOLD_STOPS = frozenset(
    {
        "budget",
        "error",
        "max_turns",
        "no_action",
        "no_tool_call",
        "output_truncated",
        "provider_error",
        "unknown_done_reason",
        "unparsed_tool_call",
    }
)


class BudgetExhausted(RuntimeError):
    """The next paid cell cannot fit under the frozen ceiling."""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _read_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return {str(key): item for key, item in value.items()}


def _write_json(path: Path, value: object) -> None:
    write_text_atomic(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _manifest_tasks(path: Path) -> list[dict[str, object]]:
    raw = _read_object(path).get("tasks")
    if not isinstance(raw, list):
        raise ValueError(f"{path} has no task list")
    tasks: list[dict[str, object]] = []
    for item in raw:
        if not isinstance(item, dict) or not isinstance(item.get("task_id"), str):
            raise ValueError(f"{path} contains an invalid task row")
        tasks.append({str(key): value for key, value in item.items()})
    return tasks


def _task_ids(path: Path) -> list[str]:
    return [cast("str", row["task_id"]) for row in _manifest_tasks(path)]


def _expected_pins(path: Path, benchmark: str) -> dict[str, str]:
    pins: dict[str, str] = {}
    for row in _manifest_tasks(path):
        task_id = cast("str", row["task_id"])
        if benchmark == "terminal-bench-2":
            git_url = row.get("git_url")
            commit = row.get("git_commit_id")
        else:
            git_url = row.get("harbor_git_url")
            commit = row.get("harbor_git_commit_id")
        if not isinstance(git_url, str) or not isinstance(commit, str):
            raise ValueError(f"{path} has no Harbor execution pin for {task_id}")
        pins[task_id] = f"git:{git_url}@{commit}"
    return pins


def _job_template(benchmark: str, jobs_dir: Path) -> JobConfig:
    dataset = (
        {"name": "terminal-bench", "version": "2.0"}
        if benchmark == "terminal-bench-2"
        else {"name": "swebench-verified", "version": "1.0"}
    )
    return JobConfig.model_validate(
        {
            "job_name": f"{EXPERIMENT_ID}-{benchmark}",
            "jobs_dir": str(jobs_dir),
            "n_concurrent_trials": 1,
            "environment": {"type": "e2b"},
            "datasets": [dataset],
            "agents": [{}],
        }
    )


async def _scorer(
    *,
    benchmark: str,
    jobs_dir: Path,
    task_ids: list[str],
    entry: PoolEntry,
    timeout_s: float,
) -> HarborScorer:
    return await HarborScorer.create(
        _job_template(benchmark, jobs_dir),
        task_ids,
        provider_config=entry.provider_config(),
        attempts=1,
        task_environment="e2b",
        harness_backend="local",
        episode_timeout_s=timeout_s,
        agent_concurrency=1,
        harbor_retries=0,
        missing_reward="zero",
    )


def _trace_path(artifact_dir: Path) -> Path:
    for candidate in (
        artifact_dir / "agent" / "wmo-run.json",
        artifact_dir / "wmo-run.json",
    ):
        if candidate.is_file():
            return candidate
    matches = sorted(artifact_dir.rglob("wmo-run.json"))
    if matches:
        return matches[0]
    raise ValueError(f"no wmo-run.json under Harbor artifact {artifact_dir}")


def _float(value: object) -> float:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0.0


def _tool_calls(trace: dict[str, object]) -> int:
    steps = trace.get("steps")
    if not isinstance(steps, list):
        return 0
    count = 0
    for step in steps:
        if not isinstance(step, dict):
            continue
        action = step.get("action")
        if isinstance(action, dict) and action.get("kind") == "tool_call":
            count += 1
    return count


def _wall_seconds(artifact_dir: Path) -> float:
    result_path = artifact_dir / "result.json"
    if not result_path.is_file():
        return 0.0
    result = TrialResult.model_validate_json(result_path.read_text(encoding="utf-8"))
    if result.started_at is None or result.finished_at is None:
        return 0.0
    return max(0.0, (result.finished_at - result.started_at).total_seconds())


def _failure_class(cell: ScoreCell, stop_reason: str) -> str:
    if cell.infra_failed:
        return "infrastructure"
    if stop_reason in SCAFFOLD_STOPS or stop_reason.startswith("agent-exception:"):
        return "scaffold"
    return "" if cell.reward == 1.0 else "task_failure"


def _outcome(
    cell: ScoreCell,
    *,
    benchmark: str,
    entry: PoolEntry,
    attempt: int,
    artifact_dir: Path,
) -> ScenarioOutcome:
    trace = _read_object(_trace_path(artifact_dir))
    usage = usage_from_trace(trace)
    steps = trace.get("steps")
    instruction = trace.get("instruction")
    stop = trace.get("stop_reason")
    stop_reason = stop if isinstance(stop, str) else ""
    return ScenarioOutcome(
        scenario_id=f"{benchmark}:{cell.task_id}",
        task=instruction if isinstance(instruction, str) else cell.task_id,
        model=entry.name,
        benchmark=benchmark,
        episode=attempt - 1,
        attempt_number=attempt,
        reward=None if cell.infra_failed else cell.reward,
        success=cell.passed and not cell.infra_failed,
        critique=cell.note,
        steps=len(steps) if isinstance(steps, list) else 0,
        tool_calls=_tool_calls(trace),
        stop_reason=stop_reason,
        usage=usage.total,
        cost_usd=exact_cost_usd(entry, usage),
        call_seconds=usage.call_seconds,
        call_input_tokens=usage.call_input_tokens,
        call_output_tokens=usage.call_output_tokens,
        call_cached_input_tokens=usage.call_cached_input_tokens,
        call_cache_write_input_tokens=usage.call_cache_write_input_tokens,
        wall_seconds=_wall_seconds(artifact_dir),
        completion_status=(
            "infrastructure_failure"
            if cell.infra_failed
            else "scored_pass"
            if cell.passed
            else "scored_failure"
        ),
        failure_class=_failure_class(cell, stop_reason),
        artifact_dir=str(artifact_dir.resolve()),
        error=(cell.note or None) if cell.infra_failed else None,
        remeasured=attempt > 1,
    )


class RunState:
    """One process's atomic matrix, ledger, and budget owner."""

    def __init__(
        self,
        *,
        root: Path,
        pool: ModelPool,
        ceiling_usd: float,
    ) -> None:
        self.root = root
        self.pool = pool
        self.ceiling_usd = ceiling_usd
        self.matrix_path = root / "outcomes.json"
        self.ledger_path = root.parent / "spend-ledger.jsonl"
        self.lock = threading.Lock()
        self.matrix = (
            OutcomeMatrix.load(self.matrix_path)
            if self.matrix_path.is_file()
            else OutcomeMatrix(pool=pool.models, outcomes=[])
        )
        if self.matrix.pool != pool.models:
            raise ValueError("the existing full matrix carries a different frozen pool")
        self.ledger = self._read_ledger()

    def _read_ledger(self) -> list[dict[str, object]]:
        if not self.ledger_path.is_file():
            return []
        rows: list[dict[str, object]] = []
        for line in self.ledger_path.read_text(encoding="utf-8").splitlines():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{self.ledger_path} contains a non-object row")
            rows.append({str(key): item for key, item in value.items()})
        return rows

    def _write_ledger(self) -> None:
        write_text_atomic(
            self.ledger_path,
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in self.ledger),
        )

    def _upsert_ledger(self, row: dict[str, object]) -> None:
        event_id = row["event_id"]
        self.ledger = [item for item in self.ledger if item.get("event_id") != event_id]
        self.ledger.append(row)
        self._write_ledger()

    def spent_and_reserved(self) -> tuple[float, float]:
        spent = 0.0
        reserved = 0.0
        for row in self.ledger:
            status = row.get("status")
            if status == "reserved":
                reserved += _float(row.get("reserved_usd"))
            elif status == "completed" or status is None:
                spent += _float(row.get("model_cost_usd"))
        return spent, reserved

    def completed(self, benchmark: str, task_id: str, arm: str) -> bool:
        scenario_id = f"{benchmark}:{task_id}"
        return any(
            row.scenario_id == scenario_id and row.model == arm and row.reward is not None
            for row in self.matrix.outcomes
        )

    def attempts(self, benchmark: str, task_id: str, arm: str) -> int:
        scenario_id = f"{benchmark}:{task_id}"
        return sum(
            row.scenario_id == scenario_id and row.model == arm for row in self.matrix.outcomes
        )

    def reserve(self, benchmark: str, task_id: str, arm: str, attempt: int) -> str:
        event_id = f"full:{benchmark}:{task_id}:{arm}:{attempt}"
        with self.lock:
            existing = next(
                (row for row in self.ledger if row.get("event_id") == event_id),
                None,
            )
            if existing is not None:
                if existing.get("status") == "reserved":
                    raise RuntimeError(
                        f"{event_id} has an unresolved reservation from an interrupted run"
                    )
                return event_id
            spent, reserved = self.spent_and_reserved()
            projected = spent + reserved + CELL_SPEND_RESERVATION_USD
            if projected > self.ceiling_usd:
                raise BudgetExhausted(
                    f"next cell reservation would reach ${projected:.2f}, "
                    f"above frozen ceiling ${self.ceiling_usd:.2f}"
                )
            self._upsert_ledger(
                {
                    "event_id": event_id,
                    "recorded_at": _utc_now(),
                    "phase": "full-matrix",
                    "benchmark": benchmark,
                    "scenario_id": f"{benchmark}:{task_id}",
                    "model": arm,
                    "attempt_number": attempt,
                    "status": "reserved",
                    "reserved_usd": CELL_SPEND_RESERVATION_USD,
                }
            )
        return event_id

    def persist(self, event_id: str, outcome: ScenarioOutcome) -> None:
        with self.lock:
            key = (outcome.scenario_id, outcome.model, outcome.attempt_number)
            self.matrix.outcomes = [
                row
                for row in self.matrix.outcomes
                if (row.scenario_id, row.model, row.attempt_number) != key
            ] + [outcome]
            self.matrix.save(self.matrix_path)
            self._upsert_ledger(
                {
                    "event_id": event_id,
                    "recorded_at": _utc_now(),
                    "phase": "full-matrix",
                    "benchmark": outcome.benchmark,
                    "scenario_id": outcome.scenario_id,
                    "model": outcome.model,
                    "attempt_number": outcome.attempt_number,
                    "status": "completed",
                    "reserved_usd": CELL_SPEND_RESERVATION_USD,
                    "usage": outcome.usage.model_dump(mode="json"),
                    "model_call_seconds": outcome.call_seconds,
                    "task_environment_wall_seconds": outcome.wall_seconds,
                    "model_cost_usd": outcome.cost_usd,
                    "task_environment_cost_usd": None,
                    "task_environment_cost_note": (
                        "E2B invoice rate is absent from Harbor artifacts"
                    ),
                    "completion_status": outcome.completion_status,
                    "failure_class": outcome.failure_class,
                    "artifact_dir": outcome.artifact_dir,
                }
            )


def _archive_infra(
    root: Path,
    artifact_dir: Path,
    *,
    benchmark: str,
    task_id: str,
    arm: str,
    attempt: int,
) -> Path:
    target = root / "infra-attempts" / benchmark / arm / task_id / f"attempt-{attempt}"
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(artifact_dir, target)
    return target


def _run_cell(
    state: RunState,
    *,
    benchmark: str,
    task_id: str,
    entry: PoolEntry,
    timeout_s: float,
) -> ScenarioOutcome | None:
    with state.lock:
        if state.completed(benchmark, task_id, entry.name):
            return None
        attempt = state.attempts(benchmark, task_id, entry.name) + 1
    while attempt <= MAX_LOGICAL_ATTEMPTS:
        event_id = state.reserve(benchmark, task_id, entry.name, attempt)
        scorer = asyncio.run(
            _scorer(
                benchmark=benchmark,
                jobs_dir=state.root / "harbor" / benchmark / entry.name / task_id,
                task_ids=[task_id],
                entry=entry,
                timeout_s=timeout_s,
            )
        )
        cell = scorer.score(default_agent("coding-router-full")).cells[0]
        artifact_dir = Path(cell.artifact_dir)
        if cell.infra_failed:
            artifact_dir = _archive_infra(
                state.root,
                artifact_dir,
                benchmark=benchmark,
                task_id=task_id,
                arm=entry.name,
                attempt=attempt,
            )
        outcome = _outcome(
            cell,
            benchmark=benchmark,
            entry=entry,
            attempt=attempt,
            artifact_dir=artifact_dir,
        )
        state.persist(event_id, outcome)
        logger.info(
            "persisted %s x %s x %s attempt %d reward=%s cost=$%.6f",
            benchmark,
            task_id,
            entry.name,
            attempt,
            outcome.reward,
            outcome.cost_usd,
        )
        if not cell.infra_failed:
            return outcome
        if attempt >= MAX_LOGICAL_ATTEMPTS:
            return outcome
        time.sleep(RETRY_DELAYS_S[attempt - 1])
        attempt += 1
    return None


def _preflight(root: Path, pool: ModelPool) -> None:
    resolved: dict[str, object] = {}
    for entry in pool.models:
        resolved[entry.name] = entry.provider_config().model_dump(mode="json")
    for benchmark in BENCHMARKS:
        manifest = root / "tasks" / f"{benchmark}.json"
        task_ids = _task_ids(manifest)
        scorer = asyncio.run(
            _scorer(
                benchmark=benchmark,
                jobs_dir=root / "full" / "preflight" / benchmark,
                task_ids=task_ids,
                entry=pool.models[0],
                timeout_s=300.0,
            )
        )
        expected = _expected_pins(manifest, benchmark)
        if scorer.task_pins != expected:
            mismatches = sorted(
                task_id
                for task_id in set(expected) | set(scorer.task_pins)
                if expected.get(task_id) != scorer.task_pins.get(task_id)
            )
            raise ValueError(f"{benchmark} execution pins differ for {mismatches[:5]}")
        resolved[benchmark] = {
            "tasks": len(task_ids),
            "execution_pins": len(scorer.task_pins),
        }
    _write_json(
        root / "full" / "preflight.json",
        {
            "verified_at": _utc_now(),
            "paid_calls": 0,
            "cells": sum(
                len(_task_ids(root / "tasks" / f"{benchmark}.json")) for benchmark in BENCHMARKS
            )
            * len(pool.models),
            "cell_spend_reservation_usd": CELL_SPEND_RESERVATION_USD,
            "resolved": resolved,
        },
    )


def _run(
    root: Path,
    *,
    pool: ModelPool,
    ceiling_usd: float,
    concurrency: int,
    timeout_s: float,
) -> None:
    for variable in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "E2B_API_KEY"):
        if not os.environ.get(variable):
            raise ValueError(f"{variable} is required for the paid matrix")
    full_root = root / "full"
    full_root.mkdir(parents=True, exist_ok=True)
    paginator = Sandbox.list(limit=E2B_ACCOUNT_CAP)
    running = paginator.next_items()
    if len(running) >= E2B_ACCOUNT_CAP or paginator.has_next:
        raise RuntimeError(
            f"E2B has at least {len(running)} active sandboxes against the frozen "
            f"{E2B_ACCOUNT_CAP}-sandbox account cap"
        )
    state = RunState(root=full_root, pool=pool, ceiling_usd=ceiling_usd)
    cells = [
        (benchmark, task_id, entry)
        for benchmark in BENCHMARKS
        for entry in pool.models
        for task_id in _task_ids(root / "tasks" / f"{benchmark}.json")
        if not state.completed(benchmark, task_id, entry.name)
    ]
    cell_iterator = iter(cells)
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures: dict[Future[ScenarioOutcome | None], tuple[str, str, str]] = {}

        def submit_next() -> bool:
            try:
                benchmark, task_id, entry = next(cell_iterator)
            except StopIteration:
                return False
            future = executor.submit(
                _run_cell,
                state,
                benchmark=benchmark,
                task_id=task_id,
                entry=entry,
                timeout_s=timeout_s,
            )
            futures[future] = (benchmark, task_id, entry.name)
            return True

        for _ in range(concurrency):
            if not submit_next():
                break
        budget_exhausted = False
        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                benchmark, task_id, arm = futures.pop(future)
                try:
                    future.result()
                except BudgetExhausted as error:
                    logger.warning("%s", error)
                    budget_exhausted = True
                except Exception:
                    logger.exception(
                        "cell failed unexpectedly: %s x %s x %s",
                        benchmark,
                        task_id,
                        arm,
                    )
                    raise
                if not budget_exhausted:
                    submit_next()
    spent, reserved = state.spent_and_reserved()
    _write_json(
        full_root / "summary.json",
        {
            "updated_at": _utc_now(),
            "cells_expected": len(pool.models)
            * sum(len(_task_ids(root / "tasks" / f"{benchmark}.json")) for benchmark in BENCHMARKS),
            "attempt_rows": len(state.matrix.outcomes),
            "gradeable_cells": sum(row.reward is not None for row in state.matrix.outcomes),
            "model_spend_usd": spent,
            "outstanding_reservations_usd": reserved,
            "spend_ceiling_usd": ceiling_usd,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(".wmo/experiments") / EXPERIMENT_ID,
    )
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout-s", type=float, default=900.0)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.concurrency < 1:
        raise ValueError("--concurrency must be at least 1")
    freeze = _read_object(args.root / "freeze-summary.json")
    pool = load_pool(args.root / "pool.toml")
    if args.preflight:
        _preflight(args.root, pool)
        return
    ceiling = freeze.get("spend_ceiling_usd")
    if not isinstance(ceiling, (int, float)) or isinstance(ceiling, bool) or ceiling <= 0:
        raise ValueError(
            "material paid sweep is disabled: record a positive user-authorized "
            "spend_ceiling_usd in freeze-summary.json"
        )
    _run(
        args.root,
        pool=pool,
        ceiling_usd=float(ceiling),
        concurrency=args.concurrency,
        timeout_s=args.timeout_s,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main()
