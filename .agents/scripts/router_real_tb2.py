"""Run the frozen nine-model Terminal-Bench 2 matrix through Harbor and E2B."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import cast

import yaml
from harbor.models.job.config import JobConfig

from wmo.agents.default import default_agent
from wmo.core.files import write_text_atomic
from wmo.evals.harbor.scorer import HarborScorer
from wmo.harness.runtime import TokenUsage as WorkerUsage
from wmo.harness.scoring import ScoreCell, ScoreReport
from wmo.optimize.outcomes import OutcomeMatrix, ScenarioOutcome
from wmo.providers.base import ProviderKind, TokenUsage
from wmo.providers.pool import PoolEntry, load_pool, pool_api_key

logger = logging.getLogger("router-real-tb2")
BENCHMARK = "terminal-bench-2"
TB2_COMMIT = "69671fbaac6d67a7ef0dfec016cc38a64ef7a77c"
MAX_ATTEMPTS = 3
RETRY_DELAYS_S = (15, 60)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one object")
    return {str(key): item for key, item in value.items()}


def _task_ids(path: Path) -> list[str]:
    raw = _json_object(path).get("tasks")
    if not isinstance(raw, list):
        raise ValueError(f"{path} has no task list")
    task_ids: list[str] = []
    for item in raw:
        if not isinstance(item, dict) or not isinstance(item.get("task_id"), str):
            raise ValueError(f"{path} contains an invalid task")
        row = cast("dict[object, object]", item)
        task_ids.append(cast("str", row["task_id"]))
    return task_ids


def _usage(cell_dir: Path) -> tuple[WorkerUsage, list[str]]:
    traces = sorted(cell_dir.rglob("wmo-run.json"))
    usage = WorkerUsage()
    for trace_path in traces:
        payload = _json_object(trace_path)
        raw = payload.get("worker_usage")
        if not isinstance(raw, dict):
            continue
        row = WorkerUsage.model_validate(raw)
        usage.input_tokens += row.input_tokens
        usage.output_tokens += row.output_tokens
        usage.cached_input_tokens += row.cached_input_tokens
        usage.cache_write_input_tokens += row.cache_write_input_tokens
        usage.reasoning_tokens += row.reasoning_tokens
        usage.calls += row.calls
        usage.call_seconds.extend(row.call_seconds)
    return usage, [str(path) for path in traces]


def _priced_usage(usage: WorkerUsage) -> TokenUsage:
    return TokenUsage(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cached_input_tokens=usage.cached_input_tokens,
        cache_write_input_tokens=usage.cache_write_input_tokens,
        reasoning_tokens=usage.reasoning_tokens,
    )


def _job_template(root: Path, jobs_dir: Path) -> JobConfig:
    path = root / "wmo/distill/configs/tb2-harbor-job-template.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a mapping")
    raw["jobs_dir"] = str(jobs_dir)
    return JobConfig.model_validate(raw)


async def _scorer(
    *,
    root: Path,
    jobs_dir: Path,
    task_ids: list[str],
    entry: PoolEntry,
    concurrency: int,
    timeout_s: float,
) -> HarborScorer:
    provider_config = entry.provider_config()
    if entry.kind is ProviderKind.AZURE_OPENAI:
        # Harbor persists its job config. The worker reads these two values from the trusted
        # standard environment so neither resolved endpoint nor deployment enters that artifact.
        provider_config = provider_config.model_copy(
            update={"endpoint": None, "deployment": None}
        )
    return await HarborScorer.create(
        _job_template(root, jobs_dir),
        task_ids,
        provider_config=provider_config,
        reward_key="reward",
        reward_mode="positive-binary",
        attempts=1,
        task_environment="e2b",
        harness_backend="e2b",
        e2b_template="",
        episode_timeout_s=timeout_s,
        agent_concurrency=concurrency,
        harbor_retries=0,
        missing_reward="zero",
    )


def _activate_entry_credentials(entry: PoolEntry) -> None:
    """Expose the selected pool route to the Harbor subprocess without persisting values."""
    if entry.kind is not ProviderKind.AZURE_OPENAI:
        return
    config = entry.provider_config()
    key = pool_api_key(entry)
    if not config.endpoint or not key:
        raise ValueError(f"{entry.name} did not resolve its Azure endpoint and key")
    os.environ["AZURE_OPENAI_ENDPOINT"] = config.endpoint
    os.environ["AZURE_OPENAI_API_KEY"] = key
    if not config.deployment:
        raise ValueError(f"{entry.name} did not resolve its Azure deployment")
    os.environ["AZURE_OPENAI_DEPLOYMENT"] = config.deployment


def _trace_task(cell: ScoreCell) -> str:
    traces = sorted(Path(cell.artifact_dir).rglob("wmo-run.json"))
    if not traces:
        return cell.task_id
    value = _json_object(traces[0]).get("instruction")
    return value if isinstance(value, str) and value else cell.task_id


def _is_tool_step(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    action = value.get("action")
    return isinstance(action, dict) and action.get("kind") == "tool_call"


def _row(cell: ScoreCell, entry: PoolEntry, attempt: int) -> dict[str, object]:
    usage, traces = _usage(Path(cell.artifact_dir))
    trace_payload = _json_object(Path(traces[0])) if traces else {}
    raw_steps = trace_payload.get("steps")
    steps = raw_steps if isinstance(raw_steps, list) else []
    tool_calls = sum(_is_tool_step(step) for step in steps)
    raw_stop = trace_payload.get("stop_reason")
    stop_reason = (
        raw_stop
        if isinstance(raw_stop, str)
        else "infrastructure_failure"
        if cell.infra_failed
        else "official_verifier"
    )
    reward = None if cell.infra_failed else cell.reward
    return {
        "scenario_id": f"{BENCHMARK}:{cell.task_id}",
        "task_id": cell.task_id,
        "task": _trace_task(cell),
        "model": entry.name,
        "attempt_number": attempt,
        "reward": reward,
        "success": bool(cell.passed and not cell.infra_failed),
        "critique": cell.note,
        "steps": len(steps),
        "tool_calls": tool_calls,
        "stop_reason": stop_reason,
        "usage": usage.model_dump(mode="json"),
        "cost_usd": entry.cost_usd(_priced_usage(usage)),
        "call_seconds": usage.call_seconds,
        "wall_seconds": 0.0,
        "completion_status": (
            "infrastructure_failure"
            if cell.infra_failed
            else "scored_pass"
            if cell.passed
            else "scored_failure"
        ),
        "failure_class": "infrastructure" if cell.infra_failed else "",
        "artifact_dir": cell.artifact_dir,
        "error": cell.note if cell.infra_failed else None,
        "remeasured": attempt > 1,
        "trace_paths": traces,
        "graded_tests": (
            cell.tests.model_dump(mode="json")
            if cell.tests is not None
            else None
        ),
    }


def _read_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path} contains a non-object row")
        rows.append({str(key): item for key, item in value.items()})
    return rows


def _number(row: dict[str, object], key: str) -> float:
    value = row.get(key, 0.0)
    return float(value) if isinstance(value, (int, float)) else 0.0


def _merge_rows(path: Path, incoming: list[dict[str, object]]) -> None:
    rows = _read_rows(path)
    keys = {
        (
            row.get("scenario_id"),
            row.get("model"),
            row.get("attempt_number"),
        )
        for row in incoming
    }
    rows = [
        row
        for row in rows
        if (row.get("scenario_id"), row.get("model"), row.get("attempt_number"))
        not in keys
    ]
    rows.extend(incoming)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomic(
        path,
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
    )


def _selected_outcomes(rows: list[dict[str, object]]) -> list[ScenarioOutcome]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        scenario_id = row.get("scenario_id")
        model = row.get("model")
        if isinstance(scenario_id, str) and isinstance(model, str):
            grouped.setdefault((scenario_id, model), []).append(row)
    outcomes: list[ScenarioOutcome] = []
    fields = set(ScenarioOutcome.model_fields)
    for attempts in grouped.values():
        ordered = sorted(attempts, key=lambda row: int(row.get("attempt_number", 0)))
        selected = next(
            (
                row
                for row in ordered
                if isinstance(row.get("reward"), (int, float))
            ),
            ordered[-1],
        )
        outcomes.append(
            ScenarioOutcome.model_validate(
                {
                    key: value
                    for key, value in selected.items()
                    if key in fields
                }
            )
        )
    return outcomes


def _persist_model_report(
    path: Path,
    *,
    entry: PoolEntry,
    scorer: HarborScorer,
    report: ScoreReport,
    attempts: int,
) -> None:
    write_text_atomic(
        path,
        json.dumps(
            {
                "model": entry.name,
                # Keep only the frozen env references. Resolved endpoints and deployments are
                # runtime configuration and must not enter durable artifacts.
                "provider": entry.model_dump(mode="json", exclude_none=True),
                "task_pins": scorer.task_pins,
                "attempts": attempts,
                "report": report.model_dump(mode="json"),
                "report_sha256_basis": "this object excluding its own hash",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )


def main() -> int:
    """Resume all selected models, retry ungradeable cells, and write the dense matrix."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--only", action="append")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--timeout-s", type=float, default=900.0)
    parser.add_argument("--budget-usd", type=float, default=2500.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    task_ids = _task_ids(args.manifest)
    pool = load_pool(args.pool)
    selected = set(args.only or [entry.name for entry in pool.models])
    unknown = selected - {entry.name for entry in pool.models}
    if unknown:
        raise ValueError(f"unknown pool models: {sorted(unknown)}")
    doc = default_agent("pi")
    rows_path = args.out_dir / "rows.jsonl"

    for entry in [item for item in pool.models if item.name in selected]:
        _activate_entry_credentials(entry)
        jobs_dir = args.out_dir / "jobs" / entry.name
        scorer = asyncio.run(
            _scorer(
                root=root,
                jobs_dir=jobs_dir,
                task_ids=task_ids,
                entry=entry,
                concurrency=args.concurrency,
                timeout_s=args.timeout_s,
            )
        )
        expected_pin = (
            f"git:https://github.com/laude-institute/terminal-bench-2.git@{TB2_COMMIT}"
        )
        if set(scorer.task_pins.values()) != {expected_pin}:
            raise ValueError(f"unexpected task pins for {entry.name}")
        if args.dry_run:
            logger.info("%s: %d tasks resolved", entry.name, len(task_ids))
            continue
        spent = sum(
            _number(row, "cost_usd")
            for row in _read_rows(rows_path)
        )
        if spent >= args.budget_usd:
            raise RuntimeError(
                f"Terminal-Bench 2 spend ${spent:.2f} reached cap ${args.budget_usd:.2f}"
            )
        final_report: ScoreReport | None = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            if attempt > 1:
                time.sleep(RETRY_DELAYS_S[attempt - 2])
            report = scorer.score(doc)
            incoming = [_row(cell, entry, attempt) for cell in report.cells]
            _merge_rows(rows_path, incoming)
            final_report = report
            infra = sum(cell.infra_failed for cell in report.cells)
            logger.info(
                "%s attempt %d: %d/%d gradeable, $%.4f cumulative",
                entry.name,
                attempt,
                len(report.cells) - infra,
                len(report.cells),
                sum(
                    _number(row, "cost_usd")
                    for row in _read_rows(rows_path)
                ),
            )
            if infra == 0:
                break
        if final_report is None:
            raise RuntimeError(f"{entry.name} produced no Harbor report")
        model_dir = args.out_dir / "reports"
        model_dir.mkdir(parents=True, exist_ok=True)
        report_path = model_dir / f"{entry.name}.json"
        _persist_model_report(
            report_path,
            entry=entry,
            scorer=scorer,
            report=final_report,
            attempts=attempt,
        )
        logger.info("%s report %s %s", entry.name, _sha256(report_path), report_path)

    outcomes = _selected_outcomes(_read_rows(rows_path))
    matrix = OutcomeMatrix(pool=pool.models, outcomes=outcomes)
    matrix.save(args.out_dir / "matrix.json")
    summary = {
        "benchmark": BENCHMARK,
        "source_commit": TB2_COMMIT,
        "tasks": len(task_ids),
        "models": len(pool.models),
        "cells_expected": len(task_ids) * len(pool.models),
        "cells_present": len(outcomes),
        "gradeable": sum(row.reward is not None for row in outcomes),
        "model_cost_usd": sum(row.cost_usd for row in outcomes),
        "environment_cost_usd": None,
        "environment_cost_note": "E2B invoice rate is not exposed in Harbor artifacts",
    }
    write_text_atomic(
        args.out_dir / "summary.json",
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
    )
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    raise SystemExit(main())
