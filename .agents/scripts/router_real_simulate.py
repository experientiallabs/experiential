"""Run the frozen candidate roster inside an Azure GPT-5.5 WMO environment."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from wmo.core.files import write_text_atomic
from wmo.core.types import Step, Trace
from wmo.engine import load_world_model
from wmo.env import WorldModelEnv
from wmo.env.scenarios import Scenario, tools_hint_from_traces
from wmo.optimize.outcomes import OutcomeMatrix, ScenarioOutcome
from wmo.optimize.sweep import CostLine, SweepPlan, execute_sweep
from wmo.providers.base import TokenUsage
from wmo.providers.pool import ModelPool, PoolEntry, load_pool, pool_api_key

logger = logging.getLogger("router-real-simulate")


def _json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _tasks(path: Path) -> list[str]:
    value = _json(path)
    if not isinstance(value, dict) or not isinstance(value.get("tasks"), list):
        raise ValueError(f"{path} has no task manifest")
    task_ids = []
    raw_tasks = value.get("tasks")
    if not isinstance(raw_tasks, list):
        raise ValueError(f"{path} has no task list")
    for row in raw_tasks:
        task_id = row.get("task_id") if isinstance(row, dict) else None
        if not isinstance(task_id, str):
            raise ValueError(f"{path} contains an invalid task row")
        task_ids.append(task_id)
    return task_ids


def _canonical_id(benchmark: str, scenario_id: str) -> str:
    if benchmark == "tau2":
        domain, separator, task_id = scenario_id.partition(":")
        return f"{domain}/{task_id}" if separator else scenario_id
    return scenario_id


def _scenarios(
    benchmark: str,
    matrix: OutcomeMatrix,
    task_ids: list[str],
) -> tuple[tuple[Scenario, ...], dict[str, str]]:
    by_id = {}
    for row in matrix.outcomes:
        by_id.setdefault(_canonical_id(benchmark, row.scenario_id), row.task)
    missing = [task_id for task_id in task_ids if task_id not in by_id]
    if missing:
        raise ValueError(f"real matrix lacks {len(missing)} frozen tasks: {missing[:3]}")
    task_to_id = {by_id[task_id]: task_id for task_id in task_ids}
    if len(task_to_id) != len(task_ids):
        raise ValueError("two frozen task ids have identical task text; mapping would be ambiguous")
    return (
        tuple(
            Scenario(task=by_id[task_id], provenance=[f"frozen:{task_id}"])
            for task_id in task_ids
        ),
        task_to_id,
    )


def _tools_hint(model_dir: Path) -> str | None:
    steps_path = model_dir / "index" / "steps.jsonl"
    steps = [
        Step.model_validate_json(line)
        for line in steps_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    hint = tools_hint_from_traces([Trace(trace_id="training-side-index", steps=steps)])
    return hint or None


def _cost_lines(
    pool: ModelPool,
    *,
    scenarios: int,
    max_steps: int,
    input_tokens: int,
    output_tokens: int,
) -> tuple[CostLine, ...]:
    calls = scenarios * max_steps
    lines = []
    for entry in pool.models:
        usage = TokenUsage(
            input_tokens=input_tokens * calls,
            output_tokens=output_tokens * calls,
        )
        price = entry.price()
        lines.append(
            CostLine(
                candidate=entry.name,
                episodes=scenarios,
                calls=calls,
                input_per_mtok=price.input_per_mtok,
                output_per_mtok=price.output_per_mtok,
                usd=entry.cost_usd(usage),
            )
        )
    return tuple(lines)


def _activate_world_model(pool: ModelPool) -> None:
    entry = pool.entry("gpt-5.5")
    config = entry.provider_config()
    key = pool_api_key(entry)
    if not config.endpoint or not config.deployment or not key:
        raise ValueError("gpt-5.5 pool route did not resolve endpoint, deployment, and key")
    os.environ["AZURE_OPENAI_ENDPOINT"] = config.endpoint
    os.environ["AZURE_OPENAI_DEPLOYMENT"] = config.deployment
    os.environ["AZURE_OPENAI_API_KEY"] = key


def _append_ledger(
    path: Path,
    *,
    benchmark: str,
    out_dir: Path,
    matrix: OutcomeMatrix,
    candidate_usd_by_model: dict[str, float],
    world_model_usd: float,
    world_model_usage_paths: list[Path],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for model in matrix.model_names():
        rows = [row for row in matrix.outcomes if row.model == model]
        lines.append(
            {
                "phase": "world-model",
                "benchmark": benchmark,
                "provider_role": "candidate",
                "model": model,
                "run": str(out_dir),
                "realized_usd": candidate_usd_by_model[model],
                "cells": len(rows),
                "source": "all simulated attempts, including infrastructure retries",
            }
        )
    lines.append(
        {
            "phase": "world-model",
            "benchmark": benchmark,
            "provider_role": "world-model-serve-and-judge",
            "model": "azure-gpt-5.5",
            "run": str(out_dir),
            "realized_usd": world_model_usd,
            "cells": len(matrix.outcomes),
            "source": (
                [str(path) for path in world_model_usage_paths]
                if world_model_usage_paths
                else "unavailable"
            ),
        }
    )
    with path.open("a", encoding="utf-8") as handle:
        for line in lines:
            handle.write(json.dumps(line, sort_keys=True) + "\n")
            handle.flush()


def _retry_plan(
    plan: SweepPlan,
    *,
    entry: PoolEntry,
    scenarios: tuple[Scenario, ...],
    out_path: Path,
) -> SweepPlan:
    pool = ModelPool(models=[entry])
    return plan.model_copy(
        update={
            "out_path": out_path,
            "pool": pool,
            "scenarios": scenarios,
            "cost_lines": _cost_lines(
                pool,
                scenarios=len(scenarios),
                max_steps=plan.max_steps,
                input_tokens=plan.assume_input_tokens,
                output_tokens=plan.assume_output_tokens,
            ),
        }
    )


def _canonical_rows(
    matrix: OutcomeMatrix,
    *,
    task_to_id: dict[str, str],
    attempt_number: int,
) -> list[ScenarioOutcome]:
    return [
        row.model_copy(
            update={
                "scenario_id": task_to_id[row.task],
                "attempt_number": attempt_number,
                "remeasured": attempt_number > 1,
            }
        )
        for row in matrix.outcomes
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=("tau2", "terminal_bench_2"), required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--real-matrix", type=Path, required=True)
    parser.add_argument("--task-manifest", type=Path, required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--spend-ledger", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--budget-usd", type=float, default=5000.0)
    args = parser.parse_args()

    matrix_path = args.out_dir / "matrix.json"
    if matrix_path.is_file():
        logger.info("simulated matrix already complete: %s", matrix_path)
        return 0
    pool = load_pool(args.pool)
    real_matrix = OutcomeMatrix.load(args.real_matrix)
    task_ids = _tasks(args.task_manifest)
    scenarios, task_to_id = _scenarios(args.benchmark, real_matrix, task_ids)
    max_steps = 100 if args.benchmark == "tau2" else 20
    raw_path = args.out_dir / "matrix-hashed.json"
    plan = SweepPlan(
        model_dir=args.model_dir,
        out_path=raw_path,
        pool=pool,
        scenarios=scenarios,
        episodes=1,
        max_steps=max_steps,
        tools_hint=_tools_hint(args.model_dir),
        history_chars=2000,
        max_concurrency=args.concurrency,
        trace_count=int(
            json.loads((args.model_dir / "provenance.json").read_text())["trace_count"]
        ),
        tiny_corpus=False,
        assume_input_tokens=2000,
        assume_output_tokens=250,
        cost_lines=_cost_lines(
            pool,
            scenarios=len(scenarios),
            max_steps=max_steps,
            input_tokens=2000,
            output_tokens=250,
        ),
    )
    if plan.total_usd > args.budget_usd:
        raise RuntimeError(
            f"candidate-side projection ${plan.total_usd:.2f} exceeds cap "
            f"${args.budget_usd:.2f}"
        )
    _activate_world_model(pool)
    world_model, _provider = load_world_model(
        args.model_dir, telemetry_root=args.out_dir / "telemetry"
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    run = execute_sweep(
        plan,
        world_model=world_model,
        env_factory=lambda: WorldModelEnv(world_model, score_on_close=True),
        on_outcome=lambda row: logger.info(
            "%s / %s: %s",
            row.model,
            task_to_id[row.task],
            "unscored" if row.reward is None else f"{row.reward:.3f}",
        ),
        runs_dir=args.out_dir / "runs",
    )
    canonical_rows = _canonical_rows(
        run.matrix, task_to_id=task_to_id, attempt_number=1
    )
    all_attempts = list(canonical_rows)
    candidate_usd_by_model = {
        model: sum(row.cost_usd for row in canonical_rows if row.model == model)
        for model in run.matrix.model_names()
    }
    world_model_usd = run.world_model_usd
    world_model_usage_paths = [run.usage_path] if run.usage_path else []
    world_model_metering_gaps = [run.metering_gap] if run.metering_gap else []
    scenario_by_task = {scenario.task: scenario for scenario in scenarios}
    current = {
        (row.model, row.scenario_id, row.episode): row for row in canonical_rows
    }
    retry_runs = 0
    for attempt_number in (2, 3):
        unscored = [row for row in current.values() if row.reward is None]
        if not unscored:
            break
        for model in sorted({row.model for row in unscored}):
            model_rows = [row for row in unscored if row.model == model]
            retry_scenarios = tuple(scenario_by_task[row.task] for row in model_rows)
            retry_dir = args.out_dir / "retries" / f"attempt-{attempt_number}" / model
            retry_plan = _retry_plan(
                plan,
                entry=pool.entry(model),
                scenarios=retry_scenarios,
                out_path=retry_dir / "matrix-hashed.json",
            )
            retry_run = execute_sweep(
                retry_plan,
                world_model=world_model,
                env_factory=lambda: WorldModelEnv(world_model, score_on_close=True),
                on_outcome=lambda row, attempt_number=attempt_number: logger.info(
                    "retry %d %s / %s: %s",
                    attempt_number,
                    row.model,
                    task_to_id[row.task],
                    "unscored" if row.reward is None else f"{row.reward:.3f}",
                ),
                runs_dir=retry_dir / "runs",
            )
            retry_runs += 1
            rows = _canonical_rows(
                retry_run.matrix,
                task_to_id=task_to_id,
                attempt_number=attempt_number,
            )
            OutcomeMatrix(pool=[pool.entry(model)], outcomes=rows).save(
                retry_dir / "matrix.json"
            )
            all_attempts.extend(rows)
            candidate_usd_by_model[model] += retry_run.candidate_usd
            world_model_usd += retry_run.world_model_usd
            if retry_run.usage_path:
                world_model_usage_paths.append(retry_run.usage_path)
            if retry_run.metering_gap:
                world_model_metering_gaps.append(retry_run.metering_gap)
            for row in rows:
                current[(row.model, row.scenario_id, row.episode)] = row
    write_text_atomic(
        args.out_dir / "attempts.jsonl",
        "".join(row.model_dump_json() + "\n" for row in all_attempts),
    )
    canonical_rows = [
        current[(row.model, row.scenario_id, row.episode)] for row in canonical_rows
    ]
    matrix = OutcomeMatrix(pool=run.matrix.pool, outcomes=canonical_rows)
    matrix.save(matrix_path)
    summary = {
        "benchmark": args.benchmark,
        "phase": "world-model",
        "world_model": "azure-gpt-5.5",
        "tasks": len(scenarios),
        "models": len(pool.models),
        "cells": len(matrix.outcomes),
        "gradeable": sum(row.reward is not None for row in matrix.outcomes),
        "attempt_rows": len(all_attempts),
        "retry_runs": retry_runs,
        "candidate_cost_usd": sum(candidate_usd_by_model.values()),
        "candidate_cost_usd_by_model": candidate_usd_by_model,
        "world_model_cost_usd": world_model_usd,
        "world_model_metering_gaps": world_model_metering_gaps,
        "world_model_usage_paths": [str(path) for path in world_model_usage_paths],
    }
    write_text_atomic(
        args.out_dir / "summary.json",
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
    )
    _append_ledger(
        args.spend_ledger,
        benchmark=args.benchmark,
        out_dir=args.out_dir,
        matrix=matrix,
        candidate_usd_by_model=candidate_usd_by_model,
        world_model_usd=world_model_usd,
        world_model_usage_paths=world_model_usage_paths,
    )
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    raise SystemExit(main())
