"""Run the paid, integrated coding-router smoke through WMO Harbor and E2B.

This runner intentionally keeps the provider cells explicit. It records the logical model
identity separately from the transport surface, so Azure OpenAI and Bedrock executions remain
comparable to the direct-provider roster without claiming that a direct credential was used.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from harbor.models.environment_type import EnvironmentType
from harbor.models.job.config import DatasetConfig, JobConfig
from harbor.models.trial.config import AgentConfig, EnvironmentConfig

from wmo.agents.default import default_agent
from wmo.evals.harbor.scorer import HarborScorer
from wmo.providers.base import ProviderConfig, ProviderKind

LOGGER = logging.getLogger("coding-router-small-agent-20260729-smoke")
PREFIX = "coding-router-small-agent-20260729"
MODEL_PRICES_USD_PER_MTOK = {
    "gpt-5.5": (5.0, 30.0),
    "gpt-5.4-mini": (0.75, 3.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-sonnet-4-6": (3.0, 15.0),
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--cap-usd", type=float, default=20_000.0)
    parser.add_argument("--episode-timeout-s", type=float, default=900.0)
    parser.add_argument("--only-cell", action="append", default=[])
    parser.add_argument("--cell-suffix", default="")
    return parser


def _load_tasks(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    task_ids = payload if isinstance(payload, list) else payload.get("task_ids")
    if not isinstance(task_ids, list) or not task_ids or any(
        not isinstance(task_id, str) or not task_id for task_id in task_ids
    ):
        raise ValueError(f"invalid smoke task manifest: {path}")
    if len(task_ids) != len(set(task_ids)):
        raise ValueError(f"duplicate smoke task ids: {path}")
    return task_ids


def _provider_cells() -> list[dict[str, Any]]:
    required = {
        "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY"),
        "ANTHROPIC_API_KEY": os.environ.get("ANTHROPIC_API_KEY"),
        "E2B_API_KEY": os.environ.get("E2B_API_KEY"),
    }
    missing = sorted(name for name, value in required.items() if not value)
    if missing:
        raise RuntimeError(f"paid smoke credentials are unavailable by env name: {missing}")
    return [
        {
            "cell_id": f"{PREFIX}-direct-openai-gpt55-v1",
            "logical_provider": "openai",
            "logical_model": "gpt-5.5",
            "transport_provider": "openai",
            "config": ProviderConfig(
                kind=ProviderKind.OPENAI,
                model_type="gpt-5.5",
                model="gpt-5.5",
            ),
        },
        {
            "cell_id": f"{PREFIX}-direct-anthropic-opus48-v3",
            "logical_provider": "anthropic",
            "logical_model": "claude-opus-4-8",
            "transport_provider": "anthropic",
            "config": ProviderConfig(
                kind=ProviderKind.ANTHROPIC,
                model_type="claude-opus-4-8",
                model="claude-opus-4-8",
            ),
        },
        {
            "cell_id": f"{PREFIX}-direct-anthropic-sonnet46-v1",
            "logical_provider": "anthropic",
            "logical_model": "claude-sonnet-4-6",
            "transport_provider": "anthropic",
            "config": ProviderConfig(
                kind=ProviderKind.ANTHROPIC,
                model_type="claude-sonnet-4-6",
                model="claude-sonnet-4-6",
            ),
        },
        {
            "cell_id": f"{PREFIX}-direct-openai-gpt54-mini-v1",
            "logical_provider": "openai",
            "logical_model": "gpt-5.4-mini",
            "transport_provider": "openai",
            "config": ProviderConfig(
                kind=ProviderKind.OPENAI,
                model_type="gpt-5.5",
                model="gpt-5.4-mini",
            ),
        },
    ]


def _job_template(root: Path, cell_id: str, task_cache: Path) -> JobConfig:
    return JobConfig(
        job_name=f"{cell_id}-job",
        jobs_dir=root / "harbor-jobs" / cell_id,
        # Keep total E2B stream pressure bounded when multiple provider cells run in parallel.
        n_concurrent_trials=1,
        environment=EnvironmentConfig(type=EnvironmentType.E2B),
        datasets=[
            DatasetConfig(
                name="terminal-bench",
                version="2.0",
                download_dir=task_cache,
            )
        ],
        agents=[AgentConfig()],
    )


def _trial_records(job_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for result_path in sorted(job_dir.glob("*/result.json")):
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        trial = next(
            (item for item in payload.get("trial_results", []) if isinstance(item, dict)),
            payload,
        )
        records.append(
            {
                "trial_result": str(result_path),
                "task_id": trial.get("task_name"),
                "trial_name": trial.get("trial_name"),
                "exception_type": (trial.get("exception_info") or {}).get("exception_type"),
                "reward": ((trial.get("verifier_result") or {}).get("rewards") or {}).get(
                    "reward"
                ),
                "token_cost": _token_cost(trial),
            }
        )
    return records


def _token_cost(trial: dict[str, Any]) -> dict[str, Any]:
    contexts: list[dict[str, Any]] = []
    if isinstance(trial.get("agent_result"), dict):
        contexts.append(trial["agent_result"])
    for step in trial.get("step_results") or []:
        if isinstance(step, dict) and isinstance(step.get("agent_result"), dict):
            contexts.append(step["agent_result"])
    totals = {"input_tokens": 0, "cache_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
    seen = False
    for context in contexts:
        seen = True
        totals["input_tokens"] += int(context.get("n_input_tokens") or 0)
        totals["cache_tokens"] += int(context.get("n_cache_tokens") or 0)
        totals["output_tokens"] += int(context.get("n_output_tokens") or 0)
        totals["cost_usd"] += float(context.get("cost_usd") or 0.0)
    return totals if seen else {key: None for key in totals}


def _append_ledger(path: Path, event: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")


def _priced_cost(record: dict[str, Any], logical_model: str) -> float:
    token_cost = record.get("token_cost") or {}
    reported = token_cost.get("cost_usd")
    if reported is not None and float(reported) > 0:
        return float(reported)
    prices = MODEL_PRICES_USD_PER_MTOK.get(logical_model)
    if prices is None:
        return float(reported or 0.0)
    input_tokens = int(token_cost.get("input_tokens") or 0)
    output_tokens = int(token_cost.get("output_tokens") or 0)
    return input_tokens * prices[0] / 1_000_000 + output_tokens * prices[1] / 1_000_000


def main() -> None:
    args = _parser().parse_args()
    root = args.artifact_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    tasks = _load_tasks(args.tasks.resolve())
    cells = _provider_cells()
    if args.cell_suffix:
        for cell in cells:
            cell["cell_id"] = f"{cell['cell_id']}{args.cell_suffix}"
    if args.only_cell:
        cells = [cell for cell in cells if cell["cell_id"] in set(args.only_cell)]
        if not cells:
            raise ValueError("none of --only-cell values matched the configured smoke cells")
    ledger = root / "spend-ledger.jsonl"
    _append_ledger(
        ledger,
        {
            "event": "smoke-authorized",
            "experiment": PREFIX,
            "cap_usd": args.cap_usd,
            "stage": "integrated-terminal-bench-smoke",
            "task_ids": tasks,
            "cells": [cell["cell_id"] for cell in cells],
            "timestamp": datetime.now(UTC).isoformat(),
        },
    )
    doc = default_agent("pi")
    for cell in cells:
        cell_id = str(cell["cell_id"])
        started = time.monotonic()
        LOGGER.info("starting paid cell=%s tasks=%s", cell_id, tasks)
        template = _job_template(root, cell_id, root / "task-cache" / cell_id)
        scorer = asyncio.run(
            HarborScorer.create(
                template,
                tasks,
                provider_config=cell["config"],
                task_environment="e2b",
                harness_backend="e2b",
                reward_key="reward",
                reward_mode="raw",
                attempts=1,
                harbor_retries=0,
                missing_reward="zero",
                agent_model_name=f"{cell['config'].kind.value}/{cell['config'].model}",
                episode_timeout_s=args.episode_timeout_s,
            )
        )
        report = scorer.score(doc)
        job_dir = scorer.candidate_job_dir(doc).resolve()
        records = _trial_records(job_dir)
        cost = sum(_priced_cost(record, str(cell["logical_model"])) for record in records)
        event = {
            "event": "smoke-cell-complete",
            "experiment": PREFIX,
            "cell_id": cell_id,
            "logical_provider": cell["logical_provider"],
            "logical_model": cell["logical_model"],
            "transport_provider": cell["transport_provider"],
            "task_ids": tasks,
            "task_pins": scorer.task_pins,
            "doc_hash": doc.doc_hash,
            "score": report.score,
            "pass_rate": report.pass_rate,
            "cells": [cell.model_dump(mode="json") for cell in report.cells],
            "trial_records": records,
            "job_dir": str(job_dir),
            "cost_usd": cost,
            "elapsed_s": round(time.monotonic() - started, 3),
            "timestamp": datetime.now(UTC).isoformat(),
        }
        _append_ledger(ledger, {"event": "smoke-cell-cost", "cell_id": cell_id, "cost_usd": cost})
        (root / f"{cell_id}-outcome.json").write_text(
            json.dumps(event, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        LOGGER.info("completed paid cell=%s score=%.4f cost_usd=%.6f", cell_id, report.score, cost)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
