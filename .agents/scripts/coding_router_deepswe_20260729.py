"""Run direct-provider WMO cells against the local DeepSWE v1.1 Harbor corpus."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
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

LOGGER = logging.getLogger("coding-router-deepswe-20260729")
PREFIX = "coding-router-deepswe-20260729"
MODEL_PRICES = {
    "gpt-5.5": (5.0, 30.0),
    "gpt-5.4-mini": (0.75, 3.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-sonnet-4-6": (3.0, 15.0),
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--task-id", action="append", default=[])
    parser.add_argument("--n-tasks", type=int)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--only-model", action="append", default=[])
    parser.add_argument(
        "--reasoning-effort",
        action="append",
        choices=("low", "medium", "high"),
        help="Run GPT models through the Responses API at these reasoning efforts.",
    )
    parser.add_argument("--cell-suffix", default="")
    parser.add_argument("--episode-timeout-s", type=float, default=5400.0)
    parser.add_argument("--cap-usd", type=float, default=20_000.0)
    return parser


def _cells() -> list[dict[str, Any]]:
    required = {
        "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY"),
        "ANTHROPIC_API_KEY": os.environ.get("ANTHROPIC_API_KEY"),
        "E2B_API_KEY": os.environ.get("E2B_API_KEY"),
    }
    missing = sorted(name for name, value in required.items() if not value)
    if missing:
        raise RuntimeError(f"DeepSWE credentials unavailable by env name: {missing}")
    return [
        {
            "cell_id": f"{PREFIX}-direct-openai-gpt55-v1",
            "logical_provider": "openai",
            "logical_model": "gpt-5.5",
            "config": ProviderConfig(
                kind=ProviderKind.OPENAI,
                model_type="gpt-5.5",
                model="gpt-5.5",
            ),
        },
        {
            "cell_id": f"{PREFIX}-direct-openai-gpt54-mini-v1",
            "logical_provider": "openai",
            "logical_model": "gpt-5.4-mini",
            "config": ProviderConfig(
                kind=ProviderKind.OPENAI,
                model_type="gpt-5.5",
                model="gpt-5.4-mini",
            ),
        },
        {
            "cell_id": f"{PREFIX}-direct-anthropic-opus48-v1",
            "logical_provider": "anthropic",
            "logical_model": "claude-opus-4-8",
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
            "config": ProviderConfig(
                kind=ProviderKind.ANTHROPIC,
                model_type="claude-sonnet-4-6",
                model="claude-sonnet-4-6",
            ),
        },
    ]


def _reasoning_cells(efforts: list[str]) -> list[dict[str, Any]]:
    required = {
        "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY"),
        "E2B_API_KEY": os.environ.get("E2B_API_KEY"),
    }
    missing = sorted(name for name, value in required.items() if not value)
    if missing:
        raise RuntimeError(f"DeepSWE credentials unavailable by env name: {missing}")
    cells: list[dict[str, Any]] = []
    for base_model in ("gpt-5.5", "gpt-5.4-mini"):
        for effort in efforts:
            short_model = base_model.replace("gpt-", "gpt").replace(".", "")
            cells.append(
                {
                    "cell_id": f"{PREFIX}-reasoning-{short_model}-{effort}-v1",
                    "logical_provider": "openai_responses",
                    "logical_model": f"{base_model}@{effort}",
                    "base_model": base_model,
                    "reasoning_effort": effort,
                    "config": ProviderConfig(
                        kind=ProviderKind.OPENAI_RESPONSES,
                        model_type=base_model,
                        model=base_model,
                        reasoning_effort=effort,
                    ),
                }
            )
    return cells


def _task_ids(root: Path, requested: list[str], n_tasks: int | None, seed: int) -> list[str]:
    available = sorted(path.parent.name for path in root.glob("*/task.toml"))
    if not available:
        raise ValueError(f"DeepSWE dataset has no task.toml files: {root}")
    if requested:
        missing = sorted(set(requested) - set(available))
        if missing:
            raise ValueError(f"DeepSWE task ids are missing: {missing}")
        if len(requested) != len(set(requested)):
            raise ValueError("DeepSWE task ids must be unique")
        return requested
    if n_tasks is None:
        return available
    if n_tasks < 1 or n_tasks > len(available):
        raise ValueError(f"--n-tasks must be between 1 and {len(available)}")
    return sorted(
        available,
        key=lambda task_id: hashlib.sha256(f"{seed}|deepswe-1.1|{task_id}".encode()).hexdigest(),
    )[:n_tasks]


def _job_template(root: Path, dataset_root: Path, cell_id: str) -> JobConfig:
    return JobConfig(
        job_name=f"{cell_id}-job",
        jobs_dir=root / "harbor-jobs" / cell_id,
        n_concurrent_trials=1,
        environment=EnvironmentConfig(type=EnvironmentType.E2B),
        datasets=[DatasetConfig(path=dataset_root)],
        agents=[AgentConfig()],
    )


def _token_cost(trial: dict[str, Any]) -> dict[str, Any]:
    contexts: list[dict[str, Any]] = []
    if isinstance(trial.get("agent_result"), dict):
        contexts.append(trial["agent_result"])
    for step in trial.get("step_results") or []:
        if isinstance(step, dict) and isinstance(step.get("agent_result"), dict):
            contexts.append(step["agent_result"])
    totals = {"input_tokens": 0, "cache_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
    if not contexts:
        return {key: None for key in totals}
    for context in contexts:
        totals["input_tokens"] += int(context.get("n_input_tokens") or 0)
        totals["cache_tokens"] += int(context.get("n_cache_tokens") or 0)
        totals["output_tokens"] += int(context.get("n_output_tokens") or 0)
        totals["cost_usd"] += float(context.get("cost_usd") or 0.0)
    return totals


def _trial_records(job_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for result_path in sorted(job_dir.glob("*/result.json")):
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        records.append(
            {
                "trial_result": str(result_path),
                "task_id": payload.get("task_name"),
                "trial_name": payload.get("trial_name"),
                "exception_type": (payload.get("exception_info") or {}).get("exception_type"),
                "reward": ((payload.get("verifier_result") or {}).get("rewards") or {}).get(
                    "reward"
                ),
                "token_cost": _token_cost(payload),
            }
        )
    return records


def _priced_cost(record: dict[str, Any], model: str) -> float:
    token_cost = record.get("token_cost") or {}
    reported = token_cost.get("cost_usd")
    if reported is not None and float(reported) > 0:
        return float(reported)
    prices = MODEL_PRICES[model.split("@", 1)[0]]
    return int(token_cost.get("input_tokens") or 0) * prices[0] / 1_000_000 + int(
        token_cost.get("output_tokens") or 0
    ) * prices[1] / 1_000_000


def main() -> None:
    args = _parser().parse_args()
    dataset_root = args.dataset_root.resolve()
    root = args.artifact_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    task_ids = _task_ids(dataset_root, args.task_id, args.n_tasks, args.sample_seed)
    cells = _reasoning_cells(args.reasoning_effort) if args.reasoning_effort else _cells()
    if args.cell_suffix:
        for cell in cells:
            cell["cell_id"] = f"{cell['cell_id']}{args.cell_suffix}"
    if args.only_model:
        allowed = set(args.only_model)
        cells = [
            cell
            for cell in cells
            if cell["logical_model"] in allowed or cell.get("base_model") in allowed
        ]
        if not cells:
            raise ValueError(f"none of --only-model values matched: {sorted(allowed)}")
    doc = default_agent("pi")
    manifest = {
        "benchmark": "deepswe-1.1",
        "dataset_root": str(dataset_root),
        "task_ids": task_ids,
        "sample_seed": args.sample_seed,
        "cells": [cell["cell_id"] for cell in cells],
        "reasoning_efforts": args.reasoning_effort or [],
        "doc_hash": doc.doc_hash,
        "timestamp": datetime.now(UTC).isoformat(),
    }
    (root / f"{PREFIX}-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for cell in cells:
        cell_id = str(cell["cell_id"])
        started = time.monotonic()
        LOGGER.info("starting cell=%s tasks=%s", cell_id, task_ids)
        scorer = asyncio.run(
            HarborScorer.create(
                _job_template(root, dataset_root, cell_id),
                task_ids,
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
        cost = sum(_priced_cost(record, str(cell.get("logical_model"))) for record in records)
        event = {
            "benchmark": "deepswe-1.1",
            "cell_id": cell_id,
            "logical_provider": cell["logical_provider"],
            "logical_model": cell["logical_model"],
            "base_model": cell.get("base_model", cell["logical_model"]),
            "reasoning_effort": cell.get("reasoning_effort"),
            "task_ids": task_ids,
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
        (root / f"{cell_id}-outcome.json").write_text(
            json.dumps(event, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        LOGGER.info("completed cell=%s score=%.4f cost_usd=%.6f", cell_id, report.score, cost)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
