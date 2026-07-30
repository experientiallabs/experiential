"""Run a fresh GPT-5.6 reasoning-effort cell on the DeepSWE v1.1 proxy."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from wmo.agents.default import default_agent
from wmo.evals.harbor.scorer import HarborScorer
from wmo.harness.doc import MAX_OUTPUT_TOKENS_ID, MAX_TURNS_ID
from wmo.providers.base import ProviderConfig, ProviderKind

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from coding_router_deepswe_20260729 import (  # noqa: E402
    _job_template,
    _task_ids,
    _trial_records,
)

LOGGER = logging.getLogger("coding-router-openai56-deepswe-20260729")
PREFIX = "coding-router-small-agent-20260729-openai56"
PRICE = {
    "gpt-5.5": (5.0, 30.0, 0.5),
    "gpt-5.6-sol": (5.0, 30.0, 0.5),
    "gpt-5.6-terra": (2.5, 15.0, 0.25),
    "gpt-5.6-luna": (1.0, 6.0, 0.1),
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--model", choices=tuple(PRICE), default="gpt-5.6-sol")
    parser.add_argument(
        "--reasoning-effort",
        choices=("low", "medium", "high", "xhigh", "max"),
        default="high",
    )
    parser.add_argument("--task-id", action="append", default=[])
    parser.add_argument("--n-tasks", type=int)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--episode-timeout-s", type=float, default=5400.0)
    parser.add_argument("--max-turns", type=int, default=60)
    return parser


def _cost(record: dict[str, Any], model: str) -> float:
    token_cost = record.get("token_cost") or {}
    reported = token_cost.get("cost_usd")
    if reported is not None and float(reported) > 0:
        return float(reported)
    input_price, output_price, cache_price = PRICE[model]
    input_tokens = int(token_cost.get("input_tokens") or 0)
    cache_tokens = min(input_tokens, int(token_cost.get("cache_tokens") or 0))
    output_tokens = int(token_cost.get("output_tokens") or 0)
    return (
        (input_tokens - cache_tokens) * input_price
        + cache_tokens * cache_price
        + output_tokens * output_price
    ) / 1_000_000


def main() -> None:
    args = _parser().parse_args()
    if not os.environ.get("OPENAI_API_KEY") or not os.environ.get("E2B_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY and E2B_API_KEY must be available by environment name")
    dataset_root = args.dataset_root.resolve()
    artifact_root = args.artifact_root.resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)
    task_ids = _task_ids(dataset_root, args.task_id, args.n_tasks, args.sample_seed)
    cell_id = f"{PREFIX}-{args.model}-{args.reasoning_effort}-v1"
    config = ProviderConfig(
        kind=ProviderKind.OPENAI_RESPONSES,
        model_type=args.model,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
    )
    if args.max_turns < 1:
        raise ValueError("--max-turns must be positive")
    doc = default_agent("pi")
    doc = doc.model_copy(
        update={
            "surfaces": [
                surface.model_copy(
                    update={"content": str(args.max_turns)}
                )
                if surface.id == MAX_TURNS_ID
                else surface.model_copy(update={"content": "16384"})
                if surface.id == MAX_OUTPUT_TOKENS_ID
                else surface
                for surface in doc.surfaces
            ]
        }
    )
    manifest = {
        "benchmark": "deepswe-1.1",
        "task_ids": task_ids,
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "max_turns": args.max_turns,
        "pricing": {
            "input_per_mtok": PRICE[args.model][0],
            "cached_input_per_mtok": PRICE[args.model][2],
            "output_per_mtok": PRICE[args.model][1],
        },
        "doc_hash": doc.doc_hash,
        "timestamp": datetime.now(UTC).isoformat(),
    }
    (artifact_root / f"{cell_id}-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    started = time.monotonic()
    LOGGER.info("starting cell=%s tasks=%s", cell_id, task_ids)
    scorer = asyncio.run(
        HarborScorer.create(
            _job_template(artifact_root, dataset_root, cell_id),
            task_ids,
            provider_config=config,
            task_environment="e2b",
            harness_backend="e2b",
            reward_key="reward",
            reward_mode="raw",
            attempts=1,
            harbor_retries=0,
            missing_reward="zero",
            agent_model_name=f"{config.kind.value}/{config.model}",
            episode_timeout_s=args.episode_timeout_s,
        )
    )
    report = scorer.score(doc)
    job_dir = scorer.candidate_job_dir(doc).resolve()
    records = _trial_records(job_dir)
    cost = sum(_cost(record, args.model) for record in records)
    event = {
        "benchmark": "deepswe-1.1",
        "cell_id": cell_id,
        "logical_provider": "openai_responses",
        "logical_model": args.model,
        "reasoning_effort": args.reasoning_effort,
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
    output = artifact_root / f"{cell_id}-outcome.json"
    output.write_text(json.dumps(event, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    LOGGER.info("completed cell=%s score=%.4f cost_usd=%.6f", cell_id, report.score, cost)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
