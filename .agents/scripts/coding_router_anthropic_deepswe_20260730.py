"""Run a fresh Anthropic Opus 5 DeepSWE proxy cell for matched partial scoring."""

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

LOGGER = logging.getLogger("coding-router-anthropic-deepswe-20260730")
PREFIX = "coding-router-small-agent-20260729-anthropic"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--task-id", action="append", default=[])
    parser.add_argument("--n-tasks", type=int)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--max-turns", type=int, default=60)
    parser.add_argument("--episode-timeout-s", type=float, default=5400.0)
    return parser


def _cost(record: dict[str, Any]) -> float:
    token_cost = record.get("token_cost") or {}
    reported = token_cost.get("cost_usd")
    if reported is not None and float(reported) > 0:
        return float(reported)
    return (
        int(token_cost.get("input_tokens") or 0) * 5.0
        + int(token_cost.get("output_tokens") or 0) * 25.0
    ) / 1_000_000


def main() -> None:
    args = _parser().parse_args()
    required = ("ANTHROPIC_API_KEY", "E2B_API_KEY")
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise RuntimeError(f"credentials unavailable by environment name: {missing}")
    if args.max_turns < 1:
        raise ValueError("--max-turns must be positive")
    dataset_root = args.dataset_root.resolve()
    artifact_root = args.artifact_root.resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)
    task_ids = _task_ids(dataset_root, args.task_id, args.n_tasks, args.sample_seed)
    cell_id = f"{PREFIX}-claude-opus-5-high-v1"
    config = ProviderConfig(
        kind=ProviderKind.ANTHROPIC,
        model_type="claude-opus-5",
        model="claude-opus-5",
    )
    doc = default_agent("pi")
    doc = doc.model_copy(
        update={
            "surfaces": [
                surface.model_copy(update={"content": str(args.max_turns)})
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
        "model": "claude-opus-5",
        "reasoning_effort": "provider-default-high-equivalent",
        "max_turns": args.max_turns,
        "pricing": {"input_per_mtok": 5.0, "output_per_mtok": 25.0},
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
    event = {
        "benchmark": "deepswe-1.1",
        "cell_id": cell_id,
        "logical_provider": "anthropic",
        "logical_model": "claude-opus-5",
        "reasoning_effort": "provider-default-high-equivalent",
        "task_ids": task_ids,
        "task_pins": scorer.task_pins,
        "doc_hash": doc.doc_hash,
        "score": report.score,
        "pass_rate": report.pass_rate,
        "cells": [cell.model_dump(mode="json") for cell in report.cells],
        "trial_records": records,
        "job_dir": str(job_dir),
        "cost_usd": sum(_cost(record) for record in records),
        "elapsed_s": round(time.monotonic() - started, 3),
        "timestamp": datetime.now(UTC).isoformat(),
    }
    output = artifact_root / f"{cell_id}-outcome.json"
    output.write_text(json.dumps(event, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    LOGGER.info("completed cell=%s score=%.4f", cell_id, report.score)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
