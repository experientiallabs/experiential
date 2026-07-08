#!/usr/bin/env python3
"""Capture telecom tau2 traces across MULTIPLE Opus models to beat the per-model Bedrock quota.

A single Opus model on Bedrock throttles (litellm ServiceUnavailableError) under sustained load — a
concurrency-8 telecom run lost ~80% of sims, and dropping concurrency didn't help because the wall
is a per-model account quota, not jitter. Telecom is the worst case (longest trajectories + a
tool-calling user simulator => most LLM calls per task).

Fix: shard the telecom task list across several Opus models (4.6 / 4.7 / 4.8), each its own
per-model quota, run them concurrently. Each shard is a DISJOINT slice of the full task list (so no
task is captured twice) and writes to its own save dir; `--auto-resume` makes a re-run retry only
that shard's still-failed tasks. Merge the shards afterward with convert_to_wmh.py + cat (see
capture_corpus.sh / the README).

This runs in the ISOLATED tau2 venv (Python 3.13, `tau2` installed); it imports `tau2`, never `wmh`.

    TAU2_DATA_DIR=$PWD/tau2-bench/data AWS_REGION=us-east-1 \
      .venv/bin/python capture_telecom_multimodel.py --total 980 --concurrency 3

-> data/simulations/capture_telecom_<modeltag>/results.json  (one per model)
Convert each like any other domain shard, then cat into packages/environment-capture/tau-bench/traces.otel.jsonl.
"""

from __future__ import annotations

import argparse
import os
import threading
from pathlib import Path

from tau2.run import run_tasks
from tau2.runner.helpers import get_tasks

# Persist each shard as data/simulations/<name>/results.json — the same layout the CLI's --save-to
# produces and convert_to_wmh.py expects. The deprecated run_tasks(save_to=) treats save_to as a
# literal path (NOT relative to data/simulations like the CLI), so we build the absolute path here.
_SIM_DIR = Path(os.environ["TAU2_DATA_DIR"]) / "simulations"

# The Opus inference-profile IDs that invoke on this account (verified ACTIVE). Each is a separate
# per-model quota, so sharding across them multiplies effective throughput. litellm's Bedrock route
# wants the `bedrock/` prefix.
DEFAULT_MODELS = [
    "bedrock/us.anthropic.claude-opus-4-6-v1",
    "bedrock/us.anthropic.claude-opus-4-7",
    "bedrock/us.anthropic.claude-opus-4-8",
]


def _model_tag(model: str) -> str:
    """A filesystem-safe tag from a model id, e.g. '...opus-4-7' -> 'opus-4-7'."""
    return model.split("/")[-1].replace("us.anthropic.claude-", "").replace(":", "_")


def _run_shard(
    model: str, tasks: list, concurrency: int, retries: int, delay: float, suffix: str = ""
) -> None:
    save_path = _SIM_DIR / f"capture_telecom_{_model_tag(model)}{suffix}" / "results.json"
    print(f"[{model}] {len(tasks)} tasks -> {save_path}", flush=True)
    run_tasks(
        domain="telecom",
        tasks=tasks,
        agent="llm_agent",
        user="user_simulator",
        llm_agent=model,
        llm_args_agent={},
        llm_user=model,
        llm_args_user={},
        num_trials=1,
        max_concurrency=concurrency,
        max_retries=retries,
        retry_delay=delay,
        auto_resume=True,
        save_to=str(save_path),
        console_display=False,
    )
    print(f"[{model}] done", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total", type=int, default=980, help="Total telecom tasks across shards.")
    parser.add_argument(
        "--offset", type=int, default=0, help="Skip this many tasks first (for disjoint top-ups)."
    )
    parser.add_argument("--concurrency", type=int, default=3, help="Per-model concurrency.")
    parser.add_argument("--retries", type=int, default=5, help="tau2 task-level retries.")
    parser.add_argument("--delay", type=float, default=5.0, help="Seconds between task retries.")
    parser.add_argument("--split", default="full", help="Telecom task split (full = 2285 tasks).")
    parser.add_argument(
        "--models", default=",".join(DEFAULT_MODELS), help="Comma-separated litellm model ids."
    )
    parser.add_argument(
        "--suffix", default="", help="Save-dir suffix to isolate a top-up run from prior shards."
    )
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    all_tasks = get_tasks("telecom", task_split_name=args.split)[args.offset : args.offset + args.total]
    print(
        f"telecom '{args.split}': tasks [{args.offset}:{args.offset + args.total}] "
        f"({len(all_tasks)}), sharding across {len(models)} models"
    )

    # Round-robin the tasks into disjoint shards so each model gets a contiguous-free, even slice and
    # no task is run twice. Round-robin (not contiguous blocks) keeps task-type mix even per model.
    shards: list[list] = [[] for _ in models]
    for i, task in enumerate(all_tasks):
        shards[i % len(models)].append(task)

    threads = [
        threading.Thread(
            target=_run_shard,
            args=(model, shard, args.concurrency, args.retries, args.delay, args.suffix),
        )
        for model, shard in zip(models, shards, strict=True)
        if shard
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print("all shards complete")


if __name__ == "__main__":
    main()
