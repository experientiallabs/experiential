"""Capture fresh REAL financebench runs on Bedrock and append them to the trace corpus.

Tasks are sharded round-robin across the given Bedrock model ids (one thread per model — the
established pattern for beating per-model throttling) and every bash transition is recorded from
real execution in the task workspace. Raw graded trajectories are also written to ``runs/`` as
JSONL (gitignored) so a capture can be inspected and resumed without re-running.

Usage (from the repo root):
    uv run python environment-capture/financebench/capture.py \
        --split train --limit 8 \
        --models us.anthropic.claude-opus-4-7,us.anthropic.claude-opus-4-8 \
        --out environment-capture/financebench/traces.otel.jsonl --append
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

from environment_capture import (
    Trajectory,
    partition_contained,
    run_capture,
    trajectory_to_spans,
    write_spans_jsonl,
)
from environment_capture.agent import BedrockBashAgent
from environment_capture.benchmarks.financebench import FinanceBenchAdapter

_HERE = Path(__file__).parent


def _model_tag(model_id: str) -> str:
    """`us.anthropic.claude-opus-4-8` -> `opus48` (for run-suffixed task ids)."""
    stem = model_id.split(".")[-1].removeprefix("claude-").removesuffix("-v1")
    return re.sub(r"[^a-z0-9]", "", stem)


def _suffix_task_id(trajectory: Trajectory, tag: str) -> Trajectory:
    """Give the emitted trace a unique id (real question unchanged) so trace ids never collide
    with the converted-cache traces (unsuffixed ids) or with other capture waves."""
    task = dataclasses.replace(trajectory.task, task_id=f"{trajectory.task.task_id}#{tag}")
    return dataclasses.replace(trajectory, task=task)


def _capture_shard(
    adapter: FinanceBenchAdapter,
    model_id: str,
    tasks: list,  # noqa: ANN001 - list[Task]; kept loose for ThreadPoolExecutor.map
    split: str,
    max_steps: int,
    run_tag: str,
) -> list[Trajectory]:
    agent = BedrockBashAgent(model_id, max_steps=max_steps)
    result = run_capture(adapter, agent, split=split, tasks=tasks)
    for failure in result.failures:
        print(f"[skip] {failure.task_id} on {model_id}: {failure.error}", file=sys.stderr)
    contained, flagged = partition_contained(result.trajectories)
    for trajectory in flagged:
        print(f"[drop] {trajectory.task.task_id}: host-escape content", file=sys.stderr)
    tag = f"{_model_tag(model_id)}-{run_tag}"
    return [_suffix_task_id(t, tag) for t in contained]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--limit", type=int, default=None, help="Cap the number of tasks")
    parser.add_argument("--skip", type=int, default=0, help="Skip the first N tasks (resume)")
    parser.add_argument(
        "--models",
        default="us.anthropic.claude-opus-4-8",
        help="Comma-separated Bedrock model ids; tasks are sharded round-robin across them",
    )
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument(
        "--run-tag",
        default="r1",
        help="Suffix for this capture wave's task ids (bump per wave: r1, r2, ...)",
    )
    parser.add_argument("--out", default=str(_HERE / "traces.otel.jsonl"))
    parser.add_argument("--append", action="store_true", help="Append to --out (default: refuse)")
    args = parser.parse_args()

    out = Path(args.out)
    if out.exists() and not args.append:
        raise SystemExit(f"{out} exists; pass --append to extend it")
    if args.split == "test":
        raise SystemExit(
            "refusing to capture the test split into a corpus: the hidden test split must stay "
            "out of world-model training data"
        )

    adapter = FinanceBenchAdapter(data_root=_HERE)
    model_ids = [m.strip() for m in args.models.split(",") if m.strip()]
    tasks = adapter.tasks(args.split)[args.skip :]
    if args.limit is not None:
        tasks = tasks[: args.limit]
    shards = [tasks[i :: len(model_ids)] for i in range(len(model_ids))]

    started = time.time()
    with ThreadPoolExecutor(max_workers=len(model_ids)) as pool:
        shard_results = list(
            pool.map(
                lambda pair: _capture_shard(
                    adapter, pair[0], pair[1], args.split, args.max_steps, args.run_tag
                ),
                zip(model_ids, shards, strict=True),
            )
        )
    trajectories = [t for shard in shard_results for t in shard]

    runs_dir = _HERE / "runs"
    runs_dir.mkdir(exist_ok=True)
    raw_path = runs_dir / f"capture-{int(started)}.jsonl"
    with raw_path.open("w", encoding="utf-8") as raw:
        for trajectory in trajectories:
            raw.write(json.dumps(asdict(trajectory), ensure_ascii=False) + "\n")

    kept = [t for t in trajectories if t.steps]
    n_spans = 0
    for index, trajectory in enumerate(kept):
        spans = trajectory_to_spans(trajectory, benchmark="financebench")
        n_spans += write_spans_jsonl(spans, out, append=args.append or index > 0)

    rewards = [t.reward or 0.0 for t in trajectories]
    mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
    print(
        f"captured {len(trajectories)} runs ({len(kept)} with transitions, "
        f"{sum(len(t.steps) for t in kept)} steps, mean reward {mean_reward:.3f}) "
        f"in {time.time() - started:.0f}s -> {out} (raw: {raw_path})"
    )


if __name__ == "__main__":
    main()
