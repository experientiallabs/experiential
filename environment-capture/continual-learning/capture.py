"""Capture fresh REAL database-exploration runs on Bedrock and append them to the trace corpus.

Tasks are sharded round-robin across the given Bedrock model ids (one thread per model — the
established pattern for beating per-model throttling). Every bash transition is recorded from real
execution against the shared read-only ``products.db`` staged into each task workspace.

Fresh runs get a run-suffixed task id (``clb-train-3#opus48-r1``) so their deterministic trace ids
never collide with the converted-cache traces or with other capture waves. Bump ``--run-tag`` for
each new wave (r1, r2, ...). Raw graded trajectories are also written to ``runs/`` as JSONL
(gitignored) so a capture can be inspected and resumed without re-running.

Usage (from the repo root; needs a fetched products.db — see fetch_data.py):
    uv run python environment-capture/continual-learning/capture.py \
        --split train --limit 8 --run-tag r1 \
        --models us.anthropic.claude-opus-4-8,us.anthropic.claude-opus-4-7 \
        --out environment-capture/continual-learning/traces.otel.jsonl --append
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
    Task,
    Trajectory,
    partition_contained,
    run_capture,
    trajectory_to_spans,
    write_spans_jsonl,
)
from environment_capture.agent import BedrockBashAgent
from environment_capture.benchmarks.continual_learning import ContinualLearningAdapter

_HERE = Path(__file__).parent
_BENCHMARK = "continual-learning"

_SYSTEM_PROMPT = """You are an autonomous data-analyst agent working in a Unix workspace. A SQLite
database is at ./database.db. Its table and column names are cryptic and its data has quality traps
(prices may be stored in integer cents, timestamps in epoch milliseconds, some values corrupted or
drifted), so EXPLORE before answering: inspect the schema with `sqlite3 database.db ".schema"`,
sample rows, and verify how each column is encoded. Use the bash tool for one focused command per
step (sqlite3 or python3) and check intermediate results rather than assuming them. When you are
confident, call submit with ONLY the final answer in the exact format the question asks for (a
number or a short string, no explanation)."""


def _model_tag(model_id: str) -> str:
    """A compact, filesystem/id-safe tag for a Bedrock model id (opus-4-8 -> opus48)."""
    stem = model_id.split(".")[-1]  # us.anthropic.claude-opus-4-8 -> claude-opus-4-8
    stem = stem.replace("claude-", "").replace("-v1", "")
    return re.sub(r"[^a-z0-9]", "", stem)


def _capture_shard(
    adapter: ContinualLearningAdapter,
    model_id: str,
    tasks: list[Task],
    split: str,
    max_steps: int,
) -> list[Trajectory]:
    """Capture one model's shard; run_capture isolates per-task failures (logged and skipped)."""
    agent = BedrockBashAgent(model_id, max_steps=max_steps, system_prompt=_SYSTEM_PROMPT)
    result = run_capture(adapter, agent, split=split, tasks=tasks)
    for failure in result.failures:
        print(f"[skip] {failure.task_id} on {model_id}: {failure.error}", file=sys.stderr)
    contained, flagged = partition_contained(result.trajectories)
    for trajectory in flagged:
        print(f"[drop] {trajectory.task.task_id}: host-escape content", file=sys.stderr)
    return contained


def _suffix_task_id(trajectory: Trajectory, run_tag: str) -> Trajectory:
    """Give the emitted trace a unique id (real question unchanged) so trace ids never collide."""
    tag = _model_tag(trajectory.model)
    new_id = f"{trajectory.task.task_id}#{tag}-{run_tag}"
    return dataclasses.replace(
        trajectory, task=dataclasses.replace(trajectory.task, task_id=new_id)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--limit", type=int, default=None, help="Cap the number of tasks")
    parser.add_argument("--skip", type=int, default=0, help="Skip the first N tasks (resume)")
    parser.add_argument("--run-tag", default="r1", help="Wave tag suffixed into fresh trace ids")
    parser.add_argument(
        "--models",
        default="us.anthropic.claude-opus-4-8",
        help="Comma-separated Bedrock model ids; tasks are sharded round-robin across them",
    )
    parser.add_argument("--max-steps", type=int, default=16)
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

    adapter = ContinualLearningAdapter(data_root=_HERE)
    model_ids = [m.strip() for m in args.models.split(",") if m.strip()]
    tasks = adapter.tasks(args.split)[args.skip :]
    if args.limit is not None:
        tasks = tasks[: args.limit]
    shards = [tasks[i :: len(model_ids)] for i in range(len(model_ids))]

    started = time.time()
    with ThreadPoolExecutor(max_workers=len(model_ids)) as pool:
        shard_results = list(
            pool.map(
                lambda pair: _capture_shard(adapter, pair[0], pair[1], args.split, args.max_steps),
                zip(model_ids, shards, strict=True),
            )
        )
    trajectories = [_suffix_task_id(t, args.run_tag) for shard in shard_results for t in shard]

    runs_dir = _HERE / "runs"
    runs_dir.mkdir(exist_ok=True)
    raw_path = runs_dir / f"capture-{int(started)}.jsonl"
    with raw_path.open("w", encoding="utf-8") as raw:
        for trajectory in trajectories:
            raw.write(json.dumps(asdict(trajectory), ensure_ascii=False) + "\n")

    kept = [t for t in trajectories if t.steps]
    n_spans = 0
    for index, trajectory in enumerate(kept):
        spans = trajectory_to_spans(trajectory, benchmark=_BENCHMARK)
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
