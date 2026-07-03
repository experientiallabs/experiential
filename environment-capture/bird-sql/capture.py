"""Capture fresh REAL bird-sql runs on Bedrock and append them to the trace corpus.

Tasks are sharded round-robin across the given Bedrock model ids (one thread per model — the
established pattern for beating per-model throttling) and every bash transition is recorded from
real execution against a fresh COPY of the task's SQLite database. Each emitted trace carries a
run-suffixed task id (``bird-train-3#opus48-r1``) built from the model and run index, so the
deterministic trace id never collides across models or repeated passes. The real task id and reward
survive in the trace metadata. Raw graded trajectories are also written to ``runs/`` as JSONL
(gitignored) so a capture can be inspected and resumed without re-running.

Usage (from the repo root; databases must be materialized first — see fetch_data.py):
    uv run python environment-capture/bird-sql/capture.py \
        --split train --limit 8 \
        --models us.anthropic.claude-opus-4-8,us.anthropic.claude-opus-4-7 \
        --out environment-capture/bird-sql/traces.otel.jsonl --append
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
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
from environment_capture.benchmarks.bird_sql import BirdSqlAdapter

_HERE = Path(__file__).parent
_BENCHMARK = "bird-sql"

_SQL_INSTRUCTIONS = (
    "\n\nThe SQLite database is ./database.db and its DDL schema is ./schema.sql. Read the schema, "
    'then explore the data with the sqlite3 CLI (e.g. `sqlite3 database.db "SELECT ..."`). When '
    "confident, call submit with your final answer set to a single SQLite SELECT query (no prose) "
    "that answers the question."
)


def _short_model(model_id: str) -> str:
    """A compact tag for a Bedrock model id, e.g. us.anthropic.claude-opus-4-8 -> opus48."""
    tail = model_id.rsplit(".", 1)[-1].removeprefix("claude-")
    return tail.replace("-", "").replace("v1", "")


def _with_sql_instructions(task: Task) -> Task:
    """Append the how-to-submit-SQL framing the generic agent prompt does not carry."""
    return dataclasses.replace(task, prompt=task.prompt + _SQL_INSTRUCTIONS)


def _capture_shard(
    adapter: BirdSqlAdapter,
    model_id: str,
    tasks: list[Task],
    split: str,
    max_steps: int,
    run_tag: str,
) -> list[Trajectory]:
    agent = BedrockBashAgent(model_id, max_steps=max_steps)
    framed = [_with_sql_instructions(t) for t in tasks]
    result = run_capture(adapter, agent, split=split, tasks=framed)
    for failure in result.failures:
        print(f"[skip] {failure.task_id} on {model_id}: {failure.error}", file=sys.stderr)
    contained, flagged = partition_contained(result.trajectories)
    for trajectory in flagged:
        print(f"[drop] {trajectory.task.task_id}: host-escape content", file=sys.stderr)
    originals = {t.task_id: t for t in tasks}
    tagged: list[Trajectory] = []
    for trajectory in contained:
        original = originals[trajectory.task.task_id]
        suffixed = dataclasses.replace(original, task_id=f"{original.task_id}#{run_tag}")
        tagged.append(
            dataclasses.replace(
                trajectory,
                task=suffixed,
                metadata={**trajectory.metadata, "base_task_id": original.task_id},
            )
        )
    return tagged


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
    parser.add_argument("--runs", type=int, default=1, help="Passes over the split (run-suffixed)")
    parser.add_argument(
        "--run-start",
        type=int,
        default=1,
        help="First run number for tag suffixes; bump past prior waves so ids never collide",
    )
    parser.add_argument("--max-steps", type=int, default=12)
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

    adapter = BirdSqlAdapter(data_root=_HERE)
    model_ids = [m.strip() for m in args.models.split(",") if m.strip()]
    tasks = adapter.tasks(args.split)[args.skip :]
    if args.limit is not None:
        tasks = tasks[: args.limit]

    started = time.time()
    all_trajectories: list[Trajectory] = []
    for run_index in range(args.runs):
        shards = [tasks[i :: len(model_ids)] for i in range(len(model_ids))]
        jobs = [
            (model_id, shard, f"{_short_model(model_id)}-r{args.run_start + run_index}")
            for model_id, shard in zip(model_ids, shards, strict=True)
        ]
        with ThreadPoolExecutor(max_workers=len(model_ids)) as pool:
            shard_results = list(
                pool.map(
                    lambda job: _capture_shard(
                        adapter, job[0], job[1], args.split, args.max_steps, job[2]
                    ),
                    jobs,
                )
            )
        all_trajectories.extend(t for shard in shard_results for t in shard)

    runs_dir = _HERE / "runs"
    runs_dir.mkdir(exist_ok=True)
    raw_path = runs_dir / f"capture-{int(started)}.jsonl"
    with raw_path.open("w", encoding="utf-8") as raw:
        for trajectory in all_trajectories:
            raw.write(json.dumps(dataclasses.asdict(trajectory), ensure_ascii=False) + "\n")

    kept = [t for t in all_trajectories if t.steps]
    n_spans = 0
    for index, trajectory in enumerate(kept):
        spans = trajectory_to_spans(trajectory, benchmark=_BENCHMARK)
        n_spans += write_spans_jsonl(spans, out, append=args.append or index > 0)

    rewards = [t.reward or 0.0 for t in all_trajectories]
    mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
    print(
        f"captured {len(all_trajectories)} runs ({len(kept)} with transitions, "
        f"{sum(len(t.steps) for t in kept)} steps, {n_spans} spans, mean reward {mean_reward:.3f}) "
        f"in {time.time() - started:.0f}s -> {out} (raw: {raw_path})"
    )


if __name__ == "__main__":
    main()
