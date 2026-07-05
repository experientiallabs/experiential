"""Capture fresh REAL AppWorld runs on Bedrock and append them to the trace corpus.

Runs under the MAIN workspace interpreter: the ``AppWorldAgent`` (Bedrock) drives the appworld-free
``AppWorldAdapter``, which launches the real ``appworld`` engine out-of-process under this
benchmark's venv (``backend/world_backend.py``). Every ``execute_python`` transition is recorded
from execution against a LIVE, stateful world, and each task is graded by AppWorld's own tests.

Tasks are sharded round-robin across the given Bedrock model ids (one thread per model — the
established pattern for beating per-model throttling; keep to 2 while other captures share Bedrock).
Each emitted trace carries a run-suffixed task id (``aw-train-3#opus48-r1``) so the deterministic
trace id never collides across models or repeated passes; the base task id and reward ride in the
trace metadata. Each shard uses its run tag as the AppWorld ``experiment_prefix`` so the per-task
world states two shards write never collide. Raw graded trajectories are also written to ``runs/``
as JSONL (gitignored) so a capture can be inspected and resumed without re-running.

The resulting ``traces.otel.jsonl`` is NOT committed: AppWorld's dataset license forbids plaintext
public redistribution of its data or derivatives (see README). It stays gitignored and is
reproducible with this script.

Usage (from the repo root; data must be materialized first — see backend/fetch_data.py):
    uv run python packages/environment-capture/appworld/capture.py \
        --split train --limit 8 \
        --models us.anthropic.claude-opus-4-8,us.anthropic.claude-opus-4-7 \
        --out packages/environment-capture/appworld/traces.otel.jsonl --append
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
from environment_capture.benchmarks.appworld import AppWorldAdapter, AppWorldAgent

_HERE = Path(__file__).parent
_BENCHMARK = "appworld"


def _short_model(model_id: str) -> str:
    """A compact tag for a Bedrock model id, e.g. us.anthropic.claude-opus-4-8 -> opus48."""
    tail = model_id.rsplit(".", 1)[-1].removeprefix("claude-")
    return tail.replace("-", "").replace("v1", "")


def _capture_shard(
    model_id: str,
    tasks: list[Task],
    split: str,
    max_steps: int,
    run_tag: str,
) -> list[Trajectory]:
    # Each shard's run tag doubles as the AppWorld experiment prefix, so two models grading the same
    # task write to disjoint experiment directories.
    adapter = AppWorldAdapter(_HERE, experiment_prefix=run_tag)
    agent = AppWorldAgent(model_id, max_steps=max_steps)
    result = run_capture(adapter, agent, split=split, tasks=tasks)
    for failure in result.failures:
        print(f"[skip] {failure.task_id} on {model_id}: {failure.error}", file=sys.stderr)
    # generic_path_markers=False: AppWorld executes in its own sandbox (SafetyGuard blocks
    # os.listdir/subprocess/open), and its SIMULATED file system legitimately uses ~/ and /home
    # paths as environment content — the generic path markers would drop those (real) trajectories
    # as false positives. The runtime identity markers (real username + home) still catch a genuine
    # leak (e.g. os.path.expanduser echoing the account), and command-level checks stay active.
    contained, flagged = partition_contained(result.trajectories, generic_path_markers=False)
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
    parser.add_argument("--split", default="train", choices=["train"])
    parser.add_argument("--limit", type=int, default=None, help="Cap the number of tasks")
    parser.add_argument("--skip", type=int, default=0, help="Skip the first N tasks (resume)")
    parser.add_argument(
        "--models",
        default="us.anthropic.claude-opus-4-8,us.anthropic.claude-opus-4-7",
        help="Comma-separated Bedrock model ids; tasks are sharded round-robin across them",
    )
    parser.add_argument("--runs", type=int, default=1, help="Passes over the split (run-suffixed)")
    parser.add_argument(
        "--run-start",
        type=int,
        default=1,
        help="First run number for tag suffixes; bump past prior waves so ids never collide",
    )
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--out", default=str(_HERE / "traces.otel.jsonl"))
    parser.add_argument("--append", action="store_true", help="Append to --out (default: refuse)")
    args = parser.parse_args()

    out = Path(args.out)
    if out.exists() and not args.append:
        raise SystemExit(f"{out} exists; pass --append to extend it")

    adapter = AppWorldAdapter(_HERE)
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
                    lambda job: _capture_shard(job[0], job[1], args.split, args.max_steps, job[2]),
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
