"""Capture fresh REAL GAIA2 runs on Bedrock and append them to the trace corpus.

Runs under the MAIN workspace interpreter: the ``Gaia2Agent`` (Bedrock) drives the ARE-free
``Gaia2Adapter``, which launches the real ARE engine out-of-process under this benchmark's venv
(``backend/world_backend.py``). Every ``execute_python`` transition is recorded from execution
against a LIVE, stateful scenario world; each task is graded by our deterministic structural
action-match (NOT the official Gaia2 LLM-judge score — see the adapter/README).

Tasks are sharded round-robin across the given Bedrock model ids (one thread per model — the
established pattern for beating per-model throttling; keep to 2 while other captures share Bedrock).
Each trajectory is emitted to ``--out`` **as soon as it is graded** (under a lock), not in one batch
at the end, so a slow/stuck Bedrock connection late in the run never discards the trajectories
already done — a partial corpus is always durable. Each emitted trace carries a run-suffixed task
id (``gaia2-train-3#opus48-r1``) so the deterministic trace id never collides; the base task id
and reward ride in the trace metadata. Bedrock reads use a bounded timeout so a hung connection
fails fast instead of wedging a shard for minutes.

Usage (from the repo root; data must be materialized first — see backend/fetch_data.py):
    uv run python packages/environment-capture/gaia2/capture.py \
        --split train --limit 40 \
        --models us.anthropic.claude-opus-4-8,us.anthropic.claude-opus-4-7 \
        --out packages/environment-capture/gaia2/traces.otel.jsonl
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import threading
import time
from pathlib import Path

import boto3
from botocore.config import Config
from environment_capture import (
    Task,
    Trajectory,
    partition_contained,
    trajectory_to_spans,
    write_spans_jsonl,
)
from environment_capture.agent import ConverseClient
from environment_capture.benchmarks.gaia2 import Gaia2Adapter, Gaia2Agent
from environment_capture.hub_push import add_hub_args, push_after_capture

_HERE = Path(__file__).parent
_BENCHMARK = "gaia2"
_TASK_ATTEMPTS = 2


def _short_model(model_id: str) -> str:
    """A compact tag for a Bedrock model id, e.g. us.anthropic.claude-opus-4-8 -> opus48."""
    tail = model_id.rsplit(".", 1)[-1].removeprefix("claude-")
    return tail.replace("-", "").replace("v1", "")


def _bounded_client(region: str = "us-east-1") -> ConverseClient:
    """A Bedrock client whose reads fail fast (a hung call must not wedge a shard for minutes)."""
    return boto3.client(
        "bedrock-runtime",
        region_name=region,
        config=Config(connect_timeout=10, read_timeout=90, retries={"max_attempts": 0}),
    )


@dataclasses.dataclass
class _Sink:
    """Thread-safe, incremental writer: appends each graded trajectory to out + raw immediately."""

    out: Path
    raw: Path
    benchmark: str
    _lock: threading.Lock = dataclasses.field(default_factory=threading.Lock)
    n_traces: int = 0
    n_steps: int = 0
    reward_sum: float = 0.0

    def emit(self, trajectory: Trajectory) -> None:
        with self._lock:
            with self.raw.open("a", encoding="utf-8") as raw:
                raw.write(json.dumps(dataclasses.asdict(trajectory), ensure_ascii=False) + "\n")
            if not trajectory.steps:
                # A step-less run (agent answered without touching the env) leaves no spans, so
                # it must not count toward the corpus totals the summary reports — the raw
                # record above still preserves it for debugging.
                return
            self.n_traces += 1
            self.n_steps += len(trajectory.steps)
            self.reward_sum += trajectory.reward or 0.0
            write_spans_jsonl(
                trajectory_to_spans(trajectory, benchmark=self.benchmark), self.out, append=True
            )


def _capture_shard(
    model_id: str,
    tasks: list[Task],
    split: str,
    max_steps: int,
    run_tag: str,
    sink: _Sink,
) -> None:
    """Run one model over its shard, emitting each graded, hygiene-clean trajectory immediately."""
    adapter = Gaia2Adapter(_HERE, experiment_prefix=run_tag)
    agent = Gaia2Agent(model_id, client=_bounded_client(), max_steps=max_steps)
    for task in tasks:
        trajectory: Trajectory | None = None
        last_error = ""
        for _attempt in range(_TASK_ATTEMPTS):
            try:
                env = adapter.open_env(task)
            except Exception as error:  # noqa: BLE001 - isolate per-task failures like run_capture
                last_error = f"{type(error).__name__}: {error}"
                continue
            try:
                run = agent.run(task, env)
            except Exception as error:  # noqa: BLE001 - isolate per-task failures like run_capture
                last_error = f"{type(error).__name__}: {error}"
                continue
            finally:
                env.close()
            try:
                reward = adapter.grade(task, run.final_answer)
            except Exception as error:  # noqa: BLE001 - a grader edge case must not kill the shard
                last_error = f"{type(error).__name__}: {error}"
                continue
            trajectory = Trajectory(
                task=task,
                steps=run.steps,
                final_answer=run.final_answer,
                reward=reward,
                model=run.model,
                split=split,
            )
            break
        if trajectory is None:
            print(f"[skip] {task.task_id} on {model_id}: {last_error}", file=sys.stderr)
            continue
        contained, flagged = partition_contained([trajectory])
        if flagged:
            print(f"[drop] {task.task_id}: host-escape content", file=sys.stderr)
            continue
        suffixed = dataclasses.replace(task, task_id=f"{task.task_id}#{run_tag}")
        sink.emit(
            dataclasses.replace(
                trajectory,
                task=suffixed,
                metadata={
                    "base_task_id": task.task_id,
                    "reward_kind": "structural-approx-not-official-gaia2",
                    "source": "meta-agents-research-environments/gaia2 (CC-BY-4.0)",
                },
            )
        )


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
    parser.add_argument("--max-steps", type=int, default=18)
    parser.add_argument("--out", default=str(_HERE / "traces.otel.jsonl"))
    parser.add_argument("--append", action="store_true", help="Append to --out (default: refuse)")
    add_hub_args(parser)
    args = parser.parse_args()

    out = Path(args.out)
    if out.exists() and not args.append:
        raise SystemExit(f"{out} exists; pass --append to extend it")

    adapter = Gaia2Adapter(_HERE)
    model_ids = [m.strip() for m in args.models.split(",") if m.strip()]
    tasks = adapter.tasks(args.split)[args.skip :]
    if args.limit is not None:
        tasks = tasks[: args.limit]

    started = time.time()
    runs_dir = _HERE / "runs"
    runs_dir.mkdir(exist_ok=True)
    sink = _Sink(out=out, raw=runs_dir / f"capture-{int(started)}.jsonl", benchmark=_BENCHMARK)

    for run_index in range(args.runs):
        shards = [tasks[i :: len(model_ids)] for i in range(len(model_ids))]
        threads = [
            threading.Thread(
                target=_capture_shard,
                args=(
                    model_id,
                    shard,
                    args.split,
                    args.max_steps,
                    f"{_short_model(model_id)}-r{args.run_start + run_index}",
                    sink,
                ),
            )
            for model_id, shard in zip(model_ids, shards, strict=True)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    mean_reward = sink.reward_sum / sink.n_traces if sink.n_traces else 0.0
    print(
        f"captured {sink.n_traces} runs with transitions ({sink.n_steps} steps, "
        f"mean reward {mean_reward:.3f}) in {time.time() - started:.0f}s -> {out} (raw: {sink.raw})"
    )
    push_after_capture("gaia2", enabled=args.push_hub, private=args.hub_private)


if __name__ == "__main__":
    main()
