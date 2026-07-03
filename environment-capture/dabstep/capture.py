"""Capture fresh REAL DABstep runs on Bedrock and append them to the trace corpus.

DABstep's train split is tiny (a handful of shared-context questions), so unlike a large benchmark
we do NOT round-robin shard: every model runs the FULL split (one thread per model — the pattern
for beating per-model throttling), which grows per-task coverage across models. Each trajectory's
task id is suffixed with the model + run tag (e.g. ``dab-train-3#opus48-r1``) so the deterministic
trace ids never collide across models or repeated runs. Grading uses the original task id; only the
emitted span carries the suffixed id. Raw graded trajectories are also written to ``runs/`` as JSONL
(gitignored) so a capture can be inspected without re-running.

The agent is given a workspace-scoped system prompt so it looks for its data under ``./data/``
instead of hunting across the host (the shared default prompt points at a ``docs/`` dir that does
not exist here). ``LocalBashEnv`` also refuses host-targeting commands and ``partition_contained``
drops any still-flagged trajectory at emit, so the corpus stays free of host filesystem content.

Usage (from the repo root, after fetch_data.py has pulled payments.csv):
    uv run python environment-capture/dabstep/capture.py \
        --models us.anthropic.claude-opus-4-8,us.anthropic.claude-opus-4-7 --runs 1 \
        --out environment-capture/dabstep/traces.otel.jsonl --append
"""

from __future__ import annotations

import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path

from environment_capture import (
    Trajectory,
    partition_contained,
    run_capture,
    trajectory_to_spans,
    write_spans_jsonl,
)
from environment_capture.agent import BedrockBashAgent
from environment_capture.benchmarks.dabstep import DabstepAdapter
from environment_capture.trajectory import Task

_HERE = Path(__file__).parent
_TASK_ATTEMPTS = 3  # a tiny split; retry transient Bedrock/network blips before giving up on a task

# Point the agent at ./data/ so it does not go hunting across the host for "the data" (the shared
# default prompt names a docs/ dir that does not exist for dabstep).
_WORKSPACE_SYSTEM_PROMPT = """You are an autonomous data-analysis agent. Your task's data files are
in the ./data/ directory of your current workspace (CSV/JSON plus a manual.md defining the business
rules you MUST follow to interpret the columns). Start by running `ls data/`; read the manual first,
then analyze with python3 + pandas. Work ONLY within this workspace using relative paths. Never run
commands that target the host filesystem (no `ls ~`, `find /`, `cd ..`, or absolute paths like
/tmp, /Users, /home): the environment BLOCKS them and a single such command invalidates the whole
run. Use the bash tool one focused command per call and check intermediate results. When confident,
call submit with the final answer in EXACTLY the format the question asks for (and nothing else)."""


def _model_tag(model_id: str) -> str:
    """Short alphanumeric id for a Bedrock model, used to keep suffixed task ids unique."""
    tail = model_id.split("claude-")[-1]
    return re.sub(r"[^a-z0-9]", "", tail)


def _suffix_task_id(trajectory: Trajectory, tag: str) -> Trajectory:
    """Re-key a graded trajectory's task id with a run suffix (after grading, before emission)."""
    task = replace(trajectory.task, task_id=f"{trajectory.task.task_id}#{tag}")
    return replace(trajectory, task=task)


def _capture_model(
    adapter: DabstepAdapter,
    model_id: str,
    tasks: list[Task],
    runs: int,
    run_start: int,
    max_steps: int,
) -> list[Trajectory]:
    """Run one model over every task; run_capture isolates and retries per-task failures.

    Runs are numbered from ``run_start`` so a top-up capture (``--run-start 2``) never reuses an
    earlier run's suffix (and thus trace id).
    """
    agent = BedrockBashAgent(model_id, max_steps=max_steps, system_prompt=_WORKSPACE_SYSTEM_PROMPT)
    tag = _model_tag(model_id)
    captured: list[Trajectory] = []
    for run_index in range(run_start, run_start + runs):
        result = run_capture(adapter, agent, split="train", tasks=tasks, attempts=_TASK_ATTEMPTS)
        for failure in result.failures:
            print(f"  [{tag} r{run_index}] {failure.task_id} skipped: {failure.error}")
        contained, flagged = partition_contained(result.trajectories)
        for trajectory in flagged:
            print(f"  [{tag} r{run_index}] {trajectory.task.task_id} dropped: host escape")
        captured.extend(_suffix_task_id(t, f"{tag}-r{run_index}") for t in contained)
    return captured


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--limit", type=int, default=None, help="Cap the number of tasks")
    parser.add_argument("--skip", type=int, default=0, help="Skip the first N tasks")
    parser.add_argument(
        "--models",
        default="us.anthropic.claude-opus-4-8",
        help="Comma-separated Bedrock model ids; each runs the FULL split",
    )
    parser.add_argument("--runs", type=int, default=1, help="Runs per model over the split")
    parser.add_argument(
        "--run-start",
        type=int,
        default=1,
        help="First run index (bump to top up a corpus without reusing earlier run suffixes)",
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

    adapter = DabstepAdapter(data_root=_HERE)
    model_ids = [m.strip() for m in args.models.split(",") if m.strip()]
    tasks = adapter.tasks(args.split)[args.skip :]
    if args.limit is not None:
        tasks = tasks[: args.limit]

    started = time.time()
    with ThreadPoolExecutor(max_workers=len(model_ids)) as pool:
        model_results = list(
            pool.map(
                lambda model_id: _capture_model(
                    adapter, model_id, tasks, args.runs, args.run_start, args.max_steps
                ),
                model_ids,
            )
        )
    trajectories = [t for result in model_results for t in result]

    runs_dir = _HERE / "runs"
    runs_dir.mkdir(exist_ok=True)
    raw_path = runs_dir / f"capture-{int(started)}.jsonl"
    with raw_path.open("w", encoding="utf-8") as raw:
        for trajectory in trajectories:
            raw.write(json.dumps(asdict(trajectory), ensure_ascii=False) + "\n")

    kept = [t for t in trajectories if t.steps]
    n_spans = 0
    for index, trajectory in enumerate(kept):
        spans = trajectory_to_spans(trajectory, benchmark="dabstep")
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
