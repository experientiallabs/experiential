"""Capture fresh REAL GAIA runs on Bedrock and append them to the trace corpus.

The in-scope validation slice is small (see ``fetch_data.py``), so like the other small-split
benchmarks we do NOT round-robin shard: every model runs the FULL train split (one thread per
model — the pattern for beating per-model throttling), and repeated runs grow per-task coverage.
Each trajectory's task id is suffixed with the model + run tag (e.g. ``gaia-train-3#opus48-r1``) so
the deterministic trace ids never collide across models or repeated runs; grading uses the base id,
only the emitted span carries the suffix. Raw graded trajectories are also written to ``runs/`` as
JSONL (gitignored) so a capture can be inspected without re-running.

The agent works offline in a workspace holding the task's single attachment; it parses the file for
real with ``python3`` (the machine's ``python3`` must have pandas/openpyxl/pdfplumber and friends
installed so xlsx/pdf/docx/pptx parse — heavy, gitignored, install once). ``LocalBashEnv`` refuses
host-targeting commands and ``partition_contained`` drops any still-flagged trajectory at emit, so
the corpus stays free of host filesystem content.

Because GAIA's terms forbid resharing its data in a crawlable format, ``traces.otel.jsonl`` is
gitignored (observations quote attachment contents): capture is reproducible from this script, but
the corpus itself is not committed.

Usage (from the repo root, after fetch_data.py has materialized the slice):
    uv run python environment-capture/gaia/capture.py \
        --models us.anthropic.claude-opus-4-8,us.anthropic.claude-opus-4-7 --runs 2 \
        --out environment-capture/gaia/traces.otel.jsonl
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
from environment_capture.benchmarks.gaia import GaiaAdapter
from environment_capture.trajectory import Task

_HERE = Path(__file__).parent
# a small split; retry transient Bedrock/network blips before giving up on a task
_TASK_ATTEMPTS = 3

# Point the agent at its single attachment in the current directory and pin GAIA's answer format so
# submissions grade against the strict normalized exact-match scorer.
_WORKSPACE_SYSTEM_PROMPT = """You are an autonomous general-assistant agent solving one question
from a single attached file. The file is in your CURRENT workspace directory. Start by running `ls`
and `file *` to see it, then parse it for real: use `python3` with pandas/openpyxl for spreadsheets,
pdfplumber or pdftotext for PDFs, python-docx/python-pptx for documents, and plain tools for CSV/
text/code. You are OFFLINE — there is no network. Work ONLY within this workspace using relative
paths; commands that target the host filesystem (no `ls ~`, `find /`, `cd ..`, or absolute paths
like /tmp, /Users, /home) are BLOCKED and a single such command invalidates the whole run. Use the
bash tool one focused command per call and check intermediate results.

When confident, call submit with your final answer as a number OR as few words as possible OR a
comma separated list. If asked for a number, do not use thousands commas or units ($ or %) unless
the question asks for them. If asked for a string, do not use articles or abbreviations and write
digits in plain text unless asked otherwise. For a list, apply those rules per element. Submit ONLY
the answer, nothing else."""


def _model_tag(model_id: str) -> str:
    """Short alphanumeric id for a Bedrock model, used to keep suffixed task ids unique."""
    tail = model_id.split("claude-")[-1]
    return re.sub(r"[^a-z0-9]", "", tail)


def _suffix_task_id(trajectory: Trajectory, tag: str) -> Trajectory:
    """Re-key a graded trajectory's task id with a run suffix (after grading, before emission)."""
    task = replace(trajectory.task, task_id=f"{trajectory.task.task_id}#{tag}")
    return replace(trajectory, task=task)


def _capture_model(
    adapter: GaiaAdapter,
    model_id: str,
    tasks: list[Task],
    runs: int,
    run_start: int,
    max_steps: int,
) -> list[Trajectory]:
    """Run one model over every task; run_capture isolates and retries per-task failures.

    Runs are numbered from ``run_start`` so a top-up capture (``--run-start 3``) never reuses an
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
    parser.add_argument(
        "--append", action="store_true", help="Append to an existing --out (default: overwrite)"
    )
    args = parser.parse_args()

    out = Path(args.out)
    if args.split == "test":
        raise SystemExit(
            "refusing to capture the test split into a corpus: the hidden test split must stay "
            "out of world-model training data"
        )

    adapter = GaiaAdapter(data_root=_HERE)
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
        spans = trajectory_to_spans(trajectory, benchmark="gaia")
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
