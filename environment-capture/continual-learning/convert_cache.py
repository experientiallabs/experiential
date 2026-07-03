"""Convert a frozen baseline cache of REAL database-exploration runs into the wmh trace corpus.

The cache holds REAL runs (real ``sqlite3``/``python3`` commands, real recorded outputs); this
re-emits them on the wmh OTel GenAI wire format with provenance in the trace metadata. The
recording harness's submission sentinel is normalized to the neutral ``SUBMIT`` by the shared
loader (see ``environment_capture.baseline_cache``). Zero-transition trajectories (the agent
submitted without running a command) produce no spans and are skipped explicitly.

Usage:
    uv run python environment-capture/continual-learning/convert_cache.py \
        --cache <path-to-baseline-cache-train-dir> --out traces.otel.jsonl
"""

from __future__ import annotations

import argparse
from pathlib import Path

from environment_capture import (
    load_baseline_cache,
    partition_contained,
    trajectory_to_spans,
    write_spans_jsonl,
)

_BENCHMARK = "continual-learning"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, help="Baseline-cache dir (manifest/tasks/traces)")
    parser.add_argument("--out", required=True, help="Output OTel GenAI JSONL path")
    args = parser.parse_args()

    trajectories = load_baseline_cache(Path(args.cache))
    contained, flagged = partition_contained(trajectories)
    for trajectory in flagged:
        print(f"[drop] {trajectory.task.task_id}: host-escape content (see hygiene.py)")
    kept = [t for t in contained if t.steps]
    skipped = len(contained) - len(kept)

    out = Path(args.out)
    n_spans = 0
    for index, trajectory in enumerate(kept):
        spans = trajectory_to_spans(trajectory, benchmark=_BENCHMARK)
        n_spans += write_spans_jsonl(spans, out, append=index > 0)

    n_steps = sum(len(t.steps) for t in kept)
    print(
        f"wrote {len(kept)} traces / {n_steps} steps / {n_spans} spans -> {out} "
        f"(skipped {skipped} zero-step, dropped {len(flagged)} host-escape trajectories)"
    )


if __name__ == "__main__":
    main()
