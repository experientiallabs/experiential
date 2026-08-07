#!/usr/bin/env python3
"""Merge a preserved incomplete TMax arm with fresh recovery episodes."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


TERMINAL_STATUSES = {
    "scored",
    "scored_after_agent_error",
    "episode_error",
    "infrastructure_setup_failed",
    "infrastructure_initial_state_failed",
}


def sha256_file(path: Path) -> str:
    """Return a file's SHA-256 digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_episodes(root: Path) -> dict[str, tuple[dict[str, Any], Path]]:
    """Load unique TMax episode results by task ID."""
    rows: dict[str, tuple[dict[str, Any], Path]] = {}
    for path in sorted(root.rglob("episode_result.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        task_id = row.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            raise ValueError(f"missing task_id: {path}")
        if task_id in rows:
            raise ValueError(f"duplicate task_id {task_id}: {root}")
        rows[task_id] = (row, path)
    if not rows:
        raise ValueError(f"no episode results: {root}")
    return rows


def select_rows(
    original: dict[str, tuple[dict[str, Any], Path]],
    recovery: dict[str, tuple[dict[str, Any], Path]],
) -> tuple[dict[str, tuple[dict[str, Any], Path]], list[str]]:
    """Replace only nonterminal original episodes with matching recovery rows."""
    extra = sorted(set(recovery) - set(original))
    if extra:
        raise ValueError(f"recovery has unexpected tasks: {extra}")
    selected = dict(original)
    replaced: list[str] = []
    for task_id, recovered in recovery.items():
        original_status = original[task_id][0].get("status")
        if original_status in TERMINAL_STATUSES:
            raise ValueError(
                f"refusing to replace terminal original task {task_id}: {original_status}"
            )
        recovery_status = recovered[0].get("status")
        if recovery_status not in TERMINAL_STATUSES:
            raise ValueError(
                f"recovery task {task_id} is not terminal: {recovery_status}"
            )
        selected[task_id] = recovered
        replaced.append(task_id)
    nonterminal = sorted(
        task_id
        for task_id, (row, _path) in selected.items()
        if row.get("status") not in TERMINAL_STATUSES
    )
    if nonterminal:
        raise ValueError(f"combined arm remains nonterminal: {nonterminal}")
    return selected, sorted(replaced)


def main() -> int:
    """Write the immutable combined arm and its merge manifest."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--recovery", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-tasks", type=int, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite: {args.out}")
    original = load_episodes(args.original)
    recovery = load_episodes(args.recovery)
    if len(original) != args.expected_tasks:
        raise ValueError(
            f"original attempted {len(original)}, expected {args.expected_tasks}"
        )
    selected, replaced = select_rows(original, recovery)

    output_paths: list[Path] = []
    for task_id in sorted(selected):
        row, source_path = selected[task_id]
        destination = args.out / "episodes" / task_id / "episode_result.json"
        destination.parent.mkdir(parents=True, exist_ok=False)
        destination.write_text(
            json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        output_paths.append(destination)
        if json.loads(destination.read_text(encoding="utf-8")) != row:
            raise ValueError(f"round-trip mismatch: {task_id}")

    rows = [row for row, _path in selected.values()]
    scored = [row for row in rows if isinstance(row.get("reward"), (int, float))]
    summary = {
        "event": "eval_complete_recovered",
        "arm": "sft-step100",
        "model": "hosted_vllm/qwen35-4b-glm52-verified17-sft-step100",
        "attempted": len(rows),
        "scored": len(scored),
        "reward_sum": sum(float(row["reward"]) for row in scored),
        "mean_reward": (
            sum(float(row["reward"]) for row in scored) / len(scored)
            if scored
            else None
        ),
        "status_counts": dict(Counter(str(row.get("status")) for row in rows)),
        "recovery_merge_used": True,
        "recovered_task_ids": replaced,
    }
    summary_path = args.out / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema": "xtoken-tmax-recovery-merge-v1",
        "original_root": str(args.original),
        "recovery_root": str(args.recovery),
        "original_task_count": len(original),
        "recovery_task_count": len(recovery),
        "combined_task_count": len(selected),
        "replaced_nonterminal_task_ids": replaced,
        "original_episode_result_sha256": {
            task_id: sha256_file(path) for task_id, (_row, path) in original.items()
        },
        "recovery_episode_result_sha256": {
            task_id: sha256_file(path) for task_id, (_row, path) in recovery.items()
        },
        "combined_episode_result_sha256": {
            path.parent.name: sha256_file(path) for path in output_paths
        },
        "summary_sha256": sha256_file(summary_path),
    }
    (args.out / "merge_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
