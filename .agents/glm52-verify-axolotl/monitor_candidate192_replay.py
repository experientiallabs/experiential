#!/usr/bin/env python3
"""Append one fail-obvious health snapshot for the candidate-192 verifier replay."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, check=False, capture_output=True, text=True)


def session_alive(name: str) -> bool:
    return run("tmux", "has-session", "-t", name).returncode == 0


def load_results(root: Path) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for path in sorted(root.glob("episodes/*/replay_result.json")):
        try:
            results.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            results.append({"status": "unreadable_result", "path": str(path)})
    return results


def gpu_health() -> list[dict[str, int]]:
    proc = run(
        "nvidia-smi",
        "--query-gpu=index,utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    )
    if proc.returncode != 0:
        return []
    rows = []
    for line in proc.stdout.splitlines():
        index, utilization, used, total = (int(value.strip()) for value in line.split(","))
        rows.append(
            {
                "index": index,
                "utilization_percent": utilization,
                "memory_mib": used,
                "memory_total_mib": total,
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--health-log", type=Path, required=True)
    args = parser.parse_args()

    results = load_results(args.run_root)
    status_counts = Counter(str(row.get("status", "missing")) for row in results)
    rewards = [float(row["reward"]) for row in results if row.get("reward") is not None]
    error_classes = Counter(
        str(row.get("error_class", "unknown"))
        for row in results
        if row.get("status") not in ("scored", "starting", None)
    )
    log_tail = ""
    if args.log.exists():
        log_tail = "\n".join(args.log.read_text(encoding="utf-8", errors="replace").splitlines()[-200:])
    lowered = log_tail.lower()
    disk = shutil.disk_usage(args.root)
    summary_path = args.run_root / "summary.json"
    snapshot = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "session_alive": session_alive(args.session),
        "results": len(results),
        "status_counts": dict(sorted(status_counts.items())),
        "scored": len(rewards),
        "perfect": sum(reward == 1.0 for reward in rewards),
        "positive": sum(reward > 0.0 for reward in rewards),
        "mean_reward": sum(rewards) / len(rewards) if rewards else None,
        "error_classes": dict(sorted(error_classes.items())),
        "summary_exists": summary_path.exists(),
        "log_signals": {
            "traceback": "traceback" in lowered,
            "oom": "out of memory" in lowered or "cuda oom" in lowered,
            "nan": "nan" in lowered,
        },
        "disk": {"free": disk.free, "used": disk.used, "total": disk.total},
        "gpus": gpu_health(),
    }
    args.health_log.parent.mkdir(parents=True, exist_ok=True)
    with args.health_log.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(snapshot, sort_keys=True) + "\n")
    print(json.dumps(snapshot, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
