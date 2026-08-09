#!/usr/bin/env python3
"""Record health and partial outcomes for two matched step-100 TBLite canaries."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, check=False, capture_output=True, text=True)


def session_alive(name: str) -> bool:
    return run("tmux", "has-session", "-t", name).returncode == 0


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
        values = [int(value.strip()) for value in line.split(",")]
        rows.append(
            {
                "index": values[0],
                "utilization_percent": values[1],
                "memory_mib": values[2],
                "memory_total_mib": values[3],
            }
        )
    return rows


def trial_rows(job: Path) -> list[dict[str, Any]]:
    rows = []
    for path in job.glob("*/result.json"):
        try:
            rows.append(json.loads(path.read_text()))
        except (json.JSONDecodeError, OSError):
            continue
    return rows


def reward(row: dict[str, Any]) -> float | None:
    value = ((row.get("verifier_result") or {}).get("rewards") or {}).get("reward")
    return float(value) if value is not None else None


def exception_type(row: dict[str, Any]) -> str | None:
    exception = row.get("exception_info") or row.get("exception")
    if isinstance(exception, dict):
        return str(exception.get("exception_type") or exception.get("type") or "unknown")
    return None


def arm_health(job: Path) -> dict[str, Any]:
    rows = trial_rows(job)
    rewards = [value for row in rows if (value := reward(row)) is not None]
    exceptions: dict[str, int] = {}
    for row in rows:
        if kind := exception_type(row):
            exceptions[kind] = exceptions.get(kind, 0) + 1
    return {
        "result_files": len(rows),
        "scored": len(rewards),
        "strict": sum(value == 1.0 for value in rewards),
        "graded_mean": sum(rewards) / len(rewards) if rewards else None,
        "exceptions": dict(sorted(exceptions.items())),
    }


def seed_health(root: Path, seed: int, step: int) -> dict[str, Any]:
    eval_root = root / f"candidate-step{step}-seed{seed}-tblite-canary10-seed0-run1"
    prefix = f"qwen35-4b-candidate-seed{seed}-step{step}-canary10-seed0"
    arm = f"candidate-seed{seed}-step{step}"
    log = root / "logs" / f"{prefix}.log"
    text = log.read_text(errors="replace") if log.exists() else ""
    return {
        "seed": seed,
        "base": arm_health(eval_root / "jobs" / f"{prefix}-base-run1"),
        "adapter": arm_health(eval_root / "jobs" / f"{prefix}-{arm}-run1"),
        "paired_report": (eval_root / "paired-vs-base-canary10.json").is_file(),
        "signals": {
            "oom": bool(re.search(r"out of memory|cuda oom", text, re.I)),
            "nan": bool(re.search(r"(?<![a-z])nan(?![a-z])", text, re.I)),
            "traceback": "traceback" in text.lower(),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--health-log", required=True, type=Path)
    parser.add_argument("--step", type=int, default=100)
    args = parser.parse_args()
    disk = shutil.disk_usage(args.root)
    snapshot = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checkpoint_step": args.step,
        "orchestrator_alive": session_alive(f"candidate-step{args.step}-two-seed-canaries"),
        "server_alive": session_alive(f"qwen35-4b-candidate-step{args.step}-seeds-serve"),
        "selection_gate_written": (
            args.root / f"candidate-step{args.step}-two-seed-canary-gate.json"
        ).is_file(),
        "seeds": [seed_health(args.root, seed, args.step) for seed in (20260809, 20260810)],
        "disk": {"free": disk.free, "used": disk.used, "total": disk.total},
        "gpus": gpu_health(),
    }
    args.health_log.parent.mkdir(parents=True, exist_ok=True)
    with args.health_log.open("a") as handle:
        handle.write(json.dumps(snapshot, sort_keys=True) + "\n")
    print(json.dumps(snapshot, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
