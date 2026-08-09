#!/usr/bin/env python3
"""Record full health for repeated matched candidate TBLite evaluations."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

TRAIN_SEED = 20260809
CONTEXT_ERROR = "maximum context length is 65536"


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
    result = []
    for line in proc.stdout.splitlines():
        values = [int(value.strip()) for value in line.split(",")]
        result.append(
            {
                "index": values[0],
                "utilization_percent": values[1],
                "memory_mib": values[2],
                "memory_total_mib": values[3],
            }
        )
    return result


def reward(row: dict[str, Any]) -> float | None:
    verifier = row.get("verifier_result") or {}
    rewards = (verifier.get("rewards") or {}) if isinstance(verifier, dict) else {}
    value = rewards.get("reward")
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0 if exception_type(row) is not None else None


def exception_type(row: dict[str, Any]) -> str | None:
    exception = row.get("exception_info") or row.get("exception")
    if not isinstance(exception, dict):
        return None
    value = exception.get("exception_type") or exception.get("type")
    return str(value) if value else None


def context_overflow_trial(trial: Path) -> bool:
    for relative in (
        Path("agent/mini-swe-agent.txt"),
        Path("agent/mini-swe-agent.trajectory.json"),
        Path("trial.log"),
    ):
        path = trial / relative
        try:
            if CONTEXT_ERROR in path.read_text(errors="replace"):
                return True
        except OSError:
            continue
    return False


def arm_health(job: Path) -> dict[str, Any]:
    rows = []
    result_paths = list(job.glob("*/result.json"))
    for path in result_paths:
        try:
            rows.append(json.loads(path.read_text()))
        except (json.JSONDecodeError, OSError):
            continue
    rewards = [value for row in rows if (value := reward(row)) is not None]
    exceptions = Counter(
        value for row in rows if (value := exception_type(row)) is not None
    )
    summary_path = job / "result.json"
    try:
        stats = json.loads(summary_path.read_text()).get("stats", {})
    except (json.JSONDecodeError, OSError):
        stats = {}
    return {
        "result_files": len(rows),
        "scored": len(rewards),
        "strict": sum(value == 1.0 for value in rewards),
        "graded_mean": sum(rewards) / len(rewards) if rewards else None,
        "exceptions": dict(sorted(exceptions.items())),
        "context_overflow_trials": sum(
            context_overflow_trial(path.parent) for path in result_paths
        ),
        "harbor": {
            key: stats.get(key)
            for key in (
                "n_completed_trials",
                "n_errored_trials",
                "n_running_trials",
                "n_pending_trials",
                "n_cancelled_trials",
                "n_retries",
            )
        },
    }


def repeat_health(root: Path, step: int, eval_seed: int) -> dict[str, Any]:
    prefix = (
        f"qwen35-4b-candidate-seed{TRAIN_SEED}-step{step}"
        f"-full100-eval-seed{eval_seed}"
    )
    eval_root = root / (
        f"candidate-step{step}-seed{TRAIN_SEED}"
        f"-tblite-full100-eval-seed{eval_seed}-run1"
    )
    arm = f"candidate-seed{TRAIN_SEED}-step{step}"
    log = root / "logs" / f"{prefix}.log"
    text = log.read_text(errors="replace") if log.exists() else ""
    return {
        "eval_seed": eval_seed,
        "base": arm_health(eval_root / "jobs" / f"{prefix}-base-run1"),
        "adapter": arm_health(eval_root / "jobs" / f"{prefix}-{arm}-run1"),
        "paired_report": (eval_root / "paired-vs-base-full100.json").is_file(),
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
    parser.add_argument("--step", type=int, required=True, choices=(100, 200))
    args = parser.parse_args()
    disk = shutil.disk_usage(args.root)
    snapshot = {
        "timestamp": datetime.now(UTC).isoformat(),
        "checkpoint_step": args.step,
        "orchestrator_alive": session_alive(
            f"candidate-step{args.step}-repeated-tblite"
        ),
        "server_alive": session_alive(
            f"qwen35-4b-candidate-step{args.step}-seeds-serve"
        ),
        "aggregate_written": (
            args.root
            / f"candidate-step{args.step}-seed{TRAIN_SEED}"
            "-tblite-repeated-eval-seeds0-2.json"
        ).is_file(),
        "repeats": [repeat_health(args.root, args.step, seed) for seed in range(3)],
        "disk": {"free": disk.free, "used": disk.used, "total": disk.total},
        "gpus": gpu_health(),
    }
    args.health_log.parent.mkdir(parents=True, exist_ok=True)
    with args.health_log.open("a") as handle:
        handle.write(json.dumps(snapshot, sort_keys=True) + "\n")
    print(json.dumps(snapshot, sort_keys=True))  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
