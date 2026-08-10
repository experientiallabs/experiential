#!/usr/bin/env python3
"""Record health and partial outcomes for two matched step-100 TBLite canaries."""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

CONTEXT_ERROR = "maximum context length is"
logger = logging.getLogger(__name__)

NAN_SIGNAL_PATTERNS = (
    r"\b(?:loss|gradient|grad_norm|logits?|probabilities|tensor)\s*(?:=|:|is)\s*nan\b",
    r"\bnan\s+(?:loss|gradient|grad_norm|logits?|probabilities|tensor)\b",
    r"\b(?:detected|found|contains?|produced|became)\s+(?:a\s+)?nan\b",
)


def run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, check=False, capture_output=True, text=True)


def session_alive(name: str) -> bool:
    return run("tmux", "has-session", "-t", name).returncode == 0


def orchestrator_alive(step: int, family: str = "candidate") -> bool:
    """Accept the original two-seed runner or a resumable single-seed runner."""
    return session_alive(f"{family}-step{step}-two-seed-canaries") or any(
        session_alive(f"{family}-step{step}-seed{seed}-resume")
        for seed in (20260809, 20260810)
    )


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
    if value is not None:
        return float(value)
    return 0.0 if exception_type(row) is not None else None


def exception_type(row: dict[str, Any]) -> str | None:
    exception = row.get("exception_info") or row.get("exception")
    if isinstance(exception, dict):
        return str(exception.get("exception_type") or exception.get("type") or "unknown")
    return None


def normalized_exception_type(row: dict[str, Any]) -> str | None:
    """Prefer the observed root cause over Harbor's command-text classifier.

    Harbor can label a mini-swe-agent context overflow as ``ApiRateLimitError``
    when the benchmark task prompt itself contains rate-limit requirements.  The
    exception message still contains vLLM's unambiguous context-window error.
    """
    raw = exception_type(row)
    exception = row.get("exception_info") or row.get("exception")
    text = json.dumps(exception, sort_keys=True) if exception is not None else ""
    if "ContextWindowExceededError" in text or CONTEXT_ERROR in text:
        return "ContextWindowExceededError"
    return raw


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


def numerical_nan_signal(text: str) -> bool:
    """Detect numerical NaNs without flagging single-run summary statistics."""
    return any(re.search(pattern, text, re.I) for pattern in NAN_SIGNAL_PATTERNS)


def token_summary(rows: list[dict[str, Any]]) -> dict[str, int | float | None]:
    """Summarize Harbor's cumulative agent token accounting."""
    inputs: list[int] = []
    outputs: list[int] = []
    for row in rows:
        agent_result = row.get("agent_result") or {}
        input_tokens = agent_result.get("n_input_tokens")
        output_tokens = agent_result.get("n_output_tokens")
        if isinstance(input_tokens, (int, float)) and not isinstance(input_tokens, bool):
            inputs.append(int(input_tokens))
        if isinstance(output_tokens, (int, float)) and not isinstance(output_tokens, bool):
            outputs.append(int(output_tokens))
    return {
        "input_tokens_total": sum(inputs),
        "output_tokens_total": sum(outputs),
        "output_tokens_mean": sum(outputs) / len(outputs) if outputs else None,
        "token_accounted_trials": len(outputs),
    }


def arm_health(job: Path) -> dict[str, Any]:
    result_paths = list(job.glob("*/result.json"))
    rows = trial_rows(job)
    rewards = [value for row in rows if (value := reward(row)) is not None]
    exceptions: dict[str, int] = {}
    raw_exceptions: dict[str, int] = {}
    for row in rows:
        if kind := normalized_exception_type(row):
            exceptions[kind] = exceptions.get(kind, 0) + 1
        if raw_kind := exception_type(row):
            raw_exceptions[raw_kind] = raw_exceptions.get(raw_kind, 0) + 1
    return {
        "result_files": len(rows),
        "scored": len(rewards),
        "strict": sum(value == 1.0 for value in rewards),
        "graded_mean": sum(rewards) / len(rewards) if rewards else None,
        "exceptions": dict(sorted(exceptions.items())),
        "raw_exceptions": dict(sorted(raw_exceptions.items())),
        "context_overflow_trials": sum(
            context_overflow_trial(path.parent) for path in result_paths
        ),
        "tokens": token_summary(rows),
    }


def seed_health(
    root: Path,
    seed: int,
    step: int,
    family: str = "candidate",
) -> dict[str, Any]:
    eval_root = root / f"{family}-step{step}-seed{seed}-tblite-canary10-seed0-run1"
    prefix = f"qwen35-4b-{family}-seed{seed}-step{step}-canary10-seed0"
    arm = f"{family}-seed{seed}-step{step}"
    logs = [
        root / "logs" / f"{prefix}.log",
        root / "logs" / f"{prefix}.resume.log",
    ]
    text = "\n".join(
        log.read_text(errors="replace") for log in logs if log.exists()
    )
    return {
        "seed": seed,
        "base": arm_health(eval_root / "jobs" / f"{prefix}-base-run1"),
        "adapter": arm_health(eval_root / "jobs" / f"{prefix}-{arm}-run1"),
        "paired_report": (eval_root / "paired-vs-base-canary10.json").is_file(),
        "signals": {
            "oom": bool(re.search(r"out of memory|cuda oom", text, re.I)),
            "nan": numerical_nan_signal(text),
            "traceback": "traceback" in text.lower(),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--health-log", required=True, type=Path)
    parser.add_argument("--step", type=int, default=100)
    parser.add_argument("--family", default="candidate")
    args = parser.parse_args()
    disk = shutil.disk_usage(args.root)
    snapshot = {
        "timestamp": datetime.now(UTC).isoformat(),
        "checkpoint_step": args.step,
        "family": args.family,
        "orchestrator_alive": orchestrator_alive(args.step, args.family),
        "server_alive": session_alive(
            f"qwen35-4b-{args.family}-step{args.step}-seeds-serve"
        ),
        "selection_gate_written": (
            args.root / f"{args.family}-step{args.step}-two-seed-canary-gate.json"
        ).is_file(),
        "seeds": [
            seed_health(args.root, seed, args.step, args.family)
            for seed in (20260809, 20260810)
        ],
        "disk": {"free": disk.free, "used": disk.used, "total": disk.total},
        "gpus": gpu_health(),
    }
    args.health_log.parent.mkdir(parents=True, exist_ok=True)
    with args.health_log.open("a") as handle:
        handle.write(json.dumps(snapshot, sort_keys=True) + "\n")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger.info(json.dumps(snapshot, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
