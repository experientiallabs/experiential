#!/usr/bin/env python3
"""Explain task-level timeout and token-use differences between TB2 arms."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

CONTEXT_ERROR = "maximum context length is"


def _timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _quantile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def load_job(job_dir: Path) -> dict[str, dict[str, Any]]:
    """Load one completed Harbor result for each unique task in a job."""
    rows: dict[str, dict[str, Any]] = {}
    for result_path in sorted(job_dir.glob("*/result.json")):
        result = json.loads(result_path.read_text())
        task_name = result["task_name"].removeprefix("terminal-bench/")
        if task_name in rows:
            raise ValueError(f"duplicate task {task_name!r} in {job_dir}")
        agent = result.get("agent_result") or {}
        rollouts = agent.get("rollout_details") or []
        completion_lengths = [
            len(completion)
            for item in rollouts
            for completion in item.get("completion_token_ids") or []
        ]
        prompt_lengths = [
            len(prompt) for item in rollouts for prompt in item.get("prompt_token_ids") or []
        ]
        log_path = result_path.with_name("trial.log")
        log_text = log_path.read_text(errors="replace") if log_path.exists() else ""
        exception = result.get("exception_info") or {}
        started_at = _timestamp(result["started_at"])
        finished_at = _timestamp(result["finished_at"])
        rows[task_name] = {
            "task_name": task_name,
            "trial_name": result["trial_name"],
            "timeout": exception.get("exception_type") == "AgentTimeoutError",
            "exception_type": exception.get("exception_type"),
            "elapsed_seconds": (finished_at - started_at).total_seconds(),
            "input_tokens": int(agent.get("n_input_tokens") or 0),
            "output_tokens": int(agent.get("n_output_tokens") or 0),
            "llm_calls": len(completion_lengths),
            "max_prompt_tokens_recorded": max(prompt_lengths, default=0),
            "max_completion_tokens": max(completion_lengths, default=0),
            "context_rejections": log_text.count(CONTEXT_ERROR),
        }
    if not rows:
        raise ValueError(f"no trial results found in {job_dir}")
    return rows


def summarize(rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Summarize one arm without dropping failed trials."""
    values = list(rows.values())
    output = [float(row["output_tokens"]) for row in values]
    calls = [float(row["llm_calls"]) for row in values]
    elapsed = [float(row["elapsed_seconds"]) for row in values]
    return {
        "task_count": len(values),
        "timeout_count": sum(bool(row["timeout"]) for row in values),
        "context_rejection_count": sum(int(row["context_rejections"]) for row in values),
        "total_input_tokens": sum(int(row["input_tokens"]) for row in values),
        "total_output_tokens": sum(int(row["output_tokens"]) for row in values),
        "llm_calls_mean": statistics.fmean(calls),
        "llm_calls_p50": _quantile(calls, 0.5),
        "llm_calls_p90": _quantile(calls, 0.9),
        "output_tokens_mean": statistics.fmean(output),
        "output_tokens_p50": _quantile(output, 0.5),
        "output_tokens_p90": _quantile(output, 0.9),
        "elapsed_seconds_mean": statistics.fmean(elapsed),
        "elapsed_seconds_p50": _quantile(elapsed, 0.5),
        "elapsed_seconds_p90": _quantile(elapsed, 0.9),
        "max_prompt_tokens_recorded": max(int(row["max_prompt_tokens_recorded"]) for row in values),
        "max_completion_tokens": max(int(row["max_completion_tokens"]) for row in values),
    }


def compare(base_job: Path, adapter_job: Path) -> dict[str, Any]:
    """Build a paired diagnostic report for two exact task sets."""
    base = load_job(base_job)
    adapter = load_job(adapter_job)
    if set(base) != set(adapter):
        raise ValueError(
            f"task mismatch: missing={sorted(set(base) - set(adapter))}, "
            f"extra={sorted(set(adapter) - set(base))}"
        )
    base_timeouts = {name for name, row in base.items() if row["timeout"]}
    adapter_timeouts = {name for name, row in adapter.items() if row["timeout"]}
    per_task = []
    for name in sorted(base):
        base_row = base[name]
        adapter_row = adapter[name]
        per_task.append(
            {
                "task_name": name,
                "base": base_row,
                "adapter": adapter_row,
                "output_token_delta": adapter_row["output_tokens"] - base_row["output_tokens"],
                "llm_call_delta": adapter_row["llm_calls"] - base_row["llm_calls"],
                "elapsed_seconds_delta": adapter_row["elapsed_seconds"]
                - base_row["elapsed_seconds"],
            }
        )
    base_summary = summarize(base)
    adapter_summary = summarize(adapter)
    return {
        "schema": "xtoken-official-tb2-timeout-gap-v1",
        "base_job": str(base_job),
        "adapter_job": str(adapter_job),
        "base": base_summary,
        "adapter": adapter_summary,
        "paired": {
            "timeout_count_delta": adapter_summary["timeout_count"] - base_summary["timeout_count"],
            "common_timeout_tasks": sorted(base_timeouts & adapter_timeouts),
            "base_only_timeout_tasks": sorted(base_timeouts - adapter_timeouts),
            "adapter_only_timeout_tasks": sorted(adapter_timeouts - base_timeouts),
            "output_token_ratio": adapter_summary["total_output_tokens"]
            / max(base_summary["total_output_tokens"], 1),
            "input_token_ratio": adapter_summary["total_input_tokens"]
            / max(base_summary["total_input_tokens"], 1),
        },
        "interpretation_boundary": (
            "AgentTimeoutError is a 2700-second model-and-harness wall-clock outcome, "
            "not an infrastructure exception. Context rejection counts are recovered "
            "HTTP 400 calls and must not be counted as task failures by themselves."
        ),
        "per_task": per_task,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-job", type=Path, required=True)
    parser.add_argument("--adapter-job", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = compare(args.base_job, args.adapter_job)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    sys.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
