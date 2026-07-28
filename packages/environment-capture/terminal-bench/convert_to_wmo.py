#!/usr/bin/env python3
"""Convert harbor Terminal-Bench trial dirs into the wmo OTel-GenAI trace corpus.

The source is real harbor trials: terminus-2 drives a task sandbox with `bash_command`
tool calls and the REAL terminal output is recorded per step in the trial's
`agent/trajectory.json` (ATIF schema). That maps directly to the harness contract: one
Step per tool call, with the real `(action) -> observation` the agent actually saw. The
environment being reconstructed is the task sandbox's shell: predict the command's real
output given the command.

This closes the capture gap filed in DECISIONS 2026-07-27 (the product could not ingest
its own benchmark rollouts): every harbor job run through `wmo optimize distill run`, a
probe, or `harbor run` directly leaves trial dirs this converter turns into a corpus
`wmo build` accepts. Mind harbor's own eviction behaviors: the scorer prunes invalid
trial dirs on resume, so convert BEFORE re-running a job into the same jobs_dir.

Like the sibling converters this is stdlib-only (no `wmo` import, no third-party deps),
reads the trial dirs in place, and writes only the produced OTel JSONL to ``--out``.

Per trial, per agent step, per `tool_calls[]` entry (paired index-wise with
`observation.results[]`):
  - action      = the real tool call (function_name + arguments, e.g. bash_command
                  {"keystrokes": "..."}).
  - observation = the real recorded terminal output for that call.
  - task        = the trial's initial user message (the full instruction the agent saw),
                  carried on the first step as gen_ai.prompt.
  - Trace.metadata = benchmark, task name, trial name, job name, verifier reward, model.

Trials with a recorded `exception_info` (no complete episode) are skipped and counted.
Zero-reward trials are INCLUDED: real failures with real consequences are exactly what a
world model must reconstruct.

Usage:
    python convert_to_wmo.py <jobs_dir_or_job_dir>... --out traces.otel.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _attr(key: str, value: str) -> dict[str, Any]:
    return {"key": key, "value": {"stringValue": value}}


def _trace_id(job_name: str, trial_name: str) -> str:
    return hashlib.sha256(f"{job_name}|{trial_name}".encode()).hexdigest()[:32]


def find_trial_dirs(roots: list[Path]) -> list[Path]:
    """Every dir under the roots holding both a result.json and an agent trajectory."""
    found: list[Path] = []
    for root in roots:
        for result in sorted(root.rglob("result.json")):
            trial_dir = result.parent
            if (trial_dir / "agent" / "trajectory.json").exists():
                found.append(trial_dir)
    return found


def _task_text(steps: list[dict[str, Any]]) -> str:
    for step in steps:
        if step.get("source") == "user":
            message = step.get("message")
            if isinstance(message, str) and message.strip():
                return message
    return ""


def _observation_contents(step: dict[str, Any], n_calls: int) -> list[tuple[str, bool]]:
    """One (content, is_error) per tool call, padded when the episode was cut short."""
    results = (step.get("observation") or {}).get("results") or []
    out: list[tuple[str, bool]] = []
    for index in range(n_calls):
        if index < len(results):
            result = results[index] or {}
            out.append((str(result.get("content") or ""), bool(result.get("is_error"))))
        else:
            out.append(("", False))
    return out


def spans_for_trial(trial_dir: Path, *, benchmark: str) -> list[dict[str, Any]]:
    """Emit ordered action/observation span pairs for one harbor trial."""
    result = json.loads((trial_dir / "result.json").read_text(encoding="utf-8"))
    if result.get("exception_info") is not None:
        return []
    trajectory = json.loads(
        (trial_dir / "agent" / "trajectory.json").read_text(encoding="utf-8")
    )
    steps = trajectory.get("steps") or []
    rewards = (result.get("verifier_result") or {}).get("rewards") or {}
    model = ((result.get("agent_info") or {}).get("model_info") or {}).get("name", "")
    job_name = ((result.get("config") or {}).get("job_id")) or trial_dir.parent.name
    trial_name = result.get("trial_name") or trial_dir.name
    trace_id = _trace_id(str(job_name), str(trial_name))
    metadata = {
        "benchmark": benchmark,
        "task": result.get("task_name", ""),
        "trial": trial_name,
        "job": job_name,
        "reward": rewards.get("reward"),
        "model": model,
    }
    task_text = _task_text(steps)

    spans: list[dict[str, Any]] = []
    ordinal = 0
    for step in steps:
        if step.get("source") != "agent":
            continue
        tool_calls = step.get("tool_calls") or []
        observations = _observation_contents(step, len(tool_calls))
        for call, (content, is_error) in zip(tool_calls, observations, strict=True):
            name = str(call.get("function_name") or "bash_command")
            action_attrs = [
                _attr("gen_ai.operation.name", "chat"),
                _attr("gen_ai.request.model", model or "terminal-agent"),
                _attr("gen_ai.tool.name", name),
                _attr("gen_ai.tool.call.arguments", json.dumps(call.get("arguments") or {})),
            ]
            if ordinal == 0 and task_text:
                action_attrs.append(_attr("gen_ai.prompt", task_text))
            if ordinal == 0:
                action_attrs.append(_attr("wmo.trace.metadata", json.dumps(metadata)))
            spans.append({
                "traceId": trace_id,
                "spanId": f"{trace_id[:12]}{ordinal:04x}a",
                "parentSpanId": "",
                "name": "chat terminal",
                "startTimeUnixNano": ordinal * 10,
                "endTimeUnixNano": ordinal * 10 + 1,
                "status": {"code": "STATUS_CODE_OK"},
                "attributes": action_attrs,
            })
            spans.append({
                "traceId": trace_id,
                "spanId": f"{trace_id[:12]}{ordinal:04x}b",
                "parentSpanId": "",
                "name": "execute_tool terminal",
                "startTimeUnixNano": ordinal * 10 + 2,
                "endTimeUnixNano": ordinal * 10 + 3,
                "status": {"code": "STATUS_CODE_ERROR" if is_error else "STATUS_CODE_OK"},
                "attributes": [
                    _attr("gen_ai.operation.name", "execute_tool"),
                    _attr("gen_ai.tool.name", name),
                    _attr("gen_ai.tool.message", content),
                ],
            })
            ordinal += 1
    return spans


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "sources",
        nargs="+",
        help="harbor jobs dirs (or single job/trial dirs) to walk for trial results",
    )
    parser.add_argument("--out", required=True, help="Output OTel JSONL path")
    parser.add_argument("--benchmark", default="terminal-bench", help="Benchmark name")
    parser.add_argument(
        "--min-tool-calls",
        type=int,
        default=1,
        help="Skip trials with fewer than this many tool calls (default 1: drop empty runs).",
    )
    args = parser.parse_args()

    trial_dirs = find_trial_dirs([Path(s) for s in args.sources])
    n_traces = n_spans = n_skipped = 0
    with Path(args.out).open("w", encoding="utf-8") as out:
        for trial_dir in trial_dirs:
            spans = spans_for_trial(trial_dir, benchmark=args.benchmark)
            if len(spans) < 2 * args.min_tool_calls:
                n_skipped += 1
                continue
            for span in spans:
                out.write(json.dumps(span) + "\n")
                n_spans += 1
            n_traces += 1
    print(
        f"wrote {n_traces} traces, {n_spans} spans -> {args.out} "
        f"(skipped {n_skipped} of {len(trial_dirs)} trial dirs)"
    )


if __name__ == "__main__":
    main()
