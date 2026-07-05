#!/usr/bin/env python3
"""Convert terminal-task agent trajectories into the wmh OTel-GenAI trace corpus.

The source is real computer-use-agent runs on a terminal/bash environment: an LLM agent issues `bash`
tool calls and the REAL command output is recorded per call — including real failures (tracebacks,
HTTP 301s, retries). That maps directly to the harness contract: one Step per tool call, with the
real `(action) -> observation` the agent actually saw. The environment being reconstructed is a Unix
shell: predict the command's real output given the command.

This is a stdlib-only converter (no `wmh` import, no third-party deps) so it stays a self-contained
capture tool. It reads the SOURCE trajectories in place and never copies them into the repo — only
the produced OTel JSONL is written to ``--out``.

Per trajectory, per `tool_calls[]` entry:
  - action      = the real tool call (name + arguments, e.g. bash {"command": "..."}).
  - observation = the real recorded `output`, with `is_error` from the call's `isError` flag.
  - task        = the trajectory's task instruction (carried on the first step as gen_ai.prompt).
  - Trace.metadata = benchmark, task_category, returncode.

`state_before` is left empty: a shell has no compact, non-leaky state snapshot to feed, and open-loop
replay reconstructs from the action + retrieved similar steps + teacher-forced history.

Expected source schema (one trajectory per JSONL line):
  {"task": "...", "task_category": "...", "returncode": 0,
   "tool_calls": [{"name": "bash", "arguments": {"command": "..."}, "output": "...", "isError": false}]}

Usage:
    python convert_to_wmh.py <trajectories.jsonl> --out traces.otel.jsonl --benchmark terminal-tasks
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _attr(key: str, value: str) -> dict[str, Any]:
    return {"key": key, "value": {"stringValue": value}}


def _as_text(value: Any) -> str:  # noqa: ANN401 - tool output is loosely-typed JSON
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _trace_id(benchmark: str, index: int) -> str:
    return hashlib.sha256(f"{benchmark}|{index}".encode()).hexdigest()[:32]


def _metadata(traj: dict[str, Any], benchmark: str) -> dict[str, Any]:
    return {
        "benchmark": benchmark,
        "task_category": traj.get("task_category", ""),
        "returncode": traj.get("returncode"),
    }


def _spans_for_trajectory(
    traj: dict[str, Any], *, benchmark: str, trace_id: str
) -> list[dict[str, Any]]:
    """Emit ordered action/observation span pairs for one trajectory's tool calls."""
    task_text = traj.get("task") if isinstance(traj.get("task"), str) else ""
    metadata = _metadata(traj, benchmark)

    spans: list[dict[str, Any]] = []
    for ordinal, tc in enumerate(traj.get("tool_calls", []) or []):
        name = tc.get("name", "bash")
        args = tc.get("arguments") or {}
        obs_content = _as_text(tc.get("output"))
        obs_error = bool(tc.get("isError", False))

        action_attrs = [
            _attr("gen_ai.operation.name", "chat"),
            _attr("gen_ai.request.model", "terminal-agent"),
            _attr("gen_ai.tool.name", str(name)),
            _attr("gen_ai.tool.call.arguments", json.dumps(args)),
        ]
        if ordinal == 0 and task_text:
            action_attrs.append(_attr("gen_ai.prompt", task_text))
        if ordinal == 0:
            action_attrs.append(_attr("wmh.trace.metadata", json.dumps(metadata)))

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
            "status": {"code": "STATUS_CODE_ERROR" if obs_error else "STATUS_CODE_OK"},
            "attributes": [
                _attr("gen_ai.operation.name", "execute_tool"),
                _attr("gen_ai.tool.name", str(name)),
                _attr("gen_ai.tool.message", obs_content),
            ],
        })
    return spans


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="Path to a terminal-task trajectories JSONL (read in place)")
    parser.add_argument("--out", required=True, help="Output OTel JSONL path")
    parser.add_argument("--benchmark", default="terminal-tasks", help="Benchmark name")
    parser.add_argument(
        "--min-tool-calls",
        type=int,
        default=1,
        help="Skip trajectories with fewer than this many tool calls (default 1: drop empty runs).",
    )
    parser.add_argument(
        "--exclude-substr",
        action="append",
        default=[],
        help=(
            "Drop any trajectory whose raw JSON contains this (case-insensitive) substring. "
            "Repeatable. Use to omit trajectories whose captured command output happens to "
            "reference source-specific paths. Whole-trajectory drop (no silent redaction of "
            "real observations)."
        ),
    )
    args = parser.parse_args()

    excludes = [s.lower() for s in args.exclude_substr]
    n_traces = n_spans = n_skipped = n_excluded = 0
    with Path(args.out).open("w", encoding="utf-8") as out:
        for line in Path(args.source).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            if any(sub in line.lower() for sub in excludes):
                n_excluded += 1
                continue
            traj = json.loads(line)
            if not isinstance(traj, dict):
                continue
            if len(traj.get("tool_calls", []) or []) < args.min_tool_calls:
                n_skipped += 1
                continue
            trace_id = _trace_id(args.benchmark, n_traces)
            spans = _spans_for_trajectory(traj, benchmark=args.benchmark, trace_id=trace_id)
            if not spans:
                n_skipped += 1
                continue
            for span in spans:
                out.write(json.dumps(span) + "\n")
                n_spans += 1
            n_traces += 1
    print(
        f"wrote {n_traces} traces, {n_spans} spans -> {args.out} "
        f"(skipped {n_skipped}, excluded {n_excluded})"
    )


if __name__ == "__main__":
    main()
