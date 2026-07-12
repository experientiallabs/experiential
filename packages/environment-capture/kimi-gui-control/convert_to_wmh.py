#!/usr/bin/env python3
"""Convert a screenpipe GUI-control agent trace dump into the wmh OTel-GenAI trace corpus.

The source is a JSONL dump of screenpipe `gui-control` agent trajectories (Kimi-K2.6 via
azure-foundry driving macOS GUI apps through the Accessibility API + shell). It can be very large
(one 9GB file holds 1000 trajectories), so this reads it STREAMING, one line at a time, and never
loads the whole file into memory.

Like the other environment-capture converters, this does NOT import `wmh` - it only emits the
OTel-GenAI span JSONL shape that `wmh.ingest.otel_genai` reads, so the world-model-harness package
stays free of any capture-side dependency. The produced `traces.otel.jsonl` is Hub-hosted, not
committed (see this corpus's README § Data & license).

What it produces, per trajectory (one trace), faithful to the contract open-loop replay needs:
  - one Step per AGENT TOOL CALL: action = the real tool call (name + arguments), observation = the
    REAL recorded tool output the agent saw (verbatim), error flag from the recorded `isError`.
  - `Trace.metadata` carries the benchmark name, the task category, the task url, and the model that
    produced the trajectory.

`state_before` is intentionally left EMPTY. The real GUI/OS state (the full accessibility tree, open
windows, filesystem) is not captured as a compact snapshot, so open-loop replay reconstructs the
environment from the action + retrieved similar past steps + the teacher-forced session history,
which is the point.

Pure-conversational turns (no tool call) are not Steps - open-loop replay scores predicted
observations for `(state, action)`, and a chat turn has no environment observation to score. A
trajectory with zero tool calls is skipped entirely.

Usage:
    python convert_to_wmh.py <source.jsonl> --out traces.otel.jsonl [--limit N]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _as_text(value: Any) -> str:  # noqa: ANN401 - source fields are loosely typed JSON
    """Render a loosely-typed value as a JSON-clean string: strings pass through, else JSON-encode.

    Tool arguments are dicts and outputs are usually strings, but neither is guaranteed. Encoding
    non-strings with json.dumps keeps the trace JSON clean end to end, so downstream can json.loads
    it (vs. a Python repr with single quotes, which needs ast.literal_eval).
    """
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _attr(key: str, value: str) -> dict[str, Any]:
    return {"key": key, "value": {"stringValue": value}}


def _trace_id(trajectory_id: str) -> str:
    return hashlib.sha256(f"kimi-gui-control|{trajectory_id}".encode()).hexdigest()[:32]


def _spans_for_trajectory(record: dict[str, Any], *, trace_id: str) -> list[dict[str, Any]]:
    """Emit ordered action/observation span pairs for one trajectory's agent tool calls."""
    tool_calls = record.get("tool_calls") or []
    task_text = _as_text(record.get("task", ""))
    metadata = {
        "benchmark": "kimi-gui-control",
        "task_category": record.get("task_category"),
        "task_url": record.get("task_url"),
        "model": record.get("model"),
        "provider": record.get("provider"),
        "returncode": record.get("returncode"),
    }

    spans: list[dict[str, Any]] = []
    for ordinal, tc in enumerate(tool_calls):
        name = tc.get("name", "")
        args = tc.get("arguments") or {}
        obs_content = _as_text(tc.get("output", ""))
        obs_error = bool(tc.get("isError", False))

        action_attrs = [
            _attr("gen_ai.operation.name", "chat"),
            _attr("gen_ai.request.model", "kimi-gui-control-agent"),
            _attr("gen_ai.tool.name", str(name)),
            _attr("gen_ai.tool.call.arguments", json.dumps(args)),
        ]
        if ordinal == 0 and task_text:
            action_attrs.append(_attr("gen_ai.prompt", task_text))
        if ordinal == 0:
            action_attrs.append(_attr("wmh.trace.metadata", json.dumps(metadata)))

        spans.append(
            {
                "traceId": trace_id,
                "spanId": f"{trace_id[:12]}{ordinal:04x}a",
                "parentSpanId": "",
                "name": "chat kimi-gui-control",
                "startTimeUnixNano": ordinal * 10,
                "endTimeUnixNano": ordinal * 10 + 1,
                "status": {"code": "STATUS_CODE_OK"},
                "attributes": action_attrs,
            }
        )
        spans.append(
            {
                "traceId": trace_id,
                "spanId": f"{trace_id[:12]}{ordinal:04x}b",
                "parentSpanId": "",
                "name": "execute_tool kimi-gui-control",
                "startTimeUnixNano": ordinal * 10 + 2,
                "endTimeUnixNano": ordinal * 10 + 3,
                "status": {"code": "STATUS_CODE_ERROR" if obs_error else "STATUS_CODE_OK"},
                "attributes": [
                    _attr("gen_ai.operation.name", "execute_tool"),
                    _attr("gen_ai.tool.name", str(name)),
                    _attr("gen_ai.tool.message", obs_content),
                ],
            }
        )
    return spans


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="Path to the source screenpipe trajectory JSONL")
    parser.add_argument("--out", required=True, help="Output OTel JSONL path")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Stop after emitting this many traces (trajectories with >=1 tool call).",
    )
    args = parser.parse_args()

    n_spans = n_traces = 0
    with (
        Path(args.source).open("r", encoding="utf-8") as src,
        Path(args.out).open("w", encoding="utf-8") as out,
    ):
        for line in src:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not record.get("tool_calls"):
                continue  # no environment observations to score -> not a Step-bearing trace
            trajectory_id = str(record.get("trajectory_id", n_traces))
            trace_id = _trace_id(trajectory_id)
            spans = _spans_for_trajectory(record, trace_id=trace_id)
            if not spans:
                continue
            for span in spans:
                out.write(json.dumps(span) + "\n")
                n_spans += 1
            n_traces += 1
            if args.limit is not None and n_traces >= args.limit:
                break
    print(f"wrote {n_traces} traces, {n_spans} spans -> {args.out}")


if __name__ == "__main__":
    main()
