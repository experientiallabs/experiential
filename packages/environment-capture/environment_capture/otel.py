"""Emit trajectories as OTel GenAI JSONL — the wmh trace wire format.

One OTLP-JSON span object per line. Per step: a `chat` action span (tool name + JSON arguments;
the first one also carries the task prompt as ``gen_ai.prompt`` and the trace metadata as
``wmh.trace.metadata``) followed by an `execute_tool` observation span (real output as
``gen_ai.tool.message``, error flag via span status). The final answer is NOT emitted as a span —
it produces no environment observation — and rides in ``wmh.trace.metadata`` instead.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

from environment_capture.trajectory import JsonValue, Trajectory

_SpanAttr = dict[str, "JsonValue"]
Span = dict[str, "JsonValue"]


def trace_id_for(benchmark: str, task_id: str) -> str:
    """Deterministic 32-hex trace id so re-conversion never duplicates traces."""
    return hashlib.sha256(f"{benchmark}|{task_id}".encode()).hexdigest()[:32]


def _attr(key: str, value: str) -> _SpanAttr:
    return {"key": key, "value": {"stringValue": value}}


def _trace_metadata(trajectory: Trajectory, benchmark: str) -> dict[str, JsonValue]:
    metadata: dict[str, JsonValue] = {
        "benchmark": benchmark,
        "task_id": trajectory.task.task_id,
    }
    if trajectory.model:
        metadata["model"] = trajectory.model
    if trajectory.split:
        metadata["split"] = trajectory.split
    if trajectory.reward is not None:
        metadata["reward"] = trajectory.reward
    if trajectory.final_answer:
        metadata["final_answer"] = trajectory.final_answer
    metadata.update(trajectory.metadata)
    return metadata


def trajectory_to_spans(trajectory: Trajectory, *, benchmark: str) -> list[Span]:
    """Emit ordered action/observation span pairs for one trajectory."""
    trace_id = trace_id_for(benchmark, trajectory.task.task_id)
    spans: list[Span] = []
    for ordinal, step in enumerate(trajectory.steps):
        action_attrs = [
            _attr("gen_ai.operation.name", "chat"),
            _attr("gen_ai.request.model", trajectory.model or benchmark),
            _attr("gen_ai.tool.name", step.action.name),
            _attr("gen_ai.tool.call.arguments", json.dumps(step.action.arguments)),
        ]
        if ordinal == 0:
            if trajectory.task.prompt:
                action_attrs.append(_attr("gen_ai.prompt", trajectory.task.prompt))
            action_attrs.append(
                _attr("wmh.trace.metadata", json.dumps(_trace_metadata(trajectory, benchmark)))
            )
        spans.append(
            {
                "traceId": trace_id,
                "spanId": f"{trace_id[:12]}{ordinal:04x}a",
                "parentSpanId": "",
                "name": f"chat {benchmark}",
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
                "name": f"execute_tool {benchmark}",
                "startTimeUnixNano": ordinal * 10 + 2,
                "endTimeUnixNano": ordinal * 10 + 3,
                "status": {"code": "STATUS_CODE_ERROR" if step.is_error else "STATUS_CODE_OK"},
                "attributes": [
                    _attr("gen_ai.operation.name", "execute_tool"),
                    _attr("gen_ai.tool.name", step.action.name),
                    _attr("gen_ai.tool.message", step.output),
                ],
            }
        )
    return spans


def write_spans_jsonl(spans: Iterable[Span], path: Path, *, append: bool = False) -> int:
    """Write spans one-JSON-object-per-line; returns the number written."""
    count = 0
    with path.open("a" if append else "w", encoding="utf-8") as out:
        for span in spans:
            out.write(json.dumps(span, ensure_ascii=False) + "\n")
            count += 1
    return count
