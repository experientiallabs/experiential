"""Write normalized `Trace`s back out as OTel-GenAI span JSONL — the inverse of `normalize`.

`wmh ingest` normalizes any source into `Trace`s, then persists them with this writer so the rest of
the pipeline (`wmh build`, `wmh eval`) reads them through the existing `otel-genai` file adapter.
Writing in the same span vocabulary the normalizer reads makes a `Trace -> file -> Trace` round-trip
lossless for everything the harness uses (action, observation+error, state_before, task, metadata).

One span per JSONL line: each `Step` becomes a `chat` (action) span plus, when it has an
observation, an `execute_tool` span. The first step's action span also carries the trace `task`
(`gen_ai.prompt`) and `wmh.trace.metadata`. Timestamps are deterministic ordinals (no wall clock),
so re-writing the same trace is byte-identical.
"""

from __future__ import annotations

import json
from pathlib import Path

from wmh.core.types import ActionKind, JsonObject, JsonValue, Step, Trace


def _attr(key: str, value: str) -> JsonObject:
    return {"key": key, "value": {"stringValue": value}}


def _action_attrs(
    step: Step, *, ordinal: int, task: str | None, metadata: JsonObject
) -> list[JsonValue]:
    attrs: list[JsonValue] = [_attr("gen_ai.operation.name", "chat")]
    action = step.action
    if action.kind == ActionKind.TOOL_CALL:
        attrs.append(_attr("gen_ai.tool.name", action.name or ""))
        attrs.append(_attr("gen_ai.tool.call.arguments", json.dumps(action.arguments)))
    else:
        attrs.append(_attr("gen_ai.completion", action.content or ""))
    if ordinal == 0 and task is not None:
        attrs.append(_attr("gen_ai.prompt", task))
    if ordinal == 0 and metadata:
        attrs.append(_attr("wmh.trace.metadata", json.dumps(metadata)))
    if step.state_before.structured:
        attrs.append(_attr("wmh.state.structured", json.dumps(step.state_before.structured)))
    if step.state_before.scratchpad:
        attrs.append(_attr("wmh.state.scratchpad", step.state_before.scratchpad))
    return attrs


def trace_to_spans(trace: Trace) -> list[JsonObject]:
    """Project one `Trace` into ordered OTLP-JSON action/observation span dicts."""
    spans: list[JsonObject] = []
    task = trace.steps[0].task if trace.steps else None
    prefix = (trace.trace_id or "trace")[:12]
    for i, step in enumerate(trace.steps):
        spans.append(
            {
                "traceId": trace.trace_id,
                "spanId": f"{prefix}{i:06x}a",
                "parentSpanId": "",
                "name": "chat",
                "startTimeUnixNano": i * 10,
                "endTimeUnixNano": i * 10 + 1,
                "status": {"code": "STATUS_CODE_OK"},
                "attributes": _action_attrs(step, ordinal=i, task=task, metadata=trace.metadata),
            }
        )
        # A final message turn (empty observation, message action) has nothing to pair; skip the
        # observation span so it round-trips as an unpaired action exactly as the normalizer emits.
        obs = step.observation
        if step.action.kind == ActionKind.MESSAGE and not obs.content and not obs.is_error:
            continue
        spans.append(
            {
                "traceId": trace.trace_id,
                "spanId": f"{prefix}{i:06x}b",
                "parentSpanId": "",
                "name": "execute_tool",
                "startTimeUnixNano": i * 10 + 2,
                "endTimeUnixNano": i * 10 + 3,
                "status": {"code": "STATUS_CODE_ERROR" if obs.is_error else "STATUS_CODE_OK"},
                "attributes": [
                    _attr("gen_ai.operation.name", "execute_tool"),
                    _attr("gen_ai.tool.name", step.action.name or "tool"),
                    _attr("gen_ai.tool.message", obs.content),
                ],
            }
        )
    return spans


def write_traces_jsonl(traces: list[Trace], path: Path) -> int:
    """Write traces as one-span-per-line OTel-GenAI JSONL. Returns the number of spans written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for trace in traces:
            for span in trace_to_spans(trace):
                f.write(json.dumps(span) + "\n")
                n += 1
    return n
