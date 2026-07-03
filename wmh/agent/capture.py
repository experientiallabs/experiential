"""Turn agent runs into traces the build pipeline ingests.

A real run (`AgentRuntime` over `E2BEnvironment`) yields `Step`s; we wrap them as a `Trace` and can
emit them in the OTel GenAI JSONL shape the `otel-genai` adapter reads — so collected traces feed
`wmh build` directly, closing the produce/consume loop. We stamp the optional `wmh.*` enrichments
(`wmh.state.*`, `wmh.trace.metadata` with the task's gold assertions) the adapter understands, so a
captured run is immediately usable for open-loop replay *and* carries the gold for closed-loop eval.
"""

from __future__ import annotations

import json
from pathlib import Path

from wmh.agent.runtime import RunResult
from wmh.agent.tasks import TaskSpec
from wmh.core.types import ActionKind, JsonObject, Step, Trace

# Nanosecond spacing between synthesized spans; only ordering matters to the adapter, so a fixed
# tick keeps span times monotonic and deterministic (no wall-clock, per repo rules).
_TICK_NANOS = 1_000_000


def run_to_trace(result: RunResult, gold: list[str] | None = None) -> Trace:
    """Wrap a run's steps as a Trace, stamping the task gold into `metadata` for later judging."""
    metadata: JsonObject = {"harness": result.harness, "stop_reason": result.stop_reason.value}
    if gold:
        metadata["gold"] = list(gold)
    return Trace(
        trace_id=f"{result.harness}:{result.task_id}",
        steps=result.steps,
        source="wmh-agent",
        metadata=metadata,
    )


def _span_for_step(
    trace_id: str, step: Step, index: int, extra_attrs: JsonObject | None = None
) -> list[dict[str, object]]:
    """Emit the OTel GenAI span(s) for one step: an LLM action span + its tool observation span.

    `extra_attrs` is merged onto the action span (used to stamp trace-level metadata on the first).
    """
    base_nano = (index + 1) * _TICK_NANOS * 4
    action = step.action
    attrs: JsonObject = dict(extra_attrs or {})
    if step.task is not None:
        attrs["gen_ai.prompt"] = step.task
    if step.state_before.scratchpad:
        attrs["wmh.state.scratchpad"] = step.state_before.scratchpad
    if step.state_before.structured:
        attrs["wmh.state.structured"] = json.dumps(step.state_before.structured)

    if action.kind == ActionKind.TOOL_CALL:
        attrs["gen_ai.operation.name"] = "invoke_agent"
        attrs["gen_ai.tool.name"] = action.name or ""
        attrs["gen_ai.tool.call.arguments"] = json.dumps(action.arguments)
    else:
        attrs["gen_ai.operation.name"] = "chat"
        attrs["gen_ai.completion"] = action.content or ""

    action_span = _span(trace_id, f"act-{index}", base_nano, base_nano + _TICK_NANOS, attrs)
    tool_attrs: JsonObject = {
        "gen_ai.operation.name": "execute_tool",
        "gen_ai.tool.name": action.name or "",
        "gen_ai.tool.message": step.observation.content,
    }
    tool_span = _span(
        trace_id,
        f"obs-{index}",
        base_nano + 2 * _TICK_NANOS,
        base_nano + 3 * _TICK_NANOS,
        tool_attrs,
        error=step.observation.is_error,
    )
    return [action_span, tool_span]


def _span(
    trace_id: str,
    span_id: str,
    start: int,
    end: int,
    attributes: JsonObject,
    *,
    error: bool = False,
) -> dict[str, object]:
    span: dict[str, object] = {
        "traceId": trace_id,
        "spanId": span_id,
        "name": attributes.get("gen_ai.operation.name", "span"),
        "startTimeUnixNano": start,
        "endTimeUnixNano": end,
        "attributes": [
            {"key": k, "value": {"stringValue": _as_str(v)}} for k, v in attributes.items()
        ],
    }
    if error:
        span["status"] = {"code": 2}
    return span


def _as_str(value: object) -> str:
    return value if isinstance(value, str) else json.dumps(value)


def trace_to_otel_lines(trace: Trace) -> list[str]:
    """Render a Trace as OTLP-JSON span lines (JSONL), carrying trace metadata on the first span."""
    lines: list[str] = []
    # Stamp trace-level metadata (harness, gold, stop_reason) on the very first span's attributes so
    # the adapter's `wmh.trace.metadata` reader picks it up (see wmh/ingest/otel_genai.py).
    for i, step in enumerate(trace.steps):
        extra: JsonObject | None = (
            {"wmh.trace.metadata": json.dumps(trace.metadata)}
            if i == 0 and trace.metadata
            else None
        )
        for span in _span_for_step(trace.trace_id, step, i, extra):
            lines.append(json.dumps(span))
    return lines


def write_otel_traces(traces: list[Trace], path: str | Path) -> Path:
    """Write traces to a `.otel.jsonl` file the `otel-genai` adapter (and `wmh build`) can read."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [line for trace in traces for line in trace_to_otel_lines(trace)]
    out.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
    return out


def gold_for(task: TaskSpec) -> list[str]:
    return task.gold
