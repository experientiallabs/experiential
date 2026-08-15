"""Tests for Langfuse trace export normalization."""

from __future__ import annotations

import json
from pathlib import Path

from wmo.simulation.ingest.langfuse import load_langfuse_file


def _trace(observations: list[dict[str, object]] | None = None) -> dict[str, object]:
    """Return one Langfuse trace with a generation, a tool observation, and a final generation.

    Args:
        observations: Declared observations, defaulting to the shared fixture observations.

    Returns:
        One Langfuse trace object.
    """
    return {
        "id": "trace-1",
        "name": "support-agent",
        "timestamp": "2026-02-01T00:00:00Z",
        "sessionId": "session-9",
        "userId": "customer-3",
        "input": {"messages": [{"role": "user", "content": "Where is my order?"}]},
        "output": {"role": "assistant", "content": "It ships tomorrow."},
        "metadata": {"provider": "openai", "tier": "gold"},
        "observations": _observations() if observations is None else observations,
    }


def _observations() -> list[dict[str, object]]:
    """Return the shared Langfuse observation fixtures in source order."""
    return [
        {
            "id": "obs-1",
            "traceId": "trace-1",
            "type": "GENERATION",
            "name": "plan",
            "startTime": "2026-02-01T00:00:01Z",
            "endTime": "2026-02-01T00:00:02Z",
            "model": "gpt-4o-mini",
            "input": [{"role": "user", "content": "Where is my order?"}],
            "output": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "function": {
                            "name": "lookup_order",
                            "arguments": '{"order": "A1"}',
                        },
                    }
                ],
            },
            "usage": {"input": 42, "output": 7},
        },
        {
            "id": "obs-2",
            "traceId": "trace-1",
            "parentObservationId": "obs-1",
            "type": "TOOL",
            "name": "lookup_order",
            "startTime": "2026-02-01T00:00:03Z",
            "endTime": "2026-02-01T00:00:04Z",
            "input": {"order": "A1"},
            "output": "ships tomorrow",
        },
        {
            "id": "obs-3",
            "traceId": "trace-1",
            "type": "SPAN",
            "name": "orchestration",
            "startTime": "2026-02-01T00:00:05Z",
        },
        {
            "id": "obs-4",
            "traceId": "trace-1",
            "type": "GENERATION",
            "name": "answer",
            "startTime": "2026-02-01T00:00:06Z",
            "model": "gpt-4o-mini",
            "output": {"role": "assistant", "content": "It ships tomorrow."},
        },
    ]


def test_load_langfuse_file_converts_generations_and_tools(tmp_path: Path) -> None:
    """Generations and tool observations become paired canonical spans."""
    path = tmp_path / "langfuse.json"
    path.write_text(json.dumps([_trace()]), encoding="utf-8")

    result = load_langfuse_file(path)

    assert result.issues == ()
    trace = result.traces[0]
    assert trace.task == "Where is my order?"
    assert trace.conversation_id == "session-9"
    assert [span.name for span in trace.spans] == [
        "agent.model_call",
        "agent.tool_call",
        "agent.model_call",
    ]
    call, tool_result, answer = trace.spans
    assert call.attributes["gen_ai.tool.call.id"] == "call-1"
    assert call.model is not None
    assert call.model.provider == "openai"
    assert call.model.model_id == "gpt-4o-mini"
    assert call.usage is not None
    assert (call.usage.input_tokens, call.usage.output_tokens) == (42, 7)
    assert tool_result.attributes["gen_ai.tool.call.id"] == "call-1"
    assert tool_result.attributes["gen_ai.tool.message"] == "ships tomorrow"
    assert tool_result.parent_span_id == call.span_id
    assert answer.attributes["gen_ai.completion"] == "It ships tomorrow."
    assert answer.attributes["wmo.customer.id"] == "customer-3"


def test_load_langfuse_file_keeps_model_name_without_provider(tmp_path: Path) -> None:
    """A model declared without a provider is retained as evidence, not as resolved identity."""
    trace = _trace()
    trace["metadata"] = {"tier": "gold"}
    path = tmp_path / "langfuse.json"
    path.write_text(json.dumps(trace), encoding="utf-8")

    result = load_langfuse_file(path)

    span = result.traces[0].spans[0]
    assert span.model is None
    assert span.attributes["gen_ai.request.model"] == "gpt-4o-mini"


def test_load_langfuse_file_retains_error_observations(tmp_path: Path) -> None:
    """An ERROR level observation becomes a structured span failure and trace outcome."""
    observations = _observations()
    observations[0]["level"] = "ERROR"
    observations[0]["statusMessage"] = "rate limited"
    trace = _trace(observations)
    path = tmp_path / "langfuse.json"
    path.write_text(json.dumps(trace), encoding="utf-8")

    result = load_langfuse_file(path)

    normalized = result.traces[0]
    assert normalized.spans[0].failure is not None
    assert normalized.spans[0].failure.message == "rate limited"
    assert normalized.outcome is not None
    assert normalized.outcome.status == "failure"


def test_load_langfuse_file_accepts_bare_observations(tmp_path: Path) -> None:
    """Observation exports without a trace wrapper group by their declared traceId."""
    path = tmp_path / "langfuse.jsonl"
    observations = [
        {
            "id": "obs-1",
            "traceId": "trace-2",
            "type": "GENERATION",
            "name": "answer",
            "startTime": "2026-02-01T00:00:01Z",
            "input": [{"role": "user", "content": "hello"}],
            "output": {"role": "assistant", "content": "hi"},
        }
    ]
    path.write_text("\n".join(json.dumps(item) for item in observations), encoding="utf-8")

    result = load_langfuse_file(path)

    assert len(result.traces) == 1
    assert result.traces[0].task == "hello"


def test_load_langfuse_file_retains_trace_without_convertible_observations(
    tmp_path: Path,
) -> None:
    """A trace whose observations are orchestration only is retained as agent evidence."""
    trace = {
        "id": "trace-3",
        "timestamp": "2026-02-01T00:00:00Z",
        "input": {"messages": [{"role": "user", "content": "hello"}]},
        "output": "hi",
        "observations": [
            {
                "id": "obs-1",
                "traceId": "trace-3",
                "type": "SPAN",
                "startTime": "2026-02-01T00:00:01Z",
            }
        ],
    }
    path = tmp_path / "langfuse.json"
    path.write_text(json.dumps(trace), encoding="utf-8")

    result = load_langfuse_file(path)

    assert [span.name for span in result.traces[0].spans] == ["agent.trace"]


def test_load_langfuse_file_excludes_observation_without_timing(tmp_path: Path) -> None:
    """An observation with no start time excludes its trace with an explicit issue."""
    observations = _observations()
    del observations[0]["startTime"]
    trace = _trace(observations)
    path = tmp_path / "langfuse.json"
    path.write_text(json.dumps(trace), encoding="utf-8")

    result = load_langfuse_file(path)

    assert result.traces == ()
    assert [issue.source_record for issue in result.issues] == ["record-1"]
