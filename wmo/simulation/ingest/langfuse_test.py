"""Tests for Langfuse trace export normalization."""

from __future__ import annotations

import json
from pathlib import Path

from wmo.simulation.ingest.langfuse import LANGFUSE_SOURCE


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


def test_load_langfuse_file_keeps_completion_and_customer_identity(tmp_path: Path) -> None:
    """The final generation keeps its completion text and the trace's customer identity."""
    path = tmp_path / "langfuse.json"
    path.write_text(json.dumps([_trace()]), encoding="utf-8")

    result = LANGFUSE_SOURCE.load(path)

    answer = result.traces[0].spans[2]
    assert answer.attributes["gen_ai.completion"] == "It ships tomorrow."
    assert answer.attributes["wmo.customer.id"] == "customer-3"


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

    result = LANGFUSE_SOURCE.load(path)

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

    result = LANGFUSE_SOURCE.load(path)

    assert [span.name for span in result.traces[0].spans] == ["agent.trace"]
