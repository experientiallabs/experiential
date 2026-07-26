"""Tests for trace-adapter registration and the OTel GenAI file parser."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from wmo.core.types import ActionKind
from wmo.ingest import get_adapter
from wmo.ingest.adapter import VendorPull
from wmo.ingest.otel_genai import VENDOR_ENDPOINT_ENV, OtelGenAIAdapter

_TESTDATA = Path(__file__).parent / "testdata"
# `wmo/ingest/otel_genai_test.py` -> repo root is parents[2].
_EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


def test_default_otel_adapter_is_registered_on_import() -> None:
    # DESIGN/README claim the OTel adapter ships registered; importing wmo.ingest must suffice.
    assert get_adapter("otel-genai").name == "otel-genai"


def test_from_file_parses_otlp_json_into_one_trace() -> None:
    traces = OtelGenAIAdapter().from_file(str(_TESTDATA / "sample_otlp.json"))

    assert len(traces) == 1
    trace = traces[0]
    assert trace.trace_id == "0af7651916cd43dd8448eb211c80319c"
    assert trace.source.endswith("sample_otlp.json")

    # 3 spans (llm+tool_call, execute_tool, final llm) -> 2 steps: paired tool call + final message.
    assert len(trace.steps) == 2

    call_step = trace.steps[0]
    assert call_step.action.kind == ActionKind.TOOL_CALL
    assert call_step.action.name == "get_weather"
    assert call_step.action.arguments == {"city": "Paris"}
    assert call_step.observation.content == "18C and sunny"
    assert call_step.observation.is_error is False
    # The originating prompt is carried onto every step's `task`.
    assert call_step.task == "What is the weather in Paris?"
    # Both the LLM span and the tool span are recorded as provenance. `BaseTraceAdapter` stamps a
    # global emission-order prefix for cross-payload uniqueness, so match on the source-id suffix.
    assert [sid.split("-", 1)[1] for sid in call_step.raw_span_ids] == [
        "b7ad6b7169203331",
        "c8be7c8270314442",
    ]

    final_step = trace.steps[1]
    assert final_step.action.kind == ActionKind.MESSAGE
    assert final_step.action.content == "It is 18C and sunny in Paris."
    # No following tool span -> empty observation.
    assert final_step.observation.content == ""


def test_from_file_parses_jsonl_with_multiple_traces() -> None:
    traces = OtelGenAIAdapter().from_file(str(_TESTDATA / "sample_spans.jsonl"))

    assert [t.trace_id for t in traces] == [
        "aaaa0000aaaa0000aaaa0000aaaa0000",
        "bbbb1111bbbb1111bbbb1111bbbb1111",
    ]

    # Trace 1: paired tool call whose execution errored.
    first = traces[0]
    assert len(first.steps) == 1
    assert first.steps[0].action.name == "rm"
    assert first.steps[0].action.arguments == {"path": "/tmp/x"}
    assert first.steps[0].observation.content == "permission denied"
    assert first.steps[0].observation.is_error is True

    # Trace 2: a lone execute_tool span with no preceding LLM span becomes a self-contained step.
    second = traces[1]
    assert len(second.steps) == 1
    assert second.steps[0].action.kind == ActionKind.TOOL_CALL
    assert second.steps[0].action.name == "search"
    assert second.steps[0].action.arguments == {"q": "otel"}
    assert second.steps[0].observation.content == "3 results"


def test_from_file_skips_corrupt_jsonl_lines(tmp_path: Path) -> None:
    good = (
        '{"traceId": "cccc", "spanId": "01", "name": "chat", '
        '"attributes": [{"key": "gen_ai.completion", "value": {"stringValue": "hi"}}]}'
    )
    path = tmp_path / "partial.jsonl"
    # A truncated middle line (crashed exporter) must not abort the whole ingest.
    path.write_text(f"{good}\n{{truncated\n{good}\n", encoding="utf-8")

    traces = OtelGenAIAdapter().from_file(str(path))

    assert len(traces) == 1
    assert traces[0].trace_id == "cccc"
    assert len(traces[0].steps) == 2  # both valid lines parsed; the corrupt one skipped


def test_state_and_metadata_attributes_populate_step_and_trace(tmp_path: Path) -> None:
    # An action span enriched with wmo.* attributes: state-before snapshot + trace metadata.
    span_llm = {
        "traceId": "dddd",
        "spanId": "01",
        "name": "chat",
        "startTimeUnixNano": 1,
        "attributes": [
            {"key": "gen_ai.operation.name", "value": {"stringValue": "chat"}},
            {"key": "gen_ai.tool.name", "value": {"stringValue": "cancel_reservation"}},
            {"key": "gen_ai.tool.call.arguments", "value": {"stringValue": '{"id": "r1"}'}},
            {"key": "gen_ai.prompt", "value": {"stringValue": "cancel r1"}},
            {
                "key": "wmo.state.structured",
                "value": {"stringValue": '{"reservations": {"r1": {"status": "confirmed"}}}'},
            },
            {"key": "wmo.state.scratchpad", "value": {"stringValue": "logged in as u1"}},
            {
                "key": "wmo.trace.metadata",
                "value": {
                    "stringValue": '{"benchmark": "tau2-bench", "task_id": "tau-train-1", '
                    '"gold": {"assertions": [{"path": "reservations.r1.status", '
                    '"equals": "cancelled"}]}}'
                },
            },
        ],
    }
    span_tool = {
        "traceId": "dddd",
        "spanId": "02",
        "name": "execute_tool",
        "startTimeUnixNano": 2,
        "attributes": [
            {"key": "gen_ai.operation.name", "value": {"stringValue": "execute_tool"}},
            {"key": "gen_ai.tool.message", "value": {"stringValue": '{"ok": true}'}},
        ],
    }
    path = tmp_path / "enriched.jsonl"
    path.write_text(json.dumps(span_llm) + "\n" + json.dumps(span_tool) + "\n", encoding="utf-8")

    traces = OtelGenAIAdapter().from_file(str(path))

    assert len(traces) == 1
    trace = traces[0]
    # Trace metadata carries benchmark name + gold (gold rides along for closed-loop later).
    assert trace.metadata["benchmark"] == "tau2-bench"
    assert trace.metadata["gold"] == {
        "assertions": [{"path": "reservations.r1.status", "equals": "cancelled"}]
    }
    # The action span's wmo.state.* snapshot becomes the step's state_before.
    step = trace.steps[0]
    assert step.state_before.structured == {"reservations": {"r1": {"status": "confirmed"}}}
    assert step.state_before.scratchpad == "logged in as u1"
    assert step.action.name == "cancel_reservation"
    assert step.observation.content == '{"ok": true}'


def test_traces_without_wmo_attributes_keep_empty_state_and_metadata() -> None:
    # Backward-compat: the bare-semconv corpus has no wmo.* attrs -> empty state/metadata, no error.
    traces = OtelGenAIAdapter().from_file(str(_TESTDATA / "sample_otlp.json"))

    assert traces[0].metadata == {}
    for step in traces[0].steps:
        assert step.state_before.structured == {}
        assert step.state_before.scratchpad == ""


def test_committed_tau2_corpus_satisfies_the_replay_contract() -> None:
    """The committed real tau2-bench corpus must parse into replay-ready traces.

    Guards the trace contract on the actual captured artifact (not a synthetic fixture): every trace
    carries benchmark + gold metadata, and every step has a real tool-call action, the real recorded
    observation, and the originating task. `state_before` is intentionally empty for tau2 — the env
    DB is huge and would leak the answer (open-loop replay must reconstruct, not look up), so the
    converter omits it; the adapter still supports `wmo.state.*` for future small-state benchmarks.
    """
    corpus = _EXAMPLES / "tau-bench" / "traces.otel.jsonl"
    if not corpus.exists():  # pragma: no cover - committed corpus; only missing in a partial slice
        pytest.skip("tau2-bench corpus not present")

    traces = OtelGenAIAdapter().from_file(str(corpus))
    assert traces, "corpus produced no traces"

    n_steps = 0
    for trace in traces:
        assert trace.metadata.get("benchmark") == "tau2-bench"
        assert "gold" in trace.metadata  # gold rides along for the deferred closed-loop eval
        assert trace.steps, f"trace {trace.trace_id} has no steps"
        for step in trace.steps:
            n_steps += 1
            assert step.action.kind == ActionKind.TOOL_CALL
            assert step.action.name  # a real tau2 tool name
            assert step.task  # the originating user instruction
    assert n_steps > 0


def test_committed_terminal_tasks_corpus_satisfies_the_replay_contract() -> None:
    """The committed terminal-tasks corpus (if present) must parse into replay-ready traces.

    Real bash tool calls with real recorded outputs (including failures). Each step has a tool-call
    action and the originating task; state_before is empty (a shell has no compact state snapshot).
    """
    corpus = _EXAMPLES / "terminal-tasks" / "traces.otel.jsonl"
    if not corpus.exists():  # pragma: no cover - committed corpus; only missing in a partial slice
        pytest.skip("terminal-tasks corpus not present")

    traces = OtelGenAIAdapter().from_file(str(corpus))
    assert traces, "corpus produced no traces"

    n_steps = 0
    for trace in traces:
        assert trace.metadata.get("benchmark") == "terminal-tasks"
        assert trace.steps, f"trace {trace.trace_id} has no steps"
        for step in trace.steps:
            n_steps += 1
            assert step.action.kind == ActionKind.TOOL_CALL
            assert step.action.name  # the real tool name (bash)
            assert step.task  # the originating task instruction
    assert n_steps > 0


def test_from_vendor_without_endpoint_raises_friendly_error() -> None:
    saved = os.environ.pop(VENDOR_ENDPOINT_ENV, None)
    try:
        with pytest.raises(ValueError, match=VENDOR_ENDPOINT_ENV):
            OtelGenAIAdapter().from_vendor(VendorPull(project="demo"))
    finally:
        if saved is not None:
            os.environ[VENDOR_ENDPOINT_ENV] = saved
