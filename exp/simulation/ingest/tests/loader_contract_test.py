"""Shared loader contract parametrized over the vendor ingest suites.

Each vendor test module keeps its fixture builders and vendor-specific quirks.
The contracts here pin the behavior every file loader must share: tool-call
pairing, model-name evidence without resolved identity, declared-error mapping,
explicit exclusion of invalid records, retention of malformed JSONL lines
through the load seam, and multi-turn history normalization.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pytest
from pydantic import JsonValue

from exp.common.core.artifacts import JsonObject
from exp.common.traces import TraceSpan
from exp.simulation.ingest.braintrust import BRAINTRUST_SOURCE
from exp.simulation.ingest.braintrust_test import _rows as _braintrust_rows
from exp.simulation.ingest.chat_json import CHAT_JSON_SOURCE
from exp.simulation.ingest.chat_json_test import _conversation as _chat_conversation
from exp.simulation.ingest.langfuse import LANGFUSE_SOURCE
from exp.simulation.ingest.langfuse_test import _observations as _langfuse_observations
from exp.simulation.ingest.langfuse_test import _trace as _langfuse_trace
from exp.simulation.ingest.langsmith import LANGSMITH_SOURCE
from exp.simulation.ingest.langsmith_test import _runs as _langsmith_runs
from exp.simulation.ingest.mastra import MASTRA_SOURCE
from exp.simulation.ingest.mastra_test import _spans as _mastra_spans
from exp.simulation.ingest.otel_genai import load_otel_genai_file
from exp.simulation.ingest.otel_genai_test import _spans as _otel_spans
from exp.simulation.ingest.otlp import TraceNormalizationResult
from exp.simulation.ingest.phoenix import PHOENIX_SOURCE
from exp.simulation.ingest.phoenix_test import _native_spans as _phoenix_spans

_TOOL_PAIR = ("agent.model_call", "agent.tool_call")

_LOADERS: dict[str, Callable[[Path], TraceNormalizationResult]] = {
    "braintrust": BRAINTRUST_SOURCE.load,
    "chat_json": CHAT_JSON_SOURCE.load,
    "langfuse": LANGFUSE_SOURCE.load,
    "langsmith": LANGSMITH_SOURCE.load,
    "mastra": MASTRA_SOURCE.load,
    "otel_genai": load_otel_genai_file,
    "phoenix": PHOENIX_SOURCE.load,
}

# Vendors whose loader is VendorSource.load; otel_genai registers its own loader.
_VENDOR_SOURCE_VENDORS = ("braintrust", "chat_json", "langfuse", "langsmith", "mastra", "phoenix")


def _written(tmp_path: Path, document: object, *, jsonl: bool = False) -> Path:
    """Write one vendor export document, or one JSON record per line when ``jsonl``."""
    if jsonl:
        assert isinstance(document, list)
        text = "\n".join(json.dumps(record) for record in document)
    else:
        text = json.dumps(document)
    path = tmp_path / ("export.jsonl" if jsonl else "export.json")
    path.write_text(text, encoding="utf-8")
    return path


def _amend(
    records: list[dict[str, object]], index: int, **changes: object
) -> list[dict[str, object]]:
    """Return vendor records with one record's declared fields replaced."""
    records[index].update(changes)
    return records


def _drop(records: list[dict[str, object]], index: int, key: str) -> list[dict[str, object]]:
    """Return vendor records with one declared field removed from one record."""
    del records[index][key]
    return records


@dataclass(frozen=True)
class _PairingCase:
    """One vendor's happy-path export and the paired-span evidence it must yield."""

    vendor: str
    document: Callable[[], object]
    task: str
    model: tuple[str, str]
    call_attributes: JsonObject
    result_attributes: JsonObject
    span_names: tuple[str, ...] = _TOOL_PAIR
    usage: tuple[int, int] | None = None
    conversation_id: str | None = None


@dataclass(frozen=True)
class _ModelEvidenceCase:
    """One vendor export naming a model without a provider and the retained evidence."""

    vendor: str
    document: Callable[[], object]
    model_name: str
    jsonl: bool = False


@dataclass(frozen=True)
class _FailureCase:
    """One vendor export with a declared error and the span that must carry it."""

    vendor: str
    document: Callable[[], object]
    failing_span: int
    message: str | None = None


@dataclass(frozen=True)
class _ExclusionCase:
    """One invalid vendor export and the exact issues its exclusion must retain."""

    vendor: str
    document: Callable[[], object]
    issues: tuple[str, ...]


_PAIRING_CASES = [
    _PairingCase(
        vendor="braintrust",
        document=_braintrust_rows,
        task="Book a table for two",
        model=("anthropic", "claude-sonnet-4"),
        call_attributes={"gen_ai.tool.name": "book_table", "gen_ai.tool.call.id": "call-7"},
        result_attributes={"gen_ai.tool.call.id": "call-7", "gen_ai.tool.message": "table booked"},
        usage=(30, 4),
    ),
    _PairingCase(
        vendor="chat_json",
        document=_chat_conversation,
        task="What is the weather in Paris?",
        model=("openai", "gpt-4o"),
        call_attributes={"gen_ai.tool.name": "get_weather", "gen_ai.tool.call.id": "call-1"},
        result_attributes={"gen_ai.tool.call.id": "call-1", "gen_ai.tool.message": "18C"},
        span_names=("agent.model_call", "agent.tool_call", "agent.model_call"),
    ),
    _PairingCase(
        vendor="langfuse",
        document=lambda: [_langfuse_trace()],
        task="Where is my order?",
        model=("openai", "gpt-4o-mini"),
        call_attributes={"gen_ai.tool.call.id": "call-1"},
        result_attributes={
            "gen_ai.tool.call.id": "call-1",
            "gen_ai.tool.message": "ships tomorrow",
        },
        span_names=("agent.model_call", "agent.tool_call", "agent.model_call"),
        usage=(42, 7),
        conversation_id="session-9",
    ),
    _PairingCase(
        vendor="langsmith",
        document=_langsmith_runs,
        task="Cancel my subscription",
        model=("openai", "gpt-4o"),
        call_attributes={
            "gen_ai.tool.name": "cancel_subscription",
            "gen_ai.tool.call.arguments": '{"plan":"pro"}',
        },
        result_attributes={"gen_ai.tool.call.id": "call-9", "gen_ai.tool.message": "cancelled"},
        usage=(12, 3),
        conversation_id="session-4",
    ),
    _PairingCase(
        vendor="mastra",
        document=lambda: {"spans": _mastra_spans()},
        task="Summarize the incident",
        model=("openai", "gpt-4.1"),
        call_attributes={
            "gen_ai.tool.name": "fetch_incident",
            "gen_ai.tool.call.arguments": '{"id":"INC-1"}',
            "gen_ai.tool.call.id": "call-3",
        },
        result_attributes={"gen_ai.tool.call.id": "call-3", "gen_ai.tool.message": "disk pressure"},
        usage=(20, 6),
        conversation_id="thread-2",
    ),
    _PairingCase(
        vendor="otel_genai",
        document=_otel_spans,
        task="Reset my password",
        model=("openai", "gpt-4o"),
        call_attributes={"gen_ai.tool.name": "reset_password", "gen_ai.tool.call.id": "call-1"},
        result_attributes={
            "gen_ai.tool.call.id": "call-1",
            "gen_ai.tool.message": "password reset",
        },
        usage=(15, 2),
    ),
    _PairingCase(
        vendor="phoenix",
        document=_phoenix_spans,
        task="Refund my order",
        model=("openai", "gpt-4o-mini"),
        call_attributes={"gen_ai.tool.name": "refund_order", "gen_ai.tool.call.id": "call-5"},
        result_attributes={"gen_ai.tool.call.id": "call-5", "gen_ai.tool.message": "refunded"},
        usage=(40, 5),
    ),
]

_MODEL_EVIDENCE_CASES = [
    _ModelEvidenceCase(
        vendor="braintrust",
        document=lambda: _amend(_braintrust_rows(), 0, metadata={"model": "claude-sonnet-4"}),
        model_name="claude-sonnet-4",
    ),
    _ModelEvidenceCase(
        vendor="langfuse",
        document=lambda: {**_langfuse_trace(), "metadata": {"tier": "gold"}},
        model_name="gpt-4o-mini",
    ),
    _ModelEvidenceCase(
        vendor="langsmith",
        document=lambda: _amend(
            _langsmith_runs(), 0, extra={"metadata": {"ls_model_name": "gpt-4o"}}
        ),
        model_name="gpt-4o",
        jsonl=True,
    ),
    _ModelEvidenceCase(
        vendor="mastra",
        document=lambda: _mastra_spans(provider=None),
        model_name="gpt-4.1",
    ),
    _ModelEvidenceCase(
        vendor="phoenix",
        document=lambda: _phoenix_spans(provider=None),
        model_name="gpt-4o-mini",
    ),
]

_FAILURE_CASES = [
    _FailureCase(
        vendor="braintrust",
        document=lambda: _amend(_braintrust_rows(), 0, error={"message": "overloaded"}),
        failing_span=0,
    ),
    _FailureCase(
        vendor="langfuse",
        document=lambda: _langfuse_trace(
            _amend(_langfuse_observations(), 0, level="ERROR", statusMessage="rate limited")
        ),
        failing_span=0,
        message="rate limited",
    ),
    _FailureCase(
        vendor="langsmith",
        document=lambda: {"runs": _amend(_langsmith_runs(), 1, error="tool timed out")},
        failing_span=1,
        message="tool timed out",
    ),
    _FailureCase(
        vendor="mastra",
        document=lambda: _amend(_mastra_spans(), 1, errorInfo={"message": "tool unavailable"}),
        failing_span=1,
    ),
]

_EXCLUSION_CASES = [
    _ExclusionCase(
        vendor="braintrust",
        document=lambda: _amend(
            _braintrust_rows(), 0, metrics={"prompt_tokens": 1, "completion_tokens": 1}
        ),
        issues=("record-1", "trace-root-1"),
    ),
    _ExclusionCase(
        vendor="langfuse",
        document=lambda: _langfuse_trace(_drop(_langfuse_observations(), 0, "startTime")),
        issues=("record-1",),
    ),
    _ExclusionCase(
        vendor="langsmith",
        document=lambda: _drop(_langsmith_runs(), 0, "start_time"),
        issues=("record-1", "trace-trace-1"),
    ),
    _ExclusionCase(
        vendor="mastra",
        document=lambda: _amend(_mastra_spans(), 0, traceId="  "),
        issues=("record-1", "trace-trace-1"),
    ),
    _ExclusionCase(
        vendor="otel_genai",
        document=lambda: _drop(_otel_spans(), 1, "end_time"),
        issues=("record-1", f"trace-{'9' * 32}"),
    ),
    _ExclusionCase(
        vendor="phoenix",
        document=lambda: _drop(_phoenix_spans(), 0, "end_time"),
        issues=("record-1", "trace-trace-1"),
    ),
]


@pytest.mark.parametrize("case", _PAIRING_CASES, ids=lambda case: case.vendor)
def test_contract_pairs_tool_calls(case: _PairingCase, tmp_path: Path) -> None:
    """Every loader pairs a tool call with its result under one canonical trace."""
    result = _LOADERS[case.vendor](_written(tmp_path, case.document()))

    assert result.issues == ()
    assert len(result.traces) == 1
    trace = result.traces[0]
    assert trace.task == case.task
    assert tuple(span.name for span in trace.spans) == case.span_names
    if case.conversation_id is not None:
        assert trace.conversation_id == case.conversation_id
    call, tool_result = trace.spans[0], trace.spans[1]
    for key, value in case.call_attributes.items():
        assert call.attributes[key] == value
    for key, value in case.result_attributes.items():
        assert tool_result.attributes[key] == value
    assert tool_result.parent_span_id == call.span_id
    assert call.model is not None
    assert (call.model.provider, call.model.model_id) == case.model
    if case.usage is not None:
        assert call.usage is not None
        assert (call.usage.input_tokens, call.usage.output_tokens) == case.usage


@pytest.mark.parametrize("case", _MODEL_EVIDENCE_CASES, ids=lambda case: case.vendor)
def test_contract_keeps_model_name_without_provider(
    case: _ModelEvidenceCase, tmp_path: Path
) -> None:
    """A model named without a provider stays evidence and resolves no identity."""
    result = _LOADERS[case.vendor](_written(tmp_path, case.document(), jsonl=case.jsonl))

    span = result.traces[0].spans[0]
    assert span.model is None
    assert span.attributes["gen_ai.request.model"] == case.model_name


@pytest.mark.parametrize("case", _FAILURE_CASES, ids=lambda case: case.vendor)
def test_contract_maps_declared_errors_to_failure_outcome(
    case: _FailureCase, tmp_path: Path
) -> None:
    """A vendor-declared error becomes a span failure and a failure trace outcome."""
    result = _LOADERS[case.vendor](_written(tmp_path, case.document()))

    trace = result.traces[0]
    failure = trace.spans[case.failing_span].failure
    assert failure is not None
    if case.message is not None:
        assert failure.message == case.message
    assert trace.outcome is not None
    assert trace.outcome.status == "failure"


@pytest.mark.parametrize("case", _EXCLUSION_CASES, ids=lambda case: case.vendor)
def test_contract_excludes_records_missing_required_fields(
    case: _ExclusionCase, tmp_path: Path
) -> None:
    """A record missing a required field is excluded with explicit issues."""
    result = _LOADERS[case.vendor](_written(tmp_path, case.document()))

    assert tuple(issue.source_record for issue in result.issues) == case.issues
    assert result.traces == ()


@pytest.mark.parametrize(
    "case",
    [case for case in _PAIRING_CASES if case.vendor in _VENDOR_SOURCE_VENDORS],
    ids=lambda case: case.vendor,
)
def test_contract_load_retains_malformed_jsonl_line_exclusions(
    case: _PairingCase, tmp_path: Path
) -> None:
    """A malformed JSONL line survives load as a line exclusion beside the valid trace."""
    path = tmp_path / "export.jsonl"
    path.write_text(json.dumps(case.document()) + "\n{not json\n", encoding="utf-8")

    result = _LOADERS[case.vendor](path)

    assert len(result.traces) == 1
    assert result.traces[0].task == case.task
    assert [issue.source_record for issue in result.issues] == ["line-2"]
    assert result.issues[0].message.startswith("invalid JSONL record")


_TURNS = (
    ("Support request", "What account email?"),
    ("customer@example.test", "Reset instructions sent."),
)


def _history(turn: int) -> list[JsonValue]:
    """Return the cumulative visible message history observed by one conversation turn.

    Args:
        turn: Zero-based turn index.

    Returns:
        Messages the model saw for that turn.
    """
    messages: list[JsonValue] = []
    for request, completion in _TURNS[:turn]:
        messages.append({"role": "user", "content": request})
        messages.append({"role": "assistant", "content": completion})
    return [*messages, {"role": "user", "content": _TURNS[turn][0]}]


_HISTORY_EXPORTS: dict[str, Callable[[], object]] = {
    "braintrust": lambda: [
        {
            "id": f"row-{turn}",
            "span_id": f"span-{turn}",
            "root_span_id": "root-1",
            "span_attributes": {"type": "llm", "name": "chat"},
            "metrics": {"start": 1_772_000_000.0 + turn * 2, "end": 1_772_000_001.0 + turn * 2},
            "metadata": {"provider": "openai", "model": "gpt-test"},
            "input": _history(turn),
            "output": {"role": "assistant", "content": _TURNS[turn][1]},
        }
        for turn in range(len(_TURNS))
    ],
    "chat_json": lambda: {
        "trace_id": "conversation-1",
        "provider": "openai",
        "model": "gpt-test",
        "messages": [
            message
            for request, completion in _TURNS
            for message in (
                {"role": "user", "content": request},
                {"role": "assistant", "content": completion},
            )
        ],
    },
    "langfuse": lambda: {
        "id": "trace-1",
        "timestamp": "2026-02-01T00:00:00Z",
        "input": {"messages": [{"role": "user", "content": _TURNS[0][0]}]},
        "metadata": {"provider": "openai"},
        "observations": [
            {
                "id": f"obs-{turn}",
                "traceId": "trace-1",
                "type": "GENERATION",
                "name": "answer",
                "startTime": f"2026-02-01T00:00:0{turn * 2}Z",
                "endTime": f"2026-02-01T00:00:0{turn * 2 + 1}Z",
                "model": "gpt-test",
                "input": _history(turn),
                "output": {"role": "assistant", "content": _TURNS[turn][1]},
            }
            for turn in range(len(_TURNS))
        ],
    },
    "langsmith": lambda: [
        {
            "id": f"run-{turn}",
            "trace_id": "trace-1",
            "run_type": "llm",
            "name": "ChatOpenAI",
            "start_time": f"2026-03-01T00:00:0{turn * 2}Z",
            "end_time": f"2026-03-01T00:00:0{turn * 2 + 1}Z",
            "inputs": {"messages": _history(turn)},
            "outputs": {"generations": [[{"text": _TURNS[turn][1]}]]},
            "extra": {"metadata": {"ls_provider": "openai", "ls_model_name": "gpt-test"}},
        }
        for turn in range(len(_TURNS))
    ],
    "mastra": lambda: {
        "spans": [
            {
                "traceId": "trace-1",
                "id": f"span-{turn}",
                "type": "model_generation",
                "name": "generate",
                "startTime": f"2026-04-01T00:00:0{turn * 2}Z",
                "endTime": f"2026-04-01T00:00:0{turn * 2 + 1}Z",
                "attributes": {"provider": "openai", "model": "gpt-test"},
                "input": {"messages": _history(turn)},
                "output": {"text": _TURNS[turn][1]},
            }
            for turn in range(len(_TURNS))
        ]
    },
    "otel_genai": lambda: [
        {
            "trace_id": "9" * 32,
            "span_id": f"{turn + 1:016x}",
            "name": "agent.model_call",
            "start_time": f"2026-06-01T00:00:0{turn * 2}Z",
            "end_time": f"2026-06-01T00:00:0{turn * 2 + 1}Z",
            "attributes": {
                "gen_ai.operation.name": "chat",
                "gen_ai.provider.name": "openai",
                "gen_ai.request.model": "gpt-test",
                "gen_ai.input.messages": json.dumps(_history(turn)),
                "gen_ai.output.messages": json.dumps(
                    [{"role": "assistant", "content": _TURNS[turn][1]}]
                ),
            },
        }
        for turn in range(len(_TURNS))
    ],
    "phoenix": lambda: [
        {
            "context": {"trace_id": "trace-1", "span_id": f"span-{turn}"},
            "name": "ChatCompletion",
            "start_time": f"2026-05-01T00:00:0{turn * 2}Z",
            "end_time": f"2026-05-01T00:00:0{turn * 2 + 1}Z",
            "attributes": {
                "openinference": {"span": {"kind": "LLM"}},
                "llm": {
                    "provider": "openai",
                    "model_name": "gpt-test",
                    "input_messages": [{"message": message} for message in _history(turn)],
                    "output_messages": [
                        {"message": {"role": "assistant", "content": _TURNS[turn][1]}}
                    ],
                },
            },
        }
        for turn in range(len(_TURNS))
    ],
}


def _span_completion(span: TraceSpan) -> str:
    """Read the assistant completion text one canonical model span carries.

    Args:
        span: Canonical model-call span.

    Returns:
        Completion text from ``gen_ai.completion``, or from the retained
        ``gen_ai.output.messages`` evidence when the loader keeps raw messages.
    """
    completion = span.attributes.get("gen_ai.completion")
    if isinstance(completion, str):
        return completion
    output = span.attributes["gen_ai.output.messages"]
    assert isinstance(output, str)
    messages = json.loads(output)
    content = messages[0]["content"]
    assert isinstance(content, str)
    return content


@pytest.mark.parametrize("vendor", sorted(_HISTORY_EXPORTS))
def test_contract_normalizes_two_turns_with_cumulative_history(vendor: str, tmp_path: Path) -> None:
    """A second turn carrying the cumulative visible history stays one two-span trace."""
    result = _LOADERS[vendor](_written(tmp_path, _HISTORY_EXPORTS[vendor]()))

    assert result.issues == ()
    assert len(result.traces) == 1
    trace = result.traces[0]
    assert trace.task == _TURNS[0][0]
    assert tuple(span.name for span in trace.spans) == ("agent.model_call", "agent.model_call")
    assert tuple(_span_completion(span) for span in trace.spans) == tuple(
        completion for _, completion in _TURNS
    )
