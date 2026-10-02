"""Format recognition preserves explicit selection for ambiguous or unknown exports."""

import json
from pathlib import Path

import pytest
from pydantic import JsonValue

from exp.common.traces.ingest.detection import detect_trace_source
from exp.common.traces.ingest.phoenix import PHOENIX_SOURCE


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"messages": [{"role": "user", "content": "Research a company"}]}, "chat-json"),
        ([{"role": "user", "content": "Research a company"}], "chat-json"),
        ({"conversations": [{"messages": []}]}, "chat-json"),
        ({"resourceSpans": [{"scopeSpans": [{"spans": []}]}]}, None),
        ([{"traceId": "a", "spanId": "b", "attributes": []}], None),
        (
            {"trace_id": "a", "span_id": "b", "attributes": {"gen_ai.operation.name": "chat"}},
            "otel-genai",
        ),
        (
            {"captures": [{"request": {"protocol": "chat_completions"}, "response": {}}]},
            "experiential",
        ),
        ({"unknown": [{"messages": []}]}, None),
        ({"data": []}, None),
        ([{"messages": []}, {"resourceSpans": []}], None),
    ],
)
def test_structural_detection(tmp_path: Path, payload: JsonValue, expected: str | None) -> None:
    """Supported envelopes are recognized while mixed or unknown formats need selection."""
    path = tmp_path / "export.json"
    path.write_text(json.dumps(payload))
    assert detect_trace_source(path) == expected


def test_openinference_envelope_requires_format_selection(tmp_path: Path) -> None:
    """A valid Phoenix export must not be sent to the incompatible GenAI OTLP loader."""
    attributes = {
        "openinference.span.kind": "LLM",
        "llm.provider": "openai",
        "llm.model_name": "gpt-test",
        "input.value": "Research a company",
        "output.value": "Company found",
    }
    span = {
        "traceId": "a" * 32,
        "spanId": "b" * 16,
        "name": "ChatCompletion",
        "startTimeUnixNano": "1778000000000000000",
        "endTimeUnixNano": "1778000001000000000",
        "attributes": [
            {"key": key, "value": {"stringValue": value}} for key, value in attributes.items()
        ],
    }
    path = tmp_path / "phoenix.json"
    path.write_text(json.dumps({"resourceSpans": [{"scopeSpans": [{"spans": [span]}]}]}))
    assert len(PHOENIX_SOURCE.load(path).traces) == 1
    assert detect_trace_source(path) is None


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("mixed", [False, True])
def test_otlp_requires_unambiguous_genai_markers(
    tmp_path: Path, wrapped: bool, mixed: bool
) -> None:
    """Direct and enveloped GenAI spans detect as OTLP unless another convention is present."""
    keys = ["gen_ai.operation.name"]
    if mixed:
        keys.append("openinference.span.kind")
    span: JsonValue = {
        "traceId": "a",
        "spanId": "b",
        "attributes": [{"key": key, "value": {"stringValue": "chat"}} for key in keys],
    }
    payload = {"resourceSpans": [{"scopeSpans": [{"spans": [span]}]}]} if wrapped else span
    path = tmp_path / "telemetry.json"
    path.write_text(json.dumps(payload))
    assert detect_trace_source(path) == (None if mixed else "otlp")


def test_jsonl_is_streamed_and_tool_payloads_do_not_change_the_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nested tool-result data is never interpreted as the surrounding export's format."""
    path = tmp_path / "export.jsonl"
    message = {"role": "tool", "content": {"resourceSpans": [], "messages": []}}
    path.write_text((json.dumps({"messages": [message]}) + "\n") * 1_000)

    def unexpected(*_args: object, **_kwargs: object) -> None:
        """Reject loading the complete export through a Path materialization helper."""
        raise AssertionError("format detection must stream the export")

    monkeypatch.setattr(Path, "read_text", unexpected)
    monkeypatch.setattr(Path, "read_bytes", unexpected)
    assert detect_trace_source(path) == "chat-json"


@pytest.mark.parametrize("data", [b"", b"{", b'{"messages": []}\nnot json', b"\xff"])
def test_malformed_data_needs_explicit_selection(tmp_path: Path, data: bytes) -> None:
    """A partial recognizable prefix never causes silent format selection."""
    path = tmp_path / "export.jsonl"
    path.write_bytes(data)
    assert detect_trace_source(path) is None


def test_unreadable_path_is_reported(tmp_path: Path) -> None:
    """A missing source stays an I/O error rather than being reported as an unknown format."""
    with pytest.raises(OSError):
        detect_trace_source(tmp_path / "missing.jsonl")
