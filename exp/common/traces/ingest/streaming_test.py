"""Streaming normalization matches canonical source evidence across archive shapes."""

import json
import tracemalloc
from pathlib import Path

import pytest

from exp.common.traces.ingest.braintrust_test import _rows
from exp.common.traces.ingest.langfuse_test import _trace as _langfuse
from exp.common.traces.ingest.langsmith_test import _runs
from exp.common.traces.ingest.mastra_test import _spans as _mastra
from exp.common.traces.ingest.otel_genai_test import _spans as _otel
from exp.common.traces.ingest.otlp_test import (
    _environment_capture_payloads,
    _payload,
    _payload_spans,
)
from exp.common.traces.ingest.persistence import ingest_traces, read_ingested_traces
from exp.common.traces.ingest.phoenix_test import _native_spans
from exp.common.traces.ingest.posthog_canonical_test import _posthog_events
from exp.common.traces.ingest.sources import load_trace_source


@pytest.mark.parametrize(
    "source,payload",
    [
        ("braintrust", _rows()),
        ("langfuse", [_langfuse()]),
        ("langsmith", _runs()),
        ("mastra", _mastra()),
        ("phoenix", _native_spans()),
        ("otel-genai", _otel()),
        ("otlp", [_payload()]),
        ("posthog", _posthog_events()),
        (
            "chat-json",
            [
                {
                    "messages": [
                        {"role": "user", "content": "Research"},
                        {"role": "assistant", "content": "Done"},
                    ]
                }
            ],
        ),
    ],
)
@pytest.mark.parametrize("jsonl", [False, True])
def test_streaming_matches_source_loader(
    tmp_path: Path, source: str, payload: list[object], jsonl: bool
) -> None:
    """Array and JSONL imports retain exact canonical content, ordering and provenance."""
    path = tmp_path / "traces.jsonl"
    path.write_text(
        "\n".join(json.dumps(item) for item in payload) if jsonl else json.dumps(payload)
    )
    expected = load_trace_source(source, path)
    summary, receipt = ingest_traces(
        "powerset", root=tmp_path / "state", source_format=source, path=path
    )
    assert receipt is not None
    assert summary.trace_count == len(expected.traces)
    assert read_ingested_traces(tmp_path / "state", receipt.import_id) == expected


@pytest.mark.parametrize("mutation", ["none", "bad-line", "envelope", "duplicate-key"])
def test_environment_profile_preserves_strict_admission(tmp_path: Path, mutation: str) -> None:
    """Only complete unambiguous direct-span JSONL enters the environment profile."""
    path = tmp_path / "trace.jsonl"
    payloads = _environment_capture_payloads()
    lines = [json.dumps(item) for item in payloads]
    if mutation == "bad-line":
        lines.append("{bad")
    elif mutation == "envelope":
        lines = [
            json.dumps({"resourceSpans": [{"scopeSpans": [{"spans": [item]}]}]})
            for item in payloads
        ]
    elif mutation == "duplicate-key":
        lines[0] = '{"name":"duplicate",' + lines[0][1:]
    path.write_text("\n".join(lines))
    expected = load_trace_source("otlp", path)
    _, receipt = ingest_traces("powerset", root=tmp_path / "state", source_format="otlp", path=path)
    assert receipt is not None
    assert read_ingested_traces(tmp_path / "state", receipt.import_id) == expected


def test_payload_memory_does_not_grow_with_archive_size(tmp_path: Path) -> None:
    """An eightfold payload increase stays within a bounded working set, with no trace cap."""
    peaks: list[int] = []
    for count in (8, 64):
        path = tmp_path / f"source-{count}.json"
        with path.open("w") as stream:
            stream.write("[")
            for index in range(count):
                if index:
                    stream.write(",")
                json.dump(
                    {
                        "id": str(index),
                        "messages": [
                            {"role": "user", "content": "x" * (128 * 1024)},
                            {"role": "assistant", "content": "done"},
                        ],
                    },
                    stream,
                )
            stream.write("]")
        tracemalloc.start()
        try:
            summary, receipt = ingest_traces(
                "powerset", root=tmp_path / f"state-{count}", source_format="chat-json", path=path
            )
            peaks.append(tracemalloc.get_traced_memory()[1])
        finally:
            tracemalloc.stop()
        assert receipt is not None and summary.trace_count == count
    assert peaks[1] < peaks[0] * 3, peaks


def test_interleaved_otlp_records_are_grouped_before_normalization(tmp_path: Path) -> None:
    """Records separated by other traces still form complete causal tool trajectories."""
    left = _payload("1" * 32)
    right = _payload("2" * 32)
    payloads = [
        item
        for pair in zip(_payload_spans(left), _payload_spans(right), strict=True)
        for item in pair
    ]
    path = tmp_path / "interleaved.jsonl"
    path.write_text("\n".join(json.dumps(payload) for payload in payloads))
    expected = load_trace_source("otlp", path)
    assert len(expected.traces) == 2 and not expected.issues
    _, receipt = ingest_traces("powerset", root=tmp_path / "state", source_format="otlp", path=path)
    assert receipt is not None
    assert read_ingested_traces(tmp_path / "state", receipt.import_id) == expected
